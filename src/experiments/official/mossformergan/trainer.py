"""Official MossFormerGAN loop. Both backwards finish before either AdamW steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data, check_publication
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.mossformergan.boundary import finish_boundary, open_boundary
from src.experiments.official.mossformergan.checks import (
    OFFICIAL_CLIP,
    OFFICIAL_DISCRIMINATOR_LEARNING_RATE,
    OFFICIAL_FAITHFUL,
    OFFICIAL_LEARNING_RATE,
    CLAIMS,
    positive_float,
    positive_int,
    require_adamw,
    refuse_amp,
    require_disjoint,
    require_group_size,
)
from src.experiments.official.mossformergan.microbatch import MicroBatch, PairGuard, open_micro_batch, share_window
from src.experiments.official.mossformergan.plateau import apply_plateau
from src.experiments.official.mossformergan.models import (
    MossFormerGANDiscriminator,
    MossFormerGANGenerator,
    require_official_architecture,
)
from src.experiments.official.mossformergan.recording import finish_run, note_micro_batch
from src.experiments.official.mossformergan.state import (
    MossProgress,
    ProgressView,
    check_grads,
    grad_state,
    write_grads,
)
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.stopping import should_stop
from src.experiments.trainers import _close_epoch_view
from src.experiments.training.control import stop_requested
from src.experiments.official.mossformergan.validation import validation_proxy
from src.experiments.training.official_objectives import require_mossformer

class MossFormerGANTrainer(ProgressView):
    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        generator: PreparedModel,
        discriminator: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        *,
        generator_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        max_logical_steps: int,
        accumulation: int = 2,
        grad_clip: float = OFFICIAL_CLIP,
        experiment_claim: str = OFFICIAL_FAITHFUL,
        validate_on_epoch_end: bool = True,
        max_epochs: int | None = None,
        stop_after_logical_steps: int | None = None,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        require_prepared(generator, MossFormerGANGenerator, "generator must be a prepared MossFormerGANGenerator")
        require_prepared(discriminator, MossFormerGANDiscriminator, "discriminator must be a prepared MossFormerGANDiscriminator")
        if generator.unwrapped().architecture != discriminator.unwrapped().architecture:
            raise BuildError("discriminator initialization architecture does not match the generator")
        if generator.unwrapped().seed != discriminator.unwrapped().seed:
            raise BuildError("generator and discriminator seeds differ")
        require_adamw(optim_g, "optim_g", lr=OFFICIAL_LEARNING_RATE)
        require_adamw(optim_d, "optim_d", lr=OFFICIAL_DISCRIMINATOR_LEARNING_RATE)
        require_disjoint(optim_g, optim_d)
        if experiment_claim not in CLAIMS:
            raise BuildError("experiment_claim must be official_faithful, leakage_safe, or research")
        require_official_architecture(generator.unwrapped().architecture, experiment_claim)
        require_mossformer(generator_objective, discriminator_objective, experiment_claim)
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        optim_g_id = getattr(optim_g, "_experiment_component_id", None)
        optim_d_id = getattr(optim_d, "_experiment_component_id", None)
        if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
            raise BuildError("optimizers have no component id")
        if not isinstance(validate_on_epoch_end, bool):
            raise BuildError("validate_on_epoch_end must be a boolean")
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.optim_g_id = optim_g_id
        self.optim_d_id = optim_d_id
        self.experiment_claim = experiment_claim
        self.max_logical_steps = positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = positive_int("accumulation", accumulation)
        self.grad_clip = positive_float("grad_clip", grad_clip)
        self.validate_on_epoch_end = validate_on_epoch_end
        self.max_epochs = None if max_epochs is None else positive_int("max_epochs", max_epochs)
        self.stop_after_logical_steps = (
            None if stop_after_logical_steps is None else positive_int("stop_after_logical_steps", stop_after_logical_steps)
        )
        self.n_fft = int(generator.unwrapped().n_fft)
        self.win = int(generator.unwrapped().win)
        self.hop = int(generator.unwrapped().hop)
        self.sample_rate = int(generator.unwrapped().sampling_rate)
        self.audio_sample_rate = self.sample_rate
        self.audio_teachers = ("dry_clean",)
        self.progress = MossProgress.initial(optim_g_id, optim_d_id, experiment_claim)
        self._guard = PairGuard()
        self._micro = MicroBatch(
            generator=generator, discriminator=discriminator, optim_g=optim_g, optim_d=optim_d,
            generator_objective=generator_objective, discriminator_objective=discriminator_objective,
            progress=self.progress, guard=self._guard, accumulation=self.accumulation, grad_clip=self.grad_clip,
            n_fft=self.n_fft, hop=self.hop, win=self.win, sample_rate=self.sample_rate,
        )
        self._restored = False
        self._skip_path: Path | None = None
        self.fail_phase: str | None = None
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators, model_selection, data=self.data, module=self.generator,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection

    def fit(self) -> RunResult:
        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        while True:
            if self.progress.phase == "pending_boundary":
                self._finish_boundary(runtime, component_id)
                if should_stop(self) or self.progress.stop_training:
                    return self._result("completed")
                continue
            if self.progress.logical_step >= self.max_logical_steps:
                return self._result("completed")
            if self._stop_requested(runtime):
                self._publish_interrupt(runtime)
                return self._result("interrupted")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            if self.max_epochs is not None and epoch >= self.max_epochs and index == 0:
                self.progress.halt_reason = "max_epochs"
                return self._result("completed")
            opened = open_micro_batch(
                self.data, runtime, self.progress, accumulation=self.accumulation, component_id=component_id,
                epoch=epoch, index=index, open_boundary=lambda: self._open_boundary(runtime, epoch),
            )
            if opened is None:
                continue
            view, batch, updates = opened
            try:
                stepped = self.consume_micro_batch(batch, runtime, update_index=index)
                index += 1
                self.data.epoch = epoch
                self.data.update_index = index
                note_micro_batch(
                    self.progress, runtime, self.journal, stepped, epoch=epoch, research=self.research,
                    lifecycle=self.lifecycle, training_control=getattr(self, "training_control", None),
                    offer=lambda event, metrics=None: self._offer(runtime, event, metrics),
                )
                if index == updates:
                    self._open_boundary(runtime, epoch)
            finally:
                _close_epoch_view(view)

    def consume_micro_batch(self, batch: Batch, runtime: ProcessRuntime, *, update_index: int) -> dict[str, Any] | None:
        """Run one micro-batch. Return the shared log when both optimizers stepped."""

        return self._micro.consume(
            batch, runtime, update_index=update_index, phase_hook=self._on_update_phase,
            step_pair=self._step_pair, journal=getattr(self, "journal", None), skip_path=self._skip_path,
        )

    def _step_pair(self, runtime: ProcessRuntime) -> dict[str, Any]:
        return self._micro._step_pair(runtime, self._on_update_phase)

    def _terms_view(self) -> Any:
        return self._micro._terms_view()

    def _validation_proxy(self, runtime: ProcessRuntime) -> Any:
        return validation_proxy(
            runtime, self.generator, self.discriminator,
            n_fft=self.n_fft, hop=self.hop, win=self.win, sample_rate=self.sample_rate,
        )

    def _apply_plateau(self, summary: dict[str, Any]) -> str:
        return apply_plateau(
            self.progress, summary, generator=self.generator, discriminator=self.discriminator,
            optim_g=self.optim_g, optim_d=self.optim_d,
        )

    def _check_claim(self) -> None:
        check_publication(self)

    def state_dict(self) -> dict[str, Any]:
        return self.progress.export(grad_state(self.generator), grad_state(self.discriminator))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        decoded = self.progress.decode(state, self.accumulation)
        generator_grads = check_grads(self.generator, decoded.generator_grads)
        discriminator_grads = check_grads(self.discriminator, decoded.discriminator_grads)
        self.progress.apply(decoded)
        write_grads(self.generator, generator_grads)
        write_grads(self.discriminator, discriminator_grads)
        self._restored = True

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        self.research.rewind(
            self.lifecycle.output_directory, restored=self._restored, phase=self.progress.phase,
            progress={"logical_step": int(self.progress.logical_step)},
        )
        refuse_amp(runtime)
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        self.journal.begin(
            restored=self._restored, logical_step=self.progress.logical_step,
            expect_rows=len(self.progress.step_log) if self._restored else None,
        )
        self._skip_path = self.journal.begin_sidecar(
            "skip_log.jsonl", restored=self._restored,
            keep_lines=len(self.progress.skip_log) if self._restored else None,
        )
        if not self._restored:
            self.progress.reset_fresh()
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        require_group_size(self.data, self.accumulation, str(component_id))
        runtime.barrier()
        return str(component_id)

    def _open_boundary(self, runtime: ProcessRuntime, epoch: int) -> None:
        open_boundary(self.progress, epoch=epoch, validate=self.validate_on_epoch_end)
        self._offer(runtime, "boundary_pending")

    def _finish_boundary(self, runtime: ProcessRuntime, component_id: str) -> None:
        finish_boundary(
            self.progress, runtime, self.data, self._micro, component_id=component_id,
            max_logical_steps=self.max_logical_steps, n_fft=self.n_fft, hop=self.hop, win=self.win,
            sample_rate=self.sample_rate, offer=lambda event, metrics=None: self._offer(runtime, event, metrics),
            research=self.research, lifecycle=self.lifecycle,
            training_control=getattr(self, "training_control", None),
        )

    def _on_update_phase(self, phase: str) -> None:
        if self.fail_phase == phase:
            raise RuntimeFailure(f"injected failure at {phase}")

    def _refuse_amp(self, runtime: ProcessRuntime) -> None:
        refuse_amp(runtime)

    def _publish_interrupt(self, runtime: ProcessRuntime) -> None:
        if self._guard.active:
            raise RuntimeFailure("refusing to checkpoint between the generator step and the discriminator step")
        if self.progress.logical_step <= 0 and self.progress.accu_count <= 0:
            return
        self._offer(runtime, "stop")

    def _offer(self, runtime: ProcessRuntime, event: str, metrics: dict[str, float] | None = None) -> None:
        if self._guard.active:
            raise RuntimeFailure("refusing to checkpoint between the generator step and the discriminator step")
        if event in ("step", "partial"):
            self.lifecycle.save_if_needed(
                event, logical_step=self.progress.logical_step,
                metrics=metrics if event == "step" else None, prepare=self._sync_ranks,
            )
            return
        self._sync_ranks(runtime)
        self.lifecycle.save_if_needed(event, logical_step=self.progress.logical_step, metrics=metrics)

    def _sync_ranks(self, runtime: ProcessRuntime) -> None:
        share_window(runtime, self.generator, self.discriminator, self.progress)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        return stop_requested(self, runtime)

    def _result(self, reason: str) -> RunResult:
        return finish_run(
            reason, research=self.research, runtime=self.runtime, lifecycle=self.lifecycle, data=self.data,
            progress=self.progress, training_control=getattr(self, "training_control", None),
            prepare=self._sync_ranks,
        )
