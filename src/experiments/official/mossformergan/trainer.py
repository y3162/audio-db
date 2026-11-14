"""Official MossFormerGAN loop for the experiment framework.

The update order is the accumulation branch of the released solver. One
successful micro-batch forwards the generator, scores the fake pair for the
generator loss, evaluates PESQ, then forwards the discriminator on detached
fakes and reals. Both losses backward before either AdamW steps. PESQ and
non-finite losses skip the micro-batch. The data cursor still advances.
Later successes keep filling the same window.

Gradient clipping runs after each successful backward, not once at the end
of the window. On more than one rank, only the new gradient is averaged.
``MPSENetGANTrainer`` is not used: its discriminator steps before the
generator backward.

Validation follows the data epoch. The discriminator stays in train mode.
The plateau is ``mossformer_validation_decision``, not ``ReduceLROnPlateau``.
``amp_fp16`` is refused: each micro-batch clips, and the two AdamW updates
are not one scaler decision.
"""

from __future__ import annotations

from src.experiments.stopping import observe_validation, should_stop

import copy
from pathlib import Path
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data, check_publication
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mossformergan.models import (
    MossFormerGANDiscriminator,
    MossFormerGANGenerator,
    require_official_architecture,
)
from src.experiments.execution.journal import StepJournal
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.selectors import scalar_metrics
from src.experiments.trainers import _close_epoch_view
from src.experiments.calc.errors import TrainingError
from src.experiments.official.mossformergan import reference as mossformer_lib
from src.experiments.official.mossformergan.reference import (
    OFFICIAL_ADAMW_WEIGHT_DECAY,
    forward_spectra,
    generator_parts,
    generator_terms,
    mossformer_validation_decision,
    reduce_validation,
    validation_report,
)
from src.experiments.training.objectives import call_objective
from src.experiments.training.official_objectives import require_mossformer

_IN_EPOCH = "in_epoch"
_PENDING = "pending_boundary"
OFFICIAL_FAITHFUL = "official_faithful"
LEAKAGE_SAFE = "leakage_safe"
_CLAIMS = frozenset({OFFICIAL_FAITHFUL, LEAKAGE_SAFE, "research"})
OFFICIAL_LEARNING_RATE = 0.0005
OFFICIAL_DISCRIMINATOR_LEARNING_RATE = 0.001
OFFICIAL_BETAS = (0.9, 0.999)
OFFICIAL_CLIP = 10.0
OFFICIAL_CHECKPOINT_INTERVAL = 5786
_COUNTER_KEYS = (
    "micro_batches_processed",
    "micro_batches_succeeded",
    "micro_batches_failed",
    "optimizer_updates",
    "official_step",
)


class MossFormerGANTrainer:
    """Generator-then-discriminator step after both backwards of a full window."""

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
        require_prepared(
            discriminator,
            MossFormerGANDiscriminator,
            "discriminator must be a prepared MossFormerGANDiscriminator",
        )
        if generator.unwrapped().architecture != discriminator.unwrapped().architecture:
            raise BuildError("discriminator initialization architecture does not match the generator")
        if generator.unwrapped().seed != discriminator.unwrapped().seed:
            raise BuildError("generator and discriminator seeds differ")
        _require_adamw(optim_g, "optim_g", lr=OFFICIAL_LEARNING_RATE)
        _require_adamw(optim_d, "optim_d", lr=OFFICIAL_DISCRIMINATOR_LEARNING_RATE)
        _require_disjoint(optim_g, optim_d)
        if experiment_claim not in _CLAIMS:
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
        self.generator_objective = generator_objective
        self.discriminator_objective = discriminator_objective
        self.optim_g_id = optim_g_id
        self.optim_d_id = optim_d_id
        self.experiment_claim = experiment_claim
        self.max_logical_steps = _positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = _positive_int("accumulation", accumulation)
        self.grad_clip = _positive_float("grad_clip", grad_clip)
        self.validate_on_epoch_end = validate_on_epoch_end
        self.max_epochs = None if max_epochs is None else _positive_int("max_epochs", max_epochs)
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = _positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self.n_fft = int(generator.unwrapped().n_fft)
        self.win = int(generator.unwrapped().win)
        self.hop = int(generator.unwrapped().hop)
        self.sample_rate = int(generator.unwrapped().sampling_rate)
        self.audio_sample_rate = self.sample_rate
        self.audio_teachers = ("dry_clean",)
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.counters = _empty_counters()
        self._accu_count = 0
        self.skip_log: list[dict[str, Any]] = []
        self.last_skip: dict[str, Any] | None = None
        self.step_log: list[dict[str, Any]] = []
        self.completed_boundaries: list[dict[str, Any]] = []
        self.boundary: dict[str, Any] | None = None
        self.phase = _IN_EPOCH
        self._best_val = float("inf")
        self._stalls = 0
        self._best_snapshot: dict[str, Any] | None = None
        self.stop_training = False
        self.halt_reason = "initialized"
        self.plateau_history: list[dict[str, Any]] = []
        self._epoch_success_start = 0
        self._epoch_update_start = 0
        self._epochs_without_step = 0
        self._window_generator: list[float] = []
        self._window_discriminator: list[float] = []
        self._window_samples: list[str] = []
        self._restored = False
        self._in_optimizer_pair = False
        self._skip_path: Path | None = None
        self.fail_phase: str | None = None
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators,
            model_selection,
            data=self.data,
            module=self.generator,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection
        self._average_calls = 0
        self.last_components: dict[str, float] | None = None

    def fit(self) -> RunResult:
        """Train until the step budget, the plateau stop, or a finite all-skip epoch."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        while True:
            if self.phase == _PENDING:
                self._finish_boundary(runtime, component_id)
                if should_stop(self):
                    return self._result("completed")
                if self.stop_training:
                    return self._result("completed")
                continue
            if self.logical_step >= self.max_logical_steps:
                return self._result("completed")
            if self._stop_requested(runtime):
                self._publish_interrupt(runtime)
                return self._result("interrupted")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            if self.max_epochs is not None and epoch >= self.max_epochs and index == 0:
                self.halt_reason = "max_epochs"
                return self._result("completed")
            opened = self._open_micro_batch(runtime, component_id, epoch, index)
            if opened is None:
                continue
            view, batch, updates = opened
            try:
                stepped = self.consume_micro_batch(batch, runtime, update_index=index)
                index += 1
                self.data.epoch = epoch
                self.data.update_index = index
                self._note_micro_batch(runtime, stepped, epoch)
                if index == updates:
                    self._open_boundary(runtime, epoch)
            finally:
                _close_epoch_view(view)

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        """Rewind metrics and the skip sidecar. Does not run an optimizer pair."""

        self.research.rewind(
            self.lifecycle.output_directory,
            restored=self._restored,
            phase=self.phase,
            progress={"logical_step": int(self.logical_step)},
        )
        self._refuse_amp(runtime)
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        self.journal.begin(
            restored=self._restored,
            logical_step=self.logical_step,
            expect_rows=len(self.step_log) if self._restored else None,
        )
        self._skip_path = self.journal.begin_sidecar(
            "skip_log.jsonl",
            restored=self._restored,
            keep_lines=len(self.skip_log) if self._restored else None,
        )
        if not self._restored:
            self.logical_step = 0
            self.optimizer_updates = {self.optim_g_id: 0, self.optim_d_id: 0}
            self.counters = _empty_counters()
            self._accu_count = 0
            self.skip_log = []
            self.last_skip = None
            self.step_log = []
            self.completed_boundaries = []
            self.boundary = None
            self.phase = _IN_EPOCH
            self._best_val = float("inf")
            self._stalls = 0
            self._best_snapshot = None
            self.stop_training = False
            self.halt_reason = "running"
            self.plateau_history = []
            self._epoch_success_start = 0
            self._epoch_update_start = 0
            self._epochs_without_step = 0
            self._clear_window_log()
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        self._require_group_size(component_id)
        runtime.barrier()
        return str(component_id)

    def _open_micro_batch(
        self,
        runtime: ProcessRuntime,
        component_id: str,
        epoch: int,
        index: int,
    ) -> tuple[Any, Batch, int] | None:
        """Return one micro-batch, or ``None`` after opening an exhausted-epoch boundary."""

        view = self.data.train_epoch(epoch, start_update=index)
        try:
            updates = int(view.updates)
            self._require_same_updates(runtime, updates, component_id)
            if updates < 1:
                raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
            if updates % self.accumulation != 0:
                raise RuntimeFailure(
                    f"component {component_id}: epoch micro-batches {updates} "
                    f"are not divisible by accumulation {self.accumulation}"
                )
            if index > updates:
                raise RuntimeFailure(f"component {component_id}: update {index} is past epoch {epoch}")
            if index == updates:
                self._open_boundary(runtime, epoch)
                _close_epoch_view(view)
                return None
            if index == 0:
                self._epoch_success_start = int(self.counters["micro_batches_succeeded"])
                self._epoch_update_start = int(self.counters["optimizer_updates"])
            batches = view.batch(index)
            if len(batches) != 1 or not isinstance(batches[0], Batch):
                raise RuntimeFailure(f"component {component_id}: expected one micro-batch")
        except Exception:
            _close_epoch_view(view)
            raise
        return view, batches[0], updates

    def _note_micro_batch(
        self,
        runtime: ProcessRuntime,
        stepped: dict[str, Any] | None,
        epoch: int,
    ) -> None:
        """Record a finished pair, or request a partial save when the window did not step."""

        if stepped is None:
            self._offer(runtime, "partial")
            return
        self.logical_step += 1
        self.step_log.append(stepped)
        self.journal.write_metrics(
            {
                "discriminator": stepped["discriminator"],
                "generator": stepped["generator"],
                "logical_step": stepped["logical_step"],
                "loss": stepped["loss"],
                "sample_id": stepped["sample_id"],
            }
        )
        self.journal.write_sample(
            {
                "logical_step": self.logical_step,
                "loss": stepped["loss"],
                "sample_id": stepped["sample_id"],
            }
        )
        step_metrics = self.research.score(
            runtime,
            output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity,
            kind="step",
            epoch=epoch,
            logical_step=self.logical_step,
            stopping=getattr(self, "training_control", None),
        )
        if step_metrics:
            self._offer(runtime, "step", metrics=step_metrics)
        else:
            self._offer(runtime, "step")

    def consume_micro_batch(
        self,
        batch: Batch,
        runtime: ProcessRuntime,
        *,
        update_index: int,
    ) -> dict[str, Any] | None:
        """Run one micro-batch. Return the shared log when both optimizers stepped."""

        if self._in_optimizer_pair:
            raise RuntimeFailure("a micro-batch cannot start between the two optimizer steps")
        self._train_mode()
        clean, noisy = self._waveforms(batch, runtime)
        self.counters["micro_batches_processed"] += 1
        self.counters["official_step"] += 1
        generator_module = self.generator.execution()
        discriminator_module = self.discriminator.execution()
        with runtime.sync_gradients([self.generator, self.discriminator], enabled=False):
            parts = generator_parts(self._terms_view(), generator_module, discriminator_module, clean, noisy)
            result = call_objective(
                self.generator_objective,
                {
                    "audio": parts["audio"],
                    "estimate": parts["audio"],
                    "gan": parts["gan"],
                    "label_mag": parts["label_mag"],
                    "magnitude": parts["magnitude"],
                    "pred_mag": parts["pred_mag"],
                    "ri": parts["ri"],
                    "target": parts["teacher"],
                    "teacher": parts["teacher"],
                    "time": parts["time"],
                },
            )
            generator_loss = result.total
            reason = None
            targets = None
            if not bool(torch.isfinite(generator_loss.detach()).all()):
                reason = "non-finite generator loss"
            else:
                targets = mossformer_lib.metric_targets(parts["teacher"], parts["audio"], self.sample_rate)
                if targets is None:
                    reason = "pesq"
            if self._skip(runtime, reason, batch, update_index):
                return None
            if targets is None:
                raise RuntimeFailure("PESQ targets are missing on a micro-batch that was not skipped")
            ones = torch.ones(clean.shape[0], device=clean.device)
            fake_detached = discriminator_module(parts["label_mag"], parts["pred_mag"].detach())
            real_score = discriminator_module(parts["label_mag"], parts["label_mag"])
            discriminator_loss = call_objective(
                self.discriminator_objective,
                {
                    "fake_score": fake_detached,
                    "ones": ones,
                    "pesq_target": targets,
                    "real_score": real_score,
                },
            ).total
            disc_reason = None if bool(torch.isfinite(discriminator_loss.detach()).all()) else "non-finite discriminator loss"
            if self._skip(runtime, disc_reason, batch, update_index):
                return None
            self.counters["micro_batches_succeeded"] += 1
            self._accu_count += 1
            self.last_components = {
                "discriminator": float(discriminator_loss.detach()),
                "generator": float(generator_loss.detach()),
            }
            for name, value in result.terms.items():
                self.last_components[name] = float(value.detach())
            self._on_update_phase("before_generator_backward")
            self._backward_average_and_clip(runtime, generator_loss / self.accumulation, self.generator)
            self._on_update_phase("after_generator_clip")
            self._on_update_phase("before_discriminator_backward")
            self._backward_average_and_clip(
                runtime, discriminator_loss / self.accumulation, self.discriminator
            )
            self._on_update_phase("after_discriminator_clip")
        self._window_generator.append(float(generator_loss.detach()))
        self._window_discriminator.append(float(discriminator_loss.detach()))
        self._window_samples.append(str(batch.sample_id))
        if self._accu_count != self.accumulation:
            self._on_update_phase("after_micro_batch")
            return None
        return self._step_pair(runtime)

    def state_dict(self) -> dict[str, Any]:
        return {
            "accu_count": int(self._accu_count),
            "best_snapshot": self._best_snapshot,
            "best_val": self._best_val,
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "counters": dict(self.counters),
            "discriminator_grads": self._grad_state(self.discriminator) if self._accu_count else {},
            "epoch_success_start": int(self._epoch_success_start),
            "epoch_update_start": int(self._epoch_update_start),
            "epochs_without_step": int(self._epochs_without_step),
            "experiment_claim": self.experiment_claim,
            "generator_grads": self._grad_state(self.generator) if self._accu_count else {},
            "halt_reason": self.halt_reason,
            "logical_step": int(self.logical_step),
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "plateau_history": list(self.plateau_history),
            "skip_log": list(self.skip_log),
            "stalls": int(self._stalls),
            "step_log": list(self.step_log),
            "stop_training": bool(self.stop_training),
            "window_discriminator": list(self._window_discriminator),
            "window_generator": list(self._window_generator),
            "window_samples": list(self._window_samples),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        logical_step = state.get("logical_step")
        updates = state.get("optimizer_updates")
        counters = state.get("counters")
        phase = state.get("phase")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
            raise RuntimeFailure("trainer state is invalid")
        if not isinstance(counters, dict) or phase not in {_IN_EPOCH, _PENDING}:
            raise RuntimeFailure("trainer counters or phase are invalid")
        if state.get("experiment_claim") != self.experiment_claim:
            raise RuntimeFailure("experiment claim does not match this trainer")
        if state.get("stop_training") not in {True, False}:
            raise RuntimeFailure("trainer stop flag is invalid")
        self.logical_step = logical_step
        self.optimizer_updates = {str(key): int(value) for key, value in updates.items()}
        self.counters = _empty_counters()
        for key in _COUNTER_KEYS:
            self.counters[key] = int(counters.get(key) or 0)
        accu = state.get("accu_count")
        if isinstance(accu, bool) or not isinstance(accu, int) or accu < 0 or accu >= self.accumulation:
            raise RuntimeFailure("accumulator count is invalid")
        self._accu_count = accu
        self._restore_grads(self.generator, state.get("generator_grads") or {})
        self._restore_grads(self.discriminator, state.get("discriminator_grads") or {})
        if self._accu_count == 0 and (state.get("generator_grads") or state.get("discriminator_grads")):
            raise RuntimeFailure("a closed window cannot carry gradients")
        self.phase = phase
        boundary = state.get("boundary")
        self.boundary = None if boundary is None else dict(boundary)
        history = state.get("plateau_history")
        completed = state.get("completed_boundaries")
        step_log = state.get("step_log")
        skip_log = state.get("skip_log")
        if not all(isinstance(item, list) for item in (history, completed, step_log, skip_log)):
            raise RuntimeFailure("trainer history is invalid")
        self.plateau_history = list(history)
        self.completed_boundaries = list(completed)
        self.step_log = list(step_log)
        self.skip_log = list(skip_log)
        self.last_skip = None if not self.skip_log else dict(self.skip_log[-1])
        self._best_val = float(state.get("best_val", float("inf")))
        self._stalls = int(state.get("stalls") or 0)
        self._best_snapshot = state.get("best_snapshot")
        self.stop_training = bool(state.get("stop_training"))
        halt = state.get("halt_reason")
        self.halt_reason = halt if isinstance(halt, str) else "resumed"
        self._epoch_success_start = int(state.get("epoch_success_start") or 0)
        self._epoch_update_start = int(state.get("epoch_update_start") or 0)
        self._epochs_without_step = int(state.get("epochs_without_step") or 0)
        self._window_generator = [float(item) for item in state.get("window_generator") or []]
        self._window_discriminator = [float(item) for item in state.get("window_discriminator") or []]
        samples = state.get("window_samples") or []
        if not isinstance(samples, list) or not all(isinstance(item, str) for item in samples):
            raise RuntimeFailure("window sample ids are invalid")
        self._window_samples = list(samples)
        if self._accu_count != len(self._window_generator) or self._accu_count != len(self._window_discriminator):
            raise RuntimeFailure("window losses do not match the accumulator")
        if len(self._window_samples) != self._accu_count:
            raise RuntimeFailure("window sample ids do not match the accumulator")
        self._restored = True

    def _on_update_phase(self, phase: str) -> None:
        """Hook for tests. An injected phase aborts the pair before the next step."""

        if self.fail_phase == phase:
            raise RuntimeFailure(f"injected failure at {phase}")

    def _step_pair(self, runtime: ProcessRuntime) -> dict[str, Any]:
        """Step generator AdamW, then discriminator AdamW, then close the AMP step."""

        self._in_optimizer_pair = True
        try:
            self._on_update_phase("before_generator_step")
            generator_result = runtime.step(self.optim_g)
            self._on_update_phase("after_generator_step")
            if not generator_result.applied:
                raise RuntimeFailure("generator optimizer did not step")
            self._on_update_phase("before_discriminator_step")
            discriminator_result = runtime.step(self.optim_d)
            self._on_update_phase("after_discriminator_step")
            if not discriminator_result.applied:
                raise RuntimeFailure("discriminator optimizer did not step")
            runtime.end_step()
            self.optim_g.zero_grad(set_to_none=True)
            self.optim_d.zero_grad(set_to_none=True)
            self._accu_count = 0
            self.counters["optimizer_updates"] += 1
            self.optimizer_updates[self.optim_g_id] += 1
            self.optimizer_updates[self.optim_d_id] += 1
            mean_g = sum(self._window_generator) / len(self._window_generator)
            mean_d = sum(self._window_discriminator) / len(self._window_discriminator)
            sample_id = ",".join(self._window_samples)
            self._clear_window_log()
            # Rank 0's scalars are stored on every rank so the step log matches.
            local_log = {
                "discriminator": mean_d,
                "generator": mean_g,
                "loss": mean_g,
                "lr_d": float(self.optim_d.param_groups[0]["lr"]),
                "lr_g": float(self.optim_g.param_groups[0]["lr"]),
                "sample_id": sample_id,
            }
        finally:
            self._in_optimizer_pair = False
        shared = runtime.gather_objects(local_log)[0]
        if not isinstance(shared, dict):
            raise RuntimeFailure("rank 0 did not return a step log")
        shared["logical_step"] = self.logical_step + 1
        self._on_update_phase("after_micro_batch")
        return shared

    def _backward_average_and_clip(
        self,
        runtime: ProcessRuntime,
        loss: torch.Tensor,
        clip_parameters: Any,
    ) -> None:
        modules = (self.generator.unwrapped(), self.discriminator.unwrapped())
        baselines: dict[int, torch.Tensor | None] = {}
        parameters: list[torch.nn.Parameter] = []
        for module in modules:
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                parameters.append(parameter)
                baselines[id(parameter)] = None if parameter.grad is None else parameter.grad.detach().clone()
        runtime.backward(loss)
        runtime.average_new_gradients(parameters, baselines)
        self._average_calls += 1
        runtime.clip_repeated(clip_parameters, self.grad_clip)

    def _skip(self, runtime: ProcessRuntime, reason: str | None, batch: Batch, update_index: int) -> bool:
        skipped = runtime.agree(reason is not None, policy="any")
        if not skipped:
            return False
        reasons = runtime.gather_objects(reason)
        case_ids = runtime.gather_objects(_case_ids(batch))
        if not all(item is None or isinstance(item, str) for item in reasons):
            raise RuntimeFailure("skip reason from another rank is not a string")
        agreed = next((item for item in reasons if item not in (None, "peer")), None)
        if agreed is None:
            agreed = "peer" if any(item is not None for item in reasons) else "pesq"
        record = {
            "case_ids_by_rank": list(case_ids),
            "count": 1,
            "official_step": int(self.counters["official_step"]),
            "reason": agreed,
            "reasons_by_rank": list(reasons),
            "update_index": int(update_index),
        }
        self.skip_log.append(record)
        self.last_skip = record
        self.counters["micro_batches_failed"] += 1
        if self._skip_path is not None:
            self.journal.write_sidecar(self._skip_path, record)
        return True

    def _open_boundary(self, runtime: ProcessRuntime, epoch: int) -> None:
        if self.phase != _IN_EPOCH:
            raise RuntimeFailure("an epoch boundary is already pending")
        self.boundary = {
            "closed": False,
            "plateau_done": False,
            "trainer_epoch": int(epoch),
            "validate": self.validate_on_epoch_end,
            "validation_done": False,
            "validation_summary": None,
        }
        self.phase = _PENDING
        self._offer(runtime, "boundary_pending")

    def _finish_boundary(self, runtime: ProcessRuntime, component_id: str) -> None:
        """Validate, apply the plateau decision, then run research evaluation."""
        if self.phase != _PENDING or self.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending MossFormerGAN boundary")
        boundary = self.boundary
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self._validate(runtime, int(boundary["trainer_epoch"]))
            boundary["validation_summary"] = summary
            boundary["validation_done"] = True
            self._offer(runtime, "boundary_pending")
        if boundary["validate"] and not boundary["plateau_done"]:
            summary = boundary.get("validation_summary")
            if not isinstance(summary, dict):
                raise RuntimeFailure(f"component {component_id}: validation summary is missing")
            action = self._apply_plateau(summary)
            boundary["plateau_action"] = action
            boundary["plateau_done"] = True
        self._discard_window()
        no_success = self.counters["micro_batches_succeeded"] == self._epoch_success_start
        no_step = self.counters["optimizer_updates"] == self._epoch_update_start
        if no_step:
            self._epochs_without_step += 1
        else:
            self._epochs_without_step = 0
        self.data.epoch = int(boundary["trainer_epoch"]) + 1
        self.data.update_index = 0
        boundary["closed"] = True
        metrics = scalar_metrics(boundary.get("validation_summary"))
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = _IN_EPOCH
        if self.stop_training:
            if self.halt_reason == "running":
                self.halt_reason = "plateau_stop"
        elif no_success:
            self.halt_reason = "no_successful_micro_batch"
            self.stop_training = True
        elif self._epochs_without_step >= self.max_logical_steps:
            self.halt_reason = "accumulation_window_never_closed"
            self.stop_training = True
        if boundary["validate"]:
            observe_validation(
                getattr(self, "training_control", None),
                metrics,
                step=self.logical_step,
                scoped=self.research.attached,
            )
            extra = self.research.score(
                runtime,
                output_directory=self.lifecycle.output_directory,
                config_identity=self.lifecycle.config_identity,
                kind="boundary",
                epoch=int(boundary["trainer_epoch"]),
                logical_step=self.logical_step,
                official=metrics,
                stopping=getattr(self, "training_control", None),
            )
            if extra:
                metrics = {**metrics, **extra}
        self._offer(runtime, "boundary", metrics=metrics)

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int) -> dict[str, Any]:
        """Run the official validation report and keep failed PESQ batches in the denominator."""
        cursor = (int(self.data.epoch), int(self.data.update_index))
        rng = runtime.capture_rng()
        summary: dict[str, Any] | None = None
        try:
            produced = list(self.data.validation_epoch(trainer_epoch))
            aligned = self._align_validation(runtime, produced)
            proxy = self._validation_proxy(runtime)
            reports: list[dict[str, Any]] = []
            for batch, alignment in aligned:
                if not isinstance(batch, Batch):
                    raise RuntimeFailure("validation did not return a batch")
                moved = runtime.move(batch)
                payload = {
                    "clean": moved.targets,
                    "noisy": moved.inputs,
                    "validation_alignment_pad": alignment,
                }
                try:
                    reports.append(validation_report(proxy, payload))
                except TrainingError as exc:
                    raise RuntimeFailure(f"validation report failed: {exc}") from exc
            try:
                summary = reduce_validation(
                    proxy, reports, collective=runtime.world_size > 1
                )
            except TrainingError as exc:
                raise RuntimeFailure(f"validation reduction failed: {exc}") from exc
        finally:
            runtime.restore_rng(rng)
            self._train_mode()
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.discriminator)
        moved = (int(self.data.epoch), int(self.data.update_index))
        if moved != cursor:
            raise RuntimeFailure("validation advanced the training cursor")
        if not isinstance(summary, dict):
            raise RuntimeFailure("validation did not return a summary")
        return summary

    def _align_validation(self, runtime: ProcessRuntime, batches: list[Batch]) -> list[tuple[Batch, bool]]:
        """Repeat a prefix when SyncBatchNorm would otherwise deadlock.

        The repeated forwards are marked and stay out of the denominator.
        CPU BatchNorm does not need them, and adding them would change the
        running statistics the official CPU path does not change.
        """

        if runtime.world_size == 1 or not self._syncbn_active():
            return [(batch, False) for batch in batches]
        counts = runtime.gather_objects(len(batches))
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in counts):
            raise RuntimeFailure("validation batch counts are not integers")
        if any(item == 0 for item in counts):
            rendered = ", ".join(str(item) for item in counts)
            raise RuntimeFailure(f"validation rank has no batch to repeat for SyncBatchNorm alignment: {rendered}")
        target = max(int(item) for item in counts)
        if all(int(item) == int(counts[0]) for item in counts):
            return [(batch, False) for batch in batches]
        padded = [(batch, False) for batch in batches]
        source = 0
        while len(padded) < target:
            padded.append((batches[source % len(batches)], True))
            source += 1
        return padded

    def _apply_plateau(self, summary: dict[str, Any]) -> str:
        """Apply one official validation decision. Schedulers are not stepped."""

        generator = (summary.get("losses") or {}).get("generator") or {}
        value = generator.get("value")
        if generator.get("status") != "ok" or value is None:
            self.plateau_history.append({"action": "absent", "logical_step": self.logical_step})
            return "absent"
        action_loss = float(value)
        self._best_val, self._stalls, action = mossformer_validation_decision(
            self._best_val, self._stalls, action_loss
        )
        if action == "improve":
            self._best_snapshot = self._capture_training_state()
        elif action == "halve":
            self._restore_best_and_halve_generator()
        elif action == "stop":
            self.stop_training = True
            self.halt_reason = "plateau_stop"
        self.plateau_history.append(
            {
                "action": action,
                "best": self._best_val,
                "logical_step": self.logical_step,
                "loss": action_loss,
                "lr_d": float(self.optim_d.param_groups[0]["lr"]),
                "lr_g": float(self.optim_g.param_groups[0]["lr"]),
                "stalls": self._stalls,
            }
        )
        return action

    def _capture_training_state(self) -> dict[str, Any]:
        """Copy both networks and both AdamW states for a later plateau restore."""
        return {
            "discriminator": {
                key: value.detach().cpu().clone()
                for key, value in self.discriminator.unwrapped().state_dict().items()
            },
            "generator": {
                key: value.detach().cpu().clone() for key, value in self.generator.unwrapped().state_dict().items()
            },
            "optim_d": copy.deepcopy(self.optim_d.state_dict()),
            "optim_g": copy.deepcopy(self.optim_g.state_dict()),
        }

    def _restore_best_and_halve_generator(self) -> None:
        """Reload the stored best pair and halve the generator learning rate."""
        snapshot = self._best_snapshot
        if snapshot is not None:
            self.generator.unwrapped().load_state_dict(snapshot["generator"], strict=True)
            self.discriminator.unwrapped().load_state_dict(snapshot["discriminator"], strict=True)
            self.optim_g.load_state_dict(snapshot["optim_g"])
            self.optim_d.load_state_dict(snapshot["optim_d"])
        self.optim_g.param_groups[0]["lr"] *= 0.5

    def _discard_window(self) -> None:
        """Drop a partial window at the epoch boundary. Official ``start_epoch`` does this."""

        self.optim_g.zero_grad(set_to_none=True)
        self.optim_d.zero_grad(set_to_none=True)
        self._accu_count = 0
        self._clear_window_log()

    def _grad_state(self, prepared: PreparedModel) -> dict[str, torch.Tensor]:
        grads = {}
        for name, parameter in prepared.unwrapped().named_parameters():
            if parameter.grad is not None:
                grads[name] = parameter.grad.detach().cpu().clone()
        return grads

    def _restore_grads(self, prepared: PreparedModel, grads: object) -> None:
        if not isinstance(grads, dict):
            raise RuntimeFailure("checkpoint gradients are not a mapping")
        named = dict(prepared.unwrapped().named_parameters())
        for parameter in named.values():
            parameter.grad = None
        for name, value in grads.items():
            if name not in named or not isinstance(value, torch.Tensor):
                raise RuntimeFailure(f"checkpoint gradient {name!r} does not match a parameter")
            parameter = named[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeFailure(f"checkpoint gradient {name} has the wrong shape")
            parameter.grad = value.detach().to(device=parameter.device, dtype=parameter.dtype)

    def _validation_proxy(self, runtime: ProcessRuntime) -> "_OfficialCall":
        return _OfficialCall(
            n_fft=self.n_fft,
            hop=self.hop,
            win=self.win,
            sample_rate=self.sample_rate,
            generator=self.generator.execution(),
            discriminator=self.discriminator.execution(),
            device=runtime.device,
            validation_requires_all_ranks=self._syncbn_active(),
        )

    def _terms_view(self) -> "_OfficialCall":
        return _OfficialCall(
            n_fft=self.n_fft,
            hop=self.hop,
            win=self.win,
            sample_rate=self.sample_rate,
            generator=None,
            discriminator=None,
            device=torch.device("cpu"),
            validation_requires_all_ranks=False,
        )

    def _waveforms(self, batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
        moved = runtime.move(batch)
        clean = moved.targets
        noisy = moved.inputs
        if not isinstance(clean, torch.Tensor) or not isinstance(noisy, torch.Tensor):
            raise RuntimeFailure("waveforms must be tensors")
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
        if int(clean.shape[-1]) <= self.n_fft // 2:
            raise RuntimeFailure(
                f"waveform length {int(clean.shape[-1])} is too short for centered STFT "
                f"(need more than {self.n_fft // 2} samples)"
            )
        return clean, noisy

    def _train_mode(self) -> None:
        self.generator.train()
        self.discriminator.train()

    def _syncbn_active(self) -> bool:
        return any(isinstance(module, torch.nn.SyncBatchNorm) for module in self.discriminator.unwrapped().modules())

    def _clear_window_log(self) -> None:
        self._window_generator = []
        self._window_discriminator = []
        self._window_samples = []

    def _refuse_amp(self, runtime: ProcessRuntime) -> None:
        if getattr(runtime, "precision", "fp32") == "amp_fp16":
            raise RuntimeFailure(
                "MossFormerGAN refuses amp_fp16. Each successful micro-batch clips, and the "
                "generator step and the discriminator step are not one scaler decision."
            )

    def _check_claim(self) -> None:
        check_publication(self)

    def _publish_interrupt(self, runtime: ProcessRuntime) -> None:
        """Save the last finished boundary. A GAN optimizer pair is not a boundary."""

        if self._in_optimizer_pair:
            raise RuntimeFailure("refusing to checkpoint between the generator step and the discriminator step")
        if self.logical_step <= 0 and self._accu_count <= 0:
            return
        self._offer(runtime, "stop")

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is None:
            return
        if int(group_size) != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _require_same_updates(self, runtime: ProcessRuntime, updates: int, component_id: str) -> None:
        counts = runtime.gather_objects(int(updates))
        if any(item != updates for item in counts):
            raise RuntimeFailure(
                f"component {component_id}: ranks have different micro-batch counts {counts}"
            )

    def _offer(
        self,
        runtime: ProcessRuntime,
        event: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Offer this state to the selector. A GAN optimizer pair is not a save point.

        A boundary and a stop make the ranks equal whether or not the selector
        saves, as they did before selectors, so validation reads the same
        buffers. Inside an epoch that runs only when a file is written.
        """

        if self._in_optimizer_pair:
            raise RuntimeFailure("refusing to checkpoint between the generator step and the discriminator step")
        if event in ("step", "partial"):
            self.lifecycle.save_if_needed(
                event,
                logical_step=self.logical_step,
                metrics=metrics if event == "step" else None,
                prepare=self._sync_ranks,
            )
            return
        self._sync_ranks(runtime)
        self.lifecycle.save_if_needed(event, logical_step=self.logical_step, metrics=metrics)

    def _sync_ranks(self, runtime: ProcessRuntime) -> None:
        # BatchNorm stats diverge after a local forward. The next DDP forward
        # would copy rank 0's buffers. Doing it before the save keeps the
        # checkpoint identical across ranks. World size 1 does not change them.
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.discriminator)
        if runtime.world_size > 1:
            # Each rank's accumulation window holds that rank's micro-batch loss.
            # The gradient is already averaged. The lists are logs, so rank 0's
            # copy is the one every rank stores. World size 1 does not enter.
            gathered = runtime.gather_objects(
                {
                    "discriminator": list(self._window_discriminator),
                    "generator": list(self._window_generator),
                    "samples": list(self._window_samples),
                }
            )
            chosen = gathered[0]
            self._window_discriminator = [float(item) for item in chosen["discriminator"]]
            self._window_generator = [float(item) for item in chosen["generator"]]
            self._window_samples = [str(item) for item in chosen["samples"]]

    def _stop_requested(self, runtime: object | None = None) -> bool:
        from src.experiments.signals import ranks_should_stop

        capped = self.stop_after_logical_steps is not None and self.logical_step >= self.stop_after_logical_steps
        return ranks_should_stop(runtime, extra=capped)

    def _result(self, reason: str) -> RunResult:
        if reason == "completed":
            epoch = getattr(self.data, "epoch", None)
            if isinstance(epoch, bool) or not isinstance(epoch, int):
                epoch = None
            extra = self.research.score(
                self.runtime,
                output_directory=self.lifecycle.output_directory,
                config_identity=self.lifecycle.config_identity,
                kind="end",
                epoch=epoch,
                logical_step=self.logical_step,
                stopping=getattr(self, "training_control", None),
            )
            if extra:
                self.lifecycle.save_if_needed("final", logical_step=self.logical_step, metrics=extra, prepare=self._sync_ranks)
            else:
                self.lifecycle.save_if_needed("final", logical_step=self.logical_step, prepare=self._sync_ranks)
        if self.halt_reason in {"initialized", "running"}:
            self.halt_reason = reason
        if reason == "completed" and should_stop(self):
            reason = "early_stopping"
        return self.lifecycle.finish(
            RunResult(
                stop_reason=reason,
                logical_step=self.logical_step,
                optimizer_updates=dict(self.optimizer_updates),
                output_dir=self.lifecycle.output_directory,
            )
        )


class _OfficialCall:
    """Binds the current task methods so ``self`` is this view, not the task.

    ``_generator_terms`` and ``validation_report`` call sibling methods. A
    ``SimpleNamespace`` does not bind those functions.
    """

    _forward = forward_spectra
    _generator_parts = generator_parts
    _generator_terms = generator_terms
    validation_report = validation_report
    reduce_validation = reduce_validation

    def __init__(
        self,
        *,
        n_fft: int,
        hop: int,
        win: int,
        sample_rate: int,
        generator: torch.nn.Module | None,
        discriminator: torch.nn.Module | None,
        device: torch.device,
        validation_requires_all_ranks: bool,
    ) -> None:
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.sample_rate = sample_rate
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.validation_requires_all_ranks = validation_requires_all_ranks


def _empty_counters() -> dict[str, int]:
    """Start the official micro-batch counters at zero."""
    return {key: 0 for key in _COUNTER_KEYS}


def _case_ids(batch: Batch) -> list[str]:
    """Read the sample ids carried by one micro-batch."""
    if batch.sample_ids:
        return [str(item) for item in batch.sample_ids]
    return [str(batch.sample_id)]


def _require_adamw(optimizer: torch.optim.Optimizer, name: str, *, lr: float) -> None:
    """Refuse an optimizer that is not AdamW."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError(f"{name} must be AdamW")
    if len(optimizer.param_groups) != 1:
        raise BuildError(f"{name} must have one parameter group")
    group = optimizer.param_groups[0]
    if float(group["lr"]) != float(lr):
        raise BuildError(f"{name} learning rate must be {lr}")
    betas = tuple(float(item) for item in group["betas"])
    if betas != OFFICIAL_BETAS:
        raise BuildError(f"{name} betas must be {OFFICIAL_BETAS}")
    if float(group["weight_decay"]) != float(OFFICIAL_ADAMW_WEIGHT_DECAY):
        raise BuildError(
            f"{name} weight decay must be the PyTorch AdamW default {OFFICIAL_ADAMW_WEIGHT_DECAY}"
        )
    if float(group["eps"]) != 1e-8 or bool(group["amsgrad"]):
        raise BuildError(f"{name} must keep AdamW eps 1e-8 and amsgrad false")


def _require_disjoint(generator: torch.optim.Optimizer, discriminator: torch.optim.Optimizer) -> None:
    """Refuse an optimizer pair that shares a parameter."""
    left = {id(parameter) for group in generator.param_groups for parameter in group["params"]}
    right = {id(parameter) for group in discriminator.param_groups for parameter in group["params"]}
    if not left or not right:
        raise BuildError("optimizer has no parameters")
    if left & right:
        raise BuildError("generator and discriminator share parameters")


def _positive_int(name: str, value: object) -> int:
    """Require a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    """Require a positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)
