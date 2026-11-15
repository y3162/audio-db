"""One MossFormerGAN micro-batch. Both backwards finish before either AdamW steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mossformergan import reference as mossformer_lib
from src.experiments.official.mossformergan.reference import generator_parts
from src.experiments.official.mossformergan.state import MossProgress
from src.experiments.official.mossformergan.view import OfficialCall
from src.experiments.runtime import ProcessRuntime
from src.experiments.training.objectives import call_objective


@dataclass
class PairGuard:
    active: bool = False


class MicroBatch:
    """Own the skippable window. The trainer still offers the checkpoint."""

    def __init__(
        self,
        *,
        generator: Any,
        discriminator: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        generator_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        progress: MossProgress,
        guard: PairGuard,
        accumulation: int,
        grad_clip: float,
        n_fft: int,
        hop: int,
        win: int,
        sample_rate: int,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.generator_objective = generator_objective
        self.discriminator_objective = discriminator_objective
        self.progress = progress
        self.guard = guard
        self.accumulation = accumulation
        self.grad_clip = grad_clip
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.sample_rate = sample_rate
        self.average_calls = 0
        self.last_components: dict[str, float] | None = None

    def consume(
        self,
        batch: Batch,
        runtime: ProcessRuntime,
        *,
        update_index: int,
        phase_hook: Any,
        step_pair: Any,
        journal: Any,
        skip_path: Any,
    ) -> dict[str, Any] | None:
        if self.guard.active:
            raise RuntimeFailure("a micro-batch cannot start between the two optimizer steps")
        self.generator.train()
        self.discriminator.train()
        clean, noisy = _waveforms(batch, runtime, self.n_fft)
        progress = self.progress
        progress.counters["micro_batches_processed"] += 1
        progress.counters["official_step"] += 1
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
            if self._skip(runtime, reason, batch, update_index, journal, skip_path):
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
            if self._skip(runtime, disc_reason, batch, update_index, journal, skip_path):
                return None
            progress.counters["micro_batches_succeeded"] += 1
            progress.accu_count += 1
            self.last_components = {
                "discriminator": float(discriminator_loss.detach()),
                "generator": float(generator_loss.detach()),
            }
            for name, value in result.terms.items():
                self.last_components[name] = float(value.detach())
            phase_hook("before_generator_backward")
            self._backward(runtime, generator_loss / self.accumulation, self.generator)
            phase_hook("after_generator_clip")
            phase_hook("before_discriminator_backward")
            self._backward(runtime, discriminator_loss / self.accumulation, self.discriminator)
            phase_hook("after_discriminator_clip")
        progress.window_generator.append(float(generator_loss.detach()))
        progress.window_discriminator.append(float(discriminator_loss.detach()))
        progress.window_samples.append(str(batch.sample_id))
        if progress.accu_count != self.accumulation:
            phase_hook("after_micro_batch")
            return None
        return step_pair(runtime)

    def _step_pair(self, runtime: ProcessRuntime, phase_hook: Any) -> dict[str, Any]:
        progress = self.progress
        self.guard.active = True
        try:
            phase_hook("before_generator_step")
            generator_result = runtime.step(self.optim_g)
            phase_hook("after_generator_step")
            if not generator_result.applied:
                raise RuntimeFailure("generator optimizer did not step")
            phase_hook("before_discriminator_step")
            discriminator_result = runtime.step(self.optim_d)
            phase_hook("after_discriminator_step")
            if not discriminator_result.applied:
                raise RuntimeFailure("discriminator optimizer did not step")
            runtime.end_step()
            self.optim_g.zero_grad(set_to_none=True)
            self.optim_d.zero_grad(set_to_none=True)
            progress.accu_count = 0
            progress.counters["optimizer_updates"] += 1
            for name in progress._optimizer_ids:
                progress.optimizer_updates[name] += 1
            mean_g = sum(progress.window_generator) / len(progress.window_generator)
            mean_d = sum(progress.window_discriminator) / len(progress.window_discriminator)
            sample_id = ",".join(progress.window_samples)
            progress.clear_window()
            local_log = {
                "discriminator": mean_d,
                "generator": mean_g,
                "loss": mean_g,
                "lr_d": float(self.optim_d.param_groups[0]["lr"]),
                "lr_g": float(self.optim_g.param_groups[0]["lr"]),
                "sample_id": sample_id,
            }
        finally:
            self.guard.active = False
        shared = runtime.gather_objects(local_log)[0]
        if not isinstance(shared, dict):
            raise RuntimeFailure("rank 0 did not return a step log")
        shared["logical_step"] = progress.logical_step + 1
        phase_hook("after_micro_batch")
        return shared

    def _backward(self, runtime: ProcessRuntime, loss: torch.Tensor, clip_parameters: Any) -> None:
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
        self.average_calls += 1
        runtime.clip_repeated(clip_parameters, self.grad_clip)

    def _skip(
        self,
        runtime: ProcessRuntime,
        reason: str | None,
        batch: Batch,
        update_index: int,
        journal: Any,
        skip_path: Any,
    ) -> bool:
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
            "official_step": int(self.progress.counters["official_step"]),
            "reason": agreed,
            "reasons_by_rank": list(reasons),
            "update_index": int(update_index),
        }
        self.progress.skip_log.append(record)
        self.progress.last_skip = record
        self.progress.counters["micro_batches_failed"] += 1
        if skip_path is not None and journal is not None:
            journal.write_sidecar(skip_path, record)
        return True

    def _terms_view(self) -> OfficialCall:
        return OfficialCall(
            n_fft=self.n_fft,
            hop=self.hop,
            win=self.win,
            sample_rate=self.sample_rate,
            generator=None,
            discriminator=None,
            device=torch.device("cpu"),
            validation_requires_all_ranks=False,
        )

    def discard_window(self) -> None:
        self.optim_g.zero_grad(set_to_none=True)
        self.optim_d.zero_grad(set_to_none=True)
        self.progress.accu_count = 0
        self.progress.clear_window()


def _waveforms(batch: Batch, runtime: ProcessRuntime, n_fft: int) -> tuple[torch.Tensor, torch.Tensor]:
    moved = runtime.move(batch)
    clean = moved.targets
    noisy = moved.inputs
    if not isinstance(clean, torch.Tensor) or not isinstance(noisy, torch.Tensor):
        raise RuntimeFailure("waveforms must be tensors")
    if clean.ndim != 2 or clean.shape != noisy.shape:
        raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
    if int(clean.shape[-1]) <= n_fft // 2:
        raise RuntimeFailure(
            f"waveform length {int(clean.shape[-1])} is too short for centered STFT "
            f"(need more than {n_fft // 2} samples)"
        )
    return clean, noisy


def share_window(runtime: ProcessRuntime, generator: Any, discriminator: Any, progress: MossProgress) -> None:
    runtime.broadcast_buffers(generator)
    runtime.broadcast_buffers(discriminator)
    if runtime.world_size <= 1:
        return
    gathered = runtime.gather_objects(
        {
            "discriminator": list(progress.window_discriminator),
            "generator": list(progress.window_generator),
            "samples": list(progress.window_samples),
        }
    )
    chosen = gathered[0]
    progress.window_discriminator = [float(item) for item in chosen["discriminator"]]
    progress.window_generator = [float(item) for item in chosen["generator"]]
    progress.window_samples = [str(item) for item in chosen["samples"]]


def open_micro_batch(
    data: Any,
    runtime: ProcessRuntime,
    progress: MossProgress,
    *,
    accumulation: int,
    component_id: str,
    epoch: int,
    index: int,
    open_boundary: Any,
) -> tuple[Any, Batch, int] | None:
    from src.experiments.trainers import _close_epoch_view

    view = data.train_epoch(epoch, start_update=index)
    try:
        updates = int(view.updates)
        counts = runtime.gather_objects(int(updates))
        if any(item != updates for item in counts):
            raise RuntimeFailure(f"component {component_id}: ranks have different micro-batch counts {counts}")
        if updates < 1:
            raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
        if updates % accumulation != 0:
            raise RuntimeFailure(
                f"component {component_id}: epoch micro-batches {updates} "
                f"are not divisible by accumulation {accumulation}"
            )
        if index > updates:
            raise RuntimeFailure(f"component {component_id}: update {index} is past epoch {epoch}")
        if index == updates:
            open_boundary()
            _close_epoch_view(view)
            return None
        if index == 0:
            progress.epoch_success_start = int(progress.counters["micro_batches_succeeded"])
            progress.epoch_update_start = int(progress.counters["optimizer_updates"])
        batches = view.batch(index)
        if len(batches) != 1 or not isinstance(batches[0], Batch):
            raise RuntimeFailure(f"component {component_id}: expected one micro-batch")
    except Exception:
        _close_epoch_view(view)
        raise
    return view, batches[0], updates


def case_ids(batch: Batch) -> list[str]:
    if batch.sample_ids:
        return [str(item) for item in batch.sample_ids]
    return [str(batch.sample_id)]


def _case_ids(batch: Batch) -> list[str]:
    return case_ids(batch)
