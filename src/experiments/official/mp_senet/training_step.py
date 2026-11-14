"""One MP-SENet GAN update.

The order matches ``MPSENetTask._official``. One generator forward is kept
until the discriminator has stepped. The generator backward then uses that
graph and the updated discriminator. CPU and distributed ranks backward the
discriminator first, then the sum of the signal and metric losses. The CUDA
single-rank path with accumulation 1 overlaps PESQ with the signal backward,
then steps the discriminator, then the metric term, then the generator.

This module does not own epochs, validation, schedulers, best selection,
research evaluation, the step journal, or checkpoints. It does not accept a
trainer. The caller passes the models, optimizers, objectives, and four
callables:

- ``phase_hook(phase, payload=None)`` at ``zero_grad``,
  ``before_generator_forward``, ``after_discriminator_step`` (the retained
  generator graphs), and ``end_step``. It must not save or advance progress.
- ``drain_previous()`` only on the CUDA overlap path, after the signal
  backward and before waiting for PESQ. The caller flushes the previous
  step's journal there.
- ``note_generator_sync(enabled)`` when a distributed generator backward
  sees a sync flag. The CUDA overlap path does not call it.
- ``note_gradients_averaged()`` immediately before ``average_gradients``.

``run`` returns only after both optimizers have stepped and ``end_step`` has
returned. A failure leaves that pair unfinished. ``amp_fp16`` is refused
because the scaler would decide overflow per optimizer.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mp_senet.models import pack_forward
from src.experiments.official.mp_senet.reference import _require_discriminator_resolution
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.training.objectives import call_objective

_LOSS_NAMES = ("total", "magnitude", "phase", "complex", "consistency", "metric", "time")


def refuse_gan_amp(runtime: ProcessRuntime) -> None:
    """Refuse FP16 AMP. The two AdamW steps are not one scaler decision."""

    if getattr(runtime, "precision", "fp32") == "amp_fp16":
        raise RuntimeFailure(
            "MP-SENet official GAN refuses amp_fp16. "
            "The current scaler decides overflow per optimizer, so the discriminator "
            "step and the generator step are not one atomic update. FP32 is unchanged."
        )


class MPSENetUpdate:
    """Discriminator-then-generator step for the injected MP-SENet objectives."""

    def __init__(
        self,
        *,
        generator: PreparedModel,
        discriminator: PreparedModel,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        signal_objective: torch.nn.Module,
        metric_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        accumulation: int,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.signal_objective = signal_objective
        self.metric_objective = metric_objective
        self.discriminator_objective = discriminator_objective
        self.accumulation = accumulation

    def run(
        self,
        batches: list[Batch],
        runtime: ProcessRuntime,
        *,
        phase_hook: Callable[..., None],
        drain_previous: Callable[[], None],
        note_generator_sync: Callable[[bool], None],
        note_gradients_averaged: Callable[[], None],
    ) -> dict[str, float]:
        """Apply one official pair, or raise before either optimizer is a safe point."""

        refuse_gan_amp(runtime)
        if len(batches) != self.accumulation:
            raise RuntimeFailure(
                f"official step received {len(batches)} micro-batches, accumulation is {self.accumulation}"
            )
        counts = runtime.gather_objects(len(batches))
        if any(item != len(batches) for item in counts):
            raise RuntimeFailure("ranks disagree on the number of micro-batches")
        self.generator.train()
        self.discriminator.train()
        if runtime.device.type == "cuda" and runtime.world_size == 1 and len(batches) == 1:
            return self._cuda_overlapped_step(
                batches[0],
                runtime,
                phase_hook=phase_hook,
                drain_previous=drain_previous,
                note_gradients_averaged=note_gradients_averaged,
            )
        return self._accumulated_step(
            batches,
            runtime,
            phase_hook=phase_hook,
            note_generator_sync=note_generator_sync,
            note_gradients_averaged=note_gradients_averaged,
        )

    def _cuda_overlapped_step(
        self,
        batch: Batch,
        runtime: ProcessRuntime,
        *,
        phase_hook: Callable[..., None],
        drain_previous: Callable[[], None],
        note_gradients_averaged: Callable[[], None],
    ) -> dict[str, float]:
        """PESQ runs while the signal backward uses the same generator graph.

        The discriminator optimizer still steps before the metric term and the
        generator optimizer.
        """

        phase_hook("zero_grad")
        self.optim_d.zero_grad(set_to_none=True)
        self.optim_g.zero_grad(set_to_none=True)
        clean, noisy = self._waveforms(batch, runtime)
        if not _inputs_ready(clean, noisy, self.generator.unwrapped().n_fft):
            raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
        phase_hook("before_generator_forward")
        try:
            parts = pack_forward(self.generator(noisy, clean))
            _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(f"generator forward failed: {type(exc).__name__}: {exc}") from exc
        pending = _schedule_metric_targets(
            clean,
            parts["audio"],
            int(self.generator.unwrapped().architecture["sampling_rate"]),
        )
        ones = torch.ones(clean.shape[0], device=clean.device)
        signal = call_objective(self.signal_objective, _signal_context(parts, clean))
        self._backward_if_finite(runtime, signal.total / self.accumulation, "generator signal loss", retain_graph=True)
        # Previous-step journal I/O overlaps the rest of the PESQ thread.
        # The caller decides whether that flush may also offer a checkpoint.
        drain_previous()
        try:
            targets = pending.result()
            real = self.discriminator(parts["clean_mag"], parts["clean_mag"])
            fake = self.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
            loss_d = call_objective(
                self.discriminator_objective,
                {"real_score": real, "fake_score": fake, "ones": ones, "pesq_target": targets},
            ).total / self.accumulation
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(f"discriminator forward failed: {type(exc).__name__}: {exc}") from exc
        self._backward_if_finite(runtime, loss_d, "discriminator loss")
        discriminator_result = runtime.step(self.optim_d)
        phase_hook("after_discriminator_step", [(clean, parts)])
        raw = self.discriminator.unwrapped().discriminator
        frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            try:
                metric = raw(parts["clean_mag"], parts["mag_hat"])
                metric_out = call_objective(self.metric_objective, {"metric": metric, "ones": ones})
            except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
                raise RuntimeFailure(f"generator loss failed: {type(exc).__name__}: {exc}") from exc
            self._backward_if_finite(runtime, metric_out.total / self.accumulation, "generator metric loss")
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        note_gradients_averaged()
        runtime.average_gradients(self.generator)
        total = signal.total + metric_out.total
        logged = torch.stack(
            [
                signal.terms["complex"].detach(),
                signal.terms["consistency"].detach(),
                loss_d.detach() * self.accumulation,
                total.detach(),
                signal.terms["magnitude"].detach(),
                metric_out.terms["metric"].detach(),
                signal.terms["phase"].detach(),
                signal.terms["time"].detach(),
            ]
        )
        host = torch.empty(logged.shape, dtype=torch.float32, pin_memory=True)
        host.copy_(logged.detach().float(), non_blocking=True)
        generator_result = runtime.step(self.optim_g)
        runtime.end_step()
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.discriminator)
        if not discriminator_result.applied or not generator_result.applied:
            raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
        complex_v, consistency_v, disc_v, total_v, magnitude_v, metric_v, phase_v, time_v = (
            float(item) for item in host.tolist()
        )
        phase_hook("end_step")
        return {
            "complex": complex_v,
            "consistency": consistency_v,
            "discriminator": disc_v,
            "generator": total_v,
            "lr_d": float(self.optim_d.param_groups[0]["lr"]),
            "lr_g": float(self.optim_g.param_groups[0]["lr"]),
            "magnitude": magnitude_v,
            "metric": metric_v,
            "phase": phase_v,
            "sample_id": batch.sample_id,
            "time": time_v,
        }

    def _accumulated_step(
        self,
        batches: list[Batch],
        runtime: ProcessRuntime,
        *,
        phase_hook: Callable[..., None],
        note_generator_sync: Callable[[bool], None],
        note_gradients_averaged: Callable[[], None],
    ) -> dict[str, float]:
        """Discriminator micro-batches, one discriminator step, then the generator."""

        phase_hook("zero_grad")
        self.optim_d.zero_grad(set_to_none=True)
        self.optim_g.zero_grad(set_to_none=True)
        stored: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        discriminator_total = 0.0
        sample_ids: list[str] = []
        with runtime.sync_gradients([self.generator], enabled=False):
            for index, batch in enumerate(batches):
                last = index == len(batches) - 1
                clean, noisy = self._waveforms(batch, runtime)
                sample_ids.append(batch.sample_id)
                ready = _inputs_ready(clean, noisy, self.generator.unwrapped().n_fft)
                if not runtime.agree(ready, policy="all"):
                    raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
                phase_hook("before_generator_forward")
                error: str | None = None
                parts: dict[str, torch.Tensor] | None = None
                try:
                    parts = pack_forward(self.generator(noisy, clean))
                    _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
                except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                errors = runtime.gather_objects(error)
                if any(errors):
                    raise RuntimeFailure("generator forward failed: " + "; ".join(item for item in errors if item))
                if parts is None:
                    raise RuntimeFailure("generator forward produced no tensors")
                stored.append((clean, parts))
                ones = torch.ones(clean.shape[0], device=clean.device)
                with runtime.sync_gradients([self.discriminator], enabled=last):
                    disc_error: str | None = None
                    loss: torch.Tensor | None = None
                    try:
                        real = self.discriminator(parts["clean_mag"], parts["clean_mag"])
                        fake = self.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
                        targets = _metric_targets(
                            clean, parts["audio"], int(self.generator.unwrapped().architecture["sampling_rate"])
                        )
                        loss = call_objective(
                            self.discriminator_objective,
                            {"real_score": real, "fake_score": fake, "ones": ones, "pesq_target": targets},
                        ).total / self.accumulation
                    except (TrainingError, RuntimeError, ValueError) as exc:
                        disc_error = f"{type(exc).__name__}: {exc}"
                    disc_errors = runtime.gather_objects(disc_error)
                    if any(disc_errors):
                        raise RuntimeFailure(
                            "discriminator forward failed: " + "; ".join(item for item in disc_errors if item)
                        )
                    if loss is None:
                        raise RuntimeFailure("discriminator loss is missing")
                    self._backward_if_finite(runtime, loss, "discriminator loss")
                    discriminator_total += float(loss.detach()) * self.accumulation
            discriminator_result = runtime.step(self.optim_d)
            phase_hook("after_discriminator_step", stored)
            raw = self.discriminator.unwrapped().discriminator
            frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
            component_totals = {name: 0.0 for name in _LOSS_NAMES}
            try:
                for parameter in frozen:
                    parameter.requires_grad_(False)
                for clean, parts in stored:
                    ones = torch.ones(clean.shape[0], device=clean.device)
                    gen_error: str | None = None
                    total: torch.Tensor | None = None
                    try:
                        metric = raw(parts["clean_mag"], parts["mag_hat"])
                        signal = call_objective(self.signal_objective, _signal_context(parts, clean))
                        metric_out = call_objective(self.metric_objective, {"metric": metric, "ones": ones})
                        total_tensor = signal.total + metric_out.total
                        losses = {
                            "total": total_tensor,
                            "magnitude": signal.terms["magnitude"],
                            "phase": signal.terms["phase"],
                            "complex": signal.terms["complex"],
                            "consistency": signal.terms["consistency"],
                            "metric": metric_out.terms["metric"],
                            "time": signal.terms["time"],
                        }
                        total = total_tensor / self.accumulation
                    except (TrainingError, RuntimeError, ValueError) as exc:
                        gen_error = f"{type(exc).__name__}: {exc}"
                    gen_errors = runtime.gather_objects(gen_error)
                    if any(gen_errors):
                        raise RuntimeFailure(
                            "generator loss failed: " + "; ".join(item for item in gen_errors if item)
                        )
                    if total is None:
                        raise RuntimeFailure("generator loss is missing")
                    synced = self.generator.gradient_sync_enabled()
                    if synced is not None:
                        note_generator_sync(synced)
                    self._backward_if_finite(runtime, total, "generator loss")
                    for name in _LOSS_NAMES:
                        component_totals[name] += float(losses[name].detach())
            finally:
                for parameter in frozen:
                    parameter.requires_grad_(True)
            note_gradients_averaged()
            runtime.average_gradients(self.generator)
        generator_result = runtime.step(self.optim_g)
        runtime.end_step()
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.discriminator)
        if not discriminator_result.applied or not generator_result.applied:
            raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
        count = len(batches)
        phase_hook("end_step")
        return {
            "complex": component_totals["complex"] / count,
            "consistency": component_totals["consistency"] / count,
            "discriminator": discriminator_total / count,
            "generator": component_totals["total"] / count,
            "lr_d": float(self.optim_d.param_groups[0]["lr"]),
            "lr_g": float(self.optim_g.param_groups[0]["lr"]),
            "magnitude": component_totals["magnitude"] / count,
            "metric": component_totals["metric"] / count,
            "phase": component_totals["phase"] / count,
            "sample_id": ",".join(sample_ids),
            "time": component_totals["time"] / count,
        }

    def _waveforms(self, batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
        batch = runtime.move(batch)
        clean = batch.targets
        noisy = batch.inputs
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
        return clean, noisy

    def _backward_if_finite(
        self,
        runtime: ProcessRuntime,
        loss: torch.Tensor,
        what: str,
        *,
        retain_graph: bool = False,
    ) -> None:
        if runtime.device.type != "cuda":
            finite = bool(torch.isfinite(loss.detach()).all())
            if not runtime.agree(finite, policy="all"):
                raise RuntimeFailure(f"{what} is not finite on every rank")
        runtime.backward(loss, retain_graph=retain_graph)


def _signal_context(parts: dict[str, torch.Tensor], clean: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the signal-term context from one generator forward."""

    return {
        "clean_audio": clean,
        "clean_complex": parts["clean_complex"],
        "clean_mag": parts["clean_mag"],
        "clean_phase": parts["clean_phase"],
        "generated_audio": parts["audio"],
        "generated_complex": parts["complex"],
        "generated_mag": parts["mag"],
        "generated_phase": parts["phase"],
        "recomputed_complex": parts["recomputed_complex"],
    }


def _schedule_metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> Any:
    """Copy waveforms without blocking, then score PESQ on a side thread."""

    from src.experiments.calc.losses import metric_targets

    clean_cpu = torch.empty(clean.shape, dtype=clean.dtype, pin_memory=True)
    enhanced_cpu = torch.empty(enhanced.shape, dtype=enhanced.dtype, pin_memory=True)
    clean_cpu.copy_(clean.detach(), non_blocking=True)
    enhanced_cpu.copy_(enhanced.detach(), non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    box: dict[str, object] = {}
    done = threading.Event()

    def run() -> None:
        try:
            event.synchronize()
            box["value"] = metric_targets(clean_cpu, enhanced_cpu, sample_rate)
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=run, name="mp-pesq", daemon=True)
    thread.start()

    class _Pending:
        def result(self) -> object:
            done.wait()
            if "error" in box:
                raise box["error"]
            return box.get("value")

    return _Pending()


def _metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    """Look up PESQ targets on the losses module so tests can patch that function."""

    from src.experiments.calc import losses as loss_lib

    return loss_lib.metric_targets(clean, enhanced, sample_rate)


def _inputs_ready(clean: torch.Tensor, noisy: torch.Tensor, n_fft: int) -> bool:
    """Accept a matching pair long enough for the STFT. CPU also requires finite values."""

    if clean.ndim != 2 or clean.shape != noisy.shape:
        return False
    if int(clean.shape[-1]) <= n_fft // 2:
        return False
    if clean.is_cuda:
        # A non-finite waveform makes the loss non-finite, which rejects the step.
        return True
    return bool(torch.isfinite(clean).all()) and bool(torch.isfinite(noisy).all())
