"""One SEMamba++ GAN update. Both AdamW steps happen here; the checkpoint does not."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.semambapp.models import pack_forward, scan_execution
from src.experiments.runtime import ProcessRuntime
from src.experiments.training.objectives import call_objective


class UpdateView:
    """Read the non-persistent GAN observations from the update that owns them."""

    @property
    def capture_gradients(self) -> bool:
        return self._update.capture_gradients

    @capture_gradients.setter
    def capture_gradients(self, value: bool) -> None:
        self._update.capture_gradients = value

    @property
    def frozen_during_generator(self) -> bool:
        return self._update.frozen_during_generator

    @property
    def generator_graph_forwards(self) -> int:
        return self._update.generator_graph_forwards

    @property
    def generator_preview_forwards(self) -> int:
        return self._update.generator_preview_forwards

    @property
    def generator_reforward_forwards(self) -> int:
        return self._update.generator_reforward_forwards

    @property
    def sync_trace(self) -> list[str]:
        return self._update.sync_trace

    @property
    def last_generator_grads(self) -> dict[str, torch.Tensor]:
        return self._update.last_generator_grads

    @property
    def last_discriminator_grads(self) -> dict[str, torch.Tensor]:
        return self._update.last_discriminator_grads


class SEMambaUpdate:
    """Own the discriminator-then-generator step and its non-persistent observations."""

    def __init__(
        self,
        *,
        generator: Any,
        mrd: Any,
        cqt: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        generator_objective: Any,
        discriminator_objective: Any,
        accumulation: int,
        grad_clip: float | None,
    ) -> None:
        self.generator = generator
        self.mrd = mrd
        self.cqt = cqt
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.generator_objective = generator_objective
        self.discriminator_objective = discriminator_objective
        self.accumulation = accumulation
        self.grad_clip = grad_clip
        self.capture_gradients = False
        self.frozen_during_generator = False
        self.generator_graph_forwards = 0
        self.generator_preview_forwards = 0
        self.generator_reforward_forwards = 0
        self.mrd_forwards = 0
        self.cqt_forwards = 0
        self.sync_trace: list[str] = []
        self.last_generator_grads: dict[str, torch.Tensor] = {}
        self.last_discriminator_grads: dict[str, torch.Tensor] = {}
        self.scan_report: dict[str, str] | None = None
        self._phase: Any = None

    def refuse_amp(self, runtime: ProcessRuntime) -> None:
        if getattr(runtime, "precision", "fp32") == "amp_fp16":
            raise RuntimeFailure(
                "SEMamba++ official GAN refuses amp_fp16. "
                "The current scaler decides overflow per optimizer, so the discriminator "
                "step and the generator step are not one atomic update. FP32 is unchanged."
            )

    def require_scan(self, runtime: ProcessRuntime) -> None:
        if self.scan_report is None:
            self.scan_report = scan_execution(self.generator.unwrapped().generator, runtime.device)
        report = self.scan_report
        if report["configured"] == "fused" and report["executed"] == "unavailable":
            raise RuntimeFailure(
                "scan_backend fused was requested but mamba-ssm is not importable. "
                "Refusing to run the reference kernel instead."
            )
        if report["configured"] == "fused" and runtime.device.type == "cuda" and report["executed"] != "fused":
            raise RuntimeFailure(
                f"scan_backend fused on CUDA resolved to {report['executed']}, not the fused kernel"
            )

    def run(
        self,
        batches: list[Batch],
        runtime: ProcessRuntime,
        phase_hook: Any,
    ) -> dict[str, float | str]:
        """Step both optimizers. The caller commits the logical step only after this returns."""

        self._phase = phase_hook
        self.refuse_amp(runtime)
        self.require_scan(runtime)
        if len(batches) != self.accumulation:
            raise RuntimeFailure(
                f"official step received {len(batches)} micro-batches, accumulation is {self.accumulation}"
            )
        counts = runtime.gather_objects(len(batches))
        if any(item != len(batches) for item in counts):
            raise RuntimeFailure("ranks disagree on the number of micro-batches")
        self.generator.train()
        self.mrd.train()
        self.cqt.train()
        graph_before = self.generator_graph_forwards
        preview_before = self.generator_preview_forwards
        reforward_before = self.generator_reforward_forwards
        self._phase("zero_grad")
        self.optim_d.zero_grad(set_to_none=True)
        reuse_graph = len(batches) == 1
        cached: list[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]] = []
        discriminator_total = None
        sample_ids: list[str] = []
        realization_ids: list[str] = []
        for index, batch in enumerate(batches):
            last = index == len(batches) - 1
            clean, noisy = _waveforms(batch, runtime)
            sample_ids.append(batch.sample_id)
            realization_ids.append(batch.realization_id)
            ready = _inputs_ready(clean, noisy)
            if not runtime.agree(ready, policy="all"):
                raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
            if reuse_graph:
                self.sync_trace.append("generator_forward_sync")
                with runtime.sync_gradients([self.generator], enabled=True):
                    parts = self._generator_forward(runtime, clean, noisy, preview=False)
                self.generator_graph_forwards += 1
                audio = parts["audio"]
                cached.append((clean, noisy, parts))
            else:
                self.sync_trace.append("generator_preview_nograd")
                with torch.no_grad():
                    preview = self._generator_forward(runtime, clean, noisy, preview=True)
                self.generator_preview_forwards += 1
                audio = preview["audio"]
                cached.append((clean, noisy, None))
            self.sync_trace.append("discriminator_sync" if last else "discriminator_nosync")
            with runtime.sync_gradients([self.mrd, self.cqt], enabled=last):
                loss = self._discriminator_loss(runtime, clean, audio.detach())
                _backward_if_finite(runtime, loss / self.accumulation, "discriminator loss")
            term = loss.detach()
            discriminator_total = term if discriminator_total is None else discriminator_total + term
            self.mrd_forwards += 1
            self.cqt_forwards += 1
        if self.grad_clip is not None:
            runtime.clip(self.mrd, self.grad_clip)
            runtime.clip(self.cqt, self.grad_clip)
        disc_norm = _grad_norm(self.mrd.unwrapped().discriminator, self.cqt.unwrapped().discriminator)
        if self.capture_gradients:
            self.last_discriminator_grads = _clone_grads(
                self.mrd.unwrapped().discriminator, self.cqt.unwrapped().discriminator
            )
        self._phase("before_discriminator_step")
        discriminator_result = runtime.step(self.optim_d)
        self._phase("after_discriminator_step")
        frozen = [
            parameter
            for module in (self.mrd.unwrapped(), self.cqt.unwrapped())
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.optim_g.zero_grad(set_to_none=True)
        component_totals: dict[str, Any] = {}
        generator_total: Any = 0.0
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            self.frozen_during_generator = True
            self._phase("before_generator_backward")
            for index, (clean, noisy, parts) in enumerate(cached):
                last = index == len(cached) - 1
                if parts is None:
                    self.sync_trace.append("generator_reforward_sync" if last else "generator_reforward_nosync")
                    with runtime.sync_gradients([self.generator], enabled=last):
                        parts = self._generator_forward(runtime, clean, noisy, preview=False)
                        self.generator_reforward_forwards += 1
                        generator_total += self._generator_backward(runtime, clean, parts, component_totals)
                else:
                    self.sync_trace.append("generator_backward_retained")
                    generator_total += self._generator_backward(runtime, clean, parts, component_totals)
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
            self.frozen_during_generator = False
        if self.grad_clip is not None:
            runtime.clip(self.generator, self.grad_clip)
        log_terms = torch.stack(
            [
                value.detach() if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=runtime.device)
                for value in component_totals.values()
            ]
        )
        if discriminator_total is None or not isinstance(generator_total, torch.Tensor):
            raise RuntimeFailure("generator loss is missing")
        log_pair = torch.stack([discriminator_total.detach(), generator_total.detach()])
        if runtime.device.type == "cuda":
            host_terms = torch.empty(log_terms.shape, dtype=torch.float32, pin_memory=True)
            host_pair = torch.empty(log_pair.shape, dtype=torch.float32, pin_memory=True)
            host_terms.copy_(log_terms.detach().float(), non_blocking=True)
            host_pair.copy_(log_pair.detach().float(), non_blocking=True)
        else:
            host_terms = log_terms.detach().float().cpu()
            host_pair = log_pair.detach().float().cpu()
        gen_norm = _grad_norm(self.generator.unwrapped().generator)
        if self.capture_gradients:
            self.last_generator_grads = _clone_grads(self.generator.unwrapped().generator)
        self._phase("before_generator_step")
        generator_result = runtime.step(self.optim_g)
        runtime.end_step()
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.mrd)
        runtime.broadcast_buffers(self.cqt)
        if not discriminator_result.applied or not generator_result.applied:
            raise RuntimeFailure("SEMamba++ requires both optimizers to apply in one logical step")
        count = len(batches)
        report = self.scan_report or {"configured": "", "executed": ""}
        self._phase("end_step")
        named = {
            name: float(number) for name, number in zip(component_totals, host_terms.tolist(), strict=True)
        }
        discriminator_value = float(host_pair[0])
        generator_value = float(host_pair[1])
        return {
            "adversarial": named["adversarial"] / count,
            "complex": named["complex"] / count,
            "consistency": named["consistency"] / count,
            "discriminator": discriminator_value / count,
            "discriminator_grad_norm": disc_norm,
            "generator": generator_value / count,
            "generator_forwards": float(self.generator_graph_forwards - graph_before),
            "generator_grad_norm": gen_norm,
            "generator_previews": float(self.generator_preview_forwards - preview_before),
            "generator_reforwards": float(self.generator_reforward_forwards - reforward_before),
            "lr_d": float(self.optim_d.param_groups[0]["lr"]),
            "lr_g": float(self.optim_g.param_groups[0]["lr"]),
            "magnitude": named["magnitude"] / count,
            "matching": named["matching"] / count,
            "mel": named["mel"] / count,
            "phase": named["phase"] / count,
            "realization_id": ",".join(realization_ids),
            "sample_id": ",".join(sample_ids),
            "scan_configured": report["configured"],
            "scan_executed": report["executed"],
        }

    def _generator_forward(
        self,
        runtime: ProcessRuntime,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        *,
        preview: bool,
    ) -> dict[str, torch.Tensor]:
        module = self.generator.unwrapped() if preview else self.generator.execution()
        error: str | None = None
        parts: dict[str, torch.Tensor] | None = None
        try:
            parts = pack_forward(module(clean, noisy))
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("generator forward failed: " + "; ".join(item for item in errors if item))
        if parts is None:
            raise RuntimeFailure("generator forward produced no tensors")
        return parts

    def _discriminator_loss(self, runtime: ProcessRuntime, clean: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        error: str | None = None
        loss: torch.Tensor | None = None
        try:
            real_q, fake_q, _real_qf, _fake_qf = self.cqt(clean, audio)
            real_r, fake_r, _real_rf, _fake_rf = self.mrd(clean, audio)
            loss = call_objective(
                self.discriminator_objective,
                {
                    "fake_cqt": fake_q,
                    "fake_mrd": fake_r,
                    "real_cqt": real_q,
                    "real_mrd": real_r,
                },
            ).total
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("discriminator forward failed: " + "; ".join(item for item in errors if item))
        if loss is None:
            raise RuntimeFailure("discriminator loss is missing")
        return loss

    def _generator_backward(
        self,
        runtime: ProcessRuntime,
        clean: torch.Tensor,
        parts: dict[str, torch.Tensor],
        components: dict[str, Any],
    ) -> torch.Tensor:
        error: str | None = None
        total: torch.Tensor | None = None
        terms: dict[str, torch.Tensor] | None = None
        try:
            band = self.cqt.unwrapped().discriminator
            mrd = self.mrd.unwrapped().discriminator
            _real_q, fake_q, real_qf, fake_qf = band(clean, parts["audio"])
            _real_r, fake_r, real_rf, fake_rf = mrd(clean, parts["audio"])
            result = call_objective(
                self.generator_objective,
                {
                    "clean_complex": parts["clean_complex"],
                    "clean_mag": parts["clean_mag"],
                    "clean_phase": parts["clean_phase"],
                    "estimate": parts["audio"],
                    "fake_cqt": fake_q,
                    "fake_cqt_features": fake_qf,
                    "fake_mrd": fake_r,
                    "fake_mrd_features": fake_rf,
                    "generated_complex": parts["complex"],
                    "generated_mag": parts["mag"],
                    "generated_phase": parts["phase"],
                    "real_cqt_features": real_qf,
                    "real_mrd_features": real_rf,
                    "recomputed_complex": parts["recomputed"],
                    "reference": clean,
                    "sample_rate": int(self.generator.unwrapped().sampling_rate),
                },
            )
            total = result.total
            terms = result.terms
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("generator loss failed: " + "; ".join(item for item in errors if item))
        if total is None or terms is None:
            raise RuntimeFailure("generator loss is missing")
        _backward_if_finite(runtime, total / self.accumulation, "generator loss")
        for name, value in terms.items():
            current = components.get(name, 0.0)
            components[name] = current + value.detach()
        return total.detach()


def _waveforms(batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
    batch = runtime.move(batch)
    clean = batch.targets
    noisy = batch.inputs
    if clean.ndim != 2 or clean.shape != noisy.shape:
        raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
    return clean, noisy


def _backward_if_finite(runtime: ProcessRuntime, loss: torch.Tensor, what: str) -> None:
    if runtime.device.type != "cuda":
        finite = bool(torch.isfinite(loss.detach()).all())
        if not runtime.agree(finite, policy="all"):
            raise RuntimeFailure(f"{what} is not finite on every rank")
    runtime.backward(loss)


def _grad_norm(*modules: torch.nn.Module) -> float:
    pieces = [
        parameter.grad.detach().square().sum()
        for module in modules
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not pieces:
        return 0.0
    values = torch.stack(pieces).detach().cpu().tolist()
    return sum(float(value) for value in values) ** 0.5


def _clone_grads(*modules: torch.nn.Module) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for module in modules:
        prefix = module.__class__.__name__
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            cloned[f"{prefix}.{name}"] = parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
    return cloned


def _inputs_ready(clean: torch.Tensor, noisy: torch.Tensor) -> bool:
    if clean.ndim != 2 or clean.shape != noisy.shape:
        return False
    if int(clean.shape[-1]) <= 1024:
        return False
    if clean.is_cuda:
        return True
    return bool(torch.isfinite(clean).all()) and bool(torch.isfinite(noisy).all())
