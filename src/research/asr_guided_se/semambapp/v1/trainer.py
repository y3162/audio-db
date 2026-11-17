"""SEMamba++ research trainer. All three switches off keeps the official update object."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import src.models.semambapp.trainer as semamba_trainer
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.runtime import ProcessRuntime
from src.experiments.training.control import weight_loss
from src.experiments.calc.errors import TrainingError
from src.models.semambapp.trainer import SEMambaPPTrainer, SEMambaUpdate
from src.research.asr_guided_se.asr import TeacherCache, fingerprint_text
from src.research.asr_guided_se.error_mask import crop_error_mask, overlapping_tokens
from src.research.asr_guided_se.semambapp.v1.config import ASR_HOP, STFT_HOP, GuidanceV1
from src.research.asr_guided_se.semambapp.v1.evaluator import predict_audio
from src.research.asr_guided_se.semambapp.v1.loss import guided_generator_loss
from src.research.asr_guided_se.tdt import mean_tdt


class GuidedSEMambaPPTrainer(SEMambaPPTrainer):
    """Official GAN loop with the v1 switches. ``experiment_claim`` is ``research``."""

    def __init__(self, *args: Any, guidance: GuidanceV1, asr: Any = None, experiment_claim: str = "research", **kwargs: Any) -> None:
        if not isinstance(guidance, GuidanceV1):
            raise BuildError("guidance must be GuidanceV1")
        if experiment_claim != "research":
            raise BuildError("ASR-guided SEMamba++ v1 requires experiment_claim research")
        kwargs.setdefault("claim_label", "research")
        super().__init__(*args, **kwargs)
        self.guidance = guidance
        self.asr = asr
        self.experiment_claim = experiment_claim
        self._hypotheses: Path | None = None
        self._cache: TeacherCache | None = None
        if guidance.uses_asr:
            self._require_asr()
        if guidance.uses_asr or guidance.asr_features_enabled:
            self._cache = TeacherCache(self._fingerprint(), writer=None)
            previous = self._update
            self._update = GuidedUpdate(
                generator=previous.generator,
                mrd=previous.mrd,
                cqt=previous.cqt,
                optim_g=previous.optim_g,
                optim_d=previous.optim_d,
                generator_objective=previous.generator_objective,
                discriminator_objective=previous.discriminator_objective,
                accumulation=previous.accumulation,
                grad_clip=previous.grad_clip,
                guidance=guidance,
                asr=asr,
                data=self.data,
                cache=self._cache,
            )

    def research_predict(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Enhanced waveform. Inputs are noisy and targets are clean."""

        module = self.generator.unwrapped()
        was_training = bool(module.training)
        module.eval()
        module.bind_batch_lookup(lambda _clean: None)
        try:
            with torch.no_grad():
                return predict_audio(module, inputs, targets)
        finally:
            module.bind_batch_lookup(None)
            if was_training:
                module.train()

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        component_id = super()._prepare_run(runtime)
        if self.asr is None:
            return component_id
        directory = Path(self.lifecycle.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        if int(getattr(runtime, "rank", 0)) == 0:
            payload = self.asr.metadata() if hasattr(self.asr, "metadata") else {"model_id": self._fingerprint()}
            (directory / "asr_decode.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        if self._cache is not None:
            path = directory / f"asr_hypotheses.rank{int(getattr(runtime, 'rank', 0))}.jsonl"
            self._hypotheses = path
            header = self.asr.metadata() if hasattr(self.asr, "metadata") else {}

            def write(record: dict[str, Any]) -> None:
                row = dict(header)
                row.update(record)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

            self._cache.writer = write
        return component_id

    def _require_asr(self) -> None:
        if self.asr is None:
            raise BuildError("this guidance setting requires the frozen Parakeet component")
        self.asr.eval()
        for parameter in self.asr.parameters():
            parameter.requires_grad_(False)
        owned = {id(parameter) for group in self.optim_g.param_groups for parameter in group["params"]}
        owned.update(id(parameter) for group in self.optim_d.param_groups for parameter in group["params"])
        if any(id(parameter) in owned for parameter in self.asr.parameters()):
            raise BuildError("ASR parameters are in an optimizer")
        if int(self.asr.samples_per_frame) != ASR_HOP:
            raise BuildError(f"ASR samples per encoder frame must be {ASR_HOP}, got {self.asr.samples_per_frame}")
        if int(self.generator.unwrapped().hop) != STFT_HOP:
            raise BuildError(f"SEMamba++ hop must be {STFT_HOP}, got {self.generator.unwrapped().hop}")
        if self.guidance.asr_features_enabled:
            held = self.generator.unwrapped().asr
            if held is not self.asr:
                raise BuildError("generator and trainer must share the frozen ASR")
            if self.guidance.encoder_layer >= int(self.asr.encoder_layers):
                raise BuildError("encoder_layer is outside the ASR encoder")
        if self.guidance.tdt_loss_enabled:
            ensure = getattr(self.asr, "ensure_tdt", None)
            if not callable(ensure):
                raise BuildError("the ASR component cannot build the TDT loss")
            ensure()

    def _fingerprint(self) -> str:
        if hasattr(self.asr, "metadata"):
            return fingerprint_text(self.asr.metadata())
        return type(self.asr).__name__

    def preflight_probe(self) -> dict[str, Any]:
        """Official probe, then drop ASR cache entries the probe created.

        Python, NumPy, and torch generators are restored by the experiment
        probe. The teacher cache is not one of those generators. Leaving the
        probe's recognitions in place would make step 1 read a cache that a
        run without the probe does not have. Numba's own device state is not
        part of this snapshot.
        """

        cache = self._cache
        items = getattr(cache, "_items", None)
        before = set(items) if isinstance(items, dict) else None
        path = self._hypotheses
        before_size = path.stat().st_size if path is not None and path.is_file() else None
        try:
            return super().preflight_probe()
        finally:
            if isinstance(items, dict) and before is not None:
                for key in list(items):
                    if key not in before:
                        del items[key]
            if path is not None and before_size is None and path.is_file():
                path.unlink()
            elif path is not None and before_size is not None and path.is_file():
                with path.open("a+b") as handle:
                    handle.truncate(before_size)


class GuidedUpdate(SEMambaUpdate):
    """Official step, with a batch lookup and a different generator loss when a switch needs it."""

    def __init__(self, *args: Any, guidance: GuidanceV1, asr: Any, data: Any, cache: TeacherCache | None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.guidance = guidance
        self.asr = asr
        self.data = data
        self.cache = cache
        self.protocol = getattr(data, "protocol", None)
        self._rows: dict[int, Any] = {}
        self._tdt_terms: list[torch.Tensor] = []
        self._mask_errors = 0
        self._mask_total = 0

    def run(self, batches, runtime, phase_hook, *, apply_optimizer: bool = True):
        self._rows.clear()
        self._tdt_terms.clear()
        self._mask_errors = 0
        self._mask_total = 0
        if self.asr is not None:
            if hasattr(self.asr, "to_device"):
                self.asr.to_device(runtime.device)
            self.asr.eval()
            for parameter in self.asr.parameters():
                parameter.requires_grad_(False)
        if self.guidance.changes_loss:
            error = None
            try:
                for batch in batches:
                    _require_batch_spans(batch)
            except (RuntimeFailure, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            errors = runtime.gather_objects(error)
            if any(errors):
                raise RuntimeFailure(
                    "crop correspondence is missing: " + "; ".join(item for item in errors if item)
                )
        module = self.generator.unwrapped()
        module.bind_batch_lookup(lambda clean: self._rows.get(id(clean)))
        original = semamba_trainer._waveforms

        def wrapped(batch, bound_runtime):
            moved = bound_runtime.move(batch)
            clean, noisy = original(moved, bound_runtime)
            self._rows[id(clean)] = moved
            return clean, noisy

        semamba_trainer._waveforms = wrapped
        try:
            result = super().run(batches, runtime, phase_hook, apply_optimizer=apply_optimizer)
        finally:
            semamba_trainer._waveforms = original
            module.bind_batch_lookup(None)
            self._rows.clear()
        observed: dict[str, float] = {}
        if self.guidance.tdt_loss_enabled and self._tdt_terms:
            folded = self._tdt_terms[0]
            for value in self._tdt_terms[1:]:
                folded = folded + value
            observed["tdt"] = float(folded.detach()) / max(len(batches), 1)
        if self._mask_total > 0:
            observed["error_mask_fraction"] = self._mask_errors / self._mask_total
        if observed:
            result = dict(result)
            result.update(observed)
        return result

    def _generator_backward(
        self,
        runtime: ProcessRuntime,
        clean: torch.Tensor,
        parts: dict[str, torch.Tensor],
        components: dict[str, Any],
        scale: int | list[float],
        offset: int,
    ) -> torch.Tensor:
        if not self.guidance.changes_loss:
            return super()._generator_backward(runtime, clean, parts, components, scale, offset)
        error: str | None = None
        total: torch.Tensor | None = None
        terms: dict[str, torch.Tensor] | None = None
        held_error: torch.Tensor | None = None
        held_content: torch.Tensor | None = None
        try:
            band = self.cqt.unwrapped().discriminator
            mrd = self.mrd.unwrapped().discriminator
            _real_q, fake_q, real_qf, fake_qf = band(clean, parts["audio"])
            _real_r, fake_r, real_rf, fake_rf = mrd(clean, parts["audio"])
            context = {
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
            }
            sample_error, content_length, tdt = self._research_terms(clean, parts["audio"])
            observed_error = sample_error.detach()
            observed_content = content_length.detach()
            if not self.guidance.tdt_loss_enabled:
                tdt = None
            if not self.guidance.error_weight_enabled:
                sample_error = None
                content_length = None
            result = guided_generator_loss(
                self.generator_objective,
                context,
                guidance=self.guidance,
                sample_error=sample_error,
                content_length=content_length,
                tdt=tdt,
                stft_hop=int(self.generator.unwrapped().hop),
            )
            total = result.total
            terms = result.terms
            held_error = observed_error
            held_content = observed_content
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("generator loss failed: " + "; ".join(item for item in errors if item))
        if total is None or terms is None:
            raise RuntimeFailure("generator loss is missing")
        semamba_trainer._backward_if_finite(runtime, weight_loss(total, scale, offset), "generator loss")
        self._reject_asr_grad(runtime)
        for name, value in terms.items():
            current = components.get(name, 0.0)
            components[name] = current + value.detach()
        if "tdt" in terms:
            self._tdt_terms.append(terms["tdt"].detach())
        if held_error is not None and held_content is not None:
            mask_errors, mask_count = _content_error_counts(held_error, held_content)
            self._mask_errors += mask_errors
            self._mask_total += mask_count
        return total.detach()

    def _research_terms(
        self,
        clean: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch = self._rows.get(id(clean))
        if batch is None or getattr(batch, "spans", None) is None:
            raise RuntimeFailure("crop correspondence is missing")
        spans = tuple(batch.spans)
        if len(spans) != clean.shape[0]:
            raise RuntimeFailure("span count does not match the micro-batch")
        ids = _realization_ids(batch, int(clean.shape[0]))
        masks: list[list[bool]] = []
        contents: list[int] = []
        targets: list[list[int]] = []
        hop = int(self.asr.samples_per_frame)
        if self.cache is None:
            raise RuntimeFailure("teacher cache is missing")
        for row, span in enumerate(spans):
            checked = _check_span(span, int(clean.shape[-1]))
            full_clean, full_noisy = self._utterance_audio(batch, row, checked, ids[row], clean)
            clean_recognition = self.cache.recognize(self.asr, full_clean, "clean")
            noisy_recognition = self.cache.recognize(self.asr, full_noisy, "noisy")
            crop_start, content_length, _window, full_length = checked
            clean_tokens = overlapping_tokens(clean_recognition.tokens, crop_start, content_length, hop)
            noisy_tokens = overlapping_tokens(noisy_recognition.tokens, crop_start, content_length, hop)
            masks.append(
                crop_error_mask(
                    clean_tokens,
                    noisy_tokens,
                    crop_start=crop_start,
                    content_length=content_length,
                    window=int(clean.shape[-1]),
                    hop=hop,
                    full_length=full_length,
                )
            )
            contents.append(content_length)
            targets.append([int(getattr(token, "token_id")) for token in clean_tokens])
        sample_error = torch.tensor(masks, device=clean.device, dtype=torch.bool)
        content = torch.tensor(contents, device=clean.device, dtype=torch.long)
        tdt = None
        if self.guidance.tdt_loss_enabled:
            per_sample = self.asr.loss_from_ids(audio, content, targets, deterministic_tdt=False)
            tdt = mean_tdt(per_sample)
        return sample_error, content, tdt

    def _utterance_audio(self, batch, row: int, span: tuple[int, int, int, int], realization_id: str, clean: torch.Tensor):
        """Full peak-normalized utterance. A crop uses the protocol; a whole row uses its prefix."""

        crop_start, content_length, window, full_length = span
        if crop_start == 0 and full_length <= window and content_length == full_length:
            return clean[row, :full_length], batch.inputs[row, :full_length]
        protocol = self.protocol
        if protocol is None or not hasattr(protocol, "utterance_for_id"):
            raise RuntimeFailure("crop correspondence is missing; the full utterance is not available")
        utterance = protocol.utterance_for_id(realization_id)
        if int(utterance.full_length) != full_length or int(utterance.example.crop_start) != crop_start:
            raise RuntimeFailure("cached utterance does not match the batch span")
        full_clean = torch.tensor(utterance.full_clean, device=clean.device, dtype=clean.dtype)
        full_noisy = torch.tensor(utterance.full_degraded, device=clean.device, dtype=clean.dtype)
        return full_clean, full_noisy

    def _reject_asr_grad(self, runtime: ProcessRuntime) -> None:
        error = None
        try:
            for parameter in self.asr.parameters():
                if parameter.requires_grad:
                    raise RuntimeFailure("ASR parameter requires grad")
                if parameter.grad is not None:
                    raise RuntimeFailure("ASR parameter received a gradient")
        except RuntimeFailure as exc:
            error = str(exc)
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("frozen ASR changed: " + "; ".join(item for item in errors if item))


def _content_error_counts(mask: torch.Tensor, content: torch.Tensor) -> tuple[int, int]:
    """Error samples inside the content. The detached mask is not a loss."""

    lengths = [int(value) for value in content.detach().to(device="cpu", dtype=torch.long).tolist()]
    cpu_mask = mask.detach().to(device="cpu")
    errors = 0
    total = 0
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        used = min(length, int(cpu_mask.shape[-1]))
        total += used
        if used:
            errors += int(cpu_mask[row, :used].sum().item())
    return errors, total


def _realization_ids(batch, count: int) -> tuple[str, ...]:
    ids = tuple(getattr(batch, "realization_ids", ()) or ())
    if len(ids) == count:
        return ids
    if count == 1 and isinstance(batch.realization_id, str) and batch.realization_id:
        return (batch.realization_id,)
    raise RuntimeFailure("realization ids do not match the micro-batch")


def _require_batch_spans(batch) -> None:
    spans = getattr(batch, "spans", None)
    if not spans:
        raise RuntimeFailure("crop correspondence is missing")
    width = int(batch.targets.shape[-1])
    if len(spans) != int(batch.targets.shape[0]):
        raise RuntimeFailure("span count does not match the micro-batch")
    for span in spans:
        _check_span(span, width)


def _check_span(span, width: int) -> tuple[int, int, int, int]:
    if not isinstance(span, tuple) or len(span) != 4:
        raise RuntimeFailure("span must be (crop_start, content_length, window, full_length)")
    values = []
    for value in span:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeFailure(f"span values must be non-negative integers, got {span}")
        values.append(int(value))
    crop_start, content_length, window, full_length = values
    if window != width:
        raise RuntimeFailure("span window does not match the waveform")
    if content_length > window or content_length > full_length:
        raise RuntimeFailure("content length does not fit the span")
    if crop_start == 0 and full_length <= window:
        if content_length != full_length:
            raise RuntimeFailure("whole-utterance content length does not match the full length")
        return crop_start, content_length, window, full_length
    if crop_start + content_length > full_length:
        raise RuntimeFailure("crop extends past the full utterance")
    return crop_start, content_length, window, full_length
