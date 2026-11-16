"""MP-SENet validation, PESQ selection, and ``best_generator.pt``.

Rank 0 scores a shadow copy and restores the training cursor and RNG.
Selection and the inference file run only when the trainer asks, after
validation and before the epoch schedulers. This module does not step an
optimizer or offer a checkpoint.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.calc.errors import TrainingError
from src.experiments.calc.losses import discriminator_loss, generator_losses, metric_targets
from src.experiments.calc.rng import seed_model_init
from src.experiments.calc.stft import mag_pha_istft, mag_pha_stft
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.execution.journal import rank_of
from src.models.mp_senet.dataset import (
    align_validation_length,  # noqa: F401  # tests patch this name; report_batch does not call it
    official_center_istft_length,
    require_discriminator_resolution,
)
from src.models.mp_senet.model import MetricDiscriminator, network_config
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.models import build_model

# --- One validation batch. ``align_validation_length`` is not used to cut the waveform. ---

OFFICIAL_PESQ_POLICY = "official_include_negative_one"
RESEARCH_PESQ_POLICY = "research_exclude_na"


def report_batch(
    *,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    spec: dict[str, Any],
    device: torch.device,
    batch: dict[str, Any],
    official: bool,
) -> dict[str, Any]:
    """Losses and reference metrics for one validation batch.

    The caller has already switched to eval mode and disabled gradients.
    ``align_validation_length`` is not used to cut this waveform.
    """

    clean, noisy = _waveforms(batch, device)
    with torch.no_grad():
        parts = _generate(clean, noisy, generator=generator, spec=spec)
    audio = parts["audio"]
    overlap = min(int(clean.shape[-1]), int(audio.shape[-1]))
    if overlap <= 0:
        time_metric = _metric(status="na", reason="iSTFT audio and reference have no overlap")
        time_value = None
    else:
        time_value = float(F.l1_loss(clean[..., :overlap], audio[..., :overlap]).detach())
        time_metric = _metric(time_value)
    magnitude = float(F.mse_loss(parts["clean_mag"], parts["mag"]).detach())
    losses: dict[str, dict[str, Any]] = {
        "time": time_metric,
        "magnitude": _metric(magnitude),
    }
    generator_value = time_value
    if official and discriminator is not None and parts.get("mag_hat") is not None and overlap > 0:
        try:
            require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
            ones = torch.ones(clean.shape[0], device=device)
            metric = discriminator(parts["clean_mag"], parts["mag_hat"])
            detailed = generator_losses(
                parts["clean_mag"],
                parts["mag"],
                parts["clean_phase"],
                parts["phase"],
                parts["clean_complex"],
                parts["complex"],
                parts["recomputed_complex"],
                clean[..., :overlap],
                audio[..., :overlap],
                metric,
                ones,
            )
            generator_value = float(detailed["total"].detach())
            for name in ("phase", "complex", "consistency", "metric"):
                losses[name] = _metric(float(detailed[name].detach()))
            real = discriminator(parts["clean_mag"], parts["clean_mag"])
            fake = discriminator(parts["clean_mag"], parts["mag_hat"])
            targets = metric_targets(clean, audio, int(spec["sampling_rate"]))
            losses["discriminator"] = _metric(float(discriminator_loss(real, fake, ones, targets).detach()))
        except (TrainingError, RuntimeError, ValueError) as exc:
            losses["discriminator"] = _metric(status="na", reason=str(exc))
    elif official and discriminator is not None:
        losses["discriminator"] = _metric(
            status="na",
            reason="iSTFT audio is too short to form the consistency spectrogram",
        )
    else:
        losses["discriminator"] = _metric(
            status="na",
            reason="reconstruction loss has no discriminator",
        )
    if generator_value is None:
        losses["generator"] = _metric(status="na", reason="iSTFT audio and reference have no overlap")
    else:
        losses["generator"] = _metric(generator_value)
    rate = int(spec["sampling_rate"])
    pesq = _validation_pesq(clean, audio, rate, official=official)
    stoi_scores = [
        _reference_score(row_clean, row_audio, rate, kind="stoi")
        for row_clean, row_audio in _rows(clean, audio)
    ]
    return {
        "losses": losses,
        "pesq": pesq,
        "stoi": _combine_scores(stoi_scores),
        "audio_lengths": _audio_length_rows(clean, audio),
        "pesq_failures": pesq.get("pesq_failures"),
        "pesq_scored": pesq.get("pesq_scored"),
        "pesq_policy": pesq.get("pesq_policy"),
    }


def official_eval_pesq(clean_utt: Any, esti_utt: Any, sample_rate: int) -> float:
    """One utterance, matching official ``eval_pesq``.

    An exception, a non-finite value, or a negative library error code becomes
    -1. The caller keeps that -1 in the mean. ImportError is not converted:
    a missing library stays unavailable.
    """

    from pesq import PesqError, pesq

    try:
        score = pesq(int(sample_rate), clean_utt, esti_utt, on_error=PesqError.RETURN_VALUES)
    except Exception:
        return -1.0
    try:
        score = float(score)
    except (TypeError, ValueError):
        return -1.0
    if not math.isfinite(score) or score < 0.0:
        return -1.0
    return score


def combine_official_pesq_values(values: list[float]) -> dict[str, Any]:
    """``np.mean`` of official per-utterance scores, including -1."""

    if not values:
        return {
            "status": "na",
            "value": None,
            "reason": "no score",
            "count": 0,
            "pesq_failures": 0,
            "pesq_scored": 0,
            "pesq_policy": OFFICIAL_PESQ_POLICY,
        }
    failures = sum(1 for value in values if not math.isfinite(float(value)) or float(value) < 0.0)
    mean = float(sum(float(value) for value in values)) / float(len(values))
    return {
        "status": "ok",
        "value": mean,
        "reason": None,
        "count": len(values),
        "pesq_failures": failures,
        "pesq_scored": len(values) - failures,
        "pesq_policy": OFFICIAL_PESQ_POLICY,
    }


def _annotate_research_pesq(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Research mean: drop ``na`` and errors. Do not rename this into the official mean."""

    summary = dict(_combine_scores(scores))
    summary["pesq_policy"] = RESEARCH_PESQ_POLICY
    if summary["status"] == "unavailable":
        summary["pesq_failures"] = None
        summary["pesq_scored"] = 0
        return summary
    summary["pesq_scored"] = sum(1 for item in scores if item["status"] == "ok" and item["value"] is not None)
    summary["pesq_failures"] = sum(1 for item in scores if item["status"] != "ok")
    return summary


def _validation_pesq(
    clean: torch.Tensor,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    official: bool,
) -> dict[str, Any]:
    pairs = list(_rows(clean, audio))
    if not official:
        scores = [_reference_score(reference, estimate, sample_rate, kind="pesq") for reference, estimate in pairs]
        return _annotate_research_pesq(scores)
    from src.evaluation.metrics.pesq import library_status

    missing = library_status()
    if missing:
        return _pesq_unavailable(str(missing), OFFICIAL_PESQ_POLICY)
    values: list[float] = []
    try:
        for reference, estimate in pairs:
            values.append(official_eval_pesq(_waveform_numpy(reference), _waveform_numpy(estimate), sample_rate))
    except ImportError as exc:
        return _pesq_unavailable(str(exc), OFFICIAL_PESQ_POLICY)
    return combine_official_pesq_values(values)


def _generate(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    *,
    generator: torch.nn.Module,
    spec: dict[str, Any],
) -> dict[str, torch.Tensor]:
    hop = int(spec["hop_size"])
    clean_mag, clean_phase, clean_complex = mag_pha_stft(
        clean, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
    )
    noisy_mag, noisy_phase, _noisy_complex = mag_pha_stft(
        noisy, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
    )
    magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
    audio = mag_pha_istft(magnitude, phase, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"])
    waveform_length = int(clean.shape[-1])
    audio_length = int(audio.shape[-1])
    if audio_length != waveform_length:
        official_length = official_center_istft_length(waveform_length, hop)
        if audio_length != official_length:
            raise TrainingError(
                f"iSTFT length {audio_length} does not match the waveform length {waveform_length}"
            )
    try:
        magnitude_hat, phase_hat, recomputed = mag_pha_stft(
            audio, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
        )
    except (RuntimeError, TrainingError):
        if audio_length == waveform_length:
            raise
        magnitude_hat = None
    else:
        magnitude_hat = (magnitude_hat, phase_hat, recomputed)
    result = {
        "clean_mag": clean_mag,
        "clean_phase": clean_phase,
        "clean_complex": clean_complex,
        "mag": magnitude,
        "phase": phase,
        "complex": complex_spectrum,
        "audio": audio,
    }
    if magnitude_hat is not None:
        hat, phase_hat, recomputed = magnitude_hat
        result["mag_hat"] = hat
        result["phase_hat"] = phase_hat
        result["recomputed_complex"] = recomputed
    return result


def _waveforms(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    clean = batch["clean"].to(device)
    noisy = batch["noisy"].to(device)
    if clean.ndim != 2 or clean.shape != noisy.shape:
        raise TrainingError(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
    return clean, noisy


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}


def _rows(clean: torch.Tensor, audio: torch.Tensor):
    return zip(clean.detach().cpu(), audio.detach().cpu(), strict=True)


def _reference_score(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int, *, kind: str) -> dict[str, Any]:
    if kind == "pesq":
        from src.evaluation.metrics.pesq import score_pesq

        result = score_pesq(reference, estimate, sample_rate)
    else:
        from src.evaluation.metrics.stoi import score_stoi

        result = score_stoi(reference, estimate, sample_rate, extended=False)
    status = result["status"]
    if status == "ok":
        return _metric(float(result["value"]))
    if status == "unavailable":
        return _metric(status="unavailable", reason=str(result["reason"]))
    return _metric(status="na", reason=str(result.get("reason") or status))


def _combine_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if any(item["status"] == "unavailable" for item in scores):
        reason = next(item["reason"] for item in scores if item["status"] == "unavailable")
        return {"status": "unavailable", "value": None, "reason": reason, "count": 0}
    values = [float(item["value"]) for item in scores if item["status"] == "ok" and item["value"] is not None]
    if values:
        return {"status": "ok", "value": sum(values) / len(values), "reason": None, "count": len(values)}
    reason = next((item["reason"] for item in scores if item["reason"]), "no score")
    return {"status": "na", "value": None, "reason": reason, "count": 0}


def _audio_length_rows(clean: torch.Tensor, audio: torch.Tensor) -> list[dict[str, int]]:
    model_input = int(clean.shape[-1])
    produced = int(audio.shape[-1])
    overlap = min(model_input, produced)
    return [
        {
            "model_input": model_input,
            "clean": model_input,
            "audio": produced,
            "pesq_reference": model_input,
            "pesq_estimate": produced,
            "time_overlap": overlap,
        }
        for _index in range(int(clean.shape[0]))
    ]


def _pesq_unavailable(reason: str, policy: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "value": None,
        "reason": reason,
        "count": 0,
        "pesq_failures": None,
        "pesq_scored": 0,
        "pesq_policy": policy,
    }


def _waveform_numpy(waveform: torch.Tensor) -> Any:
    return waveform.detach().to(dtype=torch.float32).contiguous().numpy()

# --- Mean of the rank-0 reports. Unavailable stays unavailable. ---

def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of batch-level official reports. Unavailable stays unavailable."""

    names: list[str] = []
    for report in reports:
        for name in report["losses"]:
            if name not in names:
                names.append(name)
    losses = {
        name: combine_metric([report["losses"][name] for report in reports if name in report["losses"]])
        for name in names
    }
    pesq = combine_metric([report["pesq"] for report in reports])
    policy = next((report["pesq"].get("pesq_policy") for report in reports if isinstance(report.get("pesq"), dict)), None)
    if policy is not None:
        pesq["pesq_policy"] = policy
    lengths: list[dict[str, int]] = []
    for report in reports:
        rows = report.get("audio_lengths")
        if isinstance(rows, list):
            lengths.extend(rows)
    return {
        "audio_lengths": lengths,
        "losses": losses,
        "pesq": pesq,
        "stoi": combine_metric([report["stoi"] for report in reports]),
    }


def combine_metric(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean the finite ok values. One unavailable item makes the result unavailable."""

    if any(item.get("status") == "unavailable" for item in items):
        reason = next(item.get("reason") for item in items if item.get("status") == "unavailable")
        return {"status": "unavailable", "value": None, "reason": reason, "count": 0}
    values = [float(item["value"]) for item in items if item.get("status") == "ok" and item.get("value") is not None]
    if values:
        return {"status": "ok", "value": sum(values) / len(values), "reason": None, "count": len(values)}
    reason = next((item.get("reason") for item in items if item.get("reason")), "no score")
    return {"status": "na", "value": None, "reason": reason, "count": 0}

# --- Strict PESQ selection. The trainer stores the decision. ---

def consider_best_pesq(
    epoch: int,
    summary: dict[str, Any],
    *,
    start_epoch: int,
    best_pesq: float | None,
    logical_step: int,
) -> dict[str, Any]:
    """Keep the earlier checkpoint unless PESQ is strictly higher after the start epoch.

    The returned record does not change ``best_pesq``. The trainer stores
    ``pesq``, ``epoch``, and ``logical_step`` when ``selected`` is true.
    """

    pesq = summary.get("pesq")
    if not isinstance(pesq, dict):
        pesq = {"status": "na", "value": None, "reason": "missing pesq"}
    status = pesq.get("status")
    value = pesq.get("value")
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "logical_step": int(logical_step),
        "pesq": None if value is None else float(value),
        "pesq_status": status,
        "reason": None,
        "selected": False,
    }
    if epoch < start_epoch:
        record["reason"] = "before_start_epoch"
        return record
    if status != "ok" or value is None:
        record["reason"] = "unavailable" if status == "unavailable" else "not_ok"
        return record
    score = float(value)
    if not math.isfinite(score):
        record["reason"] = "non_finite"
        return record
    if best_pesq is not None and score <= float(best_pesq):
        record["reason"] = "not_higher"
        return record
    record["selected"] = True
    record["reason"] = "improved"
    return record


def apply_pesq_selection(
    progress: Any,
    epoch: int,
    summary: dict[str, Any],
    *,
    start_epoch: int,
) -> dict[str, Any]:
    """Decide the official PESQ rule and store a strict improvement on ``progress``."""

    decision = consider_best_pesq(
        epoch,
        summary,
        start_epoch=start_epoch,
        best_pesq=progress.best_pesq,
        logical_step=progress.logical_step,
    )
    progress.remember_best(decision)
    return decision

# --- Inference weights. This is not a resume checkpoint. ---

def write_best(
    directory: Path,
    exported: dict[str, torch.Tensor],
    decision: dict[str, Any],
    experiment_claim: str,
) -> None:
    payload_path = directory / "best_generator.pt"
    meta_path = directory / "best_generator.json"
    temporary = directory / ".best_generator.pt.tmp"
    torch.save(exported, temporary)
    os.replace(temporary, payload_path)
    meta = {
        "checkpoint": payload_path.name,
        "epoch": decision["epoch"],
        "experiment_claim": experiment_claim,
        "logical_step": decision["logical_step"],
        "metric": "pesq",
        "pesq": decision["pesq"],
        "pesq_policy": "official_include_negative_one",
    }
    meta_temporary = directory / ".best_generator.json.tmp"
    meta_temporary.write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(meta_temporary, meta_path)


def publish_best(
    runtime: Any,
    directory: Path,
    decision: dict[str, Any],
    exported: dict[str, torch.Tensor],
    experiment_claim: str,
) -> None:
    """Rank 0 writes the selected generator. Every rank hears a write error."""

    error: str | None = None
    if rank_of(runtime) == 0 and decision["selected"]:
        try:
            write_best(directory, exported, decision, experiment_claim)
        except (OSError, RuntimeError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = runtime.gather_objects(error)
    failure = next((item for item in errors if item), None)
    if failure:
        raise RuntimeFailure(f"best checkpoint write failed: {failure}")

# --- Rank-0 shadow validation. The trainer chooses the epoch. ---

class _Shadow:
    """Eval-mode weights for one report. Not a training task."""

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        architecture: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.spec = architecture
        self.device = device

    def eval_mode(self) -> None:
        self.generator.eval()
        self.discriminator.eval()

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        return report_batch(
            generator=self.generator,
            discriminator=self.discriminator,
            spec=self.spec,
            device=self.device,
            batch=batch,
            official=True,
        )


class MPSENetOfficialValidator:
    """Rank-0 official report. The caller decides the epoch and what to do next."""

    def __init__(
        self,
        *,
        generator: PreparedModel,
        discriminator: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        # Kept so a caller can see which optimizer group the live model uses.
        # The shadow does not construct another optimizer from them.
        self.optim_g = optim_g
        self.scheduler_g = scheduler_g
        self._shadow: _Shadow | None = None

    def invalidate(self) -> None:
        """Drop the cached shadow. The next call builds a new one."""

        self._shadow = None

    def evaluate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        """Run the official validation report on rank 0 and share that summary."""

        cursor = (int(self.data.epoch), int(self.data.update_index))
        rng = runtime.capture_rng()
        error: str | None = None
        summary: dict[str, Any] | None = None
        try:
            if runtime.rank == 0:
                try:
                    summary = self._rank0(trainer_epoch, runtime, component_id)
                except (TrainingError, RuntimeFailure, RuntimeError, ValueError, OSError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
        finally:
            runtime.restore_rng(rng)
        errors = runtime.gather_objects(error)
        failure = next((item for item in errors if item), None)
        if failure:
            raise RuntimeFailure(f"validation failed: {failure}")
        gathered = runtime.gather_objects(summary)
        shared = gathered[0]
        if not isinstance(shared, dict):
            raise RuntimeFailure(f"component {component_id}: rank 0 did not return a validation summary")
        moved = (int(self.data.epoch), int(self.data.update_index))
        if moved != cursor:
            raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return shared

    def _rank0(self, trainer_epoch: int, runtime: ProcessRuntime, component_id: str) -> dict[str, Any]:
        shadow = self._shadow_for(runtime)
        reports: list[dict[str, Any]] = []
        for batch in self.data.validation_epoch(trainer_epoch):
            if not isinstance(batch, Batch):
                raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
            batch = runtime.move(batch)
            report = shadow.validation_report({"clean": batch.targets, "noisy": batch.inputs})
            reports.append(report)
        if not reports:
            raise RuntimeFailure(f"component {component_id}: rank 0 validation produced no batches")
        return aggregate_reports(reports)

    def _shadow_for(self, runtime: ProcessRuntime) -> _Shadow:
        if self._shadow is None:
            architecture = dict(self.generator.unwrapped().architecture)
            seed = int(self.generator.unwrapped().seed)
            self._shadow = _build_shadow(architecture, seed, runtime.device)
        shadow = self._shadow
        shadow.generator.load_state_dict(self.generator.unwrapped().generator.state_dict(), strict=True)
        shadow.discriminator.load_state_dict(
            self.discriminator.unwrapped().discriminator.state_dict(), strict=True
        )
        shadow.eval_mode()
        return shadow


def _build_shadow(architecture: dict[str, Any], seed: int, device: torch.device) -> _Shadow:
    """Build eval modules under the same seed as the reference task, without an optimizer.

    The discarded generator consumes the same initialization draws the metric
    discriminator expects. Its weights are replaced before the report.
    """

    with seed_model_init(seed):
        generator = build_model("mp_senet", network_config(architecture)).to(device)
        if "disc_dim" in architecture:
            discriminator = MetricDiscriminator(dim=int(architecture["disc_dim"])).to(device)
        else:
            discriminator = MetricDiscriminator().to(device)
    return _Shadow(generator, discriminator, architecture, device)
