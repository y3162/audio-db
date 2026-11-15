"""Official MP-SENet validation report for one batch.

The generator sees the full waveform. Official ``center=True`` iSTFT may
return the largest hop multiple that does not exceed that length. This module
does not trim the model input. ``align_validation_length`` stays importable
so a test can prove the report does not call it.

``official=True`` puts PESQ failures in the mean as -1, matching
``eval_pesq``. The research path still drops ``na`` and errors. A missing
PESQ library is ``unavailable`` on both paths, not -1 and not 0.

No optimizer is built here. The caller supplies the generator and, for the
official loss, the metric discriminator.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.calc.errors import TrainingError
from src.experiments.calc.losses import discriminator_loss, generator_losses, metric_targets
from src.experiments.calc.stft import mag_pha_istft, mag_pha_stft
from src.experiments.official.mp_senet.lengths import (
    align_validation_length,  # noqa: F401  # tests patch this name; report_batch does not call it
    official_center_istft_length,
    require_discriminator_resolution,
)

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
