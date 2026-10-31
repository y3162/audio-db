"""Render a canonical recipe to a waveform.

Additive noise uses the mean-square definition implemented by ``mix_snr``.
The stored ``noise_start`` is the crop; the seed is not consulted again.

RIR version 1 convolves causally and keeps the input length (``trim_to_input``).
Sample 0 of the reverberant signal stays aligned with sample 0 of the dry
input. The tail past the input length is discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from src.data.recipe import NOISE_START_UNIT, RecipeError, canonical_recipe, sha256_file
from src.evaluation.audio.io import audio_span
from src.evaluation.audio.mixing import POWER_EPS, mix_snr

__all__ = ["RenderResult", "plan_noise_start", "render_recipe"]


@dataclass
class RenderResult:
    observed: torch.Tensor
    reference: torch.Tensor | None
    reference_name: str
    sample_rate: int
    span: dict[str, Any]
    snr_db: float | None
    snr_db_achieved: float | None
    noise_start: int | None
    seed: int | None
    peak_scaled: bool
    clean_silent: bool
    operations: list[str] = field(default_factory=list)


def plan_noise_start(
    clean: torch.Tensor,
    noise: torch.Tensor,
    seed: int,
) -> int:
    """Crop index after the noise has been resampled to the clean rate."""
    from src.evaluation.audio.mixing import crop_start

    return crop_start(int(noise.numel()), int(clean.numel()), seed)


def prepare_noise_start(
    clean_path: str | Path,
    noise_path: str | Path,
    output_rate: int,
    seed: int,
    max_seconds: float | None,
) -> int:
    """Choose the stored crop once. Rendering uses the returned index, not the seed."""
    clean, clean_rate = torchaudio_load(Path(clean_path))
    noise, noise_rate = torchaudio_load(Path(noise_path))
    clean_wave, _span = _fit_rate((clean, clean_rate), output_rate, max_seconds)
    noise_wave, _noise_span = _fit_rate((noise, noise_rate), output_rate, None)
    return plan_noise_start(clean_wave, noise_wave, seed)


def render_recipe(recipe: dict[str, Any], root: str | Path) -> RenderResult:
    document = canonical_recipe(recipe)
    base = Path(root)
    loaded = {name: _load_source(base, item) for name, item in document["sources"].items()}
    output = document["output"]
    rate = int(output["sample_rate"])
    max_seconds = output.get("max_seconds")
    dry = None
    span = {"truncated": False, "max_seconds": max_seconds, "sample_rate": rate}
    if "clean" in loaded:
        dry, span = _fit_rate(loaded["clean"], rate, max_seconds)
    clean_signal = dry
    observed = None if dry is None else dry.clone()
    reverberant = None
    snr_db = None
    snr_achieved = None
    noise_start = None
    seed = None
    peak_scaled = False
    clean_silent = False
    names: list[str] = []
    for operation in document["operations"]:
        names.append(operation["type"])
        if operation["type"] == "rir":
            kernel = _load_rir_channel(base, document["sources"][operation["source"]], int(operation["channel"]), rate)
            if clean_signal is None or observed is None:
                raise RecipeError("rir has no clean signal")
            clean_signal = _convolve_trim(clean_signal, kernel)
            observed = _convolve_trim(observed, kernel)
            reverberant = clean_signal
        elif operation["type"] == "additive_noise":
            if observed is None:
                raise RecipeError("additive_noise has no signal to mix")
            noise = _fit_rate(loaded[operation["source"]], rate, None)[0]
            start = int(operation["noise_start"])
            if operation["noise_start_unit"] != NOISE_START_UNIT:
                raise RecipeError("noise_start_unit must be 'sample'")
            mixed = mix_snr(
                observed,
                noise,
                rate,
                rate,
                float(operation["snr_db"]),
                int(operation["seed"]),
                noise_start=start,
            )
            if mixed.noise_start != start:
                raise RecipeError("additive_noise changed the stored noise_start")
            observed = mixed.noisy
            snr_db = mixed.snr_db_requested
            snr_achieved = mixed.snr_db_achieved
            noise_start = mixed.noise_start
            seed = int(operation["seed"])
            clean_silent = mixed.clean_silent
        elif operation["type"] == "peak_normalize":
            observed, dry, clean_signal, scaled = _peak(
                observed, dry, clean_signal, float(operation["limit"])
            )
            if reverberant is not None:
                reverberant = clean_signal
            peak_scaled = peak_scaled or scaled
        else:
            raise RecipeError(f"unknown operation type: {operation['type']!r}")
    reference_name = output["reference"]
    if not document["operations"]:
        observed, reference, span = _recorded(loaded, reference_name, rate, max_seconds)
    else:
        reference = _select_reference(reference_name, dry, reverberant)
    if observed is None:
        raise RecipeError("recipe did not produce a waveform")
    return RenderResult(
        observed=observed.contiguous(),
        reference=None if reference is None else reference.contiguous(),
        reference_name=reference_name,
        sample_rate=rate,
        span=span,
        snr_db=snr_db,
        snr_db_achieved=snr_achieved,
        noise_start=noise_start,
        seed=seed,
        peak_scaled=peak_scaled,
        clean_silent=clean_silent,
        operations=names,
    )


def _recorded(loaded, reference_name: str, rate: int, max_seconds):
    if "noisy" in loaded:
        observed, span = _fit_rate(loaded["noisy"], rate, max_seconds)
    elif "clean" in loaded:
        observed, span = _fit_rate(loaded["clean"], rate, max_seconds)
    else:
        raise RecipeError("recorded recipe has no clean or noisy source")
    reference = None
    if reference_name == "dry_clean":
        reference, clean_span = _fit_rate(loaded["clean"], rate, max_seconds)
        if "noisy" not in loaded:
            span = clean_span
    elif reference_name == "none":
        reference = None
    else:
        raise RecipeError(f"recorded recipe cannot use reference {reference_name}")
    return observed, reference, span


def _select_reference(name: str, dry, reverberant):
    if name == "none":
        return None
    if name == "dry_clean":
        if dry is None:
            raise RecipeError("dry_clean reference is missing")
        return dry
    if name == "reverberant_clean":
        if reverberant is None:
            raise RecipeError("reverberant_clean reference is missing")
        return reverberant
    raise RecipeError(f"unknown reference {name}")


def _load_source(root: Path, source: dict[str, str]) -> tuple[torch.Tensor, int]:
    path = Path(os.path.normpath(root / source["path"]))
    if not path.is_file():
        raise RecipeError(f"source file not found: {source['path']}")
    digest = sha256_file(path)
    if digest != source["sha256"]:
        raise RecipeError(f"source file changed: {source['path']}")
    waveform, sample_rate = torchaudio_load(path)
    return waveform, int(sample_rate)


def torchaudio_load(path: Path) -> tuple[torch.Tensor, int]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.numel() == 0:
        raise RecipeError(f"source audio is empty: {path}")
    wave = waveform.detach().cpu().to(torch.float32).contiguous()
    if not torch.isfinite(wave).all():
        raise RecipeError(f"source audio contains non-finite samples: {path}")
    return wave, int(sample_rate)


def _fit_rate(loaded: tuple[torch.Tensor, int], rate: int, max_seconds) -> tuple[torch.Tensor, dict[str, Any]]:
    waveform, source_rate = loaded
    source_frames = int(waveform.shape[-1])
    if max_seconds is not None:
        keep = max(1, int(float(max_seconds) * source_rate))
        waveform = waveform[..., : min(source_frames, keep)]
    used = int(waveform.shape[-1])
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform[0]
    if source_rate != rate:
        waveform = AF.resample(waveform, source_rate, rate)
        if waveform.numel() == 0:
            raise RecipeError("audio is empty after resampling")
    span = audio_span(source_frames, used, source_rate, None if max_seconds is None else float(max_seconds))
    span["sample_rate"] = rate
    return waveform.contiguous(), span


def _load_rir_channel(root: Path, source: dict[str, str], channel: int, rate: int) -> torch.Tensor:
    waveform, source_rate = _load_source(root, source)
    channels = int(waveform.shape[0])
    if channel < 0 or channel >= channels:
        raise RecipeError(f"rir channel {channel} is outside 0..{channels - 1}")
    kernel = waveform[channel]
    if source_rate != rate:
        kernel = AF.resample(kernel, source_rate, rate)
    if kernel.numel() == 0:
        raise RecipeError("rir is empty after resampling")
    if float(kernel.detach().abs().sum()) < POWER_EPS:
        raise RecipeError("rir has zero energy")
    return kernel.contiguous()


def _convolve_trim(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if kernel.numel() == 0:
        raise RecipeError("rir is empty")
    width = int(kernel.numel())
    full = F.conv1d(
        signal.reshape(1, 1, -1),
        kernel.flip(0).reshape(1, 1, -1),
        padding=width - 1,
    ).reshape(-1)
    if full.numel() < signal.numel():
        raise RecipeError("convolution is shorter than the input")
    return full[: signal.numel()].contiguous()


def _peak(observed, dry, clean_signal, limit: float):
    if observed is None:
        raise RecipeError("peak_normalize has no signal")
    peak = float(observed.detach().abs().max()) if observed.numel() else 0.0
    if peak <= limit or peak <= 0.0:
        return observed, dry, clean_signal, False
    gain = limit / peak
    observed = observed * gain
    if dry is not None:
        dry = dry * gain
    if clean_signal is not None:
        clean_signal = clean_signal * gain
    return observed, dry, clean_signal, True
