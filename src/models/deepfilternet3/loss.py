"""DeepFilterNet3 loss terms and the one-batch DFStream loss.

STFT sizes and SNR limits come from the model. The multi-resolution factors
and the local-SNR window are loss parameters. ``component_losses`` is the
sequence of analysis, generator forward, waveform reconstruction, and the
injected objective. The trainer applies cosine before it calls this.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.calc.distributed import unwrap
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.stft import centered_stft
from src.experiments.errors import BuildError, RuntimeFailure
from src.models.deepfilternet3.model import DeepFilterNet3Model
from src.experiments.training.objectives import WeightedObjective, apply_weight, call_objective, require_tensor
from src.models.deepfilternet3.dsp import DFStream, vorbis_window

# --- YAML loss terms. Weights stay on the objective. ---

OFFICIAL_FACTOR = 500.0

OFFICIAL_COMPLEX_FACTOR = 500.0

OFFICIAL_GAMMA = 0.3

OFFICIAL_FFTS = (256, 512, 1024, 2048)

OFFICIAL_MAGNITUDE_FLOOR = 1e-12

OFFICIAL_LSNR_MILLISECONDS = 20

OFFICIAL_LSNR_FLOOR = 1e-12

OFFICIAL_LSNR_WEIGHT = 1e-3

def official_deepfilternet_objective() -> WeightedObjective:
    """Published spectral weight 1 and local-SNR weight 1e-3."""

    spectral = DeepFilterNetSpectralLoss(
        "spectral",
        factor=OFFICIAL_FACTOR,
        complex_factor=OFFICIAL_COMPLEX_FACTOR,
        gamma=OFFICIAL_GAMMA,
        fft_sizes=OFFICIAL_FFTS,
        magnitude_floor=OFFICIAL_MAGNITUDE_FLOOR,
    )
    local = LocalSNRLoss(
        "local_snr",
        milliseconds=OFFICIAL_LSNR_MILLISECONDS,
        energy_floor=OFFICIAL_LSNR_FLOOR,
    )
    return WeightedObjective([spectral, local], {"spectral": 1.0, "local_snr": OFFICIAL_LSNR_WEIGHT})

class DeepFilterNetSpectralLoss(torch.nn.Module):
    """Official multi-resolution magnitude and complex spectral loss.

    Each resolution's two MSEs are scaled inside this term, in FFT order.
    The objective weight is separate and is 1 for the published recipe.
    """

    def __init__(
        self,
        name: str,
        factor: float,
        complex_factor: float,
        gamma: float,
        fft_sizes: Sequence[int],
        magnitude_floor: float,
    ) -> None:
        super().__init__()
        if not isinstance(name, str) or not name:
            raise BuildError("loss term name must be a non-empty string")
        self.name = name
        self.factor = _finite(factor, "factor")
        self.complex_factor = _finite(complex_factor, "complex_factor")
        self.gamma = _finite(gamma, "gamma")
        self.fft_sizes = _even_sizes(fft_sizes)
        self.magnitude_floor = _positive(magnitude_floor, "magnitude_floor")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        return _multi_resolution_spectral(
            require_tensor(context, "estimate", self.name),
            require_tensor(context, "target", self.name),
            factor=self.factor,
            complex_factor=self.complex_factor,
            gamma=self.gamma,
            ffts=self.fft_sizes,
            magnitude_floor=self.magnitude_floor,
        )

class LocalSNRLoss(torch.nn.Module):
    """MSE between the network local SNR and the waveform-derived target.

    ``milliseconds`` and ``energy_floor`` are this term's parameters. Sample
    rate, FFT, hop, and the clamp limits are read from the context so they
    stay the model's values.
    """

    def __init__(self, name: str, milliseconds: int, energy_floor: float) -> None:
        super().__init__()
        if not isinstance(name, str) or not name:
            raise BuildError("loss term name must be a non-empty string")
        if isinstance(milliseconds, bool) or not isinstance(milliseconds, int) or milliseconds < 1:
            raise BuildError("milliseconds must be an integer >= 1")
        self.name = name
        self.milliseconds = milliseconds
        self.energy_floor = _positive(energy_floor, "energy_floor")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        predicted = require_tensor(context, "predicted_lsnr", self.name)
        target = _local_snr_db(
            require_tensor(context, "clean_complex", self.name),
            require_tensor(context, "noise_complex", self.name),
            sample_rate=_context_int(context, "sample_rate", self.name),
            n_fft=_context_int(context, "n_fft", self.name),
            hop=_context_int(context, "hop", self.name),
            low=_context_float(context, "lsnr_low", self.name),
            high=_context_float(context, "lsnr_high", self.name),
            milliseconds=self.milliseconds,
            floor=self.energy_floor,
        )
        predicted = predicted.reshape(target.shape[0], -1)
        if predicted.shape[-1] != target.shape[-1]:
            raise TrainingError(
                f"local SNR length {predicted.shape[-1]} does not match the "
                f"{self.milliseconds} ms target {target.shape[-1]}"
            )
        return F.mse_loss(predicted, target)

def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise BuildError(f"{label} must be finite")
    return number

def _positive(value: object, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise BuildError(f"{label} must be positive")
    return number

def _even_sizes(values: object) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise BuildError("fft_sizes must be a non-empty sequence of integers")
    cleaned: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2 != 0:
            raise BuildError(f"fft size {value!r} must be an even integer >= 2")
        cleaned.append(value)
    return tuple(cleaned)

def _context_int(context: Mapping[str, Any], key: str, term: str) -> int:
    if key not in context:
        raise ValueError(f"loss term {term} requires context[{key!r}]")
    value = context[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"loss term {term} context[{key!r}] must be an integer")
    return value

def _context_float(context: Mapping[str, Any], key: str, term: str) -> float:
    if key not in context:
        raise ValueError(f"loss term {term} requires context[{key!r}]")
    value = context[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"loss term {term} context[{key!r}] must be a finite number")
    return float(value)


# --- Spectral and local-SNR formulas. The comparison task keeps its own copies. ---

_MR_FACTOR = 500.0

_MR_COMPLEX = 500.0

_MR_GAMMA = 0.3

_MR_FFTS = (256, 512, 1024, 2048)

_LSNR_MS = 20

def _vorbis_istft(spectrum: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    packed = F.pad(spectrum.transpose(-1, -2), (0, 1))
    return torch.istft(packed, n_fft=n_fft, hop_length=hop, window=window, normalized=True)

def _multi_resolution_spectral(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    factor: float = _MR_FACTOR,
    complex_factor: float = _MR_COMPLEX,
    gamma: float = _MR_GAMMA,
    ffts: tuple[int, ...] = _MR_FFTS,
    magnitude_floor: float = 1e-12,
) -> torch.Tensor:
    length = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :length]
    reference = reference[..., :length]
    loss = estimate.new_zeros(())
    for n_fft in ffts:
        hop = n_fft // 4
        # The official loss registers a CPU-created Hann window and moves it
        # with the loss module; CUDA's direct window generation differs.
        window = torch.hann_window(n_fft, dtype=estimate.dtype).to(estimate.device)
        predicted = _spectral_stft(estimate, n_fft, hop, window)
        target = _spectral_stft(reference, n_fft, hop, window)
        predicted_abs = predicted.abs().clamp_min(magnitude_floor).pow(gamma)
        target_abs = target.abs().clamp_min(magnitude_floor).pow(gamma)
        loss = loss + F.mse_loss(predicted_abs, target_abs) * factor
        predicted_compressed = _compressed_complex(predicted, predicted_abs)
        target_compressed = _compressed_complex(target, target_abs)
        loss = loss + F.mse_loss(
            torch.view_as_real(predicted_compressed), torch.view_as_real(target_compressed)
        ) * complex_factor
    return loss

def _spectral_stft(waveform: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    # CUDA reflection_pad1d backward is unavailable under strict deterministic
    # algorithms. The explicit padding has identical forward values but may
    # accumulate gradients in a different order.
    if waveform.shape[-1] <= n_fft // 2 or (waveform.is_cuda and torch.are_deterministic_algorithms_enabled()):
        return centered_stft(waveform, n_fft, hop, window, normalized=True)
    return torch.stft(waveform, n_fft, hop, window=window, normalized=True, return_complex=True)

def _compressed_complex(spectrum: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    phase = _StableAngle.apply(spectrum)
    return magnitude * torch.exp(1j * phase)

class _StableAngle(torch.autograd.Function):
    """Angle whose backward stays finite when the magnitude is zero."""

    @staticmethod
    def forward(ctx, spectrum: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(spectrum)
        return torch.atan2(spectrum.imag, spectrum.real)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (spectrum,) = ctx.saved_tensors
        denom = (spectrum.real.square() + spectrum.imag.square()).clamp_min(1e-10)
        scale = grad / denom
        return torch.view_as_complex(torch.stack((-spectrum.imag * scale, spectrum.real * scale), dim=-1))

def _local_snr_db(
    clean: torch.Tensor,
    noise: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop: int,
    low: float,
    high: float,
    milliseconds: int = _LSNR_MS,
    floor: float = 1e-12,
) -> torch.Tensor:
    speech = _local_energy(torch.view_as_real(clean).unsqueeze(1), _stft_window(milliseconds, sample_rate, n_fft, hop))
    disturbance = _local_energy(
        torch.view_as_real(noise).unsqueeze(1),
        _stft_window(milliseconds, sample_rate, n_fft, hop) * 2,
    )
    ratio = speech / disturbance.clamp_min(floor)
    return (10.0 * torch.log10(ratio.clamp_min(floor))).clamp(low, high).squeeze(1)

def _stft_window(milliseconds: int, sample_rate: int, n_fft: int, hop: int) -> int:
    fft_ms = n_fft / sample_rate * 1000
    hop_ms = hop / sample_rate * 1000
    windows = 1 + (milliseconds - fft_ms) / hop_ms
    return max(int(round(windows)), 1)

def _local_energy(spectrum: torch.Tensor, windows: int) -> torch.Tensor:
    if windows % 2 == 0:
        windows += 1
    half = windows // 2
    power = spectrum.pow(2).sum(-1).sum(-1)
    power = F.pad(power, (half, half, 0, 0))
    window = torch.hann_window(windows, device=spectrum.device, dtype=power.dtype)
    return (power.unfold(-1, windows, 1) * window).sum(-1) / windows


# --- One micro-batch total. Validation and the accumulation step both call this. ---

def loss_tensor(
    stream: DFStream | None,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    module: torch.nn.Module,
    objective: torch.nn.Module,
) -> torch.Tensor:
    if stream is None:
        raise RuntimeFailure("DeepFilterNet3 stream is not ready")
    try:
        parts = component_losses(stream, module, clean, noisy, objective)
    except TrainingError as exc:
        raise RuntimeFailure(str(exc)) from exc
    total = parts["total"]
    if not bool(torch.isfinite(total).all()):
        raise RuntimeFailure("deepfilternet loss is not finite")
    return total


# --- DFStream analysis, reconstruction, and the injected objective. ---

_LSNR = 1e-3

def component_losses(
    stream: DFStream,
    module: torch.nn.Module,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    objective: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Spectral, local-SNR, and total loss for one batch.

    The call order matches ``DeepFilterNetTask._loss``. ``module`` may be a
    DDP wrapper; parameters are read from the unwrapped generator.
    """

    network = unwrap(module)
    generator = network.generator if isinstance(network, DeepFilterNet3Model) else network
    stream.reset()
    _spec, _erb, _feat, clean_complex = _analyze(stream, generator, clean)
    stream.reset()
    noisy_spec, noisy_erb, noisy_feat, noisy_complex = _analyze(stream, generator, noisy)
    produced = module(noisy_spec, noisy_erb, noisy_feat)
    if isinstance(produced, Mapping):
        enhanced = produced["enhanced"]
        lsnr = produced["lsnr"]
    else:
        enhanced, _mask, lsnr, _coefs = produced
    enhanced_complex = torch.view_as_complex(enhanced.squeeze(1))
    frames = min(enhanced_complex.shape[-2], clean_complex.shape[-2], noisy_complex.shape[-2])
    bins = min(enhanced_complex.shape[-1], clean_complex.shape[-1], noisy_complex.shape[-1])
    enhanced_complex = enhanced_complex[..., :frames, :bins]
    clean_complex = clean_complex[..., :frames, :bins]
    noisy_complex = noisy_complex[..., :frames, :bins]
    params = generator.params
    window = vorbis_window(params.fft_size).to(device=enhanced_complex.device, dtype=enhanced_complex.real.dtype)
    enhanced_wave = _vorbis_istft(enhanced_complex, params.fft_size, params.hop_size, window)
    clean_wave = _vorbis_istft(clean_complex, params.fft_size, params.hop_size, window)
    if objective is None:
        spectral = _multi_resolution_spectral(enhanced_wave, clean_wave)
        target = _local_snr_db(
            clean_complex,
            noisy_complex - clean_complex,
            sample_rate=params.sr,
            n_fft=params.fft_size,
            hop=params.hop_size,
            low=params.lsnr_min - 1,
            high=params.lsnr_max + 1,
        )
        predicted = lsnr.reshape(target.shape[0], -1)
        if predicted.shape[-1] != target.shape[-1]:
            raise RuntimeFailure(
                f"local SNR length {predicted.shape[-1]} does not match the 20 ms target {target.shape[-1]}"
            )
        local = F.mse_loss(predicted, target) * _LSNR
        return {
            "enhanced_wave": enhanced_wave.detach(),
            "local_snr": local,
            "spectral": spectral,
            "total": spectral + local,
        }
    try:
        result = call_objective(
            objective,
            {
                "estimate": enhanced_wave,
                "target": clean_wave,
                "predicted_lsnr": lsnr,
                "clean_complex": clean_complex,
                "noise_complex": noisy_complex - clean_complex,
                "sample_rate": int(params.sr),
                "n_fft": int(params.fft_size),
                "hop": int(params.hop_size),
                "lsnr_low": float(params.lsnr_min - 1),
                "lsnr_high": float(params.lsnr_max + 1),
            },
        )
    except TrainingError:
        raise
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    reported = {"enhanced_wave": enhanced_wave.detach(), "total": result.total}
    for name, value, weight in result.records:
        reported[name] = apply_weight(value, weight)
    return reported

def _analyze(stream: DFStream, generator: torch.nn.Module, waveform: torch.Tensor):
    return _analyze_features(stream, generator, waveform)

def _analyze_features(stream: DFStream, generator: torch.nn.Module, waveform: torch.Tensor):
    params = generator.params
    padded = F.pad(waveform, (0, params.fft_size))
    stream.reset()
    complex_spec = stream.analyze(padded)
    spec, erb, feat = stream.features(complex_spec)
    return spec, erb, feat, complex_spec
