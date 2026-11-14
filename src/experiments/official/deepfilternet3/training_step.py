"""DeepFilterNet3 loss for one batch.

This is the model-specific sequence of DFStream analysis, generator forward,
waveform reconstruction, and the injected objective. It is not split into
separate loss terms. The trainer decides when to call it and still applies
the cosine update before the optimizer step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.experiments.calc.distributed import unwrap
from src.experiments.calc.errors import TrainingError
from src.experiments.errors import RuntimeFailure
from src.experiments.official.deepfilternet3.model import DeepFilterNet3Model
from src.experiments.official.deepfilternet3.reference import (
    _local_snr_db,
    _multi_resolution_spectral,
    _vorbis_istft,
)
from src.experiments.training.objectives import apply_weight, call_objective
from src.se.models.deepfilternet3.dsp import DFStream, vorbis_window

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
    enhanced, _mask, lsnr, _coefs = module(noisy_spec, noisy_erb, noisy_feat)
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
