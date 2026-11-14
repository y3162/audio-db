"""Waveform estimates for research evaluation.

Official validation does not call this module. TF-GridNet and TF-Locoformer
return a waveform from ``forward``. The other generators are adapted here so
a research metric can score audio without using their training loss.
"""

from __future__ import annotations

import torch

from src.experiments.errors import RuntimeFailure


def estimate_waveform(module: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return one enhanced waveform batch. The module's parameters are not updated."""

    name = type(module).__name__
    if name == "MPSENetGANGenerator":
        return _mp_senet(module, inputs)
    if name == "SEMambaPPGenerator":
        produced = module(targets, inputs)
        audio = produced[-1]
        if not isinstance(audio, torch.Tensor):
            raise RuntimeFailure("SEMamba++ research estimate did not return audio")
        return audio
    if name == "MossFormerGANGenerator":
        return _mossformer(module, inputs, targets)
    if name == "DeepFilterNet3Model":
        raise RuntimeFailure(
            "DeepFilterNet3 waveform estimates come from the trainer stream. "
            "Research evaluation uses DeepFilterNet3Trainer.research_predict."
        )
    try:
        output = module(inputs)
    except TypeError as exc:
        raise RuntimeFailure(
            f"{name} does not accept a waveform batch. "
            "Pass a model whose forward returns enhanced audio, or research_predict on the trainer."
        ) from exc
    if not isinstance(output, torch.Tensor):
        raise RuntimeFailure(f"{name} research estimate did not return a waveform tensor")
    return output


def _mp_senet(module: torch.nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    from src.experiments.calc.stft import mag_pha_istft, mag_pha_stft

    spec = module.architecture
    noisy_mag, noisy_phase, _unused = mag_pha_stft(
        noisy, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
    )
    magnitude, phase, _complex = module.generator(noisy_mag, noisy_phase)
    return mag_pha_istft(magnitude, phase, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"])


def _mossformer(module: torch.nn.Module, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Predicted complex spectrum as a waveform.

    The official time term mixes the clean real part with the predicted
    imaginary part. This estimate uses both predicted parts and does not call
    the discriminator, so research scoring does not update its BatchNorm.
    """

    from src.experiments.official.mossformergan.reference import _compress, _istft, _stft, _uncompress

    scale = torch.sqrt(noisy.shape[-1] / (noisy.square().sum(dim=-1) + 1e-6)).unsqueeze(-1)
    normalized = noisy * scale
    noisy_spec = _compress(_stft(normalized, module.n_fft, module.hop, module.win)).permute(0, 1, 3, 2)
    real, imag = module(noisy_spec)
    real = real.permute(0, 1, 3, 2)
    imag = imag.permute(0, 1, 3, 2)
    audio = _istft(_uncompress(real, imag).squeeze(1), module.n_fft, module.hop, module.win, int(clean.shape[-1]))
    return audio
