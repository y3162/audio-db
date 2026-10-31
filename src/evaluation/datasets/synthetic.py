"""Small in-memory speech-like items for tests and smoke runs."""

from __future__ import annotations

import torch

from src.evaluation.sample import AudioSample


def make_synthetic_samples(
    count: int,
    sample_rate: int,
    seconds: float,
    seed: int,
    *,
    dataset: str = "synthetic",
) -> list[AudioSample]:
    if count < 1:
        raise ValueError("synthetic count must be positive")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    frames = int(sample_rate * seconds)
    if frames < 1:
        raise ValueError("synthetic audio length must be at least one sample")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    time = torch.arange(frames, dtype=torch.float32) / float(sample_rate)
    samples = []
    for index in range(count):
        envelope = 0.6 + 0.4 * torch.sin(2.0 * torch.pi * (3.0 + index) * time)
        clean = torch.randn(frames, generator=generator) * envelope
        noise = torch.randn(frames, generator=generator)
        clean = _scale_rms(clean, 0.05)
        noise = _scale_rms(noise, 0.05)
        samples.append(
            AudioSample(
                utterance_id=f"synthetic-{index:04d}",
                dataset=dataset,
                split="synthetic",
                sample_rate=sample_rate,
                transcript=f"sample {index}",
                speaker_id=f"spk{index % 2}",
                clean_waveform=clean,
                noise_waveform=noise,
                metadata={"synthetic": True},
            )
        )
    return samples


def _scale_rms(waveform: torch.Tensor, target: float) -> torch.Tensor:
    power = float(waveform.square().mean())
    if power <= 0.0:
        return waveform
    return waveform * (target / (power ** 0.5))
