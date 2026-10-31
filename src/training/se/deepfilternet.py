"""DeepFilterNet3 training step.

The network is ``DfNet`` from ``src.se``. Each batch builds ERB and deep-filter
features on a fresh ``DFStream``, so utterances do not share STFT or
normalization state. The released ``config.ini`` is not in this checkout.
``df/loss.py`` therefore leaves mask, spectral, and SI-SDR factors at 0 unless
a config sets them, and applies local SNR at 5e-4. This task turns on the
loss terms the source comments give as examples, with these coefficients:

- mask: factor 1, IAM, gamma 0.6, under-estimate weight 2, powers 2 and 4
- spectral magnitude: factor 1 (the comment's 1e4 is not used; no ini confirms it)
- SI-SDR: factor 0.2, the ``SdrLoss`` class default
- local SNR: factor 5e-4 against the frame SNR of clean versus noisy-clean

ASR loss stays off. AdamW uses the released defaults: lr 5e-4, betas
(0.9, 0.999), weight decay 0.05. The cosine schedule is not reproduced; the
configured scheduler is whatever the YAML selects.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.se.models.deepfilternet3.dsp import DFStream
from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import backward
from src.training.errors import TrainingError
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.spectral import si_sdr_loss

_MASK = 1.0
_SPECTRAL = 1.0
_SDR = 0.2
_LSNR = 0.0005
_GAMMA = 0.6


class DeepFilterNetTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        model = config["model"]
        if config["loss"] != "official":
            raise TrainingError("DeepFilterNet3 training uses loss official")
        architecture = dict(model.get("architecture") or {})
        self.grad_clip = config.get("grad_clip", 1.0)
        with seed_model_init(config["seed"]):
            self.generator = build_model(
                "deepfilternet3", architecture or None, variant=model.get("variant")
            ).to(device)
        params = self.generator.params
        self.stream = DFStream(
            params.sr,
            params.fft_size,
            params.hop_size,
            params.nb_erb,
            params.nb_df,
            params.min_nb_freqs,
            params.norm_tau,
            device,
        )
        self.optim_g = build_optimizer(self.generator.parameters(), config)
        self.sched_g = build_scheduler(self.optim_g, config)

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)
        self.stream.to(process.device)

    def train_mode(self) -> None:
        self.generator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()

    def learning_rates(self) -> dict[str, float]:
        return {"generator": float(self.optim_g.param_groups[0]["lr"])}

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        step_scheduler(self.sched_g, validation)

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def export_for_inference(self) -> dict[str, torch.Tensor]:
        return self.generator_state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": "official",
            "generator": self.generator_state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "sched_g": None if self.sched_g is None else self.sched_g.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loss") != "official":
            raise TrainingError("checkpoint loss does not match DeepFilterNet3")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        if self.sched_g is not None and state.get("sched_g") is not None:
            self.sched_g.load_state_dict(state["sched_g"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        self.optim_g.zero_grad(set_to_none=True)
        total = 0.0
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            loss = self._loss(self.generator, clean, noisy)
            _finite(loss, "deepfilternet")
            backward(loss / accumulation, [self.generator], last=index == len(batches) - 1)
            total += float(loss.detach())
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()
        mean = total / len(batches)
        return {"generator": mean, "deepfilter": mean}

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        with torch.no_grad():
            value = float(self._loss(unwrap(self.generator), clean, noisy).detach())
        metric = _metric(value)
        return {
            "losses": {
                "generator": metric,
                "discriminator": _metric(status="na", reason="DeepFilterNet3 has no discriminator"),
            },
            "pesq": {"status": "na", "value": None, "reason": "not computed", "count": 0},
            "stoi": {"status": "na", "value": None, "reason": "not computed", "count": 0},
        }

    def _loss(self, module: torch.nn.Module, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        self.stream.reset()
        clean_spec, clean_wave = self._analyze(clean)
        self.stream.reset()
        noisy_spec, noisy_erb, noisy_feat, noisy_complex = self._analyze_features(noisy)
        enhanced, mask, lsnr, _coefs = module(noisy_spec, noisy_erb, noisy_feat)
        enhanced_complex = torch.view_as_complex(enhanced.squeeze(1))
        enhanced_wave = self._synthesize(enhanced_complex, clean.shape[-1])
        clean_complex = torch.view_as_complex(clean_spec.squeeze(1))
        mask_loss = _mask_loss(mask, clean_complex, noisy_complex, unwrap(module).erb_fb)
        spectral = _spectral_loss(enhanced_complex, clean_complex)
        sdr = si_sdr_loss(enhanced_wave, clean_wave) * _SDR
        target_lsnr = _frame_snr(clean_wave, noisy[: clean_wave.shape[0], : clean_wave.shape[-1]] - clean_wave, mask.shape[-2] if mask.ndim >= 2 else 1)
        predicted = lsnr.reshape(target_lsnr.shape[0], -1)
        frames = min(predicted.shape[-1], target_lsnr.shape[-1])
        local = F.mse_loss(predicted[:, :frames], target_lsnr[:, :frames]) * _LSNR
        return mask_loss * _MASK + spectral * _SPECTRAL + sdr + local

    def _analyze(self, waveform: torch.Tensor):
        spec, _erb, _feat, complex_spec = self._analyze_features(waveform)
        return spec, waveform

    def _analyze_features(self, waveform: torch.Tensor):
        params = unwrap(self.generator).params
        padded = F.pad(waveform, (0, params.fft_size))
        self.stream.reset()
        complex_spec = self.stream.analyze(padded)
        spec, erb, feat = self.stream.features(complex_spec)
        return spec, erb, feat, complex_spec

    def _synthesize(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        self.stream.synthesis_memory.zero_()
        waveform = self.stream.synthesize(spectrum)
        delay = unwrap(self.generator).params.fft_size - unwrap(self.generator).params.hop_size
        return waveform[:, delay : delay + length]


def _mask_loss(mask: torch.Tensor, clean: torch.Tensor, noisy: torch.Tensor, erb_fb: torch.Tensor) -> torch.Tensor:
    iam = (clean.abs() / noisy.abs().clamp_min(1e-8)).clamp(0, 1)
    target = iam.matmul(erb_fb).clamp_min(1e-8).pow(_GAMMA)
    predicted = mask.squeeze(1).clamp_min(1e-8).pow(_GAMMA)
    frames = min(target.shape[-2], predicted.shape[-2])
    bands = min(target.shape[-1], predicted.shape[-1])
    residual = (target[..., :frames, :bands] - predicted[..., :frames, :bands]).square()
    under = torch.where(predicted[..., :frames, :bands] < target[..., :frames, :bands], 2.0, 1.0)
    weighted = residual * under
    return weighted.mean() + weighted.clamp_min(1e-13).sqrt().mean() * 10


def _spectral_loss(enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    frames = min(enhanced.shape[-2], clean.shape[-2])
    bins = min(enhanced.shape[-1], clean.shape[-1])
    pred = enhanced[..., :frames, :bins].abs().clamp_min(1e-12).pow(_GAMMA)
    ref = clean[..., :frames, :bins].abs().clamp_min(1e-12).pow(_GAMMA)
    return F.mse_loss(pred, ref)


def _frame_snr(clean: torch.Tensor, noise: torch.Tensor, frames: int) -> torch.Tensor:
    if frames < 1:
        frames = 1
    hop = max(clean.shape[-1] // frames, 1)
    count = min(frames, clean.shape[-1] // hop)
    clean_frames = clean[:, : count * hop].reshape(clean.shape[0], count, hop)
    noise_frames = noise[:, : count * hop].reshape(noise.shape[0], count, hop)
    signal = clean_frames.square().mean(dim=-1).clamp_min(1e-8)
    disturbance = noise_frames.square().mean(dim=-1).clamp_min(1e-8)
    return 10.0 * torch.log10(signal / disturbance)


def _finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
