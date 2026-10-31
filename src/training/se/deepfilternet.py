"""DeepFilterNet3 training step.

The network is ``DfNet`` from ``src.se``. Each batch builds ERB and deep-filter
features on a fresh ``DFStream``, so utterances do not share STFT or
normalization state.

The shipped DeepFilterNet3 ``config.ini`` (SHA-256
``415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290``,
next to ``model_120.ckpt.best``) sets these loss factors:

- multi-resolution spectral: magnitude 500, complex 500, gamma 0.3,
  FFT sizes 256, 512, 1024, 2048
- local SNR: 1e-3
- mask, single-resolution spectral, SI-SDR, and ASR: 0

The multi-resolution term is an MSE on Hann STFTs (``normalized=True``,
hop = n_fft / 4) of the waveforms rebuilt from the complex spectra with the
Vorbis window and ``torch.istft``. Local SNR is the 20 ms windowed ratio of
clean and noise energies, in dB, clamped to ``[lsnr_min - 1, lsnr_max + 1]``.

AdamW in that ini is lr 0.001, betas (0.9, 0.999), weight decay 1e-12,
``amsgrad=True``. The per-step cosine (warmup 3 epochs from 1e-4 to 1e-3,
then down to 1e-6, weight decay cosine toward 0.01) depends on the number of
steps in an epoch. This task does not invent that length; the YAML scheduler
is used as configured.
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
from src.se.models.deepfilternet3.dsp import vorbis_window
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.stft import centered_stft

_MR_FACTOR = 500.0
_MR_COMPLEX = 500.0
_MR_GAMMA = 0.3
_MR_FFTS = (256, 512, 1024, 2048)
_LSNR = 1e-3
_LSNR_MS = 20


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
        clean_complex = torch.view_as_complex(clean_spec.squeeze(1))
        frames = min(enhanced_complex.shape[-2], clean_complex.shape[-2], noisy_complex.shape[-2])
        bins = min(enhanced_complex.shape[-1], clean_complex.shape[-1], noisy_complex.shape[-1])
        enhanced_complex = enhanced_complex[..., :frames, :bins]
        clean_complex = clean_complex[..., :frames, :bins]
        noisy_complex = noisy_complex[..., :frames, :bins]
        params = unwrap(module).params
        window = vorbis_window(params.fft_size).to(device=enhanced_complex.device, dtype=enhanced_complex.real.dtype)
        enhanced_wave = _vorbis_istft(enhanced_complex, params.fft_size, params.hop_size, window)
        clean_td = _vorbis_istft(clean_complex, params.fft_size, params.hop_size, window)
        spectral = _multi_resolution_spectral(enhanced_wave, clean_td)
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
            raise TrainingError(
                f"local SNR length {predicted.shape[-1]} does not match the 20 ms target {target.shape[-1]}"
            )
        local = F.mse_loss(predicted, target) * _LSNR
        del mask, clean_wave
        return spectral + local

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

def _vorbis_istft(spectrum: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    packed = F.pad(spectrum.transpose(-1, -2), (0, 1))
    return torch.istft(packed, n_fft=n_fft, hop_length=hop, window=window, normalized=True)


def _multi_resolution_spectral(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    length = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :length]
    reference = reference[..., :length]
    loss = estimate.new_zeros(())
    for n_fft in _MR_FFTS:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=estimate.device, dtype=estimate.dtype)
        predicted = centered_stft(estimate, n_fft, hop, window, normalized=True)
        target = centered_stft(reference, n_fft, hop, window, normalized=True)
        predicted_abs = predicted.abs().clamp_min(1e-12).pow(_MR_GAMMA)
        target_abs = target.abs().clamp_min(1e-12).pow(_MR_GAMMA)
        loss = loss + F.mse_loss(predicted_abs, target_abs) * _MR_FACTOR
        predicted_compressed = _compressed_complex(predicted, predicted_abs)
        target_compressed = _compressed_complex(target, target_abs)
        loss = loss + F.mse_loss(
            torch.view_as_real(predicted_compressed), torch.view_as_real(target_compressed)
        ) * _MR_COMPLEX
    return loss


def _compressed_complex(spectrum: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    phase = _StableAngle.apply(spectrum)
    return torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))


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
        return torch.complex(-spectrum.imag * scale, spectrum.real * scale)


def _local_snr_db(
    clean: torch.Tensor,
    noise: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop: int,
    low: float,
    high: float,
) -> torch.Tensor:
    speech = _local_energy(torch.view_as_real(clean).unsqueeze(1), _stft_window(_LSNR_MS, sample_rate, n_fft, hop))
    disturbance = _local_energy(
        torch.view_as_real(noise).unsqueeze(1),
        _stft_window(_LSNR_MS, sample_rate, n_fft, hop) * 2,
    )
    ratio = speech / disturbance.clamp_min(1e-12)
    return (10.0 * torch.log10(ratio.clamp_min(1e-12))).clamp(low, high).squeeze(1)


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


def _finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
