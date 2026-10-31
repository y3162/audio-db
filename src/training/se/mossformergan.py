"""MossFormerGAN SE 16 kHz training step.

The generator is ``SyncANet`` from ``src.se``. Preprocessing matches the
released solver: energy normalization of the noisy waveform, Hamming window
with ``periodic=False``, ``center=True``, and magnitude compression 0.3.
The loss is ``0.1 * RI + 0.9 * magnitude + 0.2 * time + 0.05 * GAN``.
The time-domain reconstruction uses the released pairing of the clean real
part with the predicted imaginary part.

The discriminator is an independent convolutional scorer. The released model
adds a Conformer after a NISQA-style CNN; that block is not reproduced.
Targets are the same: real pairs go to 1 and fakes go to wideband PESQ
normalized by ``(score - 1) / 3.5``. A missing PESQ score aborts the step.
It is not replaced with zero.

Update order follows the accumulation branch of the released solver, which is
what the published yaml enables: backward generator, backward discriminator,
then step generator and discriminator. Discriminator learning rate is twice
the generator learning rate. AdamW weight decay is the PyTorch default 0.01
because the released constructor does not pass ``weight_decay``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import backward
from src.training.errors import TrainingError
from src.training.se.discriminators import MagnitudePairDiscriminator
from src.training.se.losses import metric_targets
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler

_RI = 0.1
_MAG = 0.9
_TIME = 0.2
_GAN = 0.05


class MossFormerGANTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        model = config["model"]
        if config["loss"] != "official":
            raise TrainingError("MossFormerGAN training uses loss official")
        architecture = dict(model.get("architecture") or {})
        self.n_fft = int(architecture.get("n_fft", 400))
        self.win = int(architecture.get("win_len", self.n_fft))
        self.hop = int(architecture.get("hop", 100))
        self.grad_clip = config.get("grad_clip", 10.0)
        self.sample_rate = int(model["sampling_rate"])
        build_config = {key: value for key, value in architecture.items() if key in {"n_fft", "win_len", "hop", "num_channel"}}
        with seed_model_init(config["seed"]):
            self.generator = build_model("mossformergan", build_config or None, variant=model.get("variant") or "se_16k").to(device)
            self.discriminator = MagnitudePairDiscriminator().to(device)
        generator_config = dict(config)
        generator_config["optimizer"] = dict(config["optimizer"])
        generator_config["optimizer"]["weight_decay"] = 0.01
        self.optim_g = build_optimizer(self.generator.parameters(), generator_config)
        self.optim_d = build_optimizer(self.discriminator.parameters(), generator_config, lr=float(config["optimizer"]["lr"]) * 2)
        self.sched_g = build_scheduler(self.optim_g, config)
        self.sched_d = build_scheduler(self.optim_d, config)
        self._stalls = 0
        self._best_loss: float | None = None

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)
        self.discriminator = wrap_module(self.discriminator, process)

    def train_mode(self) -> None:
        self.generator.train()
        self.discriminator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()
        unwrap(self.discriminator).eval()

    def learning_rates(self) -> dict[str, float]:
        return {
            "generator": float(self.optim_g.param_groups[0]["lr"]),
            "discriminator": float(self.optim_d.param_groups[0]["lr"]),
        }

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        step_scheduler(self.sched_g, validation)
        step_scheduler(self.sched_d, validation)
        if not validation:
            return
        value = (validation.get("losses") or {}).get("generator", {}).get("value")
        if value is None:
            return
        if self._best_loss is None or float(value) < self._best_loss:
            self._best_loss = float(value)
            self._stalls = 0
            return
        self._stalls += 1
        if self._stalls >= 5:
            for optimizer in (self.optim_g, self.optim_d):
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
            self._stalls = 0

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def export_for_inference(self) -> dict[str, Any]:
        return {"model": self.generator_state_dict()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": "official",
            "generator": self.generator_state_dict(),
            "discriminator": unwrap(self.discriminator).state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "sched_g": None if self.sched_g is None else self.sched_g.state_dict(),
            "sched_d": None if self.sched_d is None else self.sched_d.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loss") != "official":
            raise TrainingError("checkpoint loss does not match MossFormerGAN")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.discriminator.load_state_dict(state["discriminator"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        self.optim_d.load_state_dict(state["optim_d"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        self.optim_g.zero_grad(set_to_none=True)
        self.optim_d.zero_grad(set_to_none=True)
        gen_total = 0.0
        disc_total = 0.0
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            scale = torch.sqrt(noisy.shape[-1] / noisy.square().sum(dim=-1).clamp_min(1e-6)).unsqueeze(-1)
            normalized = noisy * scale
            real, imag, pred_mag, label_real, label_imag, label_mag, audio, teacher = self._forward(
                self.generator, normalized, clean * scale
            )
            fake_score = self.discriminator(label_mag, pred_mag)
            ones = torch.ones(clean.shape[0], device=self.device)
            gan = F.mse_loss(fake_score.flatten(), ones)
            loss_mag = F.mse_loss(pred_mag, label_mag)
            loss_ri = F.mse_loss(real, label_real) + F.mse_loss(imag, label_imag)
            length = min(audio.shape[-1], teacher.shape[-1])
            time_loss = (audio[..., :length] - teacher[..., :length]).abs().mean()
            generator_loss = _RI * loss_ri + _MAG * loss_mag + _TIME * time_loss + _GAN * gan
            _finite(generator_loss, "mossformer generator")
            targets = metric_targets(teacher[..., :length], audio[..., :length], self.sample_rate)
            if targets is None:
                raise TrainingError(
                    "PESQ could not be computed for this MossFormerGAN batch. "
                    "The discriminator target is not replaced with zero."
                )
            is_last = index == len(batches) - 1
            backward(generator_loss / accumulation, [self.generator], last=is_last)
            real_score = self.discriminator(label_mag, label_mag)
            fake_detached = self.discriminator(label_mag, pred_mag.detach())
            discriminator_loss = F.mse_loss(real_score.flatten(), ones) + F.mse_loss(
                fake_detached.flatten(), targets.to(self.device)
            )
            _finite(discriminator_loss, "mossformer discriminator")
            backward(discriminator_loss / accumulation, [self.discriminator], last=is_last)
            gen_total += float(generator_loss.detach())
            disc_total += float(discriminator_loss.detach())
        clip_gradients(unwrap(self.generator), self.grad_clip)
        clip_gradients(unwrap(self.discriminator), self.grad_clip)
        self.optim_g.step()
        self.optim_d.step()
        count = len(batches)
        return {"generator": gen_total / count, "discriminator": disc_total / count}

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        with torch.no_grad():
            scale = torch.sqrt(noisy.shape[-1] / noisy.square().sum(dim=-1).clamp_min(1e-6)).unsqueeze(-1)
            _real, _imag, _pred, _lr, _li, _lm, audio, teacher = self._forward(
                unwrap(self.generator), noisy * scale, clean * scale
            )
            length = min(audio.shape[-1], teacher.shape[-1])
            value = float((audio[..., :length] - teacher[..., :length]).abs().mean().detach())
        metric = _metric(value)
        return {
            "losses": {"generator": metric, "time": metric, "discriminator": _metric(status="na", reason="validation uses time L1")},
            "pesq": {"status": "na", "value": None, "reason": "not computed", "count": 0},
            "stoi": {"status": "na", "value": None, "reason": "not computed", "count": 0},
        }

    def _forward(self, module, noisy: torch.Tensor, clean: torch.Tensor):
        noisy_spec = _compress(_stft(noisy, self.n_fft, self.hop, self.win)).permute(0, 1, 3, 2)
        real, imag = module(noisy_spec)
        real = real.permute(0, 1, 3, 2)
        imag = imag.permute(0, 1, 3, 2)
        pred_mag = torch.sqrt(real.square() + imag.square())
        label_spec = _compress(_stft(clean, self.n_fft, self.hop, self.win))
        label_real = label_spec[:, 0:1]
        label_imag = label_spec[:, 1:2]
        label_mag = torch.sqrt(label_real.square() + label_imag.square())
        # Released loss_mossformergan_se_16k reconstructs from the clean real part
        # and the predicted imaginary part.
        reconstructed = _uncompress(label_real, imag).squeeze(1)
        audio = _istft(reconstructed, self.n_fft, self.hop, self.win, clean.shape[-1])
        return real, imag, pred_mag, label_real, label_imag, label_mag, audio, clean


def _stft(waveform: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hamming_window(win, periodic=False, device=waveform.device, dtype=waveform.dtype)
    return torch.stft(waveform, n_fft, hop, win, window=window, center=True, return_complex=False)


def _istft(spectrum: torch.Tensor, n_fft: int, hop: int, win: int, length: int) -> torch.Tensor:
    window = torch.hamming_window(win, periodic=False, device=spectrum.device, dtype=spectrum.dtype)
    return torch.istft(torch.view_as_complex(spectrum), n_fft, hop, win, window=window, center=True, length=length)


def _compress(spectrum: torch.Tensor) -> torch.Tensor:
    value = torch.complex(spectrum[..., 0], spectrum[..., 1])
    magnitude = value.abs().pow(0.3)
    return torch.stack((magnitude * torch.cos(value.angle()), magnitude * torch.sin(value.angle())), 1)


def _uncompress(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    value = torch.complex(real, imag)
    magnitude = value.abs().pow(1.0 / 0.3)
    return torch.stack((magnitude * torch.cos(value.angle()), magnitude * torch.sin(value.angle())), -1)


def _finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
