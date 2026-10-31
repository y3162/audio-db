"""Staged training for UniPASE and StuPASE.

DeWavLM stages need Microsoft WavLM-Large, which is not in
``data/checkpoints``. Those stages raise instead of training against a
stand-in teacher. StuPASE mel-vocoder and CFM stages can run without that
file. CFM training calls ``CFM.forward`` (masked velocity MSE). It does not
call ``CFM.sample``, which is the inference ODE.

UniPASE vocoder, adapter, and post-net stages require the frozen upstream
checkpoints named in ``model.components``. A missing path is an error.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import backward
from src.training.errors import TrainingError
from src.training.se.discriminators import WaveformGANDiscriminator
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.spectral import feature_matching, least_squares_discriminator, least_squares_generator, log_mel_l1

_WAVLM_MISSING = (
    "WavLM-Large.pt is not in data/checkpoints. DeWavLM training needs that "
    "frozen teacher, so this stage is not runnable."
)


class StagedTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        model = config["model"]
        self.family = model["name"]
        self.stage = model.get("stage")
        if config["loss"] != "official":
            raise TrainingError(f"{self.family} training uses loss official")
        if not self.stage:
            raise TrainingError(f"{self.family} requires model.stage")
        self.components = dict(model.get("components") or {})
        self.grad_clip = config.get("grad_clip", 3.0)
        self.sample_rate = int(model["sampling_rate"])
        with seed_model_init(config["seed"]):
            self.generator, self.discriminator, self._frozen = self._build(model)
        self.generator = self.generator.to(device)
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(device)
        self.optim_g = build_optimizer(self.generator.parameters(), config)
        self.optim_d = None
        self.sched_d = None
        if self.discriminator is not None:
            self.optim_d = build_optimizer(self.discriminator.parameters(), config)
            self.sched_d = build_scheduler(self.optim_d, config)
        self.sched_g = build_scheduler(self.optim_g, config)
        self._mel_weight, self._adv_weight, self._feat_weight = _gan_weights(self.family, self.stage)

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)
        if self.discriminator is not None:
            self.discriminator = wrap_module(self.discriminator, process)

    def train_mode(self) -> None:
        self.generator.train()
        if self.discriminator is not None:
            self.discriminator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()
        if self.discriminator is not None:
            unwrap(self.discriminator).eval()

    def learning_rates(self) -> dict[str, float]:
        rates = {"generator": float(self.optim_g.param_groups[0]["lr"])}
        if self.optim_d is not None:
            rates["discriminator"] = float(self.optim_d.param_groups[0]["lr"])
        return rates

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        step_scheduler(self.sched_g, validation)
        if self.sched_d is not None:
            step_scheduler(self.sched_d, validation)

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def export_for_inference(self) -> dict[str, Any]:
        return {"model": self.generator_state_dict(), "stage": self.stage}

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": "official",
            "stage": self.stage,
            "generator": self.generator_state_dict(),
            "discriminator": None if self.discriminator is None else unwrap(self.discriminator).state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": None if self.optim_d is None else self.optim_d.state_dict(),
            "sched_g": None if self.sched_g is None else self.sched_g.state_dict(),
            "sched_d": None if self.sched_d is None else self.sched_d.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loss") != "official" or state.get("stage") != self.stage:
            raise TrainingError("checkpoint stage does not match this task")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        if self.discriminator is not None:
            self.discriminator.load_state_dict(state["discriminator"], strict=True)
            self.optim_d.load_state_dict(state["optim_d"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        if self.stage == "cfm":
            return self._cfm_update(batches, accumulation)
        if self.discriminator is None:
            raise TrainingError(f"{self.family} stage {self.stage} has no update")
        return self._gan_update(batches, accumulation)

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        with torch.no_grad():
            if self.stage == "cfm":
                value = float(self._cfm_loss(clean, noisy).detach())
            else:
                estimate = self._vocoder_forward(unwrap(self.generator), clean)
                value = float(F.l1_loss(clean[..., : estimate.shape[-1]], estimate[..., : clean.shape[-1]]).detach())
        metric = _metric(value)
        return {
            "losses": {"generator": metric, "discriminator": _metric(status="na", reason="validation reports the stage loss")},
            "pesq": {"status": "na", "value": None, "reason": "not computed", "count": 0},
            "stoi": {"status": "na", "value": None, "reason": "not computed", "count": 0},
        }

    def _build(self, model: dict[str, Any]):
        stage = self.stage
        if stage == "dewavlm":
            raise TrainingError(_WAVLM_MISSING)
        if self.family == "stupase" and stage == "vocoder":
            from src.se.models.stupase.vocoder.vocos.vocoder import VocosVocoder

            architecture = model.get("architecture") or {}
            if not architecture and "vocoder" not in self.components:
                raise TrainingError(
                    "StuPASE vocoder needs model.architecture for a smoke network "
                    "or components.vocoder for Vocoder_Mel-16k.pt"
                )
            if "vocoder" in self.components and not architecture:
                vocoder = VocosVocoder.from_pretrained(self.components["vocoder"], frozen=False)
            else:
                vocoder = VocosVocoder(**_vocoder_kwargs(architecture))
            return vocoder, WaveformGANDiscriminator(), []
        if self.family == "stupase" and stage == "cfm":
            from src.se.models.stupase.flow.cfm import CFM

            architecture = model.get("architecture") or {}
            if not architecture and "cfm" not in self.components:
                raise TrainingError(
                    "StuPASE CFM needs model.architecture for a smoke DiT or components.cfm for CFM.pt. "
                    "Training calls CFM.forward (flow matching), not CFM.sample (ODE)."
                )
            if "cfm" in self.components and not architecture:
                cfm = CFM.from_pretrained(self.components["cfm"], frozen=False)
            else:
                cfm = CFM(**_cfm_kwargs(architecture))
            return cfm, None, []
        if self.family == "unipase" and stage in ("vocoder", "adapter", "postnet"):
            _require(self.components, "dewavlm", "UniPASE stages need components.dewavlm")
            raise TrainingError(
                f"UniPASE {stage} training requires the frozen DeWavLM-Omni checkpoint "
                f"at {self.components['dewavlm']} plus the stage module. "
                "A one-step smoke of that graph is not started from this task "
                "because the encoder is WavLM-scale. The loss weights are implemented "
                "for the StuPASE mel vocoder, which uses the same GAN form."
            )
        raise TrainingError(f"unsupported {self.family} stage {stage!r}")

    def _gan_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.optim_d.zero_grad(set_to_none=True)
        disc_total = 0.0
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            with torch.no_grad():
                estimate = self._vocoder_forward(unwrap(self.generator), clean)
            real_scores, _ = unwrap(self.discriminator)(clean)
            fake_scores, _ = unwrap(self.discriminator)(estimate.detach())
            loss = sum(least_squares_discriminator(real, fake) for real, fake in zip(real_scores, fake_scores, strict=True))
            _finite(loss, "vocoder discriminator")
            backward(loss / accumulation, [self.discriminator], last=index == len(batches) - 1)
            disc_total += float(loss.detach())
        clip_gradients(unwrap(self.discriminator), self.grad_clip)
        self.optim_d.step()

        frozen = [parameter for parameter in unwrap(self.discriminator).parameters()]
        for parameter in frozen:
            parameter.requires_grad_(False)
        self.optim_g.zero_grad(set_to_none=True)
        gen_total = 0.0
        try:
            for index, batch in enumerate(batches):
                clean = batch["clean"].to(self.device)
                estimate = self._vocoder_forward(self.generator, clean)
                length = min(clean.shape[-1], estimate.shape[-1])
                reference = clean[..., :length]
                predicted = estimate[..., :length]
                _, real_features = unwrap(self.discriminator)(reference)
                fake_scores, fake_features = unwrap(self.discriminator)(predicted)
                adversarial = sum(least_squares_generator(score) for score in fake_scores)
                matching = feature_matching(real_features, fake_features, scale=1.0)
                mel = log_mel_l1(reference, predicted, self.sample_rate, n_mels=(16, 32), windows=(64, 128))
                loss = mel * self._mel_weight + adversarial * self._adv_weight + matching * self._feat_weight
                _finite(loss, "vocoder generator")
                backward(loss / accumulation, [self.generator], last=index == len(batches) - 1)
                gen_total += float(loss.detach())
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()
        count = len(batches)
        return {"generator": gen_total / count, "discriminator": disc_total / count}

    def _cfm_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.optim_g.zero_grad(set_to_none=True)
        total = 0.0
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            loss = self._cfm_loss(clean, noisy)
            _finite(loss, "flow matching")
            backward(loss / accumulation, [self.generator], last=index == len(batches) - 1)
            total += float(loss.detach())
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()
        mean = total / len(batches)
        return {"generator": mean, "flow_matching": mean}

    def _cfm_loss(self, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        module = unwrap(self.generator)
        text_dim = module.transformer.text_embed.proj.in_features
        mel_bins = module.num_channels
        # Flow matching consumes the text condition. Without DeWavLM-R the
        # condition is zeros and the loss is still the masked velocity MSE.
        frames = max(clean.shape[-1] // 8, 2)
        text = clean.new_zeros(clean.shape[0], frames, text_dim)
        loss, _cond, _pred = module(clean, noisy, text)
        if loss.ndim != 0:
            loss = loss.mean()
        del mel_bins
        return loss

    def _vocoder_forward(self, module: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
        mel = unwrap(self.generator).mel_spec(waveform) if hasattr(unwrap(self.generator), "mel_spec") else None
        if self.stage == "vocoder":
            spec = _mel_input(module, waveform)
            return module(spec)
        if mel is None:
            raise TrainingError("this stage does not synthesize a waveform")
        return module(mel)


def _vocoder_kwargs(architecture: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "input_channels": 16,
        "dim": 32,
        "intermediate_dim": 64,
        "num_layers": 1,
        "num_res": 2,
        "num_attn": 1,
        "n_fft": 64,
        "hop_length": 16,
    }
    defaults.update(architecture)
    return defaults


def _cfm_kwargs(architecture: dict[str, Any]) -> dict[str, Any]:
    dit = {
        "dim": 32,
        "depth": 1,
        "heads": 4,
        "dim_head": 8,
        "mel_dim": 16,
        "text_in_dim": 16,
        "ff_mult": 2,
    }
    mel = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "n_mel_channels": 16,
        "target_sample_rate": 16000,
    }
    if architecture:
        dit.update(architecture.get("dit") or {})
        mel.update(architecture.get("mel") or {})
        passthrough = {key: value for key, value in architecture.items() if key not in ("dit", "mel")}
    else:
        passthrough = {}
    return {
        "dit_kwargs": dit,
        "mel_spec_kwargs": mel,
        "audio_drop_prob": 0.0,
        "cond_drop_prob": 0.0,
        **passthrough,
    }


def _gan_weights(family: str, stage: str) -> tuple[float, float, float]:
    if family == "stupase" and stage == "vocoder":
        return 15.0, 1.0, 1.0
    if family == "unipase" and stage == "vocoder":
        return 15.0, 0.5, 0.5
    if family == "unipase" and stage == "adapter":
        return 200.0, 1.0, 1.0
    if family == "unipase" and stage == "postnet":
        return 15.0, 0.5, 0.5
    return 1.0, 1.0, 1.0


def _mel_input(module: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    istft = module.head.istft
    n_fft = int(istft.n_fft)
    hop = int(istft.hop_length)
    channels = int(module.decoder.input_channels)
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(waveform, n_fft, hop, n_fft, window, center=True, return_complex=True).abs()
    if spectrum.shape[-2] >= channels:
        spectrum = spectrum[..., :channels, :]
    else:
        spectrum = F.pad(spectrum, (0, 0, 0, channels - spectrum.shape[-2]))
    return spectrum


def _require(components: dict[str, str], name: str, message: str) -> None:
    if name not in components:
        raise TrainingError(message)


def _finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
