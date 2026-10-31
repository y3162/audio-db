"""Staged training for UniPASE and StuPASE.

Each stage has its own module and optimizer. They are not fused into one step.

DeWavLM is a mean-squared error between a frozen teacher and a student, both
read at the configured transformer layer (24 in the published configs). The
teacher is WavLM-Large. A checkpoint pair is ``components.teacher`` and
``components.student``. Without those files, ``model.architecture.wavlm``
builds a smaller WavLM with the same loss. That smaller pair is not
WavLM-Large and is not a substitute for the published teacher.

StuPASE CFM training calls ``CFM.forward`` (masked velocity MSE). It does not
call ``CFM.sample``, which is the inference ODE. The text condition is zeros
unless a DeWavLM-R feature is supplied.

Vocoder and post-net steps use the published GAN order: generator, then
discriminator. AdamW betas are (0.8, 0.99) and weight decay is 0.01. The mel
term uses the published band and window lists. The linear warmup cosine
schedule (separate peak learning rates per stage) is not the epoch scheduler
selected in the YAML.

UniPASE adapter training in the unit-test path maps waveforms with frozen
linear layers. That is not DeWavLM-Omni. Pass ``components.dewavlm`` only when
that checkpoint should be loaded; this task does not start a WavLM-Large
graph unless that path is present.
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
from src.training.se.discriminators import FeatureScaleDiscriminator, WaveformGANDiscriminator
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.spectral import _mel_basis, least_squares_discriminator, least_squares_generator, log_mel_l1
from src.training.se.stft import centered_stft

_WAVLM_MISSING = (
    "DeWavLM training needs components.teacher (WavLM-Large.pt) and "
    "components.student, or model.architecture.wavlm for a smaller pair that "
    "uses the same layer-feature MSE. A missing WavLM-Large file is not replaced "
    "with another checkpoint."
)
_MEL_BANDS = (5, 10, 20, 40, 80, 160, 320)
_MEL_WINDOWS = (32, 64, 128, 256, 512, 1024, 2048)


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
        self.layer = int(model.get("output_layer", 24))
        config = dict(config)
        config["optimizer"] = dict(config["optimizer"])
        config["optimizer"]["betas"] = [0.8, 0.99]
        config["optimizer"]["weight_decay"] = 0.01
        self.config = config
        with seed_model_init(config["seed"]):
            self.generator, self.discriminator, self._frozen = self._build(model)
        self.generator = self.generator.to(device)
        self._frozen = [module.to(device).eval() for module in self._frozen]
        for module in self._frozen:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
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
        self._frozen = [module.to(process.device) for module in self._frozen]
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
        if self.stage == "dewavlm":
            return self._distill_update(batches, accumulation)
        if self.stage == "adapter":
            return self._adapter_update(batches, accumulation)
        if self.discriminator is None:
            raise TrainingError(f"{self.family} stage {self.stage} has no update")
        return self._gan_update(batches, accumulation)

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        with torch.no_grad():
            if self.stage == "cfm":
                value = float(self._cfm_loss(clean, noisy).detach())
            elif self.stage == "dewavlm":
                value = float(self._distill_loss(clean, noisy).detach())
            elif self.stage == "adapter":
                value = float(self._adapter_loss(clean, noisy)[0].detach())
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
            return _build_dewavlm(self.family, model, self.components, self.layer)
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
        if self.family == "unipase" and stage == "vocoder":
            from src.se.models.unipase.vocoder.vocos.vocoder import VocosVocoder

            architecture = model.get("architecture") or {}
            if "vocoder" in self.components and not architecture:
                vocoder = VocosVocoder.from_pretrained(self.components["vocoder"], frozen=False)
            else:
                vocoder = VocosVocoder(**_vocoder_kwargs(architecture))
            return vocoder, WaveformGANDiscriminator(), []
        if self.family == "unipase" and stage == "adapter":
            from src.se.models.unipase.adapter.vocos.adapter import VocosAdapter

            architecture = model.get("architecture") or {}
            adapter = VocosAdapter(**_adapter_kwargs(architecture))
            width = int(adapter.head.out_features)
            scales = tuple(int(item) for item in (architecture.get("disc_dims") or (8, 16)))
            features = _FrozenMelPair(width, n_mels=int(architecture.get("n_mels", 16)))
            return adapter, FeatureScaleDiscriminator(width, scales), [features]
        if self.family == "unipase" and stage == "postnet":
            from src.se.models.unipase.postnet.tfgrid_cws_res import TFGridNet

            architecture = model.get("architecture") or {}
            return TFGridNet(**_postnet_kwargs(architecture)), WaveformGANDiscriminator(), []
        raise TrainingError(f"unsupported {self.family} stage {stage!r}")

    def _gan_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        # Published vocoder and post-net scripts step the generator, then the discriminator.
        self.optim_g.zero_grad(set_to_none=True)
        gen_total = 0.0
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            estimate = self._vocoder_forward(self.generator, clean)
            length = min(clean.shape[-1], estimate.shape[-1])
            reference = clean[..., :length]
            predicted = estimate[..., :length]
            real_scores, real_features = unwrap(self.discriminator)(reference)
            fake_scores, fake_features = unwrap(self.discriminator)(predicted)
            adversarial = sum(least_squares_generator(score) for score in fake_scores)
            matching = _sum_feature_l1(real_features, fake_features)
            mel = log_mel_l1(reference, predicted, self.sample_rate, n_mels=_MEL_BANDS, windows=_MEL_WINDOWS)
            loss = mel * self._mel_weight + adversarial * self._adv_weight + matching * self._feat_weight
            _finite(loss, "vocoder generator")
            backward(loss / accumulation, [self.generator, self.discriminator], last=index == len(batches) - 1)
            gen_total += float(loss.detach())
            captured.append((reference.detach(), predicted.detach()))
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()

        self.optim_d.zero_grad(set_to_none=True)
        disc_total = 0.0
        for index, (reference, predicted) in enumerate(captured):
            real_scores, _ = unwrap(self.discriminator)(reference)
            fake_scores, _ = unwrap(self.discriminator)(predicted)
            loss = sum(least_squares_discriminator(real, fake) for real, fake in zip(real_scores, fake_scores, strict=True))
            _finite(loss, "vocoder discriminator")
            backward(loss / accumulation, [self.discriminator], last=index == len(captured) - 1)
            disc_total += float(loss.detach())
        clip_gradients(unwrap(self.discriminator), self.grad_clip)
        self.optim_d.step()
        count = len(batches)
        return {"generator": gen_total / count, "discriminator": disc_total / count}

    def _distill_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.optim_g.zero_grad(set_to_none=True)
        total = 0.0
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            loss = self._distill_loss(clean, noisy)
            _finite(loss, "dewavlm")
            backward(loss / accumulation, [self.generator], last=index == len(batches) - 1)
            total += float(loss.detach())
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()
        mean = total / len(batches)
        return {"generator": mean, "dewavlm": mean}

    def _distill_loss(self, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        if not self._frozen:
            raise TrainingError("DeWavLM training has no frozen teacher")
        student = unwrap(self.generator)(noisy)
        with torch.no_grad():
            teacher = self._frozen[0](clean)
        return F.mse_loss(student, teacher)

    def _adapter_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.optim_g.zero_grad(set_to_none=True)
        gen_total = 0.0
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            loss, target, output = self._adapter_loss(clean, noisy)
            _finite(loss, "adapter")
            backward(loss / accumulation, [self.generator, self.discriminator], last=index == len(batches) - 1)
            gen_total += float(loss.detach())
            captured.append((target.detach(), output.detach()))
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()
        self.optim_d.zero_grad(set_to_none=True)
        disc_total = 0.0
        for index, (target, output) in enumerate(captured):
            real_scores, fake_scores, _, _ = unwrap(self.discriminator)(target, output)
            loss = sum(least_squares_discriminator(real, fake) for real, fake in zip(real_scores, fake_scores, strict=True))
            _finite(loss, "adapter discriminator")
            backward(loss / accumulation, [self.discriminator], last=index == len(captured) - 1)
            disc_total += float(loss.detach())
        clip_gradients(unwrap(self.discriminator), self.grad_clip)
        self.optim_d.step()
        count = len(batches)
        return {"generator": gen_total / count, "discriminator": disc_total / count}

    def _adapter_loss(self, clean: torch.Tensor, noisy: torch.Tensor):
        if not self._frozen:
            raise TrainingError("adapter training has no feature map")
        embed_a, embed_p, target = self._frozen[0](clean, noisy)
        output = unwrap(self.generator)(embed_a, embed_p)
        real_scores, fake_scores, real_features, fake_features = unwrap(self.discriminator)(target, output)
        reconstruction = F.mse_loss(output, target)
        adversarial = sum(least_squares_generator(score) for score in fake_scores)
        matching = _sum_feature_l1(real_features, fake_features)
        loss = reconstruction * self._mel_weight + adversarial * self._adv_weight + matching * self._feat_weight
        return loss, target, output

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
        if self.stage == "postnet":
            return module(waveform, sr_in=self.sample_rate)
        if self.stage == "vocoder":
            return module(_mel_input(module, waveform))
        raise TrainingError(f"{self.stage} does not synthesize a waveform")


def _build_dewavlm(family: str, model: dict[str, Any], components: dict[str, str], layer: int):
    teacher_path = components.get("teacher")
    student_path = components.get("student")
    architecture = (model.get("architecture") or {}).get("wavlm") or {}
    if teacher_path or student_path:
        if not teacher_path or not student_path:
            raise TrainingError("DeWavLM needs both components.teacher and components.student")
        return (
            _load_wavlm(family, student_path, layer, frozen=False),
            None,
            [_load_wavlm(family, teacher_path, layer, frozen=True)],
        )
    if not architecture:
        raise TrainingError(_WAVLM_MISSING)
    student_layer = int(architecture.get("output_layer", 1))
    return (
        _WavLMTower(_make_wavlm(family, architecture), student_layer),
        None,
        [_WavLMTower(_make_wavlm(family, architecture), student_layer)],
    )


def _load_wavlm(family: str, path: str, layer: int, *, frozen: bool):
    if family == "stupase":
        from src.se.models.stupase.wavlm.feature_extractor import WavLM_feat
    elif family == "unipase":
        from src.se.models.unipase.wavlm.feature_extractor_plc import WavLM_feat
    else:
        raise TrainingError(f"unsupported DeWavLM family {family}")
    return WavLM_feat(wavlm_ckpt_path=path, output_layer=layer, load_pretrained=True, frozen=frozen)


def _make_wavlm(family: str, architecture: dict[str, Any]):
    if family == "stupase":
        from src.se.models.stupase.wavlm.WavLM import WavLM, WavLMConfig
    elif family == "unipase":
        from src.se.models.unipase.wavlm.WavLM import WavLM, WavLMConfig
    else:
        raise TrainingError(f"unsupported DeWavLM family {family}")
    values = {
        "encoder_layers": 1,
        "encoder_embed_dim": 32,
        "encoder_ffn_embed_dim": 64,
        "encoder_attention_heads": 4,
        "conv_feature_layers": "[(32, 10, 5)]",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "dropout_input": 0.0,
        "dropout_features": 0.0,
        "conv_pos": 15,
        "conv_pos_groups": 4,
        "mask_prob": 0.0,
        "relative_position_embedding": False,
    }
    values.update({key: value for key, value in architecture.items() if key != "output_layer"})
    return WavLM(WavLMConfig(values))


class _WavLMTower(torch.nn.Module):
    def __init__(self, wavlm: torch.nn.Module, layer: int) -> None:
        super().__init__()
        self.wavlm = wavlm
        self.layer = layer

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        padded = _pad_wavlm(waveform)
        result = self.wavlm.extract_features(padded, output_layer=self.layer, mask=False)[0]
        features = result["layer_reps"][self.layer]
        return F.layer_norm(features, features.shape, eps=1e-6)


def _pad_wavlm(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[-1] % 320 == 80:
        return waveform
    pad = waveform.shape[-1] // 320 * 320 + 80 - waveform.shape[-1]
    if pad < 0:
        pad += 320
    return F.pad(waveform, (0, pad))


class _FrozenMelPair(torch.nn.Module):
    """Frozen linear maps of a log-mel. This is not a WavLM encoder."""

    def __init__(self, channels: int, n_mels: int = 16, n_fft: int = 64, hop: int = 16) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop
        self.proj_a = torch.nn.Linear(n_mels, channels)
        self.proj_p = torch.nn.Linear(n_mels, channels)
        self.proj_t = torch.nn.Linear(n_mels, channels)

    def forward(self, clean: torch.Tensor, noisy: torch.Tensor):
        noisy_mel = _log_mel_frames(noisy, self.n_mels, self.n_fft, self.hop)
        clean_mel = _log_mel_frames(clean, self.n_mels, self.n_fft, self.hop)
        frames = min(noisy_mel.shape[1], clean_mel.shape[1])
        return (
            self.proj_a(noisy_mel[:, :frames]),
            self.proj_p(noisy_mel[:, :frames]),
            self.proj_t(clean_mel[:, :frames]),
        )


def _log_mel_frames(waveform: torch.Tensor, n_mels: int, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spectrum = centered_stft(waveform, n_fft, hop, window).abs()
    basis = _mel_basis(16000, n_fft, n_mels, spectrum.device, spectrum.dtype)
    mel = torch.matmul(basis, spectrum.clamp_min(0)).transpose(1, 2)
    return torch.log10(mel.clamp_min(1e-5))


def _adapter_kwargs(architecture: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "input_channels": 16,
        "dim": 32,
        "intermediate_dim": 64,
        "num_layers": 1,
        "output_channels": 16,
    }
    defaults.update(
        {key: value for key, value in architecture.items() if key not in {"disc_dims", "n_mels", "wavlm"}}
    )
    if defaults["input_channels"] != defaults["output_channels"]:
        raise TrainingError("UniPASE adapter training expects input_channels == output_channels")
    return defaults


def _postnet_kwargs(architecture: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "n_imics": 1,
        "n_layers": 1,
        "lstm_hidden_units": 16,
        "attn_n_head": 2,
        "attn_qk_output_channel": 2,
        "emb_dim": 16,
        "emb_ks": 4,
        "emb_hs": 4,
        "n_bands": 3,
    }
    defaults.update(architecture)
    return defaults


def _sum_feature_l1(real_features: list[torch.Tensor], fake_features: list[torch.Tensor]) -> torch.Tensor:
    if len(real_features) != len(fake_features) or not real_features:
        raise TrainingError("feature lists must be non-empty and the same length")
    total = real_features[0].new_zeros(())
    for real, fake in zip(real_features, fake_features, strict=True):
        total = total + F.l1_loss(fake, real.detach())
    return total


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
    spectrum = centered_stft(waveform, n_fft, hop, window).abs()
    if spectrum.shape[-2] >= channels:
        spectrum = spectrum[..., :channels, :]
    else:
        spectrum = F.pad(spectrum, (0, 0, 0, channels - spectrum.shape[-2]))
    return spectrum


def _finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
