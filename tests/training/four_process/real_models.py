"""Official and independent update steps for the four-process probe.

Each process builds one implementation. The runner loads the independent
initial state into both, so the official modules must accept that state dict.
Loss code stays on its own side: official modules call the published loss,
independent modules call ``src.training``.
"""
from types import SimpleNamespace

import torch

from tests.training.four_process.step import Step, apply_gradient_fault, finite as _finite, grads as _grads, maybe_nan


class _Counter:
    """Scheduler object with state, used when the published loop has no per-step schedule."""

    def __init__(self):
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        self.last_epoch = int(state["last_epoch"])


class _TableSchedule:
    """Per-step learning-rate table. ``apply`` runs before the optimizer step."""

    def __init__(self, optimizer, learning_rates, decays):
        self.optimizer = optimizer
        self.learning_rates = [float(value) for value in learning_rates]
        self.decays = [float(value) for value in decays]
        self.index = 0

    def apply(self):
        if self.index < 0 or self.index >= len(self.learning_rates):
            raise RuntimeError(f"schedule index {self.index} is outside {len(self.learning_rates)}")
        learning_rate = self.learning_rates[self.index]
        decay = self.decays[self.index]
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
            if float(group["weight_decay"]) > 0.0:
                group["weight_decay"] = decay

    def step(self):
        self.index += 1

    def state_dict(self):
        return {"index": self.index, "learning_rates": self.learning_rates, "decays": self.decays}

    def load_state_dict(self, state):
        if list(state["learning_rates"]) != self.learning_rates or list(state["decays"]) != self.decays:
            raise RuntimeError("refusing to resume because the scheduler table differs")
        self.index = int(state["index"])


class SEMambaPP:
    """Published discriminator-then-generator step on a narrow generator.

    The published generator is hid_feature 48, 4 TF-Mamba blocks, d_state 16,
    d_conv 4. This probe uses hid_feature 8, 1 block, d_state 4, d_conv 2 so
    four processes can keep the published CQT and multi-resolution
    discriminators. Loss weights and the single generator forward are the
    published ones. ExponentialLR can be stepped every update for diagnostics
    or held until the epoch ends, as in published training.
    """

    length = 8000
    sample_rate = 16000
    clip = None
    step_every_update = True
    update_order = ("discriminator", "generator")
    lr = 2e-4
    betas = (0.8, 0.99)
    weight_decay = 0.01
    gamma = 0.99

    def configure(self, cfg):
        backend = cfg.get("scan_backend") or "reference"
        if backend not in ("reference", "fused"):
            raise ValueError(f"unknown scan backend {backend}")
        scheduler_step = cfg.get("scheduler_step") or "update"
        if scheduler_step not in ("update", "epoch"):
            raise ValueError(f"unknown scheduler step {scheduler_step}")
        self.step_every_update = scheduler_step == "update"
        self.scan_backend = backend
        self.official_width = bool(cfg.get("official_width"))
        self.probe = {
            "version": "mel-mrd-scheduler-v1",
            "generator": ("hid_feature 48, num_tfmamba 4, d_state 16, d_conv 4, expand 2"
                          if self.official_width else "hid_feature 8, num_tfmamba 1, d_state 4, d_conv 2, expand 2"),
            "published_generator": "hid_feature 48, num_tfmamba 4, d_state 16, d_conv 4, expand 2",
            "discriminators": "published CQT and three-resolution magnitude stacks",
            "aggregation": "interleave real/fake per discriminator; sum CQT and multi-resolution losses separately",
            "scan_backend": backend,
            "length": self.length,
            "sample_rate": self.sample_rate,
            "scheduler": ("ExponentialLR gamma 0.99 stepped every probe update" if self.step_every_update
                          else "ExponentialLR gamma 0.99 held until the epoch ends"),
        }

    def _arch(self):
        return {
            "hid_feature": 48 if getattr(self, "official_width", False) else 8,
            "num_tfmamba": 4 if getattr(self, "official_width", False) else 1,
            "d_state": 16 if getattr(self, "official_width", False) else 4,
            "d_conv": 4 if getattr(self, "official_width", False) else 2,
            "expand": 2,
            "norm_epsilon": 1e-5,
            "input_channel": 2,
            "output_channel": 1,
            "n_fft": 400,
            "scan_backend": getattr(self, "scan_backend", "reference"),
        }

    def build(self, implementation, device):
        arch = self._arch()
        if implementation == "official":
            generator, band, resolution = _official_semamba(arch, device)
            self._mel = _official_mel(device)
        else:
            from src.se.models.semambapp.network import SEMambaPP as Independent
            from src.training.se.semamba_disc import MultiResolutionDiscriminator, MultiScaleSubbandCQTDiscriminator
            generator = Independent(arch).to(device)
            band = MultiScaleSubbandCQTDiscriminator(self.sample_rate).to(device)
            resolution = MultiResolutionDiscriminator().to(device)
            self._mel = None
        self._implementation = implementation
        return {"generator": generator, "discriminator_band": band, "discriminator_mrd": resolution}

    def optimizers(self, modules, lr):
        del lr
        made = {
            "discriminator": torch.optim.AdamW(
                list(modules["discriminator_band"].parameters()) + list(modules["discriminator_mrd"].parameters()),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            ),
            "generator": torch.optim.AdamW(
                modules["generator"].parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            ),
        }
        return made

    def schedulers(self, optimizers):
        return {name: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.gamma) for name, opt in optimizers.items()}

    def train_step(self, implementation, modules, optimizers, schedulers, clean, noisy, position, fault=None):
        del position
        for module in modules.values():
            module.train()
        generator = modules["generator"]
        band = modules["discriminator_band"]
        resolution = modules["discriminator_mrd"]
        optimizers["discriminator"].zero_grad(set_to_none=True)
        parts = _semamba_parts(implementation, generator, clean, noisy)
        disc = _semamba_discriminator(implementation, band, resolution, clean, parts["audio"].detach())
        disc = _finite(maybe_nan(disc, fault), "discriminator")
        disc.backward()
        grads = {}
        grads.update(_grads(band, "discriminator_band"))
        grads.update(_grads(resolution, "discriminator_mrd"))
        optimizers["discriminator"].step()
        optimizers["generator"].zero_grad(set_to_none=True)
        losses = _semamba_generator(implementation, band, resolution, clean, parts, self._mel)
        gen = _finite(losses["generator"], "generator")
        gen.backward()
        apply_gradient_fault(modules, fault)
        grads.update(_grads(generator, "generator"))
        optimizers["generator"].step()
        if self.step_every_update:
            for name in self.update_order:
                schedulers[name].step()
        payload = {"generator": gen.detach(), "discriminator": disc.detach()}
        for name, value in losses.items():
            if name != "generator":
                payload[name] = value.detach()
        return Step(parts["audio"].detach(), payload, grads, self.update_order)


class MossFormerGAN:
    """Published generator backward, discriminator backward, then both steps.

    The probe steps a counter that does not change the learning rate. Published
    training has no per-batch scheduler. PESQ failure aborts the step.
    """

    length = 8000
    sample_rate = 16000
    clip = 10.0
    step_every_update = False
    update_order = ("generator", "discriminator")
    lr = 5e-4
    betas = (0.9, 0.999)
    weight_decay = 0.01  # AdamW default. The released constructor ignores the YAML 1e-5.

    def configure(self, cfg):
        del cfg
        self.probe = {
            "generator": "published SyncANet num_channel 64, n_fft 400",
            "discriminator": "published NISQA AdaptCNN and two Conformers",
            "update": "two micro-batches: divide each loss by 2, generator backward and clip, discriminator backward and clip; then both steps",
            "scheduler": "no per-batch schedule; the released solver halves the generator learning rate only after five non-improving validations, which this four-update probe does not reach",
            "pesq": "a missing PESQ score aborts the step",
            "official_batch_pesq": "the published helper hardcodes a CUDA tensor; the probe returns the same normalization on the step device",
            "length": self.length,
            "sample_rate": self.sample_rate,
        }

    def build(self, implementation, device):
        if implementation == "official":
            generator, discriminator = _official_moss(device)
        else:
            from src.se import build_model
            from src.training.se.mossformer_discriminator import Discriminator
            generator = build_model(
                "mossformergan", {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64}, variant="se_16k"
            ).to(device)
            discriminator = Discriminator(ndf=16).to(device)
        return {"generator": generator, "discriminator": discriminator}

    def optimizers(self, modules, lr):
        del lr
        return {
            "generator": torch.optim.AdamW(
                modules["generator"].parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
            ),
            "discriminator": torch.optim.AdamW(
                modules["discriminator"].parameters(), lr=self.lr * 2, betas=self.betas, weight_decay=self.weight_decay
            ),
        }

    def schedulers(self, optimizers):
        return {name: _Counter() for name in optimizers}

    def train_step(self, implementation, modules, optimizers, schedulers, clean, noisy, position, fault=None):
        del position, schedulers
        generator = modules["generator"]
        discriminator = modules["discriminator"]
        generator.train()
        discriminator.train()
        optimizers["generator"].zero_grad(set_to_none=True)
        optimizers["discriminator"].zero_grad(set_to_none=True)
        if clean.shape[0] != 2 or noisy.shape[0] != 2:
            raise ValueError("MossFormerGAN probe requires two micro-batches")
        audios, gens, discs = [], [], []
        for clean_micro, noisy_micro in zip(clean.split(1), noisy.split(1)):
            scale = torch.sqrt(noisy_micro.shape[-1] / (noisy_micro.square().sum(dim=-1) + 1e-6)).unsqueeze(-1)
            normalized = noisy_micro * scale
            teacher = clean_micro * scale
            if implementation == "official":
                audio, gen, disc = _official_moss_losses(generator, discriminator, normalized, teacher, clean.device)
            else:
                audio, gen, disc = _independent_moss_losses(
                    generator, discriminator, normalized, teacher, self.sample_rate
                )
            gen = _finite(maybe_nan(gen, fault), "generator")
            disc = _finite(disc, "discriminator")
            (gen / 2).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), self.clip)
            (disc / 2).backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.clip)
            audios.append(audio.detach())
            gens.append(gen.detach())
            discs.append(disc.detach())
        apply_gradient_fault(modules, fault)
        grads = {}
        grads.update(_grads(generator, "generator"))
        grads.update(_grads(discriminator, "discriminator"))
        optimizers["generator"].step()
        optimizers["discriminator"].step()
        return Step(torch.cat(audios), {"generator": torch.stack(gens).mean(), "discriminator": torch.stack(discs).mean()}, grads, self.update_order)


class DeepFilterNet3:
    """Shipped 48 kHz DfNet, multi-resolution spectral loss, and per-step cosine.

    ``libdf`` is not used. Both sides analyze with the independent ``DFStream``
    and reset it on every step. The official side then runs published ``DfNet``
    and published loss modules.
    """

    length = 4800
    sample_rate = 48000
    clip = 1.0
    step_every_update = True
    update_order = ("generator",)

    def configure(self, cfg):
        import os
        mode = os.environ.get('DF_PROBE_OFFICIAL_LIBDF', '0')
        if mode not in ('0', '1', 'shared'):
            raise ValueError('unknown DeepFilterNet frontend mode')
        self.use_libdf = mode in ('1', 'shared')
        self.shared_libdf = mode == 'shared'
        expected_frontend = ({'0': 'independent DFStream; libdf analysis is not executed',
                              '1': 'official libdf vs independent DFStream',
                              'shared': 'official libdf on both sides'}[mode])
        recorded = cfg.get('probe', {}).get('frontend')
        if recorded is not None and recorded != expected_frontend:
            raise ValueError('DeepFilterNet frontend differs from the recorded probe')
        self.probe = {
            "sample_rate": self.sample_rate,
            "architecture": "shipped config.ini conv_ch 64, fft 960, hop 480",
            "frontend": expected_frontend,
            "loss": "multi-resolution magnitude 500, complex 500, gamma 0.3, local SNR 1e-3",
            "optimizer": "AdamW lr 0.001, betas (0.9, 0.999), weight decay 1e-12, amsgrad",
            "scheduler": "shipped per-step cosine, steps_per_epoch 8, epochs 120, warmup 3",
            "length": self.length,
        }

    def build(self, implementation, device):
        from src.se.models.deepfilternet3.dsp import DFStream
        from src.se.models.deepfilternet3.inference import build
        independent = build(None).to(device)
        params = independent.params
        self._stream = DFStream(
            params.sr, params.fft_size, params.hop_size, params.nb_erb, params.nb_df,
            params.min_nb_freqs, params.norm_tau, device,
        )
        self._params = params
        if self.use_libdf and (implementation == 'official' or self.shared_libdf):
            from tests.training.reference.probe_df_libdf import OfficialStream
            self._stream = OfficialStream(params)
        if implementation == "official":
            generator = _official_df(independent)
            self._official_loss = _official_df_losses(device)
        else:
            generator = independent
            self._official_loss = None
        self._implementation = implementation
        return {"generator": generator}

    def optimizers(self, modules, lr):
        del lr
        return {
            "generator": torch.optim.AdamW(
                modules["generator"].parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-12,
                amsgrad=True,
            )
        }

    def schedulers(self, optimizers):
        rates, decays = _df_schedule()
        return {"generator": _TableSchedule(optimizers["generator"], rates, decays)}

    def train_step(self, implementation, modules, optimizers, schedulers, clean, noisy, position, fault=None):
        del position
        generator = modules["generator"]
        generator.train()
        schedulers["generator"].apply()
        optimizers["generator"].zero_grad(set_to_none=True)
        audio, loss = _df_loss(implementation, generator, self._stream, self._params, clean, noisy, self._official_loss)
        loss = _finite(maybe_nan(loss, fault), "generator")
        loss.backward()
        grads = _grads(generator, "generator")
        apply_gradient_fault(modules, fault)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), self.clip)
        optimizers["generator"].step()
        if self.step_every_update:
            schedulers["generator"].step()
        return Step(audio.detach(), {"generator": loss.detach(), "discriminator": None}, grads, self.update_order)


def _allow_official_semamba_import():
    """The published loss module imports PESQ even when this step does not call it."""
    import sys
    import types
    if "pesq" in sys.modules:
        return
    try:
        import pesq  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("pesq")
        def _unavailable(*args, **kwargs):
            raise RuntimeError("PESQ is not installed; SEMamba++ probe does not score PESQ")
        module.pesq = _unavailable
        sys.modules["pesq"] = module


def _official_semamba(arch, device):
    import sys
    from pathlib import Path
    _allow_official_semamba_import()
    root = Path(__file__).resolve().parents[3] / "src" / "se" / "official" / "semambapp"
    sys.path.insert(0, str(root))
    from model.discriminator import MultiResolutionDiscriminator, MultiScaleSubbandCQTDiscriminator
    from model.semambapp import SEMambapp
    cfg = {
        "model_cfg": {
            "hid_feature": arch["hid_feature"],
            "unet_expansion": 1,
            "compress_factor": 0.3,
            "num_tfmamba": arch["num_tfmamba"],
            "d_state": arch["d_state"],
            "d_conv": arch["d_conv"],
            "expand": arch["expand"],
            "norm_epsilon": arch["norm_epsilon"],
            "beta": 2.0,
            "input_channel": 2,
            "output_channel": 1,
            "dropout": 0.1,
        },
        "stft_cfg": {"sampling_rate": 16000, "n_fft": 400, "hop_size": 100, "win_size": 400},
        "training_cfg": {"loss": {}},
    }
    return SEMambapp(cfg).to(device), MultiScaleSubbandCQTDiscriminator().to(device), MultiResolutionDiscriminator().to(device)


def _official_mel(device):
    import sys
    from pathlib import Path
    _allow_official_semamba_import()
    root = Path(__file__).resolve().parents[3] / "src" / "se" / "official" / "semambapp"
    sys.path.insert(0, str(root))
    from model.loss import MultiScaleMelSpectrogramLoss
    return MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)


def _semamba_parts(implementation, generator, clean, noisy):
    if implementation == "official":
        from model.stfts import mag_phase_istft, mag_phase_stft
        stft, istft = mag_phase_stft, mag_phase_istft
        clean_mag, clean_phase, clean_complex = stft(clean, 400, 100, 400, 0.3)
        noisy_mag, noisy_phase, _ = stft(noisy, 400, 100, 400, 0.3)
        magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
        audio = istft(magnitude, phase, 400, 100, 400, 0.3)[..., : clean.shape[-1]]
        _, _, recomputed = stft(audio, 400, 100, 400, 0.3, addeps=True)
    else:
        from src.training.se.stft import semamba_mag_phase_istft, semamba_mag_phase_stft
        clean_mag, clean_phase, clean_complex = semamba_mag_phase_stft(clean, 400, 100, 400, 0.3, addeps=False)
        noisy_mag, noisy_phase, _ = semamba_mag_phase_stft(noisy, 400, 100, 400, 0.3, addeps=False)
        magnitude, phase = generator(noisy_mag, noisy_phase)
        complex_spectrum = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
        audio = semamba_mag_phase_istft(magnitude, phase, 400, 100, 400, 0.3, length=clean.shape[-1])
        _, _, recomputed = semamba_mag_phase_stft(audio, 400, 100, 400, 0.3, addeps=True)
    return {
        "audio": audio, "mag": magnitude, "phase": phase, "complex": complex_spectrum,
        "clean_mag": clean_mag, "clean_phase": clean_phase, "clean_complex": clean_complex,
        "recomputed": recomputed,
    }


def _semamba_discriminator(implementation, band, resolution, clean, audio):
    if implementation == "official":
        from model.discriminator import discriminator_loss
        real_q, fake_q, _, _ = band(clean.unsqueeze(1), audio.unsqueeze(1))
        real_r, fake_r, _, _ = resolution(clean.unsqueeze(1), audio.unsqueeze(1))
        loss_q, _, _ = discriminator_loss(real_q, fake_q)
        loss_r, _, _ = discriminator_loss(real_r, fake_r)
        return loss_q + loss_r
    from src.training.se.spectral import least_squares_discriminator
    real_q, fake_q, _, _ = band(clean, audio)
    real_r, fake_r, _, _ = resolution(clean, audio)
    return (sum(least_squares_discriminator(real, fake) for real, fake in zip(real_q, fake_q, strict=True))
            + sum(least_squares_discriminator(real, fake) for real, fake in zip(real_r, fake_r, strict=True)))


def _semamba_generator(implementation, band, resolution, clean, parts, mel):
    if implementation == "official":
        from model.discriminator import feature_loss, generator_loss
        from model.loss import phase_losses
        real_q, fake_q, fmap_q_r, fmap_q_g = band(clean.unsqueeze(1), parts["audio"].unsqueeze(1))
        real_r, fake_r, fmap_r_r, fmap_r_g = resolution(clean.unsqueeze(1), parts["audio"].unsqueeze(1))
        gen_q, _ = generator_loss(fake_q)
        gen_r, _ = generator_loss(fake_r)
        adversarial = gen_q + gen_r
        matching = feature_loss(fmap_q_r, fmap_q_g) + feature_loss(fmap_r_r, fmap_r_g)
        ip, gd, iaf = phase_losses(parts["clean_phase"], parts["phase"], {"stft_cfg": {"n_fft": 400}})
        spectral = {
            "magnitude": torch.nn.functional.mse_loss(parts["clean_mag"], parts["mag"]),
            "phase": ip + gd + iaf,
            "complex": torch.nn.functional.mse_loss(parts["clean_complex"], parts["complex"]) * 2,
            "consistency": torch.nn.functional.mse_loss(parts["complex"], parts["recomputed"]) * 2,
        }
        mel_loss = mel(clean.unsqueeze(1), parts["audio"].unsqueeze(1))
    else:
        from src.training.se.spectral import compressed_complex_losses, feature_matching, least_squares_generator, summed_slaney_log_mel
        _, fake_q, real_qf, fake_qf = band(clean, parts["audio"])
        _, fake_r, real_rf, fake_rf = resolution(clean, parts["audio"])
        adversarial = sum(least_squares_generator(score) for score in fake_q) + sum(least_squares_generator(score) for score in fake_r)
        matching = feature_matching(real_qf, fake_qf, scale=2.0, reduction="sum") + feature_matching(real_rf, fake_rf, scale=2.0, reduction="sum")
        spectral = compressed_complex_losses(
            parts["clean_mag"], parts["mag"], parts["clean_phase"], parts["phase"],
            parts["clean_complex"], parts["complex"], parts["recomputed"], phase_mode="matrix",
        )
        mel_loss = summed_slaney_log_mel(clean, parts["audio"], 16000)
    generator = (
        adversarial + matching + mel_loss * 0.1 + spectral["magnitude"] * 0.9
        + spectral["phase"] * 0.3 + spectral["complex"] * 0.1 + spectral["consistency"] * 0.1
    )
    return {
        "generator": generator, "adversarial": adversarial, "matching": matching, "mel": mel_loss, **spectral,
    }


def _official_moss(device):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "src" / "se" / "official" / "clearervoice-studio" / "train" / "speech_enhancement"
    sys.path.insert(0, str(root))
    from models.mossformer_gan.discriminator import Discriminator
    from models.mossformer_gan.generator import SyncANet
    return SyncANet(num_channel=64, num_features=201).to(device), Discriminator(ndf=16).to(device)


def _moss_args(device):
    return SimpleNamespace(batch_size=1, win_type="hamming", win_len=400, win_inc=100, fft_len=400, device=device)


def _official_moss_losses(generator, discriminator, normalized, teacher, device):
    import sys
    from pathlib import Path
    from unittest.mock import patch
    root = Path(__file__).resolve().parents[3] / "src" / "se" / "official" / "clearervoice-studio" / "train" / "speech_enhancement"
    sys.path.insert(0, str(root))
    from losses import loss as official_loss
    from utils.misc import istft, pesq_loss, power_compress, power_uncompress, stft
    args = _moss_args(device)
    spec = power_compress(stft(normalized, args, center=True).float()).permute(0, 1, 3, 2)
    output = generator(spec)

    def on_device(clean_rows, enhanced_rows):
        values = [pesq_loss(clean, enhanced) for clean, enhanced in zip(clean_rows, enhanced_rows)]
        if -1 in values:
            return None
        return torch.tensor([(value - 1) / 3.5 for value in values], dtype=torch.float32, device=device)

    with patch.object(official_loss, "batch_pesq", side_effect=on_device):
        gen, disc = official_loss.loss_mossformergan_se_16k(args, normalized, teacher, output, torch.ones(teacher.shape[0], device=device), discriminator, device)
    if gen is None or disc is None:
        raise RuntimeError("PESQ could not be scored for this MossFormerGAN probe batch")
    real = teacher
    # The published loss consumes the scaled teacher internally. Rebuild the waveform it trains against.
    pred_imag = output[1].permute(0, 1, 3, 2)
    label_real = power_compress(stft(teacher, args, center=True).float())[:, 0, :, :].unsqueeze(1)
    audio = istft(power_uncompress(label_real, pred_imag).squeeze(1), args)
    length = min(audio.shape[-1], real.shape[-1])
    return audio[..., :length], gen, disc


def _independent_moss_losses(generator, discriminator, normalized, teacher, sample_rate):
    from src.training.se.losses import metric_targets
    from src.training.se.mossformergan import _compress, _istft, _stft, _uncompress
    noisy_spec = _compress(_stft(normalized, 400, 100, 400)).permute(0, 1, 3, 2)
    real, imag = generator(noisy_spec)
    real = real.permute(0, 1, 3, 2)
    imag = imag.permute(0, 1, 3, 2)
    pred_mag = torch.sqrt(real ** 2 + imag ** 2)
    label = _compress(_stft(teacher, 400, 100, 400))
    label_real = label[:, 0, :, :].unsqueeze(1)
    label_imag = label[:, 1, :, :].unsqueeze(1)
    label_mag = torch.sqrt(label_real ** 2 + label_imag ** 2)
    ones = torch.ones(teacher.shape[0], device=teacher.device)
    gan = torch.nn.functional.mse_loss(discriminator(label_mag, pred_mag).flatten(), ones)
    loss_mag = torch.nn.functional.mse_loss(pred_mag, label_mag)
    loss_ri = torch.nn.functional.mse_loss(real, label_real) + torch.nn.functional.mse_loss(imag, label_imag)
    audio = _istft(_uncompress(label_real, imag).squeeze(1), 400, 100, 400, teacher.shape[-1])
    length = min(audio.shape[-1], teacher.shape[-1])
    time_loss = (audio[..., :length] - teacher[..., :length]).abs().mean()
    gen = 0.1 * loss_ri + 0.9 * loss_mag + 0.2 * time_loss + 0.05 * gan
    targets = metric_targets(teacher[..., :length], audio[..., :length], sample_rate)
    if targets is None:
        raise RuntimeError("PESQ could not be scored for this MossFormerGAN probe batch")
    fake = discriminator(label_mag, pred_mag.detach())
    real_score = discriminator(label_mag, label_mag)
    disc = torch.nn.functional.mse_loss(real_score.flatten(), ones) + torch.nn.functional.mse_loss(
        fake.flatten(), targets.to(teacher.device)
    )
    return audio[..., :length], gen, disc


def _official_df(independent):
    import os
    import sys
    import types
    from pathlib import Path
    package = Path(__file__).resolve().parents[3] / "src" / "se" / "official" / "deepfilternet" / "DeepFilterNet" / "df"
    df = types.ModuleType("df")
    df.__path__ = [str(package)]
    df.__package__ = "df"
    sys.modules["df"] = df
    libdf = types.ModuleType("libdf")
    libdf.unit_norm_init = lambda *args, **kwargs: None
    libdf.DF = type("DF", (), {})
    sys.modules["libdf"] = libdf
    loguru = types.ModuleType("loguru")

    class _Level:
        no = 30

    class _Logger:
        def level(self, name):
            return _Level()

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru
    sys.path.insert(0, str(package.parent))
    from df.config import config
    from df.deepfilternet3 import DfNet
    config_path = Path(os.environ.get("DF_REFERENCE_CONFIG", str(Path(__file__).resolve().parents[3] / "data" / "checkpoints" / "deepfilternet3" / "DeepFilterNet3" / "config.ini")))
    if not config_path.is_file():
        raise FileNotFoundError(f"DeepFilterNet3 reference config not found: {config_path}")
    config.load(str(config_path))
    official = DfNet(independent.erb_fb.detach().clone(), independent.mask.erb_inv_fb.detach().clone())
    official.load_state_dict(independent.state_dict(), strict=True)
    return official.to(independent.erb_fb.device)


def _official_df_losses(device):
    from df.loss import LocalSnrLoss, MultiResSpecLoss
    from df.modules import local_snr
    return {
        "spectral": MultiResSpecLoss([256, 512, 1024, 2048], 0.3, 500.0, 500.0).to(device),
        "local": LocalSnrLoss(1e-3).to(device),
        "local_snr": local_snr,
    }


def _df_schedule():
    from src.training.se.deepfilternet import deepfilter_cosine_schedules
    config = {
        "optimizer": {"lr": 1e-3, "weight_decay": 1e-12},
        "scheduler": {
            "steps_per_epoch": 8, "epochs": 120, "lr_min": 1e-6, "lr_warmup": 1e-4,
            "warmup_epochs": 3, "weight_decay_end": 0.01,
        },
    }
    return deepfilter_cosine_schedules(config)


def _df_loss(implementation, module, stream, params, clean, noisy, official_loss):
    import torch.nn.functional as F
    from src.se.models.deepfilternet3.dsp import vorbis_window
    from src.training.se.deepfilternet import _local_snr_db, _multi_resolution_spectral, _vorbis_istft
    stream.reset()
    clean_spec, _erb, _feat, clean_complex = _df_analyze(stream, params, clean)
    stream.reset()
    noisy_spec, noisy_erb, noisy_feat, noisy_complex = _df_analyze(stream, params, noisy)
    enhanced, _mask, lsnr, _coefs = module(noisy_spec, noisy_erb, noisy_feat)
    enhanced_complex = torch.view_as_complex(enhanced.squeeze(1))
    clean_c = torch.view_as_complex(clean_spec.squeeze(1))
    frames = min(enhanced_complex.shape[-2], clean_c.shape[-2], noisy_complex.shape[-2])
    bins = min(enhanced_complex.shape[-1], clean_c.shape[-1], noisy_complex.shape[-1])
    enhanced_complex = enhanced_complex[..., :frames, :bins]
    clean_c = clean_c[..., :frames, :bins]
    noisy_c = noisy_complex[..., :frames, :bins]
    window = vorbis_window(params.fft_size).to(device=enhanced_complex.device, dtype=enhanced_complex.real.dtype)
    enhanced_wave = _vorbis_istft(enhanced_complex, params.fft_size, params.hop_size, window)
    clean_td = _vorbis_istft(clean_c, params.fft_size, params.hop_size, window)
    if implementation == "official":
        spectral = official_loss["spectral"](enhanced_wave, clean_td)
        window_frames = _df_windows(params)
        target = official_loss["local_snr"](
            clean_c.unsqueeze(1), (noisy_c - clean_c).unsqueeze(1),
            window_size=window_frames, db=True, window_size_ns=window_frames * 2,
        )[0].squeeze(1).clamp(params.lsnr_min - 1, params.lsnr_max + 1)
        loss = spectral + official_loss["local"](lsnr.reshape(target.shape[0], -1).unsqueeze(-1), target)
    else:
        spectral = _multi_resolution_spectral(enhanced_wave, clean_td)
        target = _local_snr_db(
            clean_c, noisy_c - clean_c, sample_rate=params.sr, n_fft=params.fft_size, hop=params.hop_size,
            low=params.lsnr_min - 1, high=params.lsnr_max + 1,
        )
        predicted = lsnr.reshape(target.shape[0], -1)
        loss = spectral + F.mse_loss(predicted, target) * 1e-3
    return enhanced_wave, loss


def _df_analyze(stream, params, waveform):
    import torch.nn.functional as F
    stream.reset()
    complex_spec = stream.analyze(F.pad(waveform, (0, params.fft_size)))
    spec, erb, feat = stream.features(complex_spec)
    return spec, erb, feat, complex_spec


def _df_windows(params):
    fft_ms = params.fft_size / params.sr * 1000
    hop_ms = params.hop_size / params.sr * 1000
    return max(int(round(1 + (20 - fft_ms) / hop_ms)), 1)
