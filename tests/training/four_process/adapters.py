"""Model-specific constructors and update paths for the four-process probe.

The shared runner stores modules, optimizers, schedulers, and checkpoints.
Loss formulas and update order stay in these adapters.
"""
import random

import numpy as np
import torch

from tests.training.four_process.step import Step, apply_gradient_fault, finite as _finite, grads as _grads, maybe_nan

class Mock:
    """Single linear map. Each step consumes Python, NumPy, and PyTorch RNGs."""
    length = 8
    clip = None
    step_every_update = True
    update_order = ("generator",)
    def build(self, implementation, device):
        del implementation
        return {"generator": torch.nn.Linear(8, 8).to(device)}
    def optimizers(self, modules, lr):
        return {"generator": torch.optim.Adam(modules["generator"].parameters(), lr=lr)}
    def schedulers(self, optimizers):
        return {"generator": torch.optim.lr_scheduler.StepLR(optimizers["generator"], step_size=1, gamma=0.9)}
    def train_step(self, implementation, modules, optimizers, schedulers, clean, noisy, position, fault=None):
        del implementation
        py = random.random()
        np_draw = float(np.random.random())
        torch_draw = torch.randn((), device=noisy.device)
        shift = torch_draw + float(py + np_draw)
        gen = modules["generator"]
        opt = optimizers["generator"]
        gen.train()
        opt.zero_grad(set_to_none=True)
        out = gen(noisy + position)
        loss = _finite(maybe_nan((out - clean + shift).square().mean(), fault), "generator")
        loss.backward()
        apply_gradient_fault(modules, fault)
        grads = _grads(gen, "generator")
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(gen.parameters(), self.clip)
        opt.step()
        if self.step_every_update:
            schedulers["generator"].step()
        return Step(out.detach(), {"generator": loss.detach(), "discriminator": None}, grads, self.update_order)

class TFGridNet:
    length = 2048
    clip = 5.0
    step_every_update = True
    update_order = ("generator",)
    def build(self, implementation, device):
        from tests.training.reference import compare_tf_gridnet as reference
        if implementation == "official":
            net = reference.reference_v1()(
                input_dim=0, n_srcs=1, use_builtin_complex=True, **reference.ARCHITECTURE,
            ).to(device)
            criterion = reference._official_loss_class()(
                window_sz=list(reference.WINDOWS),
                hop_sz=None,
                eps=1e-8,
                time_domain_weight=0.5,
                normalize_variance=False,
                reduction="sum",
            ).to(device)
        else:
            net = reference.build_model("tf_gridnet", reference.ARCHITECTURE, variant="ms-snsd").to(device)
            criterion = None
        self._criterion = criterion
        self._reference = reference
        return {"generator": net}
    def optimizers(self, modules, lr):
        return {"generator": torch.optim.Adam(modules["generator"].parameters(), lr=lr)}
    def schedulers(self, optimizers):
        return {"generator": torch.optim.lr_scheduler.StepLR(optimizers["generator"], step_size=1, gamma=0.9)}
    def train_step(self, implementation, modules, optimizers, schedulers, clean, noisy, position, fault=None):
        del position
        reference = self._reference
        net = modules["generator"]
        opt = optimizers["generator"]
        net.train()
        opt.zero_grad(set_to_none=True)
        if implementation == "official":
            out = net(noisy, torch.tensor([noisy.shape[-1]], device=noisy.device))[0][0]
            loss = self._criterion(clean, out).mean()
        else:
            from src.training.se.spectral import multi_resolution_l1
            out = net(noisy)
            loss = multi_resolution_l1(
                clean, out, reference.WINDOWS, normalize_variance=False,
            )
        loss = _finite(maybe_nan(loss, fault), "generator")
        loss.backward()
        apply_gradient_fault(modules, fault)
        grads = _grads(net, "generator")
        torch.nn.utils.clip_grad_norm_(net.parameters(), self.clip)
        opt.step()
        if self.step_every_update:
            schedulers["generator"].step()
        return Step(out.detach(), {"generator": loss.detach(), "discriminator": None}, grads, self.update_order)

class MPSENet:
    """Official order: discriminator (PESQ target, detached magnitude), then generator.

    ``scheduler_step=update`` steps ExponentialLR after every probe update so a
    resumed worker has a scheduler state that is not the initial state.
    ``scheduler_step=epoch`` leaves the learning rate unchanged for updates
    inside one epoch. Published ``train.py`` steps both schedulers once per
    completed epoch, after the training loop, not after each batch.
    """
    length = 8000
    clip = None
    step_every_update = True
    def configure(self, cfg):
        mode = cfg.get("scheduler_step") or "update"
        if mode not in ("update", "epoch"):
            raise RuntimeError(f"unsupported MP-SENet scheduler step {mode}")
        self.step_every_update = mode == "update"
        self.probe = {
            "scheduler": (
                "ExponentialLR gamma 0.99 stepped every probe update"
                if self.step_every_update
                else "ExponentialLR gamma 0.99 held for the epoch; published training steps once per completed epoch"
            ),
            "scheduler_step": mode,
            "width": "dense_channel 16, one TS-Conformer; published width is 64 and four blocks",
        }
    update_order = ("discriminator", "generator")
    lr = 5e-4
    betas = (0.8, 0.99)
    weight_decay = 0.01
    gamma = 0.99
    def hyper(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            dense_channel=16,
            num_tsconformers=1,
            n_fft=400,
            hop_size=100,
            win_size=400,
            compress_factor=0.3,
            beta=2.0,
            sampling_rate=16000,
        )
    def build(self, implementation, device):
        h = self.hyper()
        if implementation == "official":
            generator, discriminator = _official_mp(h, device)
        else:
            from src.se import build_model
            from src.training.se.discriminator import MetricDiscriminator
            generator = build_model("mp_senet", vars(h)).to(device)
            discriminator = MetricDiscriminator().to(device)
        self._h = h
        self._implementation = implementation
        return {"generator": generator, "discriminator": discriminator}
    def optimizers(self, modules, lr):
        del lr
        made = {}
        for name in ("discriminator", "generator"):
            made[name] = torch.optim.AdamW(
                modules[name].parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        return made
    def schedulers(self, optimizers):
        return {
            name: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.gamma)
            for name, opt in optimizers.items()
        }
    def train_step(self, implementation, modules, optimizers, schedulers, clean, noisy, position, fault=None):
        del position
        from src.training.se.losses import discriminator_loss, generator_losses, metric_targets
        h = self._h
        generator = modules["generator"]
        discriminator = modules["discriminator"]
        generator.train()
        discriminator.train()
        optimizers["discriminator"].zero_grad(set_to_none=True)
        with torch.no_grad():
            detached = _mp_spectra(implementation, generator, clean, noisy, h)
        ones = torch.ones(clean.shape[0], device=clean.device)
        real = discriminator(detached["clean_mag"], detached["clean_mag"])
        fake = discriminator(detached["clean_mag"], detached["mag_hat"])
        targets = metric_targets(clean, detached["audio"], h.sampling_rate)
        if targets is None:
            raise RuntimeError("PESQ could not be scored for this MP-SENet probe batch")
        d_loss = _finite(maybe_nan(discriminator_loss(real, fake, ones, targets), fault), "discriminator")
        d_loss.backward()
        grads = _grads(discriminator, "discriminator")
        optimizers["discriminator"].step()
        frozen = [p for p in discriminator.parameters() if p.requires_grad]
        for parameter in frozen:
            parameter.requires_grad_(False)
        try:
            optimizers["generator"].zero_grad(set_to_none=True)
            parts = _mp_spectra(implementation, generator, clean, noisy, h)
            metric = discriminator(parts["clean_mag"], parts["mag_hat"])
            losses = generator_losses(
                parts["clean_mag"], parts["mag"], parts["clean_phase"], parts["phase"],
                parts["clean_complex"], parts["complex"], parts["recomputed_complex"],
                clean, parts["audio"], metric, ones,
            )
            g_loss = _finite(losses["total"], "generator")
            g_loss.backward()
            apply_gradient_fault(modules, fault)
            grads.update(_grads(generator, "generator"))
            optimizers["generator"].step()
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        if self.step_every_update:
            for name in self.update_order:
                schedulers[name].step()
        payload = {"generator": g_loss.detach(), "discriminator": d_loss.detach()}
        for name, value in losses.items():
            if name != "total":
                payload[name] = value.detach()
        return Step(parts["audio"].detach(), payload, grads, self.update_order)

def _official_mp(h, device):
    import sys
    import types
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "src" / "se" / "official" / "mp-senet"
    joblib = types.ModuleType("joblib")
    joblib.Parallel = lambda *args, **kwargs: None
    joblib.delayed = lambda fn: fn
    sys.modules.setdefault("joblib", joblib)
    sys.modules.setdefault("librosa", types.ModuleType("librosa"))
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda *args, **kwargs: None
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pylab", types.ModuleType("matplotlib.pylab"))
    sys.path.insert(0, str(root))
    from models.discriminator import MetricDiscriminator as OfficialDiscriminator
    from models.model import MPNet
    generator = MPNet(h, num_tsblocks=h.num_tsconformers).to(device)
    discriminator = OfficialDiscriminator().to(device)
    return generator, discriminator

def _mp_spectra(implementation, generator, clean, noisy, h):
    if implementation == "official":
        from dataset import mag_pha_istft as istft
        from dataset import mag_pha_stft as stft
    else:
        from src.training.se.stft import mag_pha_istft as istft
        from src.training.se.stft import mag_pha_stft as stft
    noisy_mag, noisy_phase, _ = stft(noisy, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
    audio = istft(magnitude, phase, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    if audio.shape[-1] != clean.shape[-1]:
        raise RuntimeError(f"iSTFT length {audio.shape[-1]} != waveform {clean.shape[-1]}")
    clean_mag, clean_phase, clean_complex = stft(clean, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    mag_hat, _, recomputed = stft(audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    return {
        "audio": audio,
        "mag": magnitude,
        "phase": phase,
        "complex": complex_spectrum,
        "clean_mag": clean_mag,
        "clean_phase": clean_phase,
        "clean_complex": clean_complex,
        "mag_hat": mag_hat,
        "recomputed_complex": recomputed,
    }

from tests.training.four_process.real_models import DeepFilterNet3, MossFormerGAN, SEMambaPP

ADAPTERS = {
    "mock": Mock(),
    "tf_gridnet": TFGridNet(),
    "mp_senet": MPSENet(),
    "semambapp": SEMambaPP(),
    "mossformergan": MossFormerGAN(),
    "deepfilternet3": DeepFilterNet3(),
}
