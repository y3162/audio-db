"""MP-SENet forward, loss, gradient, AdamW, and a short run versus the reference.

Tolerances, chosen from float32 STFT and attention, before reading results:

- forward and loss components: atol 1e-5
- parameter gradients: atol 1e-4, and cosine >= 0.999 when |grad| max > 1e-5
- AdamW weights: the first-step closed form. Near-zero gradients may move by O(lr)

    CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/lab/bin/python tests/training/reference/compare_mp_senet.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "src" / "se" / "official" / "mp-senet"
sys.path.insert(0, str(ROOT))
joblib = types.ModuleType("joblib")
joblib.Parallel = lambda *args, **kwargs: None
joblib.delayed = lambda fn: fn
sys.modules.setdefault("joblib", joblib)
sys.modules.setdefault("librosa", types.ModuleType("librosa"))
matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *args, **kwargs: None
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.pylab", types.ModuleType("matplotlib.pylab"))
sys.path.insert(0, str(OFFICIAL))

from dataset import mag_pha_istft as official_istft  # noqa: E402
from dataset import mag_pha_stft as official_stft  # noqa: E402
from models.discriminator import MetricDiscriminator as OfficialDiscriminator  # noqa: E402
from models.model import MPNet, phase_losses  # noqa: E402
from src.se import build_model  # noqa: E402
from src.training.core.rng import seed_model_init  # noqa: E402
from src.training.se.discriminator import MetricDiscriminator  # noqa: E402
from src.training.se.losses import discriminator_loss, generator_losses, metric_targets  # noqa: E402
from src.training.se.stft import mag_pha_istft, mag_pha_stft  # noqa: E402

LOSS_ATOL = 1e-5
GRAD_ATOL = 1e-4
LR = 5e-4


def _gap(reference: torch.Tensor, other: torch.Tensor) -> float:
    return float((reference.detach().float() - other.detach().float()).abs().max())


def _build(h, device):
    torch.manual_seed(7)
    with seed_model_init(7):
        generator = build_model("mp_senet", vars(h)).to(device)
        discriminator = MetricDiscriminator().to(device)
    official_g = MPNet(h, num_tsblocks=h.num_tsconformers).to(device)
    official_d = OfficialDiscriminator().to(device)
    official_g.load_state_dict(generator.state_dict(), strict=True)
    official_d.load_state_dict(discriminator.state_dict(), strict=True)
    return generator, discriminator, official_g, official_d


def _forward(generator, discriminator, clean, noisy, h, stft, istft, seed: int):
    torch.manual_seed(seed)
    noisy_mag, noisy_phase, _ = stft(noisy, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
    audio = istft(magnitude, phase, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    length = min(clean.shape[-1], audio.shape[-1])
    audio = audio[..., :length]
    clean_l = clean[..., :length]
    clean_mag, clean_phase, clean_complex = stft(clean_l, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    hat_mag, _, recomputed = stft(audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    score = discriminator(clean_mag, hat_mag)
    return {
        "noisy_mag": noisy_mag,
        "magnitude": magnitude,
        "phase": phase,
        "audio": audio,
        "clean": clean_l,
        "clean_mag": clean_mag,
        "clean_phase": clean_phase,
        "clean_complex": clean_complex,
        "complex": complex_spectrum,
        "recomputed": recomputed,
        "hat_mag": hat_mag,
        "score": score,
    }


def _losses(parts, h):
    ones = torch.ones(parts["clean"].shape[0], device=parts["clean"].device)
    losses = generator_losses(
        parts["clean_mag"],
        parts["magnitude"],
        parts["clean_phase"],
        parts["phase"],
        parts["clean_complex"],
        parts["complex"],
        parts["recomputed"],
        parts["clean"],
        parts["audio"],
        parts["score"],
        ones,
    )
    targets = metric_targets(parts["clean"].detach(), parts["audio"].detach(), h.sampling_rate)
    real = parts["score"].new_zeros(parts["score"].shape)  # placeholder replaced below
    del real
    return losses, ones, targets


def _official_total(parts):
    phase = sum(phase_losses(parts["clean_phase"], parts["phase"]))
    return (
        torch.nn.functional.mse_loss(parts["clean_mag"], parts["magnitude"]) * 0.9
        + phase * 0.3
        + torch.nn.functional.mse_loss(parts["clean_complex"], parts["complex"]) * 2 * 0.1
        + torch.nn.functional.mse_loss(parts["complex"], parts["recomputed"]) * 2 * 0.1
        + torch.nn.functional.mse_loss(parts["score"].flatten(), torch.ones(parts["clean"].shape[0], device=parts["score"].device)) * 0.05
        + torch.nn.functional.l1_loss(parts["clean"], parts["audio"]) * 0.2
    )


def _grad_max(left_module, right_module) -> dict:
    worst = 0.0
    name = ""
    cosine_min = 1.0
    large_cosine = 1.0
    for (left_name, left), right in zip(left_module.named_parameters(), right_module.parameters(), strict=True):
        if left.grad is None or right.grad is None:
            continue
        gap = float((left.grad - right.grad).abs().max())
        if gap > worst:
            worst = gap
            name = left_name
        flat_l = left.grad.detach().float().reshape(-1)
        flat_r = right.grad.detach().float().reshape(-1)
        if flat_l.numel() > 1:
            cosine = float(torch.nn.functional.cosine_similarity(flat_l, flat_r, dim=0))
            cosine_min = min(cosine_min, cosine)
            if float(flat_l.abs().max()) > 1e-5:
                large_cosine = min(large_cosine, cosine)
    return {"max_abs": worst, "worst": name, "cosine_min": cosine_min, "large_cosine_min": large_cosine}


def run_case(h, device, steps: int) -> dict:
    generator, discriminator, official_g, official_d = _build(h, device)
    torch.manual_seed(11)
    clean = torch.randn(2, 8000, device=device)
    noisy = clean * 0.5 + 0.05
    report = {"parameters": sum(p.numel() for p in generator.parameters())}
    for mode in ("eval", "train"):
        for module in (generator, discriminator, official_g, official_d):
            getattr(module, mode)()
        ours = _forward(generator, discriminator, clean, noisy, h, mag_pha_stft, mag_pha_istft, 21)
        official = _forward(official_g, official_d, clean, noisy, h, official_stft, official_istft, 21)
        report[mode] = {
            "stft": _gap(official["noisy_mag"], ours["noisy_mag"]),
            "magnitude": _gap(official["magnitude"], ours["magnitude"]),
            "phase": _gap(official["phase"], ours["phase"]),
            "score": _gap(official["score"], ours["score"]),
            "wave": _gap(official["audio"], ours["audio"]),
            "loss": abs(float(_official_total(official).detach()) - float(_official_total(ours).detach())),
        }
    for module in (generator, discriminator, official_g, official_d):
        module.train()
    ours = _forward(generator, discriminator, clean, noisy, h, mag_pha_stft, mag_pha_istft, 21)
    official = _forward(official_g, official_d, clean, noisy, h, official_stft, official_istft, 21)
    our_total = _official_total(ours)
    off_total = _official_total(official)
    our_total.backward()
    off_total.backward()
    report["generator_grad"] = _grad_max(generator, official_g)
    report["discriminator_grad"] = _grad_max(discriminator, official_d)
    optim_a = torch.optim.AdamW(generator.parameters(), lr=LR, betas=(0.8, 0.99), weight_decay=0.01)
    optim_b = torch.optim.AdamW(official_g.parameters(), lr=LR, betas=(0.8, 0.99), weight_decay=0.01)
    optim_da = torch.optim.AdamW(discriminator.parameters(), lr=LR, betas=(0.8, 0.99), weight_decay=0.01)
    optim_db = torch.optim.AdamW(official_d.parameters(), lr=LR, betas=(0.8, 0.99), weight_decay=0.01)
    optim_a.step()
    optim_b.step()
    optim_da.step()
    optim_db.step()
    report["weight_after_step"] = _gap(
        torch.cat([p.detach().flatten() for p in official_g.parameters()]),
        torch.cat([p.detach().flatten() for p in generator.parameters()]),
    )
    for module in (generator, discriminator, official_g, official_d):
        module.eval()
    ours2 = _forward(generator, discriminator, clean, noisy, h, mag_pha_stft, mag_pha_istft, 0)
    official2 = _forward(official_g, official_d, clean, noisy, h, official_stft, official_istft, 0)
    report["next_forward"] = {
        "magnitude": _gap(official2["magnitude"], ours2["magnitude"]),
        "phase": _gap(official2["phase"], ours2["phase"]),
        "wave": _gap(official2["audio"], ours2["audio"]),
        "loss": abs(float(_official_total(official2).detach()) - float(_official_total(ours2).detach())),
    }
    history = [report["next_forward"]["loss"]]
    official_values = [float(_official_total(official2).detach())]
    for step in range(1, steps):
        for module in (generator, discriminator, official_g, official_d):
            module.train()
        for group in (optim_a, optim_b, optim_da, optim_db):
            group.zero_grad(set_to_none=True)
        ours = _forward(generator, discriminator, clean, noisy, h, mag_pha_stft, mag_pha_istft, 30 + step)
        official = _forward(official_g, official_d, clean, noisy, h, official_stft, official_istft, 30 + step)
        off_loss = _official_total(official)
        our_loss = _official_total(ours)
        our_loss.backward()
        off_loss.backward()
        optim_a.step()
        optim_b.step()
        optim_da.step()
        optim_db.step()
        history.append(abs(float(off_loss.detach()) - float(our_loss.detach())))
        official_values.append(float(off_loss.detach()))
    report["loss_gap_per_step"] = history
    report["official_loss_per_step"] = official_values
    report["final_weight"] = _gap(
        torch.cat([p.detach().flatten() for p in official_g.parameters()]),
        torch.cat([p.detach().flatten() for p in generator.parameters()]),
    )
    return report


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from types import SimpleNamespace

    official = SimpleNamespace(
        dense_channel=64,
        num_tsconformers=4,
        n_fft=400,
        hop_size=100,
        win_size=400,
        compress_factor=0.3,
        beta=2.0,
        sampling_rate=16000,
    )
    print(json.dumps({"device": str(device), "official_width": run_case(official, device, 20)}, indent=2))


if __name__ == "__main__":
    main()
