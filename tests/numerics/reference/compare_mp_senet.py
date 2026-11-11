"""MP-SENet forward, GAN update, and a short speech run versus the reference.

Tolerances, fixed before reading the measured errors:

- forward and loss components: atol 1e-5
- parameter gradients: atol 1e-4, and cosine >= 0.999 when |grad| max > 1e-5
- AdamW weights: first-step closed form. Near-zero gradients may move by O(lr)

The update order is the official ``train.py`` order: discriminator (real target
1, fake target normalized PESQ, enhanced magnitude detached), then generator.
The discriminator is not stepped again after the generator backward.

Official ``AdaptiveMaxPool2d`` has no deterministic CUDA backward, so this
comparison does not enable ``torch.use_deterministic_algorithms``. Launch with
``CUBLAS_WORKSPACE_CONFIG=:4096:8``.

    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/lab/bin/python tests/numerics/reference/compare_mp_senet.py
"""

from __future__ import annotations

import sys
import types
import ast
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.numerics.reference.common import (  # noqa: E402
    configured_steps,
    environment,
    gap,
    grad_report,
    load_cached,
    optim_report,
    reference_scores,
    rng_restore,
    rng_snapshot,
    save_report,
    weight_report,
)

OFFICIAL = ROOT / "src" / "se" / "official" / "mp-senet"
matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *args, **kwargs: None
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.pylab", types.ModuleType("matplotlib.pylab"))
sys.path.insert(0, str(OFFICIAL))

from dataset import mag_pha_istft as official_istft  # noqa: E402
from dataset import mag_pha_stft as official_stft  # noqa: E402
from models.discriminator import MetricDiscriminator as OfficialDiscriminator  # noqa: E402
from models.discriminator import batch_pesq as official_batch_pesq  # noqa: E402
from models.model import MPNet, phase_losses as official_phase_losses  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from src.se import build_model  # noqa: E402
from src.experiments.calc.rng import seed_model_init  # noqa: E402
from src.experiments.official.mp_senet.discriminator import MetricDiscriminator  # noqa: E402
from src.experiments.calc.losses import discriminator_loss, generator_losses, metric_targets  # noqa: E402
from src.experiments.official.mp_senet.reference import MPSENetTask  # noqa: E402
from src.experiments.calc.stft import mag_pha_istft, mag_pha_stft  # noqa: E402

LOSS_ATOL = 1e-5
GRAD_ATOL = 1e-4
LR = 5e-4
BETAS = (0.8, 0.99)
WEIGHT_DECAY = 0.01
LOSS_NAMES = ("total", "magnitude", "phase", "complex", "consistency", "metric", "time")


def _official_train_statements(first: int, last: int):
    """Compile the actual loss statements from the checked-out official train.py."""
    path = OFFICIAL / "train.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    statements = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.If))
                  and first <= node.lineno <= last and node.end_lineno <= last
                  and node.lineno != 178]  # metric_g is evaluated by the caller, with matched RNG
    # Keep only top-level statements in the requested span, not children of an if.
    statements = [node for node in statements if not any(
        isinstance(parent, ast.If) and parent.lineno < node.lineno <= parent.end_lineno
        for parent in statements)]
    statements.sort(key=lambda node: node.lineno)
    if not statements or statements[0].lineno != first or statements[-1].end_lineno != last:
        raise RuntimeError("Official train.py loss statements changed; inspect before comparison")
    return compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), str(path), "exec")


_OFFICIAL_D = _official_train_statements(136, 144)
_OFFICIAL_G = _official_train_statements(167, 181)


def _official_pesq(clean, audio):
    # The reference module is imported from a checkout rather than installed;
    # joblib subprocesses cannot import its top-level ``models`` package.
    # Use joblib's thread backend while retaining the official scoring function.
    from joblib import parallel_backend
    with parallel_backend("threading"):
        return official_batch_pesq(list(clean.detach().cpu().numpy()),
                                   list(audio.detach().cpu().numpy()))


def _official_components(parts, metric, ones, clean):
    values = dict(F=F, phase_losses=official_phase_losses, clean_mag=parts["clean_mag"],
                  mag_g=parts["magnitude"], clean_pha=parts["clean_phase"], pha_g=parts["phase"],
                  clean_com=parts["clean_complex"], com_g=parts["complex"],
                  com_g_hat=parts["recomputed"], clean_audio=clean, audio_g=parts["audio"],
                  metric_g=metric, one_labels=ones)
    exec(_OFFICIAL_G, values)
    return {name: values[key] for name, key in {
        "total": "loss_gen_all", "magnitude": "loss_mag", "phase": "loss_pha",
        "complex": "loss_com", "consistency": "loss_stft", "metric": "loss_metric",
        "time": "loss_time"}.items()}


def _official_discriminator_loss(real, fake, ones, target):
    values = dict(F=F, one_labels=ones, metric_r=real, metric_g=fake,
                  batch_pesq_score=target, device=real.device)
    exec(_OFFICIAL_D, values)
    return values["loss_disc_all"]


def _hyper():
    from types import SimpleNamespace

    return SimpleNamespace(
        dense_channel=64,
        num_tsconformers=4,
        n_fft=400,
        hop_size=100,
        win_size=400,
        compress_factor=0.3,
        beta=2.0,
        sampling_rate=16000,
    )


def _build(h, device, seed: int = 7):
    torch.manual_seed(seed)
    with seed_model_init(seed):
        generator = build_model("mp_senet", vars(h)).to(device)
        discriminator = MetricDiscriminator().to(device)
    official_g = MPNet(h, num_tsblocks=h.num_tsconformers).to(device)
    official_d = OfficialDiscriminator().to(device)
    official_g.load_state_dict(generator.state_dict(), strict=True)
    official_d.load_state_dict(discriminator.state_dict(), strict=True)
    return generator, discriminator, official_g, official_d


def _optimizers(generator, discriminator, official_g, official_d):
    return (
        torch.optim.AdamW(generator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY),
        torch.optim.AdamW(discriminator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY),
        torch.optim.AdamW(official_g.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY),
        torch.optim.AdamW(official_d.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY),
    )


def _spectra(generator, clean, noisy, h, stft, istft):
    noisy_mag, noisy_phase, _ = stft(noisy, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
    audio = istft(magnitude, phase, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    if audio.shape[-1] != clean.shape[-1]:
        raise RuntimeError(f"iSTFT length {audio.shape[-1]} != waveform {clean.shape[-1]}")
    clean_mag, clean_phase, clean_complex = stft(clean, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    hat_mag, _, recomputed = stft(audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    return {
        "noisy_mag": noisy_mag,
        "magnitude": magnitude,
        "phase": phase,
        "audio": audio,
        "clean_mag": clean_mag,
        "clean_phase": clean_phase,
        "clean_complex": clean_complex,
        "complex": complex_spectrum,
        "recomputed": recomputed,
        "hat_mag": hat_mag,
    }


def _paired_spectra(generator, official_g, clean, noisy, h):
    snap = rng_snapshot()
    ours = _spectra(generator, clean, noisy, h, mag_pha_stft, mag_pha_istft)
    rng_restore(snap)
    official = _spectra(official_g, clean, noisy, h, official_stft, official_istft)
    return ours, official


def _components(parts, score, ones, clean):
    # generator_losses takes clean audio. The call above cannot omit it.
    return generator_losses(
        parts["clean_mag"],
        parts["magnitude"],
        parts["clean_phase"],
        parts["phase"],
        parts["clean_complex"],
        parts["complex"],
        parts["recomputed"],
        clean,
        parts["audio"],
        score,
        ones,
    )


def _loss_gap(official: dict, ours: dict) -> dict[str, float]:
    return {
        name: abs(float(official[name].detach()) - float(ours[name].detach()))
        for name in LOSS_NAMES
    }


def _values(losses: dict) -> dict[str, float]:
    return {name: float(losses[name].detach()) for name in LOSS_NAMES}


def gan_step(modules, optimizers, clean, noisy, h) -> dict:
    """One official-order update. Discriminator gradients are read before the generator backward."""
    generator, discriminator, official_g, official_d = modules
    optim_g, optim_d, optim_og, optim_od = optimizers
    for module in modules:
        module.train()
    for optim in optimizers:
        optim.zero_grad(set_to_none=True)
    ours, official = _paired_spectra(generator, official_g, clean, noisy, h)
    ours["audio"].retain_grad()
    official["audio"].retain_grad()
    ours_real, official_real = _paired_clean_and_fake(discriminator, official_d, ours, official)
    ones_ours = torch.ones(clean.shape[0], device=clean.device)
    ones_official = torch.ones(clean.shape[0], device=clean.device)
    pesq_ours = metric_targets(clean, ours["audio"].detach(), h.sampling_rate)
    pesq_official = _official_pesq(clean, official["audio"])
    loss_d_ours = discriminator_loss(ours_real[0], ours_real[1], ones_ours, pesq_ours)
    loss_d_official = _official_discriminator_loss(official_real[0], official_real[1], ones_official, pesq_official)
    loss_d_ours.backward()
    loss_d_official.backward()
    discriminator_grad = grad_report(official_d, discriminator, atol=GRAD_ATOL)
    optim_od.step()
    optim_d.step()
    discriminator_weight_after_d = weight_report(official_d, discriminator)
    for optim in (optim_g, optim_og):
        optim.zero_grad(set_to_none=True)
    snap = rng_snapshot()
    ours_metric = discriminator(ours["clean_mag"], ours["hat_mag"])
    rng_restore(snap)
    official_metric = official_d(official["clean_mag"], official["hat_mag"])
    loss_g_ours = _components(ours, ours_metric, ones_ours, clean)
    loss_g_official = _official_components(official, official_metric, ones_official, clean)
    loss_g_ours["total"].backward()
    loss_g_official["total"].backward()
    generator_grad = grad_report(official_g, generator, atol=GRAD_ATOL)
    optim_og.step()
    optim_g.step()
    return {
        "forward": {
            "stft": gap(official["noisy_mag"], ours["noisy_mag"]),
            "magnitude": gap(official["magnitude"], ours["magnitude"]),
            "phase": gap(official["phase"], ours["phase"]),
            "wave": gap(official["audio"], ours["audio"]),
            "real_score": gap(official_real[0], ours_real[0]),
            "fake_score": gap(official_real[1], ours_real[1]),
            "metric_score": gap(official_metric, ours_metric),
        },
        "pesq_target": _pesq_gap(pesq_official, pesq_ours),
        "discriminator_loss": abs(float(loss_d_official.detach()) - float(loss_d_ours.detach())),
        "official_discriminator_loss": float(loss_d_official.detach()),
        "generator_loss_gap": _loss_gap(loss_g_official, loss_g_ours),
        "waveform_gradient": gap(official["audio"].grad, ours["audio"].grad),
        "official_generator_loss": _values(loss_g_official),
        "independent_generator_loss": _values(loss_g_ours),
        "discriminator_grad": discriminator_grad,
        "generator_grad": generator_grad,
        "discriminator_weight_after_d": discriminator_weight_after_d,
        "generator_weight": weight_report(official_g, generator),
        "discriminator_weight": weight_report(official_d, discriminator),
        "generator_optim": optim_report(optim_og, optim_g),
        "discriminator_optim": optim_report(optim_od, optim_d),
    }


def _paired_clean_and_fake(discriminator, official_d, ours, official):
    snap = rng_snapshot()
    real_ours = discriminator(ours["clean_mag"], ours["clean_mag"])
    fake_ours = discriminator(ours["clean_mag"], ours["hat_mag"].detach())
    rng_restore(snap)
    real_official = official_d(official["clean_mag"], official["clean_mag"])
    fake_official = official_d(official["clean_mag"], official["hat_mag"].detach())
    return (real_ours, fake_ours), (real_official, fake_official)


def _pesq_gap(official, ours) -> dict:
    if official is None and ours is None:
        return {"status": "na", "reason": "pesq failed on both sides", "max_abs": None}
    if official is None or ours is None:
        return {
            "status": "mismatch",
            "reason": "pesq failed on one side only",
            "official_none": official is None,
            "independent_none": ours is None,
        }
    return {"status": "ok", **gap(official, ours.to(official.device))}


def _forward_modes(modules, clean, noisy, h) -> dict:
    generator, discriminator, official_g, official_d = modules
    report = {}
    for mode in ("eval", "train"):
        for module in modules:
            getattr(module, mode)()
        snap = rng_snapshot()
        ours = _spectra(generator, clean, noisy, h, mag_pha_stft, mag_pha_istft)
        ours_score = discriminator(ours["clean_mag"], ours["hat_mag"])
        rng_restore(snap)
        official = _spectra(official_g, clean, noisy, h, official_stft, official_istft)
        official_score = official_d(official["clean_mag"], official["hat_mag"])
        ones = torch.ones(clean.shape[0], device=clean.device)
        ours_loss = _components(ours, ours_score, ones, clean)
        official_loss = _official_components(official, official_score, ones, clean)
        report[mode] = {
            "stft": gap(official["noisy_mag"], ours["noisy_mag"])["max_abs"],
            "magnitude": gap(official["magnitude"], ours["magnitude"])["max_abs"],
            "phase": gap(official["phase"], ours["phase"])["max_abs"],
            "score": gap(official_score, ours_score)["max_abs"],
            "wave": gap(official["audio"], ours["audio"])["max_abs"],
            "loss": _loss_gap(official_loss, ours_loss),
        }
    return report


def _task_matches(modules, clean, noisy, h, device) -> dict:
    generator, discriminator, _official_g, _official_d = modules
    config = {
        "seed": 7,
        "loss": "official",
        "optimizer": {"lr": LR, "betas": list(BETAS), "weight_decay": WEIGHT_DECAY},
        "scheduler": {"gamma": 0.99},
        "model": vars(h),
    }
    task = MPSENetTask(config, device)
    task.generator.load_state_dict(generator.state_dict(), strict=True)
    task.discriminator.load_state_dict(discriminator.state_dict(), strict=True)
    task.optim_g = torch.optim.AdamW(task.generator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    task.optim_d = torch.optim.AdamW(task.discriminator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    held = rng_snapshot()
    metrics = task.train_update([{"clean": clean, "noisy": noisy}], 1)
    rng_restore(held)
    reference_modules = _build(h, device)
    reference_modules[0].load_state_dict(generator.state_dict(), strict=True)
    reference_modules[1].load_state_dict(discriminator.state_dict(), strict=True)
    reference_opt = _optimizers(*reference_modules)
    manual = gan_step(reference_modules, reference_opt, clean, noisy, h)
    return {
        "task_losses": metrics,
        "weight_versus_manual_independent": weight_report(reference_modules[0], task.generator),
        "discriminator_weight_versus_manual": weight_report(reference_modules[1], task.discriminator),
        "manual_generator_loss": manual["official_generator_loss"]["total"],
    }


def _probe(modules, clean, noisy, sample_rate) -> dict:
    generator, _discriminator, official_g, _official_d = modules
    for module in modules:
        module.eval()
    with torch.no_grad():
        snap = rng_snapshot()
        ours = _spectra(generator, clean, noisy, _hyper(), mag_pha_stft, mag_pha_istft)
        rng_restore(snap)
        official = _spectra(official_g, clean, noisy, _hyper(), official_stft, official_istft)
    clean_cpu = clean[0].detach().cpu()
    return {
        "wave": gap(official["audio"], ours["audio"]),
        "official": reference_scores(clean_cpu, official["audio"][0], sample_rate),
        "independent": reference_scores(clean_cpu, ours["audio"][0], sample_rate),
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h = _hyper()
    report = {
        "environment": environment(device),
        "official_commit": "89932cfe90d1dacb8e170e4a331d762462c21792",
        "tolerances": {"loss_atol": LOSS_ATOL, "grad_atol": GRAD_ATOL, "large_cosine_min": 0.999},
        "optimizer": {"name": "adamw", "lr": LR, "betas": list(BETAS), "weight_decay": WEIGHT_DECAY},
        "scheduler": "ExponentialLR is per epoch and is not stepped during these 20 optimizer updates",
        "update_order": "discriminator (PESQ target, detached magnitude) then generator",
        "batch_size": 2,
        "segment_size": 32000,
        "note": "Official config.json uses batch 4. This run uses batch 2 of the smoke set.",
    }
    modules = _build(h, device)
    report["parameters"] = sum(parameter.numel() for parameter in modules[0].parameters())
    torch.manual_seed(11)
    synthetic_clean = torch.randn(2, 8000, device=device)
    synthetic_noisy = synthetic_clean * 0.5 + 0.05
    report["synthetic_forward"] = _forward_modes(modules, synthetic_clean, synthetic_noisy, h)
    try:
        train = load_cached("mp_train")
        valid = load_cached("valid")
        report["input_protocol"] = "historical_speech_cache_mp_train"
    except FileNotFoundError as exc:
        generator = torch.Generator().manual_seed(11)
        clean = torch.randn(4, 8000, generator=generator)
        noisy = clean * 0.5 + 0.05 * torch.randn(4, 8000, generator=generator)
        train = {"clean": clean, "noisy": noisy, "case_id": [f"synthetic-mp-{i}" for i in range(4)]}
        valid = {"clean": clean[:1], "noisy": noisy[:1], "case_id": ["synthetic-mp-valid"]}
        report["input_protocol"] = "synthetic_fixed_seed"
        report["input_protocol_detail"] = (
            "torch.Generator seed 11; 4 utterances x 8000 samples. "
            "Not speech_cache.pt and not training.duckdb."
        )
        report["cache_missing"] = str(exc)
    report["train_case_ids"] = train["case_id"]
    report["valid_case_id"] = valid["case_id"]
    clean = train["clean"][:2].to(device)
    noisy = train["noisy"][:2].to(device)
    modules = _build(h, device)
    report["task_versus_manual"] = _task_matches(modules, clean, noisy, h, device)
    modules = _build(h, device)
    optimizers = _optimizers(*modules)
    history = []
    steps = configured_steps(20)
    report["steps"] = steps
    for step in range(steps):
        start = (2 * step) % train["clean"].shape[0]
        indices = [start, (start + 1) % train["clean"].shape[0]]
        clean_batch = train["clean"][indices].to(device)
        noisy_batch = train["noisy"][indices].to(device)
        case_ids = [train["case_id"][index] for index in indices]
        record = gan_step(modules, optimizers, clean_batch, noisy_batch, h)
        record["step"] = step
        record["case_id"] = case_ids
        history.append(
            {
                "step": step,
                "case_id": case_ids,
                "official_generator": record["official_generator_loss"]["total"],
                "independent_generator": record["independent_generator_loss"]["total"],
                "generator_gap": record["generator_loss_gap"]["total"],
                "discriminator_gap": record["discriminator_loss"],
                "official_discriminator": record["official_discriminator_loss"],
                "wave_max_abs": record["forward"]["wave"]["max_abs"],
                "generator_grad_max_abs": record["generator_grad"]["max_abs"],
                "generator_large_cosine": record["generator_grad"]["large_cosine_min"],
                "generator_weight_max_abs": record["generator_weight"]["max_abs"],
                "pesq_target": record["pesq_target"],
            }
        )
        report["last_step"] = record
        report["history"] = history
        save_report("mp_senet.json", report)
        print(f"mp-senet step {step} gen {record['official_generator_loss']['total']:.6f} gap {record['generator_loss_gap']['total']:.6e}", flush=True)
    report["probe_case_id"] = valid["case_id"]
    report["probe"] = _probe(modules, valid["clean"].to(device), valid["noisy"].to(device), 16000)
    path = save_report("mp_senet.json", report)
    print(path)


if __name__ == "__main__":
    main()
