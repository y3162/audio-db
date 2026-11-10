"""One SEMamba++ training step, and a short run, against the reference modules.

Tolerances are fixed before reading the measured errors:

- loss components and discriminator outputs: atol 1e-5
- generator outputs and parameter gradients: atol 1e-4, cosine >= 0.999
- AdamW weights after one step: atol 1e-4, except elements whose gradient
  absolute value is below 1e-6, where AdamW epsilon can amplify the sign

    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/semamba/bin/python tests/numerics/reference/compare_semamba_step.py narrow
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import torch

if "pesq" not in sys.modules:
    sys.modules["pesq"] = types.ModuleType("pesq")
    sys.modules["pesq"].pesq = lambda *args, **kwargs: 0.0

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "src" / "se" / "official" / "semambapp"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from model.discriminator import (  # noqa: E402
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from model.discriminator import MultiResolutionDiscriminator as OfficialMRD  # noqa: E402
from model.discriminator import MultiScaleSubbandCQTDiscriminator as OfficialCQT  # noqa: E402
from model.loss import MultiScaleMelSpectrogramLoss, phase_losses  # noqa: E402
from model.semambapp import SEMambapp  # noqa: E402
from model.stfts import mag_phase_istft, mag_phase_stft  # noqa: E402
from src.se.models.semambapp.network import SEMambaPP  # noqa: E402
from src.experiments.official.semambapp.discriminator import (  # noqa: E402
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from src.experiments.calc.spectral import compressed_complex_losses, feature_matching, summed_slaney_log_mel  # noqa: E402
from src.experiments.calc.spectral import least_squares_discriminator, least_squares_generator
from src.experiments.calc.stft import semamba_mag_phase_istft, semamba_mag_phase_stft  # noqa: E402

LOSS_ATOL = 1e-5
GRAD_ATOL = 1e-4
COSINE_MIN = 0.999
LR = 2e-4
BETAS = (0.8, 0.99)
WEIGHT_DECAY = 0.01


def _cfg(width: int, layers: int, state: int, conv: int) -> dict:
    return {
        "model_cfg": {
            "hid_feature": width,
            "unet_expansion": 1,
            "compress_factor": 0.3,
            "num_tfmamba": layers,
            "d_state": state,
            "d_conv": conv,
            "expand": 2,
            "norm_epsilon": 1e-5,
            "beta": 2.0,
            "input_channel": 2,
            "output_channel": 1,
            "dropout": 0.1,
        },
        "stft_cfg": {"sampling_rate": 16000, "n_fft": 400, "hop_size": 100, "win_size": 400},
        "training_cfg": {"loss": {"adv_g": 1.0, "fm_g": 1.0, "mel": 0.1, "magnitude": 0.9, "phase": 0.3, "complex": 0.1, "consistancy": 0.1}},
    }


def _scan_backend(device: torch.device) -> str:
    requested = os.environ.get("AUDIO_DB_SEMAMBA_SCAN", "reference")
    if requested not in ("reference", "fused"):
        raise SystemExit(f"AUDIO_DB_SEMAMBA_SCAN must be reference or fused, got {requested}")
    if requested == "fused" and device.type != "cuda":
        raise SystemExit("fused scan requires CUDA")
    return requested


def _flat(cfg: dict, scan_backend: str) -> dict:
    values = dict(cfg["model_cfg"])
    values["n_fft"] = 400
    values["scan_backend"] = scan_backend
    return values


def _gap(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    delta = (reference.detach().float() - other.detach().float()).abs()
    left = reference.detach().float().reshape(-1)
    right = other.detach().float().reshape(-1)
    cosine = 1.0 if left.numel() < 2 else float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "cosine": cosine}


def _pair(device: torch.device, width: int, layers: int, state: int, conv: int):
    cfg = _cfg(width, layers, state, conv)
    official_g = SEMambapp(cfg).to(device)
    independent_g = SEMambaPP(_flat(cfg, _scan_backend(device))).to(device)
    official_cqt = OfficialCQT().to(device)
    independent_cqt = MultiScaleSubbandCQTDiscriminator().to(device)
    official_mrd = OfficialMRD().to(device)
    independent_mrd = MultiResolutionDiscriminator().to(device)
    independent_g.load_state_dict(official_g.state_dict(), strict=True)
    independent_cqt.load_state_dict(official_cqt.state_dict(), strict=True)
    independent_mrd.load_state_dict(official_mrd.state_dict(), strict=True)
    return cfg, official_g, independent_g, official_cqt, independent_cqt, official_mrd, independent_mrd


def _official_parts(generator, clean, noisy, cfg):
    n_fft, hop, win = 400, 100, 400
    clean_mag, clean_phase, clean_complex = mag_phase_stft(clean, n_fft, hop, win, 0.3)
    noisy_mag, noisy_phase, _ = mag_phase_stft(noisy, n_fft, hop, win, 0.3)
    magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
    audio = mag_phase_istft(magnitude, phase, n_fft, hop, win, 0.3)
    audio = audio[..., : clean.shape[-1]]
    _, _, recomputed = mag_phase_stft(audio, n_fft, hop, win, 0.3, addeps=True)
    return {
        "clean_mag": clean_mag,
        "clean_phase": clean_phase,
        "clean_complex": clean_complex,
        "mag": magnitude,
        "phase": phase,
        "complex": complex_spectrum,
        "recomputed": recomputed,
        "audio": audio,
    }


def _independent_parts(generator, clean, noisy):
    clean_mag, clean_phase, clean_complex = semamba_mag_phase_stft(clean, 400, 100, 400, 0.3, addeps=False)
    noisy_mag, noisy_phase, _ = semamba_mag_phase_stft(noisy, 400, 100, 400, 0.3, addeps=False)
    magnitude, phase = generator(noisy_mag, noisy_phase)
    complex_spectrum = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    audio = semamba_mag_phase_istft(magnitude, phase, 400, 100, 400, 0.3, length=clean.shape[-1])
    _, _, recomputed = semamba_mag_phase_stft(audio, 400, 100, 400, 0.3, addeps=True)
    return {
        "clean_mag": clean_mag,
        "clean_phase": clean_phase,
        "clean_complex": clean_complex,
        "mag": magnitude,
        "phase": phase,
        "complex": complex_spectrum,
        "recomputed": recomputed,
        "audio": audio,
    }


def _official_losses(parts, clean, cqt, mrd, mel, cfg):
    real_q, fake_q, fmap_q_r, fmap_q_g = cqt(clean.unsqueeze(1), parts["audio"].unsqueeze(1))
    real_r, fake_r, fmap_r_r, fmap_r_g = mrd(clean.unsqueeze(1), parts["audio"].unsqueeze(1))
    loss_q, _, _ = discriminator_loss(real_q, fake_q)
    loss_r, _, _ = discriminator_loss(real_r, fake_r)
    gen_q, _ = generator_loss(fake_q)
    gen_r, _ = generator_loss(fake_r)
    adversarial = gen_q + gen_r
    matching = feature_loss(fmap_q_r, fmap_q_g) + feature_loss(fmap_r_r, fmap_r_g)
    ip, gd, iaf = phase_losses(parts["clean_phase"], parts["phase"], cfg)
    spectral = {
        "magnitude": torch.nn.functional.mse_loss(parts["clean_mag"], parts["mag"]),
        "phase": ip + gd + iaf,
        "complex": torch.nn.functional.mse_loss(parts["clean_complex"], parts["complex"]) * 2,
        "consistency": torch.nn.functional.mse_loss(parts["complex"], parts["recomputed"]) * 2,
    }
    mel_loss = mel(clean.unsqueeze(1), parts["audio"].unsqueeze(1))
    weights = cfg["training_cfg"]["loss"]
    generator = (
        adversarial * weights["adv_g"]
        + matching * weights["fm_g"]
        + mel_loss * weights["mel"]
        + spectral["magnitude"] * weights["magnitude"]
        + spectral["phase"] * weights["phase"]
        + spectral["complex"] * weights["complex"]
        + spectral["consistency"] * weights["consistancy"]
    )
    return {
        "discriminator": loss_q + loss_r,
        "generator": generator,
        "adversarial": adversarial,
        "matching": matching,
        "mel": mel_loss,
        **spectral,
    }


def _independent_losses(parts, clean, cqt, mrd):
    real_q, real_qf = cqt(clean)
    fake_q, fake_qf = cqt(parts["audio"])
    real_r, real_rf = mrd(clean)
    fake_r, fake_rf = mrd(parts["audio"])
    discriminator = sum(
        least_squares_discriminator(real, fake)
        for real, fake in zip(real_q + real_r, fake_q + fake_r, strict=True)
    )
    adversarial = sum(least_squares_generator(score) for score in fake_q + fake_r)
    matching = feature_matching(real_qf + real_rf, fake_qf + fake_rf, scale=2.0, reduction="sum")
    spectral = compressed_complex_losses(
        parts["clean_mag"],
        parts["mag"],
        parts["clean_phase"],
        parts["phase"],
        parts["clean_complex"],
        parts["complex"],
        parts["recomputed"],
        phase_mode="matrix",
    )
    mel_loss = summed_slaney_log_mel(clean, parts["audio"], 16000)
    generator = (
        adversarial
        + matching
        + mel_loss * 0.1
        + spectral["magnitude"] * 0.9
        + spectral["phase"] * 0.3
        + spectral["complex"] * 0.1
        + spectral["consistency"] * 0.1
    )
    return {
        "discriminator": discriminator,
        "generator": generator,
        "adversarial": adversarial,
        "matching": matching,
        "mel": mel_loss,
        **spectral,
    }


def _grad_report(official, independent) -> dict[str, float | str | int]:
    worst = 0.0
    worst_name = ""
    cosine_min = 1.0
    cosine_name = ""
    cosine_scale = 0.0
    large_cosine_min = 1.0
    compared = 0
    large = 0
    independent_named = dict(independent.named_parameters())
    for name, left in official.named_parameters():
        right = independent_named[name]
        if left.grad is None or right.grad is None:
            continue
        stats = _gap(left.grad, right.grad)
        scale = float(left.grad.detach().abs().max())
        compared += 1
        if stats["cosine"] < cosine_min:
            cosine_min = stats["cosine"]
            cosine_name = name
            cosine_scale = scale
        if scale > 1e-5:
            large += 1
            large_cosine_min = min(large_cosine_min, stats["cosine"])
        if stats["max_abs"] > worst:
            worst = stats["max_abs"]
            worst_name = name
    return {
        "max_abs": worst,
        "worst": worst_name,
        "cosine_min": cosine_min,
        "cosine_name": cosine_name,
        "cosine_grad_max": cosine_scale,
        "large_cosine_min": large_cosine_min,
        "large_compared": large,
        "compared": compared,
    }


def _weight_report(official, independent) -> dict[str, float | str]:
    worst = 0.0
    worst_name = ""
    independent_named = dict(independent.named_parameters())
    for name, left in official.named_parameters():
        gap = float((left - independent_named[name]).abs().max())
        if gap > worst:
            worst = gap
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


def one_step(
    device: torch.device,
    width: int,
    layers: int,
    state: int,
    conv: int,
    steps: int,
    batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict:
    cfg, official_g, independent_g, official_cqt, independent_cqt, official_mrd, independent_mrd = _pair(
        device, width, layers, state, conv
    )
    mel = MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
    torch.manual_seed(23)
    if batches is None:
        clean = torch.randn(1, 8000, device=device)
        noisy = clean * 0.6 + 0.05 * torch.randn_like(clean)
        batches = [(clean, noisy)] * steps
    clean, noisy = batches[0]
    with torch.no_grad():
        probe = torch.randn(1, 8000, device=device)
        mel_gap = abs(float(mel(probe.unsqueeze(1), (probe * 0.5).unsqueeze(1)) - summed_slaney_log_mel(probe, probe * 0.5, 16000)))
    optim_g_o = torch.optim.AdamW(official_g.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    optim_d_o = torch.optim.AdamW(
        list(official_cqt.parameters()) + list(official_mrd.parameters()), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    optim_g_i = torch.optim.AdamW(independent_g.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    optim_d_i = torch.optim.AdamW(
        list(independent_cqt.parameters()) + list(independent_mrd.parameters()), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    history = []
    first = None
    for index in range(steps):
        clean, noisy = batches[index]
        for module in (official_g, independent_g, official_cqt, independent_cqt, official_mrd, independent_mrd):
            module.train()
        official_parts = _official_parts(official_g, clean, noisy, cfg)
        independent_parts = _independent_parts(independent_g, clean, noisy)
        official_loss = _official_losses(official_parts, clean, official_cqt, official_mrd, mel, cfg)
        independent_loss = _independent_losses(independent_parts, clean, independent_cqt, independent_mrd)
        # Discriminator step uses detached audio. Recompute that graph explicitly.
        for optim in (optim_d_o, optim_d_i, optim_g_o, optim_g_i):
            optim.zero_grad(set_to_none=True)
        official_d = _official_losses(
            {**official_parts, "audio": official_parts["audio"].detach()}, clean, official_cqt, official_mrd, mel, cfg
        )["discriminator"]
        independent_d = _independent_losses(
            {**independent_parts, "audio": independent_parts["audio"].detach()}, clean, independent_cqt, independent_mrd
        )["discriminator"]
        official_d.backward()
        independent_d.backward()
        disc_grad = {
            "cqt": _grad_report(official_cqt, independent_cqt),
            "mrd": _grad_report(official_mrd, independent_mrd),
        }
        optim_d_o.step()
        optim_d_i.step()
        for optim in (optim_g_o, optim_g_i):
            optim.zero_grad(set_to_none=True)
        official_parts = _official_parts(official_g, clean, noisy, cfg)
        independent_parts = _independent_parts(independent_g, clean, noisy)
        official_loss = _official_losses(official_parts, clean, official_cqt, official_mrd, mel, cfg)
        independent_loss = _independent_losses(independent_parts, clean, independent_cqt, independent_mrd)
        official_loss["generator"].backward()
        independent_loss["generator"].backward()
        gen_grad = _grad_report(official_g, independent_g)
        optim_g_o.step()
        optim_g_i.step()
        record = {
            "step": index,
            "losses": {
                name: abs(float(official_loss[name].detach()) - float(independent_loss[name].detach()))
                for name in ("generator", "discriminator", "adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")
            },
            "wave": _gap(official_parts["audio"], independent_parts["audio"]),
            "generator_grad": gen_grad,
            "discriminator_grad": disc_grad,
            "generator_weight": _weight_report(official_g, independent_g),
            "cqt_weight": _weight_report(official_cqt, independent_cqt),
            "mrd_weight": _weight_report(official_mrd, independent_mrd),
        }
        # The discriminator scalar above is the generator-step graph. Replace it
        # with the detached discriminator objective that was optimized.
        record["losses"]["discriminator"] = abs(float(official_d.detach()) - float(independent_d.detach()))
        record["official_losses"] = {
            name: float(official_loss[name].detach())
            for name in ("generator", "discriminator", "adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")
        }
        record["official_losses"]["discriminator"] = float(official_d.detach())
        history.append(record)
        if first is None:
            first = record
    return {
        "width": width,
        "layers": layers,
        "scan_backend": _scan_backend(device),
        "mel_gap": mel_gap,
        "first": first,
        "last": history[-1],
        "generator_loss_gap": [item["losses"]["generator"] for item in history],
        "discriminator_loss_gap": [item["losses"]["discriminator"] for item in history],
        "official_generator_loss": [item["official_losses"]["generator"] for item in history],
        "official_discriminator_loss": [item["official_losses"]["discriminator"] for item in history],
    }


def _top_grads(official, independent, limit: int = 12) -> list[dict]:
    ranked = []
    named = dict(independent.named_parameters())
    for name, left in official.named_parameters():
        right = named[name]
        if left.grad is None or right.grad is None:
            continue
        stats = _gap(left.grad, right.grad)
        scale = float(left.grad.detach().abs().max())
        ranked.append(
            {
                "name": name,
                "max_abs": stats["max_abs"],
                "relative": stats["max_abs"] / max(scale, 1e-12),
                "grad_max": scale,
                "cosine": stats["cosine"],
            }
        )
    ranked.sort(key=lambda item: item["max_abs"], reverse=True)
    return ranked[:limit]


def _block_outputs(official_g, independent_g, clean, noisy, cfg) -> list[dict]:
    captured = {"official": [], "independent": []}

    def hook(bucket):
        def record(_module, _inputs, output):
            bucket.append(output.detach())
        return record

    handles = [block.register_forward_hook(hook(captured["official"])) for block in official_g.TSMamba]
    handles += [block.register_forward_hook(hook(captured["independent"])) for block in independent_g.TSMamba]
    try:
        with torch.no_grad():
            _official_parts(official_g, clean, noisy, cfg)
            _independent_parts(independent_g, clean, noisy)
    finally:
        for handle in handles:
            handle.remove()
    return [
        _gap(left, right)
        for left, right in zip(captured["official"], captured["independent"], strict=True)
    ]


def diagnose(device: torch.device) -> dict:
    """Separate spectral, mel, and adversarial gradients at the official width."""
    cfg, official_g, independent_g, official_cqt, independent_cqt, official_mrd, independent_mrd = _pair(
        device, 48, 4, 16, 4
    )
    mel = MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
    torch.manual_seed(23)
    clean = torch.randn(1, 8000, device=device)
    noisy = clean * 0.6 + 0.05 * torch.randn_like(clean)
    for module in (official_g, independent_g, official_cqt, independent_cqt, official_mrd, independent_mrd):
        module.train()
    blocks = _block_outputs(official_g, independent_g, clean, noisy, cfg)
    reports = {}
    objectives = {
        "spectral": lambda loss: (
            loss["magnitude"] * 0.9 + loss["phase"] * 0.3 + loss["complex"] * 0.1 + loss["consistency"] * 0.1
        ),
        "mel": lambda loss: loss["mel"] * 0.1,
        "adversarial": lambda loss: loss["adversarial"] + loss["matching"],
        "full": lambda loss: loss["generator"],
    }
    for name, select in objectives.items():
        for module in (official_g, independent_g):
            module.zero_grad(set_to_none=True)
        official_parts = _official_parts(official_g, clean, noisy, cfg)
        independent_parts = _independent_parts(independent_g, clean, noisy)
        official_loss = _official_losses(official_parts, clean, official_cqt, official_mrd, mel, cfg)
        independent_loss = _independent_losses(independent_parts, clean, independent_cqt, independent_mrd)
        chosen_official = select(official_loss)
        chosen_independent = select(independent_loss)
        loss_gap = abs(float(chosen_official.detach()) - float(chosen_independent.detach()))
        chosen_official.backward()
        chosen_independent.backward()
        reports[name] = {
            "grad": _grad_report(official_g, independent_g),
            "top": _top_grads(official_g, independent_g),
            "loss_gap": loss_gap,
            "wave": _gap(official_parts["audio"], independent_parts["audio"]),
        }
    return {
        "block_outputs": blocks,
        "forward_wave": reports["full"]["wave"],
        "ablation": reports,
        "tolerances": {"grad_atol": GRAD_ATOL, "cosine_min": COSINE_MIN},
    }


def _steps(default: int) -> int:
    raw = os.environ.get("AUDIO_DB_NUMERICS_STEPS")
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 1:
        raise SystemExit("AUDIO_DB_NUMERICS_STEPS must be an integer >= 1")
    return value


def _out_dir() -> Path:
    raw = os.environ.get("AUDIO_DB_NUMERICS_OUT")
    if raw:
        return Path(raw)
    return ROOT / "data" / "results" / "training" / "official-compare-20261101"


def _speech_batches(device: torch.device, steps: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    path = ROOT / "data" / "results" / "training" / "official-compare-20261101" / "semamba_waves.pt"
    saved = torch.load(path, map_location=device, weights_only=True)
    pairs = list(zip(saved["clean"], saved["noisy"], strict=True))
    return [pairs[index % len(pairs)] for index in range(steps)]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "narrow"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if mode == "narrow":
        report = one_step(device, 8, 1, 4, 2, _steps(5))
    elif mode == "official":
        report = one_step(device, 48, 4, 16, 4, _steps(20))
    elif mode == "diagnose":
        report = diagnose(device)
    elif mode == "speech":
        speech_steps = _steps(20)
        report = one_step(device, 48, 4, 16, 4, speech_steps, batches=_speech_batches(device, speech_steps))
        report["audio"] = "libri_demand_smoke_train segments of 8000 samples"
    else:
        raise SystemExit(f"unknown mode {mode}")
    report["device"] = str(device)
    report["tolerances"] = {"loss_atol": LOSS_ATOL, "grad_atol": GRAD_ATOL, "cosine_min": COSINE_MIN}
    out = _out_dir() / f"semamba_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(report if mode == "diagnose" else {"first_gap": report.get("generator_loss_gap", [None])[0], "last_gap": report.get("generator_loss_gap", [None])[-1]}, indent=2))


if __name__ == "__main__":
    main()
