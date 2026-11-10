"""SEMamba++ 3-step trace: native official STFT versus the shared deterministic STFT.

Both runs use the published width. The shared run points the official
mag/phase STFT and iSTFT at the audio-db index-reflection pair, so the two
generators see the same transform. Outputs go to AUDIO_DB_NUMERICS_OUT.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-continue"))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from tests.numerics.reference import compare_semamba_step as sem  # noqa: E402

FLOOR = 1e-6


def _masked(official, independent) -> dict:
    named = dict(independent.named_parameters())
    worst = 0.0
    worst_name = ""
    large = 0.0
    large_name = ""
    tiny = 0.0
    tiny_name = ""
    for name, left in official.named_parameters():
        right = named[name]
        delta = (left.detach() - right.detach()).abs()
        value = float(delta.max())
        if value > worst:
            worst = value
            worst_name = name
        if left.grad is None:
            continue
        mask = left.grad.detach().abs() >= FLOOR
        if bool(mask.any()):
            found = float(delta[mask].max())
            if found > large:
                large = found
                large_name = name
        small = ~mask
        if bool(small.any()):
            found = float(delta[small].max())
            if found > tiny:
                tiny = found
                tiny_name = name
    return {
        "max_abs": worst,
        "worst": worst_name,
        "large_grad_max_abs": large,
        "large_grad_worst": large_name,
        "tiny_grad_max_abs": tiny,
        "tiny_grad_worst": tiny_name,
        "tiny_grad_floor": FLOOR,
    }


def _element(official, independent, name: str) -> dict:
    left = dict(official.named_parameters())[name]
    right = dict(independent.named_parameters())[name]
    delta = (left.detach() - right.detach()).abs()
    index = int(delta.view(-1).argmax())
    grad_l = None if left.grad is None else float(left.grad.detach().view(-1)[index])
    grad_r = None if right.grad is None else float(right.grad.detach().view(-1)[index])
    return {
        "name": name,
        "index": index,
        "weight_abs": float(delta.view(-1)[index]),
        "grad_official": grad_l,
        "grad_independent": grad_r,
        "param_official": float(left.detach().view(-1)[index]),
    }


def _hook_blocks(official_g, independent_g):
    captured: dict[str, list] = {"official": [], "independent": []}

    def hook(bucket):
        def record(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(value):
                bucket.append(value.detach())

        return record

    handles = [block.register_forward_hook(hook(captured["official"])) for block in official_g.TSMamba]
    handles += [block.register_forward_hook(hook(captured["independent"])) for block in independent_g.TSMamba]
    return handles, captured


def _block_gaps(captured: dict[str, list]) -> list[float]:
    gaps = []
    for left, right in zip(captured["official"], captured["independent"], strict=True):
        gaps.append(float((left - right).abs().max()))
    return gaps


def _share_stft() -> None:
    def istft(mag, pha, n_fft, hop, win, compress_factor=0.3, center=True):
        del center
        return sem.semamba_mag_phase_istft(mag, pha, n_fft, hop, win, compress_factor)

    sem.mag_phase_stft = sem.semamba_mag_phase_stft
    sem.mag_phase_istft = istft


def _run(device: torch.device, shared: bool) -> dict:
    if shared:
        _share_stft()
    cfg, official_g, independent_g, official_cqt, independent_cqt, official_mrd, independent_mrd = sem._pair(
        device, 48, 4, 16, 4
    )
    mel = sem.MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
    torch.manual_seed(23)
    clean = torch.randn(1, 8000, device=device)
    noisy = clean * 0.6 + 0.05 * torch.randn_like(clean)
    optim_g_o = torch.optim.AdamW(official_g.parameters(), lr=sem.LR, betas=sem.BETAS, weight_decay=sem.WEIGHT_DECAY)
    optim_d_o = torch.optim.AdamW(
        list(official_cqt.parameters()) + list(official_mrd.parameters()),
        lr=sem.LR,
        betas=sem.BETAS,
        weight_decay=sem.WEIGHT_DECAY,
    )
    optim_g_i = torch.optim.AdamW(independent_g.parameters(), lr=sem.LR, betas=sem.BETAS, weight_decay=sem.WEIGHT_DECAY)
    optim_d_i = torch.optim.AdamW(
        list(independent_cqt.parameters()) + list(independent_mrd.parameters()),
        lr=sem.LR,
        betas=sem.BETAS,
        weight_decay=sem.WEIGHT_DECAY,
    )
    rows = []
    for index in range(3):
        for module in (official_g, independent_g, official_cqt, independent_cqt, official_mrd, independent_mrd):
            module.train()
        for optim in (optim_d_o, optim_d_i, optim_g_o, optim_g_i):
            optim.zero_grad(set_to_none=True)
        official_parts = sem._official_parts(official_g, clean, noisy, cfg)
        independent_parts = sem._independent_parts(independent_g, clean, noisy)
        official_loss = sem._official_losses(official_parts, clean, official_cqt, official_mrd, mel, cfg)
        independent_loss = sem._independent_losses(independent_parts, clean, independent_cqt, independent_mrd)
        official_d = sem._official_losses(
            {**official_parts, "audio": official_parts["audio"].detach()}, clean, official_cqt, official_mrd, mel, cfg
        )["discriminator"]
        independent_d = sem._independent_losses(
            {**independent_parts, "audio": independent_parts["audio"].detach()},
            clean,
            independent_cqt,
            independent_mrd,
        )["discriminator"]
        official_d.backward()
        independent_d.backward()
        optim_d_o.step()
        optim_d_i.step()
        for optim in (optim_g_o, optim_g_i):
            optim.zero_grad(set_to_none=True)
        handles, captured = _hook_blocks(official_g, independent_g)
        try:
            official_parts = sem._official_parts(official_g, clean, noisy, cfg)
            independent_parts = sem._independent_parts(independent_g, clean, noisy)
        finally:
            for handle in handles:
                handle.remove()
        blocks = _block_gaps(captured)
        official_loss = sem._official_losses(official_parts, clean, official_cqt, official_mrd, mel, cfg)
        independent_loss = sem._independent_losses(independent_parts, clean, independent_cqt, independent_mrd)
        official_loss["generator"].backward()
        independent_loss["generator"].backward()
        grad = sem._grad_report(official_g, independent_g)
        optim_g_o.step()
        optim_g_i.step()
        weight = _masked(official_g, independent_g)
        row = {
            "step": index,
            "block_output_max_abs": blocks,
            "wave_max_abs": float((official_parts["audio"].detach() - independent_parts["audio"].detach()).abs().max()),
            "generator_loss_gap": abs(float(official_loss["generator"].detach()) - float(independent_loss["generator"].detach())),
            "discriminator_loss_gap": abs(float(official_d.detach()) - float(independent_d.detach())),
            "generator_grad_max_abs": grad["max_abs"],
            "generator_grad_worst": grad["worst"],
            "large_cosine_min": grad["large_cosine_min"],
            "weight": weight,
            "weight_element": _element(official_g, independent_g, str(weight["worst"])),
        }
        rows.append(row)
        print("semamba", "shared" if shared else "native", row["step"], row["generator_loss_gap"], row["generator_grad_max_abs"], row["weight"]["max_abs"], flush=True)
    return {"shared_deterministic_stft": shared, "steps": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    native = _run(device, False)
    torch.cuda.empty_cache()
    shared = _run(device, True)
    report = {
        "device": str(device),
        "scan_backend": os.environ.get("AUDIO_DB_SEMAMBA_SCAN", "reference"),
        "official_commit": "cd369c89a8106072c077b468417b765b227afd74",
        "width": {"hid_feature": 48, "num_tfmamba": 4, "d_state": 16, "d_conv": 4},
        "samples": 8000,
        "batch_size": 1,
        "native": native,
        "shared_deterministic_stft": shared,
    }
    path = OUT / "semamba_trace.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
