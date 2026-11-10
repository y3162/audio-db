"""One official-versus-official SEMamba++ step at the published width.

This is the self-repeat floor for the fused generator. It does not compare
audio-db. A sign flip on a sub-1e-6 gradient here means the later cross-run
weight gap is inside the official CUDA backward noise.
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
from model.semambapp import SEMambapp  # noqa: E402


def _clone_step(device: torch.device) -> dict:
    cfg = sem._cfg(48, 4, 16, 4)
    torch.manual_seed(23)
    left = SEMambapp(cfg).to(device)
    right = SEMambapp(cfg).to(device)
    right.load_state_dict(left.state_dict(), strict=True)
    cqt_l = sem.OfficialCQT().to(device)
    cqt_r = sem.OfficialCQT().to(device)
    cqt_r.load_state_dict(cqt_l.state_dict(), strict=True)
    mrd_l = sem.OfficialMRD().to(device)
    mrd_r = sem.OfficialMRD().to(device)
    mrd_r.load_state_dict(mrd_l.state_dict(), strict=True)
    mel = sem.MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
    clean = torch.randn(1, 8000, device=device)
    noisy = clean * 0.6 + 0.05 * torch.randn_like(clean)
    optim_l = torch.optim.AdamW(left.parameters(), lr=sem.LR, betas=sem.BETAS, weight_decay=sem.WEIGHT_DECAY)
    optim_r = torch.optim.AdamW(right.parameters(), lr=sem.LR, betas=sem.BETAS, weight_decay=sem.WEIGHT_DECAY)
    for module in (left, right, cqt_l, cqt_r, mrd_l, mrd_r):
        module.train()
    parts_l = sem._official_parts(left, clean, noisy, cfg)
    parts_r = sem._official_parts(right, clean, noisy, cfg)
    loss_l = sem._official_losses(parts_l, clean, cqt_l, mrd_l, mel, cfg)["generator"]
    loss_r = sem._official_losses(parts_r, clean, cqt_r, mrd_r, mel, cfg)["generator"]
    loss_l.backward()
    loss_r.backward()
    grad = sem._grad_report(left, right)
    name = "mask_decoder.mask_conv.1.bias"
    bias_l = dict(left.named_parameters())[name]
    bias_r = dict(right.named_parameters())[name]
    index = int((bias_l.grad - bias_r.grad).abs().view(-1).argmax())
    optim_l.step()
    optim_r.step()
    weight = float((bias_l.detach() - bias_r.detach()).abs().max())
    return {
        "wave_max_abs": float((parts_l["audio"].detach() - parts_r["audio"].detach()).abs().max()),
        "loss_gap": abs(float(loss_l.detach()) - float(loss_r.detach())),
        "grad_max_abs": grad["max_abs"],
        "grad_worst": grad["worst"],
        "bias_name": name,
        "bias_grad_left": float(bias_l.grad.detach().view(-1)[index]),
        "bias_grad_right": float(bias_r.grad.detach().view(-1)[index]),
        "bias_weight_max_abs_after_step": weight,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report = {"device": str(device), "scan_note": "official SEMambapp only", "step": _clone_step(device)}
    path = OUT / "semamba_self.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(report["step"], indent=2))


if __name__ == "__main__":
    main()
