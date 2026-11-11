"""Audio-db-versus-audio-db TF-Locoformer steps, with float64 loss gaps.

Same inputs, width, and AdamW as level1_locoformer_oo.py. Thresholds are unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path("/tmp/audio-db-level1-20261111/locoformer")

from src.experiments.calc.spectral import multi_resolution_l1  # noqa: E402
from src.se.models.tf_locoformer.separator import TFLocoformerSeparator  # noqa: E402
from tests.numerics.reference.common import gap, grad_report, optim_report, weight_report  # noqa: E402
from tests.numerics.reference.level1_locoformer_ulp import DNS, _terms, _tone, _wave_from  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    options = {key: value for key, value in DNS.items() if key not in ("n_fft", "hop_length", "normalize_variance")}
    torch.manual_seed(3)
    left = TFLocoformerSeparator(**options).to(device)
    right = TFLocoformerSeparator(**options).to(device)
    right.load_state_dict(left.state_dict(), strict=True)
    opt_l = torch.optim.AdamW(left.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    opt_r = torch.optim.AdamW(right.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    rows = []
    for step in range(3):
        clean, noisy = _tone(step, 64000, device)
        left.train()
        right.train()
        opt_l.zero_grad(set_to_none=True)
        opt_r.zero_grad(set_to_none=True)
        wave_l = _wave_from(left, noisy)
        wave_r = _wave_from(right, noisy)
        loss_l = multi_resolution_l1(clean, wave_l, time_domain_weight=0.5, normalize_variance=False)
        loss_r = multi_resolution_l1(clean, wave_r, time_domain_weight=0.5, normalize_variance=False)
        loss_l.backward()
        loss_r.backward()
        torch.nn.utils.clip_grad_norm_(left.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(right.parameters(), 5.0)
        opt_l.step()
        opt_r.step()
        terms_l = _terms(clean, wave_l.detach(), torch.float64)
        terms_r = _terms(clean, wave_r.detach(), torch.float64)
        rows.append(
            {
                "step": step + 1,
                "port_loss_gap": abs(float(loss_l.detach()) - float(loss_r.detach())),
                "wave": gap(wave_l.detach(), wave_r.detach()),
                "grad_max_abs": grad_report(left, right, atol=1e-4)["max_abs"],
                "weight": weight_report(left, right),
                "optim_max_abs": optim_report(opt_l, opt_r)["max_abs"],
                "float64_gap": {key: abs(terms_l[key] - terms_r[key]) for key in terms_l},
            }
        )
        print(
            rows[-1]["step"],
            rows[-1]["port_loss_gap"],
            rows[-1]["wave"]["max_abs"],
            rows[-1]["float64_gap"]["total"],
            flush=True,
        )
    path = OUT / "locoformer_aa_float64.json"
    path.write_text(json.dumps({"comparison": "audiodb_versus_audiodb", "rows": rows}, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
