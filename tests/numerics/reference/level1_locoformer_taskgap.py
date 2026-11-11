"""Compare the training wrapper with the official separator frontend.

The shared-frontend run matched exactly. This checks TFLocoformer.forward,
which is what TFLocoformerTask.train_update calls. Rules stay those in
VALIDATION_PLAN.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-level1-20261111/locoformer"))

from src.experiments.calc.optim import clip_gradients  # noqa: E402
from src.experiments.calc.rng import seed_model_init  # noqa: E402
from src.experiments.calc.spectral import multi_resolution_l1  # noqa: E402
from src.experiments.official.tf_locoformer.reference import TFLocoformerTask  # noqa: E402
from tests.numerics.reference.common import gap, grad_report, optim_report, weight_report  # noqa: E402
from tests.numerics.reference.level1_locoformer_ulp import (  # noqa: E402
    DNS,
    _loss_pair,
    _separator_class,
    _tone,
    _wave_from,
)
from tests.numerics.reference.compare_tf_gridnet import _official_loss_class  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    windows = [256, 512, 768, 1024]
    criterion = _official_loss_class()(
        window_sz=windows,
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        normalize_variance=False,
        reduction="sum",
    ).to(device)
    official_cls = _separator_class()
    options = {key: value for key, value in DNS.items() if key not in ("n_fft", "hop_length", "normalize_variance")}
    torch.manual_seed(3)
    official = official_cls(**options).to(device)
    config = {
        "seed": 0,
        "loss": "official",
        "grad_clip": 5.0,
        "optimizer": {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "tf_locoformer",
            "variant": "dns",
            "sampling_rate": 16000,
            "architecture": dict(DNS),
            "stft": {},
            "components": {},
        },
    }
    with seed_model_init(0):
        task = TFLocoformerTask(config, device)
    task.generator.separator.load_state_dict(official.state_dict(), strict=True)
    clean, noisy = _tone(2, 64000, device)
    official.eval()
    task.generator.eval()
    with torch.no_grad():
        via_helper = _wave_from(official, noisy)
        via_port_helper = _wave_from(task.generator.separator, noisy)
        via_task = task.generator(noisy)
    report = {
        "forward_eval": {
            "helper_vs_port_helper": gap(via_helper, via_port_helper),
            "helper_vs_task_forward": gap(via_helper, via_task),
            "port_helper_vs_task_forward": gap(via_port_helper, via_task),
            "loss": _loss_pair(criterion, clean, via_helper, via_task),
        }
    }
    print("forward", report["forward_eval"]["helper_vs_task_forward"], report["forward_eval"]["loss"]["port_gap"], flush=True)
    official.train()
    task.generator.train()
    official_opt = torch.optim.AdamW(official.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    rows = []
    for step in range(3):
        clean, noisy = _tone(step, 64000, device)
        official_opt.zero_grad(set_to_none=True)
        official_wave = _wave_from(official, noisy)
        official_loss = multi_resolution_l1(clean, official_wave, time_domain_weight=0.5, normalize_variance=False)
        official_loss.backward()
        torch.nn.utils.clip_grad_norm_(official.parameters(), 5.0)
        task.train_mode()
        task.optim_g.zero_grad(set_to_none=True)
        task_wave = task.generator(noisy)
        task_loss = multi_resolution_l1(clean, task_wave, time_domain_weight=0.5, normalize_variance=False)
        task_loss.backward()
        clip_gradients(task.generator, task.grad_clip)
        task.optim_g.step()
        official_opt.step()
        rows.append(
            {
                "step": step + 1,
                "task_loss": float(task_loss.detach()),
                "official_port_loss": float(official_loss.detach()),
                "port_loss_gap": abs(float(task_loss.detach()) - float(official_loss.detach())),
                "wave_after": gap(official_wave.detach(), task_wave.detach()),
                "grad": grad_report(official, task.generator.separator, atol=1e-4),
                "weight": weight_report(official, task.generator.separator),
                "optim": optim_report(official_opt, task.optim_g),
                "loss_pair": _loss_pair(criterion, clean, official_wave.detach(), task_wave.detach()),
            }
        )
        print(
            "task",
            rows[-1]["step"],
            rows[-1]["port_loss_gap"],
            rows[-1]["wave_after"]["max_abs"],
            rows[-1]["weight"]["max_abs"],
            flush=True,
        )
    report["steps"] = rows
    path = OUT / "locoformer_taskgap.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
