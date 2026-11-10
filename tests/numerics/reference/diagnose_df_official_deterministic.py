"""Record whether DeepFilterNet3's official CUDA step can run deterministically."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "DF_REFERENCE_CONFIG",
    str(ROOT / "data/results/training/final-h100-20261102-cpu/df-config.ini"),
)

from tests.numerics.four_process.compare import _batch, seed_probe  # noqa: E402
from tests.numerics.four_process.real_models import DeepFilterNet3  # noqa: E402

SOURCE = ROOT / "data/results/training/df-cuda-paths-20261103/native"
OUTPUT = ROOT / "data/results/training/df-cuda-paths-20261103/official-deterministic.json"


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(SOURCE / "input.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((SOURCE / "config.json").read_text())
    adapter = DeepFilterNet3()
    adapter.configure(cfg)
    device = torch.device("cuda:0")
    seed_probe(cfg["seed"], True)
    modules = adapter.build("official", device)
    for name, module in modules.items():
        module.load_state_dict(payload["initial"][name], strict=True)
    clean, noisy, _items = _batch(SOURCE, cfg, 0, None)
    shape_note = {"clean": list(clean.shape), "noisy": list(noisy.shape)}
    if clean.ndim == 1:
        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
    clean = clean.to(device)
    noisy = noisy.to(device)
    optim = adapter.optimizers(modules, None)
    sched = adapter.schedulers(optim)
    torch.use_deterministic_algorithms(True)
    error = None
    try:
        adapter.train_step("official", modules, optim, sched, clean, noisy, 0)
        torch.cuda.synchronize()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    OUTPUT.write_text(json.dumps({"shape": shape_note, "official_deterministic_error": error}, indent=2) + "\n")
    print(json.dumps({"shape": shape_note, "official_deterministic_error": error}, indent=2))


if __name__ == "__main__":
    main()
