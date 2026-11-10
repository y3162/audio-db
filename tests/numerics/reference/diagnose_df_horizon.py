"""Show whether the DeepFilterNet3 20-step gap is one-ulp amplification.

Trains the published and independent networks on the saved probe waveform.
A third published network starts one float32 ulp away on the parameter that
differed most at step 1. Scheduler and optimizer match the four-process probe.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.numerics.four_process.real_models import DeepFilterNet3, _df_loss


def build(kind, initial):
    adapter = DeepFilterNet3()
    adapter.configure({})
    modules = adapter.build(kind, torch.device("cpu"))
    modules["generator"].load_state_dict(initial, strict=True)
    return adapter, modules


def step(adapter, kind, modules, optim, sched, clean, noisy):
    modules["generator"].train()
    sched.apply()
    optim.zero_grad(set_to_none=True)
    _wave, loss = _df_loss(
        kind, modules["generator"], adapter._stream, adapter._params, clean, noisy, adapter._official_loss,
    )
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in modules["generator"].named_parameters() if param.grad is not None}
    torch.nn.utils.clip_grad_norm_(modules["generator"].parameters(), 1.0)
    optim.step()
    sched.step()
    return float(loss.detach()), grads


def main() -> None:
    source = Path(sys.argv[1])
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    prepared = torch.load(source / "input.pt", map_location="cpu", weights_only=False)
    clean, noisy = prepared["clean"], prepared["noisy"]
    torch.set_num_threads(1)
    official_a, modules_o = build("official", prepared["initial"]["generator"])
    independent, modules_i = build("independent", prepared["initial"]["generator"])
    perturbed = official_a
    modules_p = {"generator": copy.deepcopy(modules_o["generator"])}
    target = modules_p["generator"].get_parameter("erb_dec.conv0_out.0.weight")
    with torch.no_grad():
        flat = target.view(-1)
        flat[0] = torch.nextafter(flat[0], flat[0] + 1)
    runs = {
        "official": (official_a, "official", modules_o),
        "independent": (independent, "independent", modules_i),
        "perturbed": (perturbed, "official", modules_p),
    }
    optimizers = {}
    schedulers = {}
    for name, (adapter, _kind, modules) in runs.items():
        optimizers[name] = adapter.optimizers(modules, None)["generator"]
        schedulers[name] = adapter.schedulers({"generator": optimizers[name]})["generator"]
    for index in range(1, steps + 1):
        losses = {}
        grads = {}
        for name, (adapter, kind, modules) in runs.items():
            losses[name], grads[name] = step(adapter, kind, modules, optimizers[name], schedulers[name], clean, noisy)
        best = None
        for key, left in grads["official"].items():
            right = grads["independent"][key]
            delta = float((left - right).abs().max())
            if best is None or delta > best[0]:
                best = (delta, key, float(left.norm()), float(right.norm()))
        perturb = None
        for key, left in grads["official"].items():
            right = grads["perturbed"][key]
            delta = float((left - right).abs().max())
            if perturb is None or delta > perturb[0]:
                perturb = (delta, key, float(left.norm()), float(right.norm()))
        print(
            f"step {index} loss official {losses['official']:.6f} independent {losses['independent']:.6f} "
            f"perturbed {losses['perturbed']:.6f} "
            f"cross {best[0]:.3e} {best[1]} norms {best[2]:.3f}/{best[3]:.3f} "
            f"ulp {perturb[0]:.3e} {perturb[1]} norms {perturb[2]:.3f}/{perturb[3]:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
