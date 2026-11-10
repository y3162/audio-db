"""200-update SEMamba++ run, official or independent, one process.

The published config uses batch 8 and a 48000-sample segment. This comparison
uses batch 1 and 8000 samples so the fused kernel fits in a 16 GB Quadro and
the generator graph is not held eight times. The loss, AdamW, ExponentialLR,
and discriminator-then-generator order follow ``train.py``. Time L1 and the
metric weight are not in that executed sum.

    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        .venvs/semamba/bin/python tests/numerics/reference/train_semamba_200.py \
        official data/results/training/semamba-200step-20261101/official-a 200
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.numerics.reference.compare_semamba_quality import _require_environment
from tests.numerics.reference.compare_semamba_step import (
    _cfg,
    _official_losses,
    _official_parts,
)
from model.discriminator import MultiResolutionDiscriminator as OfficialMRD
from model.discriminator import MultiScaleSubbandCQTDiscriminator as OfficialCQT
from model.loss import MultiScaleMelSpectrogramLoss
from model.semambapp import SEMambapp
from src.evaluation.metrics.sisdr import score_si_sdr
from src.experiments.calc.rng import capture_rng, seed_training_stream
from src.experiments.calc.dataset import load_recipe_dataset
from src.experiments.calc.sampler import epoch_indices
from src.experiments.calc.semambapp import SEMambaTask
from tests.numerics.supply import tensor_sha256

DB = ROOT / "data/db/semamba-voicebank-train-20261101-compat.duckdb"
INIT = ROOT / "data/results/training/semamba-repro-20261101/run2/checkpoints/step_0000_official.pt"
DETAIL = {0, 1, 2, 5, 10, 20, 50, 100, 200}
GRAD_STEPS = {1, 2}
CHECKPOINT_STEPS = {0, 1, 10, 20, 50, 100, 200}
VAL_ITEMS = 8
LR = 2e-4
BETAS = (0.8, 0.99)
WEIGHT_DECAY = 0.01


def main() -> None:
    side, out, steps = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    if side not in ("official", "independent") or steps < 1 or out.exists():
        raise SystemExit("usage: train_semamba_200.py official|independent NEW_DIR STEPS")
    _require_environment()
    out.mkdir(parents=True)
    device = torch.device("cuda")
    initial = torch.load(INIT, map_location="cpu", weights_only=False)
    train = load_recipe_dataset(DB, "voicebank_28_train", role="train", seed=1234, sample_rate=16000, segment_size=8000, training=True)
    valid = load_recipe_dataset(DB, "voicebank_28_valid", role="validation", seed=1234, sample_rate=16000, segment_size=None, training=False)
    if (len(train), len(valid)) != (10700, 872):
        raise SystemExit(f"unexpected split sizes {len(train)} {len(valid)}")
    bundle = _official(device, initial) if side == "official" else _independent(device, initial)
    seed_training_stream(1234, cuda=True)
    val_ids = [valid.examples[index].case_id for index in range(VAL_ITEMS)]
    (out / "config.json").write_text(
        json.dumps(
            {
                "side": side,
                "steps": steps,
                "seed": 1234,
                "segment_size": 8000,
                "global_batch_size": 1,
                "micro_batch_size": 1,
                "accumulation_steps": 1,
                "published_batch_size": 8,
                "published_segment_size": 48000,
                "optimizer": {"name": "adamw", "lr": LR, "betas": list(BETAS), "weight_decay": WEIGHT_DECAY},
                "scheduler": {"name": "exponential", "gamma": 0.99, "step": "once per finished epoch"},
                "grad_clip": None,
                "train_set": "voicebank_28_train",
                "validation_set": "voicebank_28_valid",
                "validation_items": val_ids,
                "init": str(INIT),
                "init_updates": initial.get("updates_completed"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_state(out, 0, bundle)
    _validate(out, 0, side, bundle, valid, device)
    order = None
    epoch = -1
    for step in range(1, steps + 1):
        new_epoch, cursor = divmod(step - 1, len(train))
        if new_epoch != epoch:
            epoch = new_epoch
            train.set_epoch(epoch)
            order = epoch_indices(len(train), 1234, epoch, 1, 0)
        batch = train.collate([order[cursor]])
        record = {
            "step": step,
            "epoch": epoch,
            "case_id": batch["case_id"][0],
            "clean_sha256": tensor_sha256(batch["clean"]),
            "noisy_sha256": tensor_sha256(batch["noisy"]),
        }
        metrics, grads = _update(side, bundle, batch, device, capture_grads=step in GRAD_STEPS)
        record.update(metrics)
        with (out / "loss.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step in CHECKPOINT_STEPS or step == steps:
            _save_state(out, step, bundle)
        if step in GRAD_STEPS and grads is not None:
            torch.save(grads, out / f"grads_{step:08d}.pt")
        if step in DETAIL:
            _validate(out, step, side, bundle, valid, device)
        if step in DETAIL or step == steps:
            print(side, step, record.get("generator"), record.get("discriminator"), flush=True)


def _official(device, initial):
    cfg = _cfg(48, 4, 16, 4)
    generator, cqt, mrd = SEMambapp(cfg), OfficialCQT(), OfficialMRD()
    generator.load_state_dict(initial["generator"], strict=True)
    cqt.load_state_dict(initial["cqt"], strict=True)
    mrd.load_state_dict(initial["mrd"], strict=True)
    generator.to(device).train()
    cqt.to(device).train()
    mrd.to(device).train()
    optim_g = torch.optim.AdamW(generator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    optim_d = torch.optim.AdamW(list(cqt.parameters()) + list(mrd.parameters()), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    return {
        "kind": "official",
        "cfg": cfg,
        "generator": generator,
        "cqt": cqt,
        "mrd": mrd,
        "optim_g": optim_g,
        "optim_d": optim_d,
        "sched_g": torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.99),
        "sched_d": torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.99),
        "mel": MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device),
    }


def _independent(device, initial):
    config = {
        "seed": 1234,
        "loss": "official",
        "grad_clip": None,
        "optimizer": {"name": "adamw", "lr": LR, "betas": [0.8, 0.99], "weight_decay": WEIGHT_DECAY},
        "scheduler": {"name": "exponential", "gamma": 0.99},
        "model": {
            "name": "semambapp",
            "sampling_rate": 16000,
            "variant": None,
            "architecture": {"hid_feature": 48, "num_tfmamba": 4, "d_state": 16, "d_conv": 4, "expand": 2, "n_fft": 400},
            "stft": {"hop": 100, "win": 400},
        },
    }
    task = SEMambaTask(config, device)
    task.generator.load_state_dict(initial["generator"], strict=True)
    task.discriminator_band.load_state_dict(initial["cqt"], strict=True)
    task.discriminator_mrd.load_state_dict(initial["mrd"], strict=True)
    return {"kind": "independent", "task": task}


def _update(side, bundle, batch, device, *, capture_grads: bool):
    if side == "independent":
        task = bundle["task"]
        task.capture_update = True
        task.capture_gradients = capture_grads
        metrics = task.train_update([batch], 1)
        finite = all(value == value and value not in (float("inf"), float("-inf")) for value in metrics.values())
        metrics = {**metrics, "finite": finite, "lr_g": task.learning_rates()["generator"], "lr_d": task.learning_rates()["discriminator"]}
        return metrics, task.last_capture if capture_grads else None
    clean = batch["clean"].to(device)
    noisy = batch["noisy"].to(device)
    modules = (bundle["generator"], bundle["cqt"], bundle["mrd"])
    for module in modules:
        module.train()
    bundle["optim_d"].zero_grad(set_to_none=True)
    parts = _official_parts(bundle["generator"], clean, noisy, bundle["cfg"])
    detached = {**parts, "audio": parts["audio"].detach()}
    disc = _official_losses(detached, clean, bundle["cqt"], bundle["mrd"], bundle["mel"], bundle["cfg"])
    disc["discriminator"].backward()
    disc_norm = _norm(bundle["cqt"], bundle["mrd"])
    disc_grads = _clone(bundle["cqt"], bundle["mrd"]) if capture_grads else None
    bundle["optim_d"].step()
    bundle["optim_g"].zero_grad(set_to_none=True)
    generated = _official_losses(parts, clean, bundle["cqt"], bundle["mrd"], bundle["mel"], bundle["cfg"])
    generated["generator"].backward()
    gen_norm = _norm(bundle["generator"])
    gen_grads = _clone(bundle["generator"]) if capture_grads else None
    bundle["optim_g"].step()
    metrics = {
        name: float(generated[name].detach())
        for name in ("generator", "adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")
    }
    metrics["discriminator"] = float(disc["discriminator"].detach())
    metrics["generator_grad_norm"] = gen_norm
    metrics["discriminator_grad_norm"] = disc_norm
    metrics["finite"] = all(torch.isfinite(generated[name]).all() for name in ("generator", "adversarial", "mel"))
    metrics["lr_g"] = float(bundle["optim_g"].param_groups[0]["lr"])
    metrics["lr_d"] = float(bundle["optim_d"].param_groups[0]["lr"])
    grads = {"generator_grad": gen_grads, "discriminator_grad": disc_grads, "audio": parts["audio"].detach().cpu()} if capture_grads else None
    return metrics, grads


def _validate(out, step, side, bundle, valid, device):
    rows = []
    waves = []
    for index in range(VAL_ITEMS):
        batch = valid.collate([index])
        clean = batch["clean"].to(device)
        noisy = batch["noisy"].to(device)
        with torch.no_grad():
            if side == "official":
                bundle["generator"].eval()
                audio = _official_parts(bundle["generator"], clean, noisy, bundle["cfg"])["audio"]
                bundle["generator"].train()
            else:
                bundle["task"].eval_mode()
                audio = bundle["task"]._generate(clean, noisy, bundle["task"].generator)["audio"]
                bundle["task"].train_mode()
        enhanced = audio[0].detach().cpu()
        teacher = batch["clean"][0].cpu()
        observed = batch["noisy"][0].cpu()
        length = min(enhanced.numel(), teacher.numel(), observed.numel())
        rows.append(
            {
                "case_id": batch["case_id"][0],
                "length": int(teacher.numel()),
                "common": length,
                "enhanced": score_si_sdr(teacher, enhanced),
                "noisy": score_si_sdr(teacher, observed),
                "cosine": _cosine(teacher[:length], enhanced[:length]),
            }
        )
        waves.append({"case_id": batch["case_id"][0], "clean": teacher, "noisy": observed, "enhanced": enhanced.cpu()})
    torch.save({"step": step, "rows": waves}, out / f"val_{step:08d}.pt")
    (out / f"val_{step:08d}.json").write_text(json.dumps(rows), encoding="utf-8")


def _save_state(out, step, bundle):
    if bundle["kind"] == "official":
        payload = {
            "generator": bundle["generator"].state_dict(),
            "cqt": bundle["cqt"].state_dict(),
            "mrd": bundle["mrd"].state_dict(),
            "optim_g": bundle["optim_g"].state_dict(),
            "optim_d": bundle["optim_d"].state_dict(),
            "sched_g": bundle["sched_g"].state_dict(),
            "sched_d": bundle["sched_d"].state_dict(),
        }
    else:
        payload = bundle["task"].state_dict()
    payload["rng"] = capture_rng(cuda=True)
    payload["step"] = step
    torch.save(payload, out / f"state_{step:08d}.pt")


def _norm(*modules) -> float:
    total = 0.0
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().square().sum())
    return total**0.5


def _clone(*modules):
    cloned = {}
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.grad is not None:
                cloned[f"{module.__class__.__name__}.{name}"] = parameter.grad.detach().float().cpu()
    return cloned


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denom = left.norm() * right.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denom)


if __name__ == "__main__":
    main()
