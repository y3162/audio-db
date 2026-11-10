"""SEMamba++ at the published 3-second segment and global batch 8.

The published loop is one batch of 8 waveforms, each 48000 samples, with no
gradient accumulation. This driver keeps that loop when it fits. When a native
batch does not fit, later stages use micro-batch 1 and scale each mean loss by
the accumulation count. Each stage writes a new file and leaves existing
outputs alone.

    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/semamba/bin/python tests/numerics/reference/run_semamba_official_batch.py \\
        prepare data/results/training/semamba-phaseA-20261101
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.numerics.reference.compare_semamba_quality import _require_environment
from tests.numerics.reference.compare_semamba_step import _official_losses, _official_parts
from tests.numerics.reference.train_semamba_200 import (
    DB,
    INIT,
    _clone,
    _independent,
    _norm,
    _official,
)
from src.experiments.calc.rng import capture_rng, seed_training_stream
from src.experiments.calc.distributed import unwrap
from src.experiments.calc.dataset import load_recipe_dataset
from src.experiments.calc.sampler import epoch_indices
from tests.numerics.supply import tensor_sha256

SEGMENT = 48000
SEED = 1234
COMPONENTS = ("generator", "adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")


def main() -> None:
    stage = sys.argv[1]
    out = Path(sys.argv[2])
    if stage == "prepare":
        _prepare(out)
        return
    _require_environment()
    if stage == "probe":
        suffix = sys.argv[5] if len(sys.argv) > 5 else ""
        _probe(out, sys.argv[3], int(sys.argv[4]), suffix)
        return
    if stage == "step":
        _step(out, sys.argv[3])
        return
    if stage == "equiv":
        _equiv(out, sys.argv[3], int(sys.argv[4]))
        return
    if stage == "repeat":
        _repeat(out, sys.argv[3], int(sys.argv[4]))
        return
    if stage == "short":
        _short(out, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else "")
        return
    if stage == "resume":
        _resume(out, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else "")
        return
    raise SystemExit("stage must be prepare, probe, step, equiv, short, or resume")


def _prepare(out: Path) -> None:
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    train = load_recipe_dataset(
        DB, "voicebank_28_train", role="train", seed=SEED, sample_rate=16000, segment_size=SEGMENT, training=True
    )
    train.set_epoch(0)
    order = epoch_indices(len(train), SEED, 0, 1, 0)
    picked = order[:64]
    batch = train.collate(picked)
    if tuple(batch["clean"].shape) != (64, SEGMENT) or tuple(batch["noisy"].shape) != (64, SEGMENT):
        raise SystemExit(f"unexpected crop shape {tuple(batch['clean'].shape)}")
    if not torch.isfinite(batch["clean"]).all() or not torch.isfinite(batch["noisy"]).all():
        raise SystemExit("non-finite waveform")
    rows = []
    for index, case_id in enumerate(batch["case_id"]):
        clean = batch["clean"][index]
        noisy = batch["noisy"][index]
        rows.append(
            {
                "position": index,
                "dataset_index": int(picked[index]),
                "case_id": case_id,
                "samples": int(clean.numel()),
                "clean_sha256": tensor_sha256(clean),
                "noisy_sha256": tensor_sha256(noisy),
                "cosine": _cosine(clean, noisy),
                "clean_energy": float(clean.square().sum()),
                "noisy_energy": float(noisy.square().sum()),
            }
        )
    torch.save({"clean": batch["clean"], "noisy": batch["noisy"], "case_id": list(batch["case_id"])}, out / "waves.pt")
    (out / "prepare.json").write_text(
        json.dumps(
            {
                "segment_size": SEGMENT,
                "sample_rate": 16000,
                "seed": SEED,
                "epoch": 0,
                "published_batch_size": 8,
                "published_segment_size": 48000,
                "normalization": "training dataset energy normalization, shared by clean and noisy",
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("prepared", len(rows), "crops", flush=True)


def _probe(out: Path, side: str, batch_size: int, suffix: str = "") -> None:
    destination = out / f"probe-{side}-b{batch_size}{suffix}.json"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    waves = _waves(out)
    started = time.time()
    record = {"side": side, "batch_size": batch_size, "segment_size": SEGMENT}
    try:
        bundle = _load(side)
        micro = [_slice(waves, index) for index in range(batch_size)]
        torch.cuda.reset_peak_memory_stats()
        metrics, _ = _update(side, bundle, [_one(micro)], 1, capture=False)
        torch.cuda.synchronize()
        record.update(
            {
                "ok": True,
                "native": True,
                "seconds": time.time() - started,
                "peak_allocated": int(torch.cuda.max_memory_allocated()),
                "metrics": metrics,
            }
        )
    except RuntimeError as exc:
        record.update({"ok": False, "seconds": time.time() - started, "error": str(exc).splitlines()[-1][:500]})
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("side", "batch_size", "ok")}), flush=True)


def _step(out: Path, side: str) -> None:
    destination = out / f"step-{side}.pt"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    waves = _waves(out)
    bundle = _load(side)
    seed_training_stream(SEED, cuda=True)
    before = _generator_state(side, bundle)
    metrics, capture = _update(side, bundle, [_slice(waves, 0)], 1, capture=True)
    torch.cuda.synchronize()
    payload = {
        "side": side,
        "metrics": metrics,
        "clean": waves["clean"][:1].cpu(),
        "noisy": waves["noisy"][:1].cpu(),
        "audio": capture["audio"][0].cpu() if side == "independent" else capture["audio"].cpu(),
        "generator_before": before,
        "generator_after": _generator_state(side, bundle),
        "generator_grad": capture["generator_grad"],
        "discriminator_grad": capture["discriminator_grad"],
        "state": _full_state(side, bundle),
    }
    torch.save(payload, destination)
    summary = {"side": side, "generator": metrics["generator"], "discriminator": metrics["discriminator"], "finite": metrics["finite"]}
    (out / f"step-{side}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def _equiv(out: Path, side: str, batch_size: int) -> None:
    destination = out / f"equiv-{side}-b{batch_size}.pt"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    waves = _waves(out)
    native = _load(side)
    native_metrics, native_capture = _update(side, native, [_stack(waves, batch_size)], 1, capture=True)
    native_weight = _generator_state(side, native)
    native_audio = _audio_tensor(side, native_capture)
    native_gen = native_capture["generator_grad"]
    native_disc = native_capture["discriminator_grad"]
    del native, native_capture
    torch.cuda.empty_cache()
    split = _load(side)
    split_metrics, split_capture = _update(
        side,
        split,
        [_slice(waves, index) for index in range(batch_size)],
        batch_size,
        capture=True,
    )
    torch.cuda.synchronize()
    report = {
        "side": side,
        "batch_size": batch_size,
        "native_metrics": native_metrics,
        "split_metrics": split_metrics,
        "generator_grad": _grad_gap(native_gen, split_capture["generator_grad"]),
        "discriminator_grad": _grad_gap(native_disc, split_capture["discriminator_grad"]),
        "generator_weight": _tensor_gap(native_weight, _generator_state(side, split)),
        "audio": _tensor_gap(native_audio, _audio_tensor(side, split_capture)),
    }
    torch.save({"report": report}, destination)
    (out / f"equiv-{side}-b{batch_size}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"side": side, "batch": batch_size, "generator_grad": report["generator_grad"]}), flush=True)


def _repeat(out: Path, side: str, batch_size: int) -> None:
    destination = out / f"repeat-{side}-b{batch_size}.json"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    waves = _waves(out)
    first = _load(side)
    _, first_capture = _update(side, first, [_stack(waves, batch_size)], 1, capture=True)
    first_grad = first_capture["generator_grad"]
    del first, first_capture
    torch.cuda.empty_cache()
    second = _load(side)
    _, second_capture = _update(side, second, [_stack(waves, batch_size)], 1, capture=True)
    report = {"side": side, "batch_size": batch_size, "generator_grad": _grad_gap(first_grad, second_capture["generator_grad"])}
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def _short(out: Path, side: str, steps: int, accumulation: int, suffix: str) -> None:
    destination = out / f"short-{side}{suffix}"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    destination.mkdir()
    waves = _waves(out)
    count = int(waves["clean"].shape[0])
    (destination / "config.json").write_text(
        json.dumps({"side": side, "steps": steps, "accumulation": accumulation, "crops": count, "segment_size": SEGMENT}, indent=2),
        encoding="utf-8",
    )
    bundle = _load(side)
    seed_training_stream(SEED, cuda=True)
    _save_train_state(destination, 0, side, bundle)
    for step in range(1, steps + 1):
        start = (step - 1) * accumulation
        if start + accumulation > count:
            raise SystemExit("not enough prepared crops for this step")
        batches = [_slice(waves, start + offset) for offset in range(accumulation)]
        metrics, _ = _update(side, bundle, batches, accumulation, capture=False)
        record = {"step": step, "case_id": [waves["case_id"][start + offset] for offset in range(accumulation)]}
        record.update(metrics)
        with (destination / "loss.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step in {1, 4, 5, steps}:
            _save_train_state(destination, step, side, bundle)
        print(side, step, metrics["generator"], metrics["discriminator"], flush=True)


def _resume(out: Path, side: str, source_step: int, target_step: int, suffix: str) -> None:
    source = out / f"short-{side}{suffix}" / f"state_{source_step:08d}.pt"
    destination = out / f"resume-{side}{suffix}-{source_step:04d}-{target_step:04d}.pt"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    config = json.loads((out / f"short-{side}{suffix}" / "config.json").read_text(encoding="utf-8"))
    accumulation = int(config["accumulation"])
    saved = torch.load(source, map_location="cpu", weights_only=False)
    bundle = _load(side, saved)
    _restore_rng(saved["rng"])
    waves = _waves(out)
    restored = _generator_state(side, bundle)
    gap = _tensor_gap(restored, saved["generator"])
    for step in range(source_step + 1, target_step + 1):
        start = (step - 1) * accumulation
        batches = [_slice(waves, start + offset) for offset in range(accumulation)]
        metrics, _ = _update(side, bundle, batches, accumulation, capture=False)
        print(side, "resume", step, metrics["generator"], flush=True)
    continuous = torch.load(out / f"short-{side}{suffix}" / f"state_{target_step:08d}.pt", map_location="cpu", weights_only=False)
    report = {
        "restore_generator": gap,
        "continued_generator": _tensor_gap(_generator_state(side, bundle), continuous["generator"]),
    }
    torch.save({"report": report, "generator": _generator_state(side, bundle)}, destination)
    (destination.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def _update(side: str, bundle: dict, batches: list[dict], accumulation: int, *, capture: bool):
    if side == "independent":
        task = bundle["task"]
        task.capture_update = True
        task.capture_gradients = capture
        metrics = task.train_update(batches, accumulation)
        finite = all(value == value and value not in (float("inf"), float("-inf")) for value in metrics.values())
        return {**metrics, "finite": finite}, task.last_capture
    device = torch.device("cuda")
    for module in (bundle["generator"], bundle["cqt"], bundle["mrd"]):
        module.train()
    bundle["optim_d"].zero_grad(set_to_none=True)
    cached = []
    disc_total = 0.0
    for batch in batches:
        clean = batch["clean"].to(device)
        noisy = batch["noisy"].to(device)
        if accumulation == 1:
            parts = _official_parts(bundle["generator"], clean, noisy, bundle["cfg"])
            cached.append((clean, noisy, parts))
            audio = parts["audio"]
        else:
            with torch.no_grad():
                preview = _official_parts(bundle["generator"], clean, noisy, bundle["cfg"])
            cached.append((clean, noisy, None))
            parts = preview
        # Preserve the generator graph for its later update; the discriminator
        # update must not backpropagate through the generated waveform.
        disc_parts = {**parts, "audio": parts["audio"].detach()}
        disc = _official_losses(disc_parts, clean, bundle["cqt"], bundle["mrd"], bundle["mel"], bundle["cfg"])
        (disc["discriminator"] / accumulation).backward()
        disc_total += float(disc["discriminator"].detach())
    disc_grads = _clone(unwrap(bundle["cqt"]), unwrap(bundle["mrd"])) if capture else None
    bundle["optim_d"].step()
    frozen = [parameter for module in (bundle["cqt"], bundle["mrd"]) for parameter in module.parameters()]
    for parameter in frozen:
        parameter.requires_grad_(False)
    bundle["optim_g"].zero_grad(set_to_none=True)
    totals = {name: 0.0 for name in COMPONENTS}
    audios = []
    try:
        for clean, noisy, parts in cached:
            if parts is None:
                parts = _official_parts(bundle["generator"], clean, noisy, bundle["cfg"])
            generated = _official_losses(parts, clean, unwrap(bundle["cqt"]), unwrap(bundle["mrd"]), bundle["mel"], bundle["cfg"])
            (generated["generator"] / accumulation).backward()
            for name in COMPONENTS:
                totals[name] += float(generated[name].detach())
            audios.append(parts["audio"].detach().cpu())
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)
    gen_grads = _clone(unwrap(bundle["generator"])) if capture else None
    gen_norm = _norm(bundle["generator"])
    bundle["optim_g"].step()
    count = len(batches)
    metrics = {name: totals[name] / count for name in COMPONENTS}
    metrics["discriminator"] = disc_total / count
    metrics["generator_grad_norm"] = gen_norm
    metrics["finite"] = all(value == value and value not in (float("inf"), float("-inf")) for value in metrics.values())
    return metrics, {"audio": torch.cat(audios, dim=0), "generator_grad": gen_grads, "discriminator_grad": disc_grads}


def _load(side: str, state: dict | None = None):
    device = torch.device("cuda")
    initial = torch.load(INIT, map_location="cpu", weights_only=False) if state is None else state
    if side == "official":
        bundle = _official(device, {"generator": initial["generator"], "cqt": initial["cqt"], "mrd": initial["mrd"]})
        if state is not None:
            bundle["optim_g"].load_state_dict(state["optim_g"])
            bundle["optim_d"].load_state_dict(state["optim_d"])
            bundle["sched_g"].load_state_dict(state["sched_g"])
            bundle["sched_d"].load_state_dict(state["sched_d"])
        return bundle
    bundle = _independent(device, {"generator": initial["generator"], "cqt": initial.get("cqt", initial.get("discriminator_band")), "mrd": initial.get("mrd", initial.get("discriminator_mrd"))})
    if state is not None:
        bundle["task"].optim_g.load_state_dict(state["optim_g"])
        bundle["task"].optim_d.load_state_dict(state["optim_d"])
        if state.get("sched_g") is not None:
            bundle["task"].sched_g.load_state_dict(state["sched_g"])
            bundle["task"].sched_d.load_state_dict(state["sched_d"])
    return bundle


def _full_state(side: str, bundle: dict) -> dict:
    if side == "official":
        return {
            "generator": bundle["generator"].state_dict(),
            "cqt": bundle["cqt"].state_dict(),
            "mrd": bundle["mrd"].state_dict(),
            "optim_g": bundle["optim_g"].state_dict(),
            "optim_d": bundle["optim_d"].state_dict(),
            "sched_g": bundle["sched_g"].state_dict(),
            "sched_d": bundle["sched_d"].state_dict(),
            "rng": capture_rng(cuda=True),
        }
    state = bundle["task"].state_dict()
    state["rng"] = capture_rng(cuda=True)
    return state


def _save_train_state(out: Path, step: int, side: str, bundle: dict) -> None:
    payload = _full_state(side, bundle)
    payload["step"] = step
    torch.save(payload, out / f"state_{step:08d}.pt")


def _generator_state(side: str, bundle: dict) -> dict[str, torch.Tensor]:
    module = bundle["generator"] if side == "official" else bundle["task"].generator
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _waves(out: Path) -> dict:
    return torch.load(out / "waves.pt", map_location="cpu", weights_only=False)


def _slice(waves: dict, index: int) -> dict:
    return {"clean": waves["clean"][index : index + 1], "noisy": waves["noisy"][index : index + 1], "case_id": [waves["case_id"][index]]}


def _stack(waves: dict, count: int) -> dict:
    return {"clean": waves["clean"][:count], "noisy": waves["noisy"][:count], "case_id": list(waves["case_id"][:count])}


def _one(batches: list[dict]) -> dict:
    return {
        "clean": torch.cat([batch["clean"] for batch in batches], dim=0),
        "noisy": torch.cat([batch["noisy"] for batch in batches], dim=0),
        "case_id": [case for batch in batches for case in batch["case_id"]],
    }


def _audio_tensor(side: str, capture: dict) -> torch.Tensor:
    if side == "independent":
        return torch.cat([audio.cpu() for audio in capture["audio"]], dim=0)
    return capture["audio"].cpu()


def _grad_gap(left: dict, right: dict) -> dict:
    return _tensor_gap(left, right)


def _tensor_gap(left: dict | torch.Tensor, right: dict | torch.Tensor) -> dict:
    if isinstance(left, torch.Tensor):
        delta = (left.detach().float() - right.detach().float()).abs()
        return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}
    shared = sorted(set(left) & set(right))
    missing = sorted(set(left) ^ set(right))
    max_abs = 0.0
    worst = None
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for key in shared:
        if not torch.is_tensor(left[key]):
            continue
        a = left[key].detach().float().reshape(-1)
        b = right[key].detach().float().reshape(-1)
        delta = float((a - b).abs().max())
        if delta >= max_abs:
            max_abs = delta
            worst = key
        dot += float(torch.dot(a, b))
        left_sq += float(torch.dot(a, a))
        right_sq += float(torch.dot(b, b))
    denom = (left_sq * right_sq) ** 0.5
    cosine = 1.0 if denom == 0.0 else dot / denom
    return {"max_abs": max_abs, "worst": worst, "cosine": cosine, "missing": missing}


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denom = left.norm() * right.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denom)


def _restore_rng(state: dict) -> None:
    from src.experiments.calc.rng import restore_rng

    restore_rng(state, cuda=True)


if __name__ == "__main__":
    main()
