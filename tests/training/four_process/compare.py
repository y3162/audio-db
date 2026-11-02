"""Sequential four-process training comparison. Results are trusted local torch files."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)

from src.training.core.checkpoint import save_atomic
from src.training.core.rng import capture_rng, restore_rng

RUNS = ("official_a", "official_b", "independent_a", "independent_b")
PAIRS = ((0, 1), (2, 3), (0, 2), (1, 3))
FORMAT = "audio-db-four-process-v1"

def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()

from tests.training.four_process.adapters import ADAPTERS

def identity(cfg):
    keys = (
        "model", "seed", "device", "fault", "fault_run", "dataset", "database",
        "segment_size", "sample_rate", "epoch", "order",
    )
    return {key: cfg.get(key) for key in keys}

def seed_probe(seed, cuda):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def rng_parts(cuda):
    state = capture_rng(cuda=cuda)
    return {
        "torch": digest(state["torch"]),
        "python": digest(state["python"]),
        "numpy": digest(state["numpy"]),
        "cuda": digest(state["cuda"]) if cuda else None,
    }

def module_digest(modules):
    return {name: digest(module.state_dict()) for name, module in modules.items()}

def prepare(path, cfg):
    seed_probe(cfg["seed"], False)
    adapter = ADAPTERS[cfg["model"]]
    modules = adapter.build("independent", torch.device("cpu"))
    state = {name: module.state_dict() for name, module in modules.items()}
    payload = {"initial": state, "kind": "synthetic"}
    if cfg.get("dataset"):
        from tests.training.four_process.recipes import describe_item, load_probe_dataset, planned_indices
        dataset = load_probe_dataset(cfg)
        dataset.set_epoch(int(cfg["epoch"]))
        indices = planned_indices(dataset, cfg)
        items, cleans, noisies = [], [], []
        for index in indices:
            item, clean, noisy = describe_item(dataset, index)
            items.append(item)
            cleans.append(clean)
            noisies.append(noisy)
        payload.update(kind="recipe", indices=indices, items=items, clean=cleans, noisy=noisies)
    else:
        generator = torch.Generator().manual_seed(cfg["seed"] + 1)
        length = adapter.length
        clean = torch.randn(1, length, generator=generator)
        noisy = clean * 0.5 + 0.05
        payload.update(clean=clean, noisy=noisy, items=[{
            "case_id": "NA: synthetic input",
            "recipe_sha256": None,
            "sample_rate": 16000,
            "file_sample_rate": None,
            "sources": {},
            "clean_length": length,
            "noisy_length": length,
            "crop_start": None,
            "segment_size": None,
            "batch_index": 0,
            "clean_sha256": digest(clean),
            "noisy_sha256": digest(noisy),
        }])
    torch.save(payload, path)

def _batch(root, cfg, position, fault):
    data = torch.load(root / "input.pt", map_location="cpu", weights_only=False)
    if data["kind"] == "synthetic":
        clean, noisy = data["clean"], data["noisy"]
        items = data["items"]
    else:
        from tests.training.four_process.recipes import describe_item, load_probe_dataset
        dataset = load_probe_dataset(cfg)
        dataset.set_epoch(int(cfg["epoch"]))
        index = int(data["indices"][position])
        item, clean, noisy = describe_item(dataset, index)
        if item["case_id"] != data["items"][position]["case_id"]:
            raise RuntimeError("loaded case_id does not match the prepared batch")
        if item["clean_sha256"] != data["items"][position]["clean_sha256"] or item["noisy_sha256"] != data["items"][position]["noisy_sha256"]:
            raise RuntimeError("loaded waveform hash does not match the prepared batch")
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        items = [item]
    if fault == "input":
        noisy = noisy.clone()
        noisy[0, 0] += 1
        items = [dict(items[0], noisy_sha256=digest(noisy), case_id=items[0]["case_id"])]
    if fault == "case":
        items = [dict(items[0], case_id="wrong-case")]
    return clean, noisy, items

def _apply_weight_fault(modules):
    with torch.no_grad():
        next(modules["generator"].parameters()).add_(1)

def _snapshot_modules(modules):
    return {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for name, module in modules.items()}

def _load_modules(modules, state):
    for name, module in modules.items():
        module.load_state_dict(state[name], strict=True)

def worker(root, name):
    cfg = json.loads((root / "config.json").read_text())
    run = root / name
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(cfg["device"])
    cuda = device.type == "cuda"
    implementation = name.split("_")[0]
    fault = cfg.get("fault") if name == cfg.get("fault_run") else None
    if fault == "exception":
        raise RuntimeError("injected exception")
    adapter = ADAPTERS[cfg["model"]]
    seed_probe(cfg["seed"], False)
    modules = adapter.build(implementation, device)
    prepared = torch.load(root / "input.pt", map_location="cpu", weights_only=False)
    _load_modules(modules, prepared["initial"])
    if fault == "weight":
        _apply_weight_fault(modules)
    lr = 0.002 if fault == "optimizer" else 0.001
    optimizers = adapter.optimizers(modules, lr)
    schedulers = adapter.schedulers(optimizers)
    seed_probe(cfg["seed"], cuda)
    if fault == "rng":
        torch.randn(1)
        random.random()
        np.random.random()
    checkpoint_path = run / "checkpoint.pt"
    records, snapshots = [], []
    position = 0
    epoch = int(cfg["epoch"])
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        _require_checkpoint(checkpoint, cfg, root, name, modules, fault)
        _load_modules(modules, checkpoint["modules"])
        for key, opt in optimizers.items():
            opt.load_state_dict(checkpoint["optimizers"][key])
        for key, sched in schedulers.items():
            sched.load_state_dict(checkpoint["schedulers"][key])
        restore_rng(checkpoint["rng"], cuda=cuda)
        records = checkpoint["records"]
        snapshots = checkpoint["snapshots"]
        position = int(checkpoint["data_position"])
        epoch = int(checkpoint["epoch"])
        if position != len(records):
            raise RuntimeError("checkpoint data position does not match saved records")
    else:
        _reject_foreign_config(cfg, root)
    if position == 0:
        _clean0, _noisy0, items0 = _batch(root, cfg, 0, fault)
        initial = _initial(cfg, modules, optimizers, schedulers, cuda, items0)
    else:
        initial = checkpoint["initial"]
        if digest(initial) != digest(checkpoint["initial"]):
            raise RuntimeError("refusing to resume because the initial record changed")
    start = time.monotonic()
    while position < int(cfg["steps"]):
        clean, noisy, items = _batch(root, cfg, position, fault)
        clean, noisy = clean.to(device), noisy.to(device)
        step = adapter.train_step(
            implementation, modules, optimizers, schedulers, clean, noisy, position, fault,
        )
        if any(not torch.isfinite(grad).all() for grad in step.gradients.values()):
            raise ValueError("nonfinite gradient")
        loss = step.losses["generator"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        disc = step.losses.get("discriminator")
        grad_norm = math.sqrt(sum(float(v.double().square().sum()) for v in step.gradients.values()))
        state = _snapshot_modules(modules)
        forward = step.forward.detach().cpu()
        snapshot = {"forward": forward, "grad": step.gradients, "weight": state}
        snapshots.append(snapshot)
        position += 1
        components = {
            key: float(value.detach()) for key, value in step.losses.items()
            if key not in ("generator", "discriminator") and torch.is_tensor(value)
        }
        records.append({
            "step": position,
            "loss": float(loss.detach()),
            "generator_loss": float(loss.detach()),
            "discriminator_loss": None if disc is None else float(disc.detach()),
            "components": components,
            "gradient_norm": grad_norm,
            "lr": {key: float(opt.param_groups[0]["lr"]) for key, opt in optimizers.items()},
            "model_hash": digest(state),
            "optimizer_hash": digest({key: opt.state_dict() for key, opt in optimizers.items()}),
            "scheduler_hash": digest({key: sched.state_dict() for key, sched in schedulers.items()}),
            "rng_hash": digest(_rng_tensors(cuda)),
            "rng_parts": rng_parts(cuda),
            "data_position": position,
            "epoch": epoch,
            "global_step": position,
            "update_order": list(step.update_order),
            "elapsed_seconds": time.monotonic() - start,
            "run_id": root.name,
            "model": cfg["model"],
            "implementation": implementation,
            "replicate": name[-1],
            "seed": cfg["seed"],
            "case_id": [item["case_id"] for item in items],
            "input_hash": {"clean": digest(clean.detach().cpu()), "noisy": digest(noisy.detach().cpu())},
            "batch": items,
            "snapshot_hash": digest(snapshot),
        })
        _write_checkpoint(checkpoint_path, cfg, root, name, modules, optimizers, schedulers, cuda, records, snapshots, position, epoch, initial, fault)
    result = _result(name, cfg, implementation, initial, records, snapshots, modules, optimizers, schedulers, cuda, position, epoch)
    if fault != "missing":
        torch.save(result, run / "result.pt")

def _rng_tensors(cuda):
    state = capture_rng(cuda=cuda)
    return {"cpu": state["torch"], "cuda": state.get("cuda")}

def _initial(cfg, modules, optimizers, schedulers, cuda, position_items):
    items = position_items or []
    digests = module_digest(modules)
    return {
        "generator": digests.get("generator"),
        "discriminator": digests.get("discriminator", "NA"),
        "modules": digests,
        "optimizer": digest({key: opt.state_dict() for key, opt in optimizers.items()}),
        "scheduler": digest({key: sched.state_dict() for key, sched in schedulers.items()}),
        "rng": digest(_rng_tensors(cuda)),
        "rng_parts": rng_parts(cuda),
        "clean": None if not items else items[0]["clean_sha256"],
        "noisy": None if not items else items[0]["noisy_sha256"],
        "seed": cfg["seed"],
        "case_id": [item["case_id"] for item in items],
        "batch": items,
        "input_order": [item.get("batch_index", 0) for item in items],
        "crop": [item.get("crop_start") for item in items],
        "sample_rate": cfg.get("sample_rate") or 16000,
        "batch_size": 1,
        "accumulation": 1,
        "dtype": "float32",
        "tf32": False,
        "amp": False,
        "update_order": list(ADAPTERS[cfg["model"]].update_order),
        "optimizers": sorted(optimizers),
    }

def _write_checkpoint(path, cfg, root, name, modules, optimizers, schedulers, cuda, records, snapshots, position, epoch, initial, fault):
    payload = {
        "format": FORMAT,
        "run_name": root.name,
        "worker": name,
        "config": identity(cfg),
        "input_sha256": cfg["input_sha256"],
        "fault": fault,
        "modules": _snapshot_modules(modules),
        "optimizers": {key: opt.state_dict() for key, opt in optimizers.items()},
        "schedulers": {key: sched.state_dict() for key, sched in schedulers.items()},
        "global_step": position,
        "epoch": epoch,
        "data_position": position,
        "rng": capture_rng(cuda=cuda),
        "records": records,
        "snapshots": snapshots,
        "initial": initial,
        "settings": {
            "model": cfg["model"],
            "seed": cfg["seed"],
            "device": cfg["device"],
            "update_order": list(ADAPTERS[cfg["model"]].update_order),
        },
    }
    save_atomic(path, payload)

def _reject_foreign_config(cfg, root):
    if cfg.get("input_sha256") != file_digest(root / "input.pt"):
        raise RuntimeError("refusing to start because input.pt does not match the recorded hash")

def _require_checkpoint(checkpoint, cfg, root, name, modules, fault):
    if checkpoint.get("format") != FORMAT or checkpoint.get("worker") != name or checkpoint.get("run_name") != root.name:
        raise RuntimeError("refusing to resume a checkpoint from a different run")
    if checkpoint.get("config") != identity(cfg) or checkpoint.get("input_sha256") != cfg.get("input_sha256"):
        raise RuntimeError("refusing to resume because the training settings or input hash differ")
    if checkpoint.get("input_sha256") != file_digest(root / "input.pt"):
        raise RuntimeError("refusing to resume because input.pt changed")
    if checkpoint.get("fault") != fault:
        raise RuntimeError("refusing to resume because the injected fault differs")
    prepared = torch.load(root / "input.pt", map_location="cpu", weights_only=False)
    current = {key: digest(module.state_dict()) for key, module in modules.items()}
    # modules currently hold the prepared initial weights, after the same pre-loop fault
    expected = checkpoint["initial"]["modules"]
    if current != expected:
        raise RuntimeError("refusing to resume because the initial weights differ")
    del prepared

def _result(name, cfg, implementation, initial, records, snapshots, modules, optimizers, schedulers, cuda, position, epoch):
    return {
        "name": name,
        "pid": os.getpid(),
        "model": cfg["model"],
        "implementation": implementation,
        "replicate": name[-1],
        "initial": initial,
        "records": records,
        "snapshots": snapshots,
        "modules": _snapshot_modules(modules),
        "optimizer": {key: opt.state_dict() for key, opt in optimizers.items()},
        "scheduler": {key: sched.state_dict() for key, sched in schedulers.items()},
        "rng": _rng_tensors(cuda),
        "rng_full_hash": digest(capture_rng(cuda=cuda)),
        "rng_parts": rng_parts(cuda),
        "data_position": position,
        "epoch": epoch,
        "global_step": position,
    }

def vectors(left, right):
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("snapshot keys mismatch")
        pairs = [(left[key], right[key]) for key in sorted(left)]
        if not pairs:
            return {"max_abs": 0.0, "relative_l2": 0.0, "cosine": None, "magnitude_bins": {}}
        aa = torch.cat([x.detach().cpu().double().reshape(-1) for x, _ in pairs])
        bb = torch.cat([y.detach().cpu().double().reshape(-1) for _, y in pairs])
    else:
        aa = left.detach().cpu().double().reshape(-1)
        bb = right.detach().cpu().double().reshape(-1)
    if aa.shape != bb.shape or not torch.isfinite(aa).all() or not torch.isfinite(bb).all():
        raise ValueError("invalid snapshot")
    delta = aa - bb
    norm = aa.norm().item()
    bn = bb.norm().item()
    def measure(x, y):
        n = x.norm().item() * y.norm().item()
        return {
            "count": x.numel(),
            "max_abs": (x - y).abs().max().item() if x.numel() else None,
            "cosine": float(x.dot(y) / n) if n > 1e-24 else None,
        }
    bins = {}
    for low, high in ((0, 1e-8), (1e-8, 1e-6), (1e-6, 1e-4), (1e-4, 1e-2), (1e-2, float("inf"))):
        mask = (aa.abs() >= low) & (aa.abs() < high)
        bins[f"{low:g}..{high:g}"] = measure(aa[mask], bb[mask])
    return {
        "max_abs": delta.abs().max().item() if delta.numel() else 0.0,
        "relative_l2": delta.norm().item() / norm if norm else None,
        "cosine": float(aa.dot(bb) / (norm * bn)) if norm * bn > 1e-24 else None,
        "magnitude_bins": bins,
    }

def _flat_weights(state):
    flat = {}
    for module_name, weights in state.items():
        for key, value in weights.items():
            flat[f"{module_name}.{key}"] = value
    return flat

def report(root):
    cfg = json.loads((root / "config.json").read_text())
    runs, errors = {}, []
    for name in RUNS:
        try:
            status = json.loads((root / name / "status.json").read_text())
            if status["returncode"] != 0:
                raise ValueError(f"exit {status['returncode']}")
            if status.get("result_sha256") and file_digest(root / name / "result.pt") != status["result_sha256"]:
                raise ValueError("result file hash mismatch")
            result = torch.load(root / name / "result.pt", map_location="cpu", weights_only=False)
            if result["name"] != name or result["model"] != cfg["model"] or len(result["records"]) != cfg["steps"] or len(result["snapshots"]) != cfg["steps"]:
                raise ValueError("invalid result identity or length")
            for index, rec in enumerate(result["records"]):
                if rec["step"] != index + 1 or not math.isfinite(rec["loss"]) or not math.isfinite(rec["gradient_norm"]):
                    raise ValueError("invalid metrics")
                if digest(result["snapshots"][index]["weight"]) != rec["model_hash"]:
                    raise ValueError("model hash mismatch")
                if "snapshot_hash" in rec and digest(result["snapshots"][index]) != rec["snapshot_hash"]:
                    raise ValueError("snapshot hash mismatch")
            if digest(result["optimizer"]) != result["records"][-1]["optimizer_hash"]:
                raise ValueError("final state hash mismatch")
            if digest(result["scheduler"]) != result["records"][-1]["scheduler_hash"] or digest(result["rng"]) != result["records"][-1]["rng_hash"]:
                raise ValueError("final state hash mismatch")
            if result["data_position"] != cfg["steps"] or result["global_step"] != cfg["steps"]:
                raise ValueError("final data position mismatch")
            runs[name] = result
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    comparisons, conditions = {}, {}
    if len(runs) == 4:
        base = runs[RUNS[0]]["initial"]
        source = torch.load(root / "input.pt", map_location="cpu", weights_only=False)
        if base["generator"] != digest(source["initial"]["generator"]):
            errors.append("generator: differs from prepared input")
        if source["kind"] == "synthetic":
            if base["clean"] != digest(source["clean"]) or base["noisy"] != digest(source["noisy"]):
                errors.append("waveform: differs from prepared input")
        for name in RUNS:
            conditions[name] = {key: runs[name]["initial"].get(key) == value for key, value in base.items()}
            if not all(conditions[name].values()):
                errors.append(f"{name}: initial conditions mismatch")
        if len({item["pid"] for item in runs.values()}) != 4:
            errors.append("process IDs are not distinct")
        for ia, ib in PAIRS:
            left, right = runs[RUNS[ia]], runs[RUNS[ib]]
            steps = []
            try:
                for rec_a, rec_b, snap_a, snap_b in zip(left["records"], right["records"], left["snapshots"], right["snapshots"]):
                    step = {
                        "step": rec_a["step"],
                        "loss_difference": abs(rec_a["loss"] - rec_b["loss"]),
                        "generator_loss_difference": abs(rec_a["generator_loss"] - rec_b["generator_loss"]),
                        "discriminator_loss_difference": _optional_abs(rec_a["discriminator_loss"], rec_b["discriminator_loss"]),
                        "gradient_norm_difference": abs(rec_a["gradient_norm"] - rec_b["gradient_norm"]),
                        "forward": vectors(snap_a["forward"], snap_b["forward"]),
                        "grad": vectors(snap_a["grad"], snap_b["grad"]),
                        "weight": vectors(_flat_weights(snap_a["weight"]), _flat_weights(snap_b["weight"])),
                        "case_id": rec_a["case_id"],
                        "batch": rec_a["batch"],
                        "update_order": rec_a["update_order"],
                        "rng_parts_match": rec_a["rng_parts"] == rec_b["rng_parts"],
                    }
                    if rec_a["case_id"] != rec_b["case_id"] or rec_a["input_hash"] != rec_b["input_hash"]:
                        raise ValueError("input mismatch")
                    steps.append(step)
                label = f"{RUNS[ia]} vs {RUNS[ib]}"
                comparisons[label] = steps
                if cfg["model"] == "mock" and any(
                    step["loss_difference"] != 0 or any(step[key]["max_abs"] != 0 for key in ("forward", "grad", "weight"))
                    for step in steps
                ):
                    errors.append(f"{label}: mock numerical mismatch")
                if any(step["update_order"] != list(ADAPTERS[cfg["model"]].update_order) for step in steps):
                    errors.append(f"{label}: update order mismatch")
            except Exception as exc:
                errors.append(f"comparison: {type(exc).__name__}: {exc}")
    result = {
        "valid": not errors,
        "errors": errors,
        "conditions": conditions,
        "comparisons": comparisons,
        "pids": {key: value["pid"] for key, value in runs.items()},
    }
    (root / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result

def _optional_abs(left, right):
    if left is None and right is None:
        return None
    return abs(left - right)

def _statuses(root, cfg):
    for name in cfg["order"]:
        run = root / name
        run.mkdir(exist_ok=True)
        start = time.monotonic()
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (run / "stdout.log").open("w") as out, (run / "stderr.log").open("w") as err:
            try:
                completed = subprocess.run(
                    [sys.executable, __file__, "worker", str(root), name],
                    cwd=ROOT, env=env, stdout=out, stderr=err, timeout=cfg["timeout"],
                )
                code, timed_out = completed.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = -1, True
        result_path = run / "result.pt"
        (run / "status.json").write_text(json.dumps({
            "returncode": code,
            "timeout": timed_out,
            "seconds": time.monotonic() - start,
            "result_sha256": file_digest(result_path) if result_path.exists() else None,
        }, indent=2) + "\n")

def launch(root, cfg):
    root.mkdir(parents=True, exist_ok=False)
    cfg = dict(cfg)
    cfg["epoch"] = int(cfg.get("epoch") or 0)
    cfg["sample_rate"] = int(cfg.get("sample_rate") or 16000)
    cfg["segment_size"] = int(cfg.get("segment_size") or ADAPTERS[cfg["model"]].length)
    prepare(root / "input.pt", cfg)
    cfg["input_sha256"] = file_digest(root / "input.pt")
    cfg["directory"] = str(root)
    (root / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (root / "environment.json").write_text(json.dumps(_environment(cfg), indent=2) + "\n")
    _statuses(root, cfg)
    return report(root)

def resume(root, steps):
    cfg = json.loads((root / "config.json").read_text())
    if int(steps) < int(cfg["steps"]):
        raise RuntimeError("refusing to resume with fewer steps than the original run")
    if cfg.get("input_sha256") != file_digest(root / "input.pt"):
        raise RuntimeError("refusing to resume because input.pt changed")
    cfg["steps"] = int(steps)
    if cfg.get("dataset"):
        from tests.training.four_process.recipes import load_probe_dataset, planned_indices
        dataset = load_probe_dataset(cfg)
        dataset.set_epoch(int(cfg["epoch"]))
        indices = planned_indices(dataset, cfg)
        prepared = torch.load(root / "input.pt", map_location="cpu", weights_only=False)
        if list(prepared["indices"]) != list(indices[:len(prepared["indices"])]):
            raise RuntimeError("refusing to resume because the dataset order changed")
        if len(prepared["indices"]) < int(steps):
            raise RuntimeError("prepared recipe batches do not cover the resumed step count")
    (root / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    _statuses(root, cfg)
    return report(root)

def _environment(cfg):
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": cfg["device"],
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip(),
    }

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("directory")
    run.add_argument("--model", choices=sorted(ADAPTERS), default="mock")
    run.add_argument("--steps", type=int, default=1)
    run.add_argument("--seed", type=int, default=3)
    run.add_argument("--device", default="cpu")
    run.add_argument("--timeout", type=int, default=600)
    run.add_argument("--order", nargs=4, default=list(RUNS))
    run.add_argument("--fault")
    run.add_argument("--fault-run", default="independent_b")
    run.add_argument("--dataset", default=None)
    run.add_argument("--database", default=None)
    run.add_argument("--segment-size", type=int, default=None)
    run.add_argument("--sample-rate", type=int, default=16000)
    run.add_argument("--epoch", type=int, default=0)
    resumed = sub.add_parser("resume")
    resumed.add_argument("directory")
    resumed.add_argument("--steps", type=int, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("directory")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("directory")
    worker_parser.add_argument("name")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    try:
        if args.cmd == "worker":
            worker(root, args.name)
            return
        if args.cmd == "report":
            result = report(root)
        elif args.cmd == "resume":
            result = resume(root, args.steps)
        else:
            if sorted(args.order) != sorted(RUNS) or args.steps < 1:
                parser.error("invalid order or steps")
            if bool(args.dataset) != bool(args.database):
                parser.error("dataset and database are set together")
            cfg = vars(args)
            cfg["segment_size"] = args.segment_size or ADAPTERS[args.model].length
            result = launch(root, cfg)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print(json.dumps({"valid": result["valid"], "errors": result["errors"], "pids": result["pids"]}, indent=2))
    sys.exit(0 if result["valid"] else 1)

if __name__ == "__main__":
    main()
