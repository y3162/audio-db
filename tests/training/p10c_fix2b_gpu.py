"""Fix.2-B GPU checks for MossFormerGAN. Not part of the CPU unittest suite.

Predetermined before the first GPU attempt. Failed attempts stay in the log.

R5 repeats: 3 synthetic, then 3 real VoiceBank batches, same process group.
Each repeat is a new official-width initialization. One 824-item validation
follows the real repeats in that same process. Prefix validation inside a
repeat is 1 batch and is not the 824-item result.

Resume tolerances, fixed before comparing a resumed run with a continuous run:
- case ids, plan hash, global_step, official_step, learning rate,
  accumulation count, and integer BatchNorm counters: exact
- parameter atol 1e-3
- Adam moment atol 1e-3
- floating BatchNorm buffer atol 1e-4
- saved accumulation gradient atol 1e-4
- rank-to-rank parameter atol 1e-6

If two continuous runs differ by more than that atol, a resumed run is still
consistent when its delta is at most twice the continuous-vs-continuous delta.
A larger gap is a restore mismatch, not CUDA noise.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("P10C_OUT", "/tmp/p10c-fix2b"))
PY = ROOT / ".venvs" / "se" / "bin" / "python"
DATASET = "mossformergan.vb28.ondemand.official@1"
GPUS = os.environ.get("P10C_GPUS", "6,7")

R5_SYNTHETIC_REPEATS = 3
R5_REAL_REPEATS = 3
WEIGHT_ATOL = 1e-3
OPTIM_ATOL = 1e-3
BN_ATOL = 1e-4
GRAD_ATOL = 1e-4
RANK_ATOL = 1e-6


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _children(pid: int) -> list[int]:
    found = []
    node = Path(f"/proc/{pid}/task/{pid}/children")
    if not node.is_file():
        return found
    for part in node.read_text(encoding="utf-8").split():
        child = int(part)
        found.append(child)
        found.extend(_children(child))
    return found


def _signal_workers(pid: int, sig: int) -> list[int]:
    targets = []
    for child in _children(pid):
        command = Path(f"/proc/{child}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        if "p10c_fix2b_gpu.py" in command and "worker" in command:
            targets.append(child)
    for child in targets:
        os.kill(child, sig)
    return targets


def _launch(phase: str, log_path: Path, extra: dict[str, str], timeout: float) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(extra)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["P10C_OUT"] = str(OUT)
    command = [
        str(PY),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(Path(__file__).resolve()),
        "worker",
        phase,
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = started + timeout
        while time.time() < deadline:
            code = process.poll()
            if code is not None:
                return {"returncode": code, "elapsed_sec": time.time() - started, "pid": process.pid}
            time.sleep(1)
        process.kill()
        process.wait(timeout=30)
        return {"returncode": None, "elapsed_sec": time.time() - started, "pid": process.pid, "killed_by_driver": True}


def _base_config(checkpoint_dir: Path, **overrides) -> dict:
    data = {
        "pipeline": "reproduction",
        "dataset": DATASET,
        "allow_test_selection": True,
        "load_workers": 4,
        "prefetch": True,
        "verify_on_read": False,
    }
    if "max_updates_per_epoch" in overrides:
        data["max_updates_per_epoch"] = overrides.pop("max_updates_per_epoch")
    if "validation_max_batches" in overrides:
        data["validation_max_batches"] = overrides.pop("validation_max_batches")
    config = {
        "seed": 777,
        "deterministic": False,
        "global_batch_size": 4,
        "micro_batch_size_per_gpu": 1,
        "required_world_size": 2,
        "experiment_profile": "full",
        "experiment_claim": "official_faithful",
        "max_steps": 350053,
        "segment_size": 32000,
        "loss": "official",
        "grad_clip": 10.0,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_interval": 1,
        "validation_interval": 2893,
        "best_checkpoint_start_epoch": 40,
        "verification_max_updates": 3,
        "optimizer": {"name": "adamw", "lr": 0.0005, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "mossformergan",
            "variant": "se_16k",
            "sampling_rate": 16000,
            "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
        },
        "data": data,
        "runtime": {"device": "cuda", "allow_busy_gpu": False},
    }
    config.update(overrides)
    return config


def _config_path(directory: Path) -> Path:
    """Keep the YAML outside the checkpoint directory so a new run is allowed."""
    return directory.parent / f"{directory.name}.yaml"


def _dump_config(path: Path, config: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _latest_checkpoint(directory: Path) -> Path | None:
    files = sorted(directory.glob("step_*.pt"))
    return files[-1] if files else None


def _wait_loadable(directory: Path, timeout: float) -> Path | None:
    deadline = time.time() + timeout
    seen = None
    while time.time() < deadline:
        latest = _latest_checkpoint(directory)
        if latest is not None and latest != seen:
            seen = latest
            size = latest.stat().st_size
            time.sleep(1)
            if latest.is_file() and latest.stat().st_size == size and size > 0:
                return latest
        time.sleep(0.5)
    return seen


# --- checkpoint comparison (CPU tensors only) ---

def _flat_diff(left, right, atol: float) -> dict:
    import torch

    if not torch.is_tensor(left) or not torch.is_tensor(right):
        return {"ok": left == right, "kind": type(left).__name__}
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"ok": False, "reason": "shape", "left": list(left.shape), "right": list(right.shape)}
    if not torch.is_floating_point(left):
        same = torch.equal(left.cpu(), right.cpu())
        return {"ok": same, "exact": True}
    value = float((left.detach().cpu().float() - right.detach().cpu().float()).abs().max())
    return {"ok": value <= atol, "max_abs": value}


def _named_diff(left: dict, right: dict, atol: float) -> dict:
    import torch

    missing = sorted(set(left) ^ set(right))
    worst = 0.0
    worst_name = None
    integer_mismatch = []
    for name in sorted(set(left) & set(right)):
        item = _flat_diff(left[name], right[name], atol)
        if item.get("exact") and not item["ok"]:
            integer_mismatch.append(name)
        value = item.get("max_abs")
        if value is not None and value > worst:
            worst = value
            worst_name = name
        if item.get("reason") == "shape":
            return {"ok": False, "reason": f"shape {name}", "detail": item}
    return {
        "ok": not missing and not integer_mismatch and worst <= atol,
        "max_abs": worst,
        "worst": worst_name,
        "missing": missing,
        "integer_mismatch": integer_mismatch,
    }


def _optim_diff(left: dict, right: dict) -> dict:
    import torch

    lrs = [group["lr"] for group in left["param_groups"]]
    other = [group["lr"] for group in right["param_groups"]]
    state_keys = set(left["state"]) ^ set(right["state"])
    worst = 0.0
    worst_name = None
    step_mismatch = False
    for key in sorted(set(left["state"]) & set(right["state"]), key=lambda item: str(item)):
        for name in ("exp_avg", "exp_avg_sq"):
            if name not in left["state"][key] and name not in right["state"][key]:
                continue
            value = float(
                (left["state"][key][name].cpu().float() - right["state"][key][name].cpu().float()).abs().max()
            )
            if value > worst:
                worst = value
                worst_name = f"{key}:{name}"
        if "step" in left["state"][key] or "step" in right["state"][key]:
            a = left["state"][key].get("step")
            b = right["state"][key].get("step")
            if torch.is_tensor(a):
                a = a.cpu()
            if torch.is_tensor(b):
                b = b.cpu()
            if not torch.equal(torch.as_tensor(a).cpu(), torch.as_tensor(b).cpu()):
                step_mismatch = True
    return {
        "ok": lrs == other and not state_keys and not step_mismatch and worst <= OPTIM_ATOL,
        "lr": lrs,
        "other_lr": other,
        "max_abs": worst,
        "worst": worst_name,
        "state_key_mismatch": sorted(str(item) for item in state_keys),
        "step_mismatch": step_mismatch,
    }


def _rng_diff(left: dict, right: dict) -> dict:
    import numpy as np
    import torch

    python_same = left["python"] == right["python"]
    numpy_same = np.array_equal(left["numpy"][1], right["numpy"][1]) and left["numpy"][2] == right["numpy"][2]
    torch_same = torch.equal(left["torch"].cpu(), right["torch"].cpu())
    cuda_left = left.get("cuda") or []
    cuda_right = right.get("cuda") or []
    cuda_same = len(cuda_left) == len(cuda_right) and all(
        torch.equal(a.cpu(), b.cpu()) for a, b in zip(cuda_left, cuda_right, strict=True)
    )
    return {
        "python": python_same,
        "numpy": bool(numpy_same),
        "torch": bool(torch_same),
        "cuda": bool(cuda_same),
        "ok": bool(python_same and numpy_same and torch_same and cuda_same),
    }


def _bn_diff(left: dict, right: dict) -> dict:
    import torch

    floating = {}
    integer = {}
    worst = 0.0
    worst_name = None
    for name in sorted(set(left) & set(right)):
        if "running_" not in name and not name.endswith("num_batches_tracked"):
            continue
        a = left[name]
        b = right[name]
        if not torch.is_tensor(a):
            continue
        if not torch.is_floating_point(a):
            integer[name] = bool(torch.equal(a.cpu(), b.cpu()))
            continue
        value = float((a.cpu().float() - b.cpu().float()).abs().max())
        floating[name] = value
        if value > worst:
            worst = value
            worst_name = name
    return {
        "ok": worst <= BN_ATOL and all(integer.values()),
        "max_abs": worst,
        "worst": worst_name,
        "integer": integer,
    }


def _restore_env(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = previous


def compare_checkpoints(left_path: Path, right_path: Path) -> dict:
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    from src.training.core.checkpoint import load_checkpoint

    left = load_checkpoint(left_path, torch.device("cpu"))
    right = load_checkpoint(right_path, torch.device("cpu"))
    task_l = left["task"]
    task_r = right["task"]
    acc_l = task_l.get("accumulation") or {}
    acc_r = task_r.get("accumulation") or {}
    try:
        return {
        "global_step": [left["global_step"], right["global_step"]],
        "global_step_equal": left["global_step"] == right["global_step"],
        "data_state_equal": left.get("data_state") == right.get("data_state"),
        "plan_hash": [
            (left.get("data_state") or {}).get("plan_hash"),
            (right.get("data_state") or {}).get("plan_hash"),
        ],
        "official_step": [acc_l.get("counters", {}).get("official_step"), acc_r.get("counters", {}).get("official_step")],
        "accu_count": [acc_l.get("count"), acc_r.get("count")],
        "counters": [acc_l.get("counters"), acc_r.get("counters")],
        "generator": _named_diff(task_l["generator"], task_r["generator"], WEIGHT_ATOL),
        "discriminator": _named_diff(task_l["discriminator"], task_r["discriminator"], WEIGHT_ATOL),
        "optim_g": _optim_diff(task_l["optim_g"], task_r["optim_g"]),
        "optim_d": _optim_diff(task_l["optim_d"], task_r["optim_d"]),
        "generator_bn": _bn_diff(task_l["generator"], task_r["generator"]),
        "discriminator_bn": _bn_diff(task_l["discriminator"], task_r["discriminator"]),
        "generator_grads": _named_diff(acc_l.get("generator_grads") or {}, acc_r.get("generator_grads") or {}, GRAD_ATOL),
        "discriminator_grads": _named_diff(
            acc_l.get("discriminator_grads") or {}, acc_r.get("discriminator_grads") or {}, GRAD_ATOL
        ),
        "rng": _rng_diff(left["rng"], right["rng"]),
        "plateau": [task_l.get("plateau"), task_r.get("plateau")],
        "validation_pending": [left.get("validation_pending"), right.get("validation_pending")],
        "learning_rate_g": [task_l["optim_g"]["param_groups"][0]["lr"], task_r["optim_g"]["param_groups"][0]["lr"]],
        }
    finally:
        _restore_env(previous)


def _train_case_ids(directory: Path) -> list[dict]:
    path = directory / "metrics.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") in ("train", "pesq_skip", "validation"):
            rows.append(
                {
                    "type": record.get("type"),
                    "global_step": record.get("global_step"),
                    "case_ids": record.get("case_ids"),
                    "losses": record.get("losses"),
                    "learning_rate": record.get("learning_rate"),
                    "validation_committed": record.get("validation_committed"),
                    "validation_coverage": record.get("validation_coverage"),
                }
            )
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _judge(control: dict, resumed: dict) -> dict:
    """Apply the predetermined rule. Control is continuous-vs-continuous."""

    def margin(section: str, atol: float) -> dict:
        resumed_section = resumed.get(section) or {}
        structural = bool(
            resumed_section.get("missing")
            or resumed_section.get("integer_mismatch")
            or resumed_section.get("step_mismatch")
            or resumed_section.get("state_key_mismatch")
            or resumed_section.get("reason")
        )
        control_gap = float((control.get(section) or {}).get("max_abs") or 0.0)
        resume_gap = float(resumed_section.get("max_abs") or 0.0)
        limit = atol if control_gap <= atol else 2.0 * control_gap
        if structural:
            kind = "restore_mismatch"
        elif resume_gap <= atol:
            kind = "within_atol"
        elif resume_gap <= limit:
            kind = "within_continuous_spread"
        else:
            kind = "restore_mismatch"
        return {"control_max_abs": control_gap, "resume_max_abs": resume_gap, "limit": limit, "kind": kind}

    sections = {
        "generator": margin("generator", WEIGHT_ATOL),
        "discriminator": margin("discriminator", WEIGHT_ATOL),
        "optim_g": margin("optim_g", OPTIM_ATOL),
        "optim_d": margin("optim_d", OPTIM_ATOL),
        "generator_grads": margin("generator_grads", GRAD_ATOL),
        "discriminator_grads": margin("discriminator_grads", GRAD_ATOL),
        "generator_bn": margin("generator_bn", BN_ATOL),
        "discriminator_bn": margin("discriminator_bn", BN_ATOL),
    }
    exact_ok = (
        resumed.get("global_step_equal")
        and resumed.get("data_state_equal")
        and resumed.get("official_step", [None, None])[0] == resumed.get("official_step", [None, 1])[1]
        and resumed.get("accu_count", [None, None])[0] == resumed.get("accu_count", [None, 1])[1]
        and resumed.get("learning_rate_g", [None, None])[0] == resumed.get("learning_rate_g", [None, 1])[1]
    )
    float_ok = all(item["kind"] != "restore_mismatch" for item in sections.values())
    return {"exact_counters": bool(exact_ok), "sections": sections, "consistent": bool(exact_ok and float_ok)}


# --- worker ---

def _mark(path: Path, name: str, rank: int, **extra) -> None:
    import torch

    row = {"name": name, "rank": rank, "time": time.time(), **extra}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        row["allocated"] = int(torch.cuda.memory_allocated())
        row["reserved"] = int(torch.cuda.memory_reserved())
        row["peak"] = int(torch.cuda.max_memory_allocated())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _synthetic(index: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(1000 + index)
    return {
        "noisy": torch.randn(1, 32000, generator=generator),
        "clean": torch.randn(1, 32000, generator=generator),
        "case_id": [f"synthetic-{index}"],
        "lengths": torch.tensor([32000]),
    }


def _open_data():
    from src.reproduction.pipeline.training_data import open_training_data

    return open_training_data(
        dataset=DATASET,
        allow_test_selection=True,
        seed=777,
        global_batch=4,
        micro_batch=1,
        accumulation=2,
        world_size=2,
        segment_size=32000,
        model_sample_rate=16000,
        load_workers=2,
        verify_on_read=False,
    )


def _task_config(root: Path) -> dict:
    return {
        "seed": 777,
        "loss": "official",
        "grad_clip": 10.0,
        "optimizer": {"name": "adamw", "lr": 5e-4, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "mossformergan",
            "variant": "se_16k",
            "sampling_rate": 16000,
            "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
        },
        "checkpoint_dir": str(root),
    }


def _new_task(device, process):
    from src.training.se.mossformergan import MossFormerGANTask

    task = MossFormerGANTask(_task_config(OUT / "r5-task"), device)
    task.distribute(process)
    return task


def _rank_delta(task, device) -> dict:
    import torch
    import torch.distributed as dist

    from src.training.core.distributed import unwrap

    deltas = {}
    for label, module in (("generator", task.generator), ("discriminator", task.discriminator)):
        worst = 0.0
        for parameter in unwrap(module).parameters():
            other = parameter.detach().clone()
            dist.broadcast(other, src=0)
            if dist.get_rank() != 0:
                worst = max(worst, float((parameter.detach() - other).abs().max()))
        for name, buffer in unwrap(module).named_buffers():
            other = buffer.detach().clone()
            dist.broadcast(other, src=0)
            if dist.get_rank() != 0 and torch.is_floating_point(buffer):
                worst = max(worst, float((buffer.detach() - other).abs().max()))
        packed = torch.tensor([worst], device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)
        deltas[label] = float(packed.item())
    return deltas


def _micro(task, batch, marks: Path, label: str, rank: int) -> dict:
    _mark(marks, f"{label}:before", rank, shape=list(batch["clean"].shape))
    metrics = task.train_update([batch], 2)
    _mark(
        marks,
        f"{label}:after",
        rank,
        update_applied=metrics["update_applied"],
        official_step=task.counters["official_step"],
        accu=task._accu_count,
        succeeded=task.counters["micro_batches_succeeded"],
        failed=task.counters["micro_batches_failed"],
        generator_loss=metrics["generator"],
        discriminator_loss=metrics["discriminator"],
    )
    return metrics


def _worker_r5() -> None:
    import torch
    import torch.distributed as dist

    from src.reproduction.pipeline.planner import validation_refs
    from src.training.core.checkpoint import save_atomic
    from src.training.core.distributed import setup_process, unwrap
    from src.training.core.gpu import assert_gpu_idle
    from src.training.core.rng import capture_rng
    from src.training.se import mossformergan as moss_module

    rank = int(os.environ["LOCAL_RANK"])
    assert_gpu_idle(rank, allow_busy=False)
    process = setup_process("cuda")
    device = process.device
    marks = OUT / "r5" / f"rank{process.rank}-boundaries.jsonl"
    summary = OUT / "r5" / f"rank{process.rank}.json"
    data = _open_data()
    stream = data.epoch(0, rank=process.rank, world_size=2, start_update=0)
    real_group = stream.render(0)
    original_pesq = moss_module.metric_targets
    gate = {"mode": "real", "once": set()}

    def gated(clean, enhanced, sample_rate):
        this = dist.get_rank()
        if gate.get("nan") == "d":
            gate["nan"] = None
            return torch.full((clean.shape[0],), float("nan"))
        if this in gate["once"]:
            gate["once"].discard(this)
            return None
        if gate["mode"] == "fixed":
            return torch.full((clean.shape[0],), 0.5)
        return original_pesq(clean, enhanced, sample_rate)

    moss_module.metric_targets = gated
    original_terms = moss_module.MossFormerGANTask._generator_terms

    def terms(self, *args, **kwargs):
        result = original_terms(self, *args, **kwargs)
        if gate.get("nan") == "g":
            gate["nan"] = None
            result = dict(result)
            result["generator"] = result["generator"] + float("nan")
        return result

    moss_module.MossFormerGANTask._generator_terms = terms
    repeats = []
    try:
        plan = [("synthetic", index) for index in range(R5_SYNTHETIC_REPEATS)]
        plan += [("real", index) for index in range(R5_REAL_REPEATS)]
        for kind, index in plan:
            row = {"kind": kind, "index": index, "status": "running"}
            task = None
            try:
                task = _new_task(device, process)
                _mark(marks, f"{kind}-{index}:init", process.rank, syncbn=_syncbn_count(task))
                if kind == "synthetic":
                    batches = [_synthetic(index * 10 + offset) for offset in range(6)]
                    gate["mode"] = "fixed"
                else:
                    batches = [real_group[0], real_group[1], real_group[0], real_group[1], real_group[0], real_group[1]]
                    gate["mode"] = "real"
                _micro(task, batches[0], marks, f"{kind}-{index}:success-a", process.rank)
                _micro(task, batches[1], marks, f"{kind}-{index}:success-b", process.rank)
                gate["once"] = {0}
                _micro(task, batches[2], marks, f"{kind}-{index}:rank0-pesq-fail", process.rank)
                gate["once"] = set()
                _micro(task, batches[3], marks, f"{kind}-{index}:refill-a", process.rank)
                _micro(task, batches[4], marks, f"{kind}-{index}:refill-b", process.rank)
                task.eval_mode()
                _mark(marks, f"{kind}-{index}:validation-before", process.rank)
                report = task.validation_report(batches[0])
                _mark(
                    marks,
                    f"{kind}-{index}:validation-after",
                    process.rank,
                    pesq_gate=report.get("pesq_gate"),
                    included=report.get("generator_loss_included"),
                )
                task.train_mode()
                spec = torch.rand(1, 1, 201, 19, device=device)
                _ = unwrap(task.discriminator)(spec, spec)
                _mark(marks, f"{kind}-{index}:syncbn-forward", process.rank)
                if process.rank == 0:
                    payload = {
                        "format": "audio-db-training-v1",
                        "global_step": int(task.counters["optimizer_updates"]),
                        "epoch": 0,
                        "micro_batches_consumed": 0,
                        "task": task.state_dict(),
                        "rng": capture_rng(cuda=True),
                        "note": "fix2b r5 probe; not a trainer resume checkpoint",
                    }
                    save_atomic(OUT / "r5" / f"{kind}-{index}.pt", payload)
                dist.barrier()
                _mark(marks, f"{kind}-{index}:checkpoint", process.rank)
                task.train_mode()
                _micro(task, batches[5], marks, f"{kind}-{index}:next-update", process.rank)
                deltas = _rank_delta(task, device)
                row.update(
                    {
                        "status": "PASS",
                        "counters": dict(task.counters),
                        "accu_count": task._accu_count,
                        "rank_delta": deltas,
                        "learning_rate": task.learning_rates(),
                        "skip_reasons": [item["reason"] for item in task.skip_log],
                    }
                )
            except Exception as exc:
                row.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})
                _mark(marks, f"{kind}-{index}:exception", process.rank, error=str(exc))
                repeats.append(row)
                _write(summary, {"repeats": repeats, "aborted": True})
                raise
            finally:
                del task
                torch.cuda.empty_cache()
            repeats.append(row)
            _write(summary, {"repeats": repeats})
        matrix_task = _new_task(device, process)
        conditions = [
            ("all_success", ["ok", "ok"]),
            ("rank0", ["rank0"]),
            ("rank1", ["rank1"]),
            ("both", ["both", "both"]),
            ("success_fail_success", ["ok", "rank0", "ok"]),
            ("fail_success_success", ["both", "ok", "ok"]),
            ("nan_generator", ["nan_g"]),
            ("nan_discriminator", ["nan_d"]),
        ]
        matrix = []
        for name, actions in conditions:
            matrix_task.start_epoch(resume_mid_epoch=False)
            before_counters = dict(matrix_task.counters)
            before_hash = _weight_hash(matrix_task)
            gate["mode"] = "fixed"
            gate["once"] = set()
            gate["nan"] = None
            applied = []
            for action in actions:
                gate["once"] = {"rank0": {0}, "rank1": {1}, "both": {0, 1}}.get(action, set())
                gate["nan"] = {"nan_g": "g", "nan_d": "d"}.get(action)
                applied.append(matrix_task.train_update([real_group[0]], 2)["update_applied"])
            after = dict(matrix_task.counters)
            matrix.append(
                {
                    "name": name,
                    "actions": actions,
                    "update_applied": applied,
                    "processed": after["micro_batches_processed"] - before_counters["micro_batches_processed"],
                    "succeeded": after["micro_batches_succeeded"] - before_counters["micro_batches_succeeded"],
                    "failed": after["micro_batches_failed"] - before_counters["micro_batches_failed"],
                    "optimizer_updates": after["optimizer_updates"] - before_counters["optimizer_updates"],
                    "official_step_delta": after["official_step"] - before_counters["official_step"],
                    "accu_count": matrix_task._accu_count,
                    "weights_unchanged": _weight_hash(matrix_task) == before_hash,
                    "rank_delta": _rank_delta(matrix_task, device),
                }
            )
            _mark(marks, f"matrix:{name}", process.rank)
        _write(summary, {"repeats": repeats, "pesq_matrix": matrix})
        del matrix_task
        refs = validation_refs(
            data.store,
            seed=777,
            epoch=0,
            world_size=2,
            policy=data.policy,
            rank=process.rank,
        )
        full = {"planned": len(refs), "completed": 0, "case_ids_match": None}
        task = _new_task(device, process)
        task.eval_mode()
        seen = []
        _mark(marks, "full824:before", process.rank, planned=len(refs))
        for batch_index, batch in enumerate(data.validation(rank=process.rank, world_size=2, epoch=0)):
            seen.append(batch["case_id"][0])
            task.validation_report(batch)
            full["completed"] = batch_index + 1
            if batch_index % 20 == 0:
                _mark(marks, "full824:progress", process.rank, completed=full["completed"])
        _mark(marks, "full824:after", process.rank, completed=full["completed"])
        full["case_ids_match"] = seen == [item.case_id for item in refs]
        full["per_rank"] = len(seen)
        full["status"] = "PASS" if len(seen) == 412 and full["case_ids_match"] else "FAIL"
        dist.barrier()
        _write(summary, {"repeats": repeats, "pesq_matrix": matrix, "full_validation": full, "plan_hash": stream.plan_hash})
    finally:
        data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _weight_hash(task) -> str:
    from src.training.core.distributed import unwrap

    digest = hashlib.sha256()
    for module in (unwrap(task.generator), unwrap(task.discriminator)):
        for name, parameter in module.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _syncbn_count(task) -> int:
    import torch

    from src.training.core.distributed import unwrap

    return sum(isinstance(module, torch.nn.SyncBatchNorm) for module in unwrap(task.discriminator).modules())


def _install_trainer_hooks() -> None:
    import torch
    import torch.distributed as dist

    from src.training.core.distributed import unwrap
    from src.training.se.mossformergan import MossFormerGANTask

    fail_steps = {int(part) for part in os.environ.get("P10C_FAIL_STEPS", "").split(",") if part}
    fail_rank = os.environ.get("P10C_FAIL_RANK", "")
    original_pesq = __import__("src.training.se.mossformergan", fromlist=["metric_targets"]).metric_targets
    holder: dict = {}

    def gated(clean, enhanced, sample_rate):
        task = holder.get("task")
        step = None if task is None else int(task.counters["official_step"])
        rank = dist.get_rank() if dist.is_initialized() else 0
        if step in fail_steps and (fail_rank == "all" or str(rank) == fail_rank):
            return None
        return original_pesq(clean, enhanced, sample_rate)

    if fail_steps:
        __import__("src.training.se.mossformergan", fromlist=["metric_targets"]).metric_targets = gated

    original_init = MossFormerGANTask.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        holder["task"] = self

    MossFormerGANTask.__init__ = init
    original_update = MossFormerGANTask.train_update
    original_start = MossFormerGANTask.start_epoch
    original_backward = MossFormerGANTask._backward_average_and_clip

    def backward(self, loss, module):
        original_backward(self, loss, module)
        if os.environ.get("P10C_RANK_CHECK") != "1":
            return
        total = 0.0
        for parameter in unwrap(module).parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().pow(2).sum())
        path = Path(self.config["checkpoint_dir"]) / f"grad-rank{dist.get_rank()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"norm": total ** 0.5, "official_step": self.counters["official_step"]}) + "\n")

    def start_epoch(self, resume_mid_epoch: bool = False):
        before = int(getattr(self, "_accu_count", 0))
        grads = any(parameter.grad is not None for parameter in unwrap(self.generator).parameters())
        original_start(self, resume_mid_epoch=resume_mid_epoch)
        if os.environ.get("P10C_LOG_EPOCH") != "1":
            return
        path = Path(self.config["checkpoint_dir"]) / f"epoch-rank{dist.get_rank()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "resume_mid_epoch": resume_mid_epoch,
                        "accu_before": before,
                        "accu_after": int(self._accu_count),
                        "grads_before": grads,
                        "grads_after": any(parameter.grad is not None for parameter in unwrap(self.generator).parameters()),
                    }
                )
                + "\n"
            )

    def train_update(self, batches, accumulation):
        result = original_update(self, batches, accumulation)
        if os.environ.get("P10C_RANK_CHECK") != "1" or not dist.is_initialized():
            return result
        worst = 0.0
        device = unwrap(self.generator).parameters().__next__().device
        for module in (unwrap(self.generator), unwrap(self.discriminator)):
            tensors = list(module.parameters()) + [
                buffer for buffer in module.buffers() if torch.is_floating_point(buffer)
            ]
            for parameter in tensors:
                other = parameter.detach().clone()
                dist.broadcast(other, src=0)
                if dist.get_rank() != 0:
                    worst = max(worst, float((parameter.detach() - other).abs().max()))
        packed = torch.tensor([worst], device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)
        path = Path(self.config["checkpoint_dir"]) / f"rank-delta-{dist.get_rank()}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "global_counters": dict(self.counters),
                        "accu": self._accu_count,
                        "rank_delta": float(packed.item()),
                        "update_applied": result["update_applied"],
                        "case_ids": [case for batch in batches for case in batch["case_id"]],
                        "lr": self.learning_rates(),
                    }
                )
                + "\n"
            )
        return result

    MossFormerGANTask._backward_average_and_clip = backward
    MossFormerGANTask.start_epoch = start_epoch
    MossFormerGANTask.train_update = train_update


def _worker_train() -> None:
    import torch.distributed as dist

    from src.training.core.gpu import assert_gpu_idle
    from src.training.se.run import execute

    assert_gpu_idle(int(os.environ["LOCAL_RANK"]), allow_busy=False)
    _install_trainer_hooks()
    resume = os.environ.get("P10C_RESUME") or None
    execute(os.environ["P10C_CONFIG"], resume=None if not resume else resume, strict_resume=bool(resume))
    del dist


def _worker_crash(mode: str) -> None:
    import torch
    import torch.distributed as dist

    from src.training.core.distributed import ProcessInfo, unwrap
    from src.training.core.gpu import assert_gpu_idle

    local = int(os.environ["LOCAL_RANK"])
    assert_gpu_idle(local, allow_busy=False)
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=20))
    rank = dist.get_rank()
    device = torch.device("cuda", local)
    process = ProcessInfo(rank, dist.get_world_size(), local, device, True)
    log = OUT / "crash" / mode / f"rank{rank}.json"
    _write(log, {"status": "started", "mode": mode, "rank": rank})
    task = _new_task(device, process)
    batch = _synthetic(0)
    victim = 0 if mode.endswith("0") else 1
    dist.barrier()
    if mode.startswith("before"):
        if rank == victim:
            _write(log, {"status": "raising", "where": "before_forward"})
            raise RuntimeError(f"intentional exception on rank {rank} before SyncBatchNorm")
        _write(log, {"status": "entering_validation"})
        task.eval_mode()
        task.validation_report(batch)
    else:
        if rank == victim:
            for module in unwrap(task.discriminator).modules():
                if isinstance(module, torch.nn.SyncBatchNorm):
                    def _boom(_input, _module=module):
                        raise RuntimeError(f"intentional SyncBatchNorm exception on rank {rank}")

                    module.forward = _boom
                    break
        _write(log, {"status": "entering_syncbn"})
        task.eval_mode()
        task.validation_report(batch)
    _write(log, {"status": "returned"})


def _worker() -> None:
    phase = sys.argv[2]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        if phase == "r5":
            _worker_r5()
        elif phase == "train":
            _worker_train()
        elif phase == "crash":
            _worker_crash(sys.argv[3])
        else:
            raise SystemExit(f"unknown phase {phase}")
    except Exception:
        failed = OUT / "workers" / f"{phase}-rank{os.environ.get('RANK', 'x')}.traceback"
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _run_train(name: str, config: dict, extra: dict[str, str], timeout: float, resume: Path | None = None) -> dict:
    directory = Path(config["checkpoint_dir"])
    config_path = _config_path(directory)
    _dump_config(config_path, config)
    env = {"P10C_CONFIG": str(config_path), "P10C_RANK_CHECK": "1", **extra}
    if resume is not None:
        env["P10C_RESUME"] = str(resume)
    launch = _launch("train", OUT / "logs" / f"{name}.log", env, timeout)
    checkpoint = _latest_checkpoint(directory)
    return {
        "name": name,
        "launch": launch,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": None if checkpoint is None else _file_sha256(checkpoint),
        "status_file": _read_json(directory / "status.json"),
        "cases": _train_case_ids(directory),
    }


def _read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _interrupt(name: str, config: dict, extra: dict[str, str], how: str) -> dict:
    directory = Path(config["checkpoint_dir"])
    config_path = _config_path(directory)
    _dump_config(config_path, config)
    env = {"P10C_CONFIG": str(config_path), "P10C_RANK_CHECK": "1", **extra}
    log_path = OUT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_env = os.environ.copy()
    command_env.update(env)
    command_env["PYTHONPATH"] = str(ROOT) + os.pathsep + command_env.get("PYTHONPATH", "")
    command_env["CUDA_VISIBLE_DEVICES"] = GPUS
    command_env["CUDA_LAUNCH_BLOCKING"] = "1"
    command_env["PYTHONUNBUFFERED"] = "1"
    command_env["P10C_OUT"] = str(OUT)
    command = [
        str(PY),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(Path(__file__).resolve()),
        "worker",
        "train",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=command_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + 900
        checkpoint = None
        while time.time() < deadline:
            if process.poll() is not None:
                checkpoint = _latest_checkpoint(directory)
                break
            latest = _latest_checkpoint(directory)
            if latest is not None:
                size = latest.stat().st_size
                time.sleep(1)
                if latest.is_file() and latest.stat().st_size == size and size > 0:
                    checkpoint = latest
                    break
            time.sleep(0.5)
        if checkpoint is None:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            return {
                "name": name,
                "error": "no checkpoint appeared",
                "returncode": process.poll(),
                "launch_pid": process.pid,
            }
        targets = []
        if process.poll() is None:
            sig = signal.SIGTERM if how == "sigterm" else signal.SIGKILL
            targets = _signal_workers(process.pid, sig)
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                process.kill()
                code = None
        else:
            code = process.returncode
        return {
            "name": name,
            "signal": how,
            "signaled_pids": targets,
            "returncode": code,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "status_file": _read_json(directory / "status.json"),
            "cases": _train_case_ids(directory),
        }


def _pending_probe(config: dict) -> dict:
    directory = Path(config["checkpoint_dir"])
    config_path = _config_path(directory)
    _dump_config(config_path, config)
    env = os.environ.copy()
    env.update({"P10C_CONFIG": str(config_path), "P10C_RANK_CHECK": "1", "P10C_LOG_EPOCH": "1"})
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["P10C_OUT"] = str(OUT)
    log_path = OUT / "logs" / "pending-kill.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PY), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", str(Path(__file__).resolve()), "worker", "train"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 900
        found = None
        while time.time() < deadline and process.poll() is None:
            for path in directory.glob("step_*.pt"):
                previous = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    import torch

                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except Exception:
                    payload = None
                finally:
                    _restore_env(previous)
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload, dict) and payload.get("validation_pending"):
                    found = path
                    break
            if found is not None:
                break
            time.sleep(0.3)
        if found is None:
            process.kill()
            process.wait(timeout=30)
            return {"error": "validation_pending checkpoint was not observed", "status": _read_json(directory / "status.json")}
        targets = _signal_workers(process.pid, signal.SIGKILL)
        process.wait(timeout=60)
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import torch

            payload = torch.load(found, map_location="cpu", weights_only=False)
        finally:
            _restore_env(previous)
        return {
            "checkpoint": str(found),
            "still_pending": bool(payload.get("validation_pending")),
            "global_step": payload.get("global_step"),
            "signaled_pids": targets,
            "sha256": _file_sha256(found),
        }


def driver() -> None:
    import yaml

    del yaml
    OUT.mkdir(parents=True, exist_ok=True)
    _write(
        OUT / "plan.json",
        {
            "r5_synthetic_repeats": R5_SYNTHETIC_REPEATS,
            "r5_real_repeats": R5_REAL_REPEATS,
            "full_validation_runs": 1,
            "weight_atol": WEIGHT_ATOL,
            "optimizer_atol": OPTIM_ATOL,
            "bn_atol": BN_ATOL,
            "grad_atol": GRAD_ATOL,
            "rank_atol": RANK_ATOL,
            "gpus": GPUS,
            "dataset": DATASET,
            "note": "Smoke caps stay identical between an interrupted run and its resume. They are not the 121-epoch launch.",
        },
    )
    results: dict = {"phases": []}
    if os.environ.get("P10C_SKIP_R5") == "1":
        print("phase r5 skipped", flush=True)
        r5 = {"skipped": True, "returncode": 0}
    else:
        print("phase r5", flush=True)
        r5 = _launch("r5", OUT / "logs" / "r5.log", {}, timeout=3600)
    r5_summary = {
        "launch": r5,
        "rank0": _read_json(OUT / "r5" / "rank0.json"),
        "rank1": _read_json(OUT / "r5" / "rank1.json"),
    }
    results["r5"] = r5_summary
    _write(OUT / "results.json", results)

    def train_pair(label: str, overrides: dict, extra: dict, how: str) -> None:
        continuous_a = OUT / label / "continuous-a"
        continuous_b = OUT / label / "continuous-b"
        interrupted = OUT / label / "interrupted"
        config_a = _base_config(continuous_a, **overrides)
        config_b = _base_config(continuous_b, **overrides)
        config_i = _base_config(interrupted, **overrides)
        first = _run_train(f"{label}-continuous-a", config_a, extra, timeout=1800)
        second = _run_train(f"{label}-continuous-b", config_b, extra, timeout=1800)
        stopped = _interrupt(f"{label}-interrupt", config_i, extra, how)
        resumed = None
        comparison = None
        latest = _latest_checkpoint(interrupted)
        if latest is not None:
            resumed = _run_train(
                f"{label}-resume",
                config_i,
                extra,
                timeout=1800,
                resume=latest,
            )
            if first.get("checkpoint") and second.get("checkpoint") and resumed.get("checkpoint"):
                control = compare_checkpoints(Path(first["checkpoint"]), Path(second["checkpoint"]))
                against = compare_checkpoints(Path(first["checkpoint"]), Path(resumed["checkpoint"]))
                comparison = {"control": control, "resumed_vs_continuous_a": against, "judgment": _judge(control, against)}
        results[label] = {
            "continuous_a": first,
            "continuous_b": second,
            "interrupted": stopped,
            "resumed": resumed,
            "comparison": comparison,
            "cases_equal_prefix": _case_prefix(first, stopped, resumed),
        }
        _write(OUT / "results.json", results)

    print("phase happy", flush=True)
    train_pair("happy", {"verification_max_updates": 3, "checkpoint_interval": 1}, {"P10C_LOG_EPOCH": "1"}, "sigterm")
    print("phase partial", flush=True)
    train_pair(
        "partial",
        {"verification_max_updates": 2, "checkpoint_interval": 1},
        {"P10C_FAIL_STEPS": "1", "P10C_FAIL_RANK": "0", "P10C_LOG_EPOCH": "1"},
        "sigkill",
    )
    epoch_dir = OUT / "epoch"
    epoch_config = _base_config(
        epoch_dir,
        verification_max_updates=2,
        checkpoint_interval=1,
        max_updates_per_epoch=3,
        validation_max_batches=2,
    )
    print("phase epoch", flush=True)
    results["epoch_fail"] = _run_train(
        "epoch-fail",
        epoch_config,
        {"P10C_FAIL_STEPS": "4,5", "P10C_FAIL_RANK": "all", "P10C_LOG_EPOCH": "1"},
        timeout=1800,
    )
    _write(OUT / "results.json", results)
    residual_dir = OUT / "residual"
    residual_config = _base_config(
        residual_dir,
        verification_max_updates=1,
        checkpoint_interval=1,
        max_updates_per_epoch=1,
        validation_max_batches=2,
    )
    results["epoch_residual"] = _run_train(
        "epoch-residual",
        residual_config,
        {"P10C_FAIL_STEPS": "1", "P10C_FAIL_RANK": "all", "P10C_LOG_EPOCH": "1"},
        timeout=1800,
    )
    _write(OUT / "results.json", results)
    pending_dir = OUT / "pending"
    pending_config = _base_config(
        pending_dir,
        verification_max_updates=2,
        checkpoint_interval=0,
        max_updates_per_epoch=2,
        validation_max_batches=80,
    )
    print("phase pending", flush=True)
    killed = _pending_probe(pending_config)
    resumed_pending = None
    if killed.get("still_pending") and killed.get("checkpoint"):
        resumed_pending = _run_train("pending-resume", pending_config, {"P10C_LOG_EPOCH": "1"}, timeout=1800, resume=Path(killed["checkpoint"]))
    results["pending"] = {"killed": killed, "resumed": resumed_pending, "cases": _train_case_ids(pending_dir)}
    _write(OUT / "results.json", results)

    print("phase crash", flush=True)
    results["crash"] = _crash_suite()
    before = OUT / "kill-before"
    before_config = _base_config(before, verification_max_updates=5, checkpoint_interval=0)
    results["kill_before_checkpoint"] = _kill_during_init(before_config)
    _write(OUT / "results.json", results)
    print(json.dumps({"out": str(OUT), "keys": sorted(results)}), flush=True)


def _case_prefix(first: dict, stopped: dict, resumed: dict | None) -> dict:
    def ids(payload):
        rows = []
        for row in payload.get("cases") or []:
            if row["type"] == "train":
                rows.append(row["case_ids"])
        return rows

    continuous = ids(first)
    interrupt = ids(stopped)
    resume_rows = [] if resumed is None else ids(resumed)
    return {
        "continuous": continuous,
        "interrupted_matches_prefix": interrupt == continuous[: len(interrupt)] if interrupt else False,
        "resumed_matches_continuous": resume_rows == continuous,
    }


def _crash_suite() -> dict:
    outcomes = {}
    for mode in ("before0", "before1", "inside0", "inside1"):
        log_path = OUT / "logs" / f"crash-{mode}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = GPUS
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["P10C_OUT"] = str(OUT)
        command = [
            str(PY),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "2",
            str(Path(__file__).resolve()),
            "worker",
            "crash",
            mode,
        ]
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=180)
                killed = False
            except subprocess.TimeoutExpired:
                process.kill()
                code = None
                killed = True
        text = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        outcomes[mode] = {
            "returncode": code,
            "elapsed_sec": time.time() - started,
            "driver_killed": killed,
            "log_tail": text,
            "rank_logs": {
                "0": _read_json(OUT / "crash" / mode / "rank0.json"),
                "1": _read_json(OUT / "crash" / mode / "rank1.json"),
            },
        }
    return outcomes


def _kill_during_init(config: dict) -> dict:
    directory = Path(config["checkpoint_dir"])
    config_path = _config_path(directory)
    _dump_config(config_path, config)
    env = os.environ.copy()
    env["P10C_CONFIG"] = str(config_path)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["P10C_OUT"] = str(OUT)
    log_path = OUT / "logs" / "kill-before.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PY), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", str(Path(__file__).resolve()), "worker", "train"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 900
        while time.time() < deadline and process.poll() is None:
            if "batch plan" in log_path.read_text(encoding="utf-8", errors="replace"):
                time.sleep(3)
                break
            time.sleep(0.5)
        targets = _signal_workers(process.pid, signal.SIGKILL)
        process.wait(timeout=60)
    steps = list(directory.glob("step_*.pt"))
    temps = list(directory.glob(".step_*.tmp")) + list(directory.glob("*.tmp"))
    return {
        "signaled_pids": targets,
        "step_files": [path.name for path in steps],
        "temp_files": [path.name for path in temps],
        "status": _read_json(directory / "status.json"),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        _worker()
    else:
        driver()
