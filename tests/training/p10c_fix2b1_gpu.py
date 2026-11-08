"""Fix.2-B.1 same-checkpoint resume comparison and the Fix.2-B operational rerun.

The plan file is written before the first training process starts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path("/tmp/p10c-fix2b1")
GPUS = "0,6"

PLAN = {
    "continuous_runs_per_mode": 2,
    "resumes_from_same_step1_per_mode": 2,
    "optimizer_updates": 3,
    "seed": 777,
    "world_size": 2,
    "dataset": "mossformergan.vb28.ondemand.official@1",
    "atol": {
        "parameter": 1e-3,
        "adam_moment": 1e-3,
        "bn_float": 1e-4,
        "accumulation_gradient": 1e-4,
        "rank_to_rank": 1e-6,
        "restore_only": 0.0,
    },
    "rule": (
        "Discrete fields are exact. Restore-only weights, optimizer moments, "
        "BatchNorm buffers, accumulation gradients, and per-rank RNG are exact. "
        "A same-checkpoint resume passes a floating section when its max abs "
        "versus the source continuous checkpoint is within that section's atol, "
        "or within twice the max abs between the two resumes of that same "
        "checkpoint. The factor of two is the existing continuation-noise rule. "
        "Tolerances are not widened. Independent continuous runs are reported "
        "separately and are not the resume denominator."
    ),
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _git_meta() -> dict:
    def run(args: list[str]) -> str:
        completed = subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)
        return completed.stdout.strip()

    return {
        "head": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": run(["git", "status", "--short"]),
    }


def _section_gap(report: dict, key: str) -> float:
    section = report.get(key) or {}
    if section.get("missing") or section.get("integer_mismatch") or section.get("step_mismatch"):
        return float("inf")
    return float(section.get("max_abs") or 0.0)


def _judge_pair(source_gap: dict, resume_gap: dict, atol: dict) -> dict:
    mapping = {
        "generator": atol["parameter"],
        "discriminator": atol["parameter"],
        "optim_g": atol["adam_moment"],
        "optim_d": atol["adam_moment"],
        "generator_bn": atol["bn_float"],
        "discriminator_bn": atol["bn_float"],
        "generator_grads": atol["accumulation_gradient"],
        "discriminator_grads": atol["accumulation_gradient"],
    }
    sections = {}
    ok = True
    for key, limit in mapping.items():
        noise = _section_gap(resume_gap, key)
        observed = _section_gap(source_gap, key)
        if noise == float("inf") or observed == float("inf"):
            passed = False
        elif observed <= limit or observed == 0.0:
            passed = True
        else:
            passed = noise > 0.0 and observed <= noise * 2.0
        sections[key] = {"source": observed, "resume_pair": noise, "atol": limit, "pass": passed}
        ok = ok and passed
    discrete = {
        "global_step": bool(source_gap.get("global_step_equal")),
        "data_state": bool(source_gap.get("data_state_equal")),
        "rng_by_rank": bool((source_gap.get("rng_by_rank") or {}).get("ok")),
    }
    return {"pass": ok and all(discrete.values()), "sections": sections, "discrete": discrete}


def _copy_tree_files(directory: Path, destination: Path, names: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = directory / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _live_names(directory: Path) -> list[str]:
    names = [
        "step_00000002.pt",
        "step_00000003.pt",
        "metrics.jsonl",
        "forward-rank0.jsonl",
        "forward-rank1.jsonl",
        "rank-delta-0.jsonl",
        "rank-delta-1.jsonl",
        "epoch-rank0.jsonl",
        "epoch-rank1.jsonl",
    ]
    return [name for name in names if (directory / name).is_file()]


def _restore_worker() -> None:
    import torch
    import torch.distributed as dist

    from src.training.core.checkpoint import load_checkpoint
    from src.training.core.distributed import setup_process
    from src.training.core.rng import capture_rank_rng, restore_rank_rng, resume_rng_plan, set_determinism
    from src.training.se.config import load_training_config
    from src.training.se.run import build_task

    config = load_training_config(os.environ["P10C_CONFIG"])
    set_determinism(bool(config["deterministic"]), cuda=True)
    process = setup_process("cuda")
    task = build_task(config, process.device)
    payload = load_checkpoint(Path(os.environ["P10C_RESUME"]), process.device)
    task.load_state_dict(payload["task"])
    kind, state = resume_rng_plan(
        payload.get("rng"),
        payload.get("rng_by_rank"),
        rank=process.rank,
        world_size=process.world_size,
    )
    if kind == "legacy":
        raise SystemExit("restore-check expected per-rank RNG")
    restore_rank_rng(state, cuda=True)
    current = capture_rank_rng(cuda=True)

    def flat_tensors(value, prefix=""):
        found = {}
        if torch.is_tensor(value):
            found[prefix or "value"] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                found.update(flat_tensors(item, child))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                found.update(flat_tensors(item, f"{prefix}[{index}]"))
        return found

    def exact(left, right) -> dict:
        missing = sorted(set(left) ^ set(right))
        worst = 0.0
        worst_name = None
        for name in sorted(set(left) & set(right)):
            a = left[name]
            b = right[name]
            if not torch.is_tensor(a) or not torch.is_tensor(b):
                continue
            if not torch.is_floating_point(a):
                if not torch.equal(a.cpu(), b.cpu()):
                    worst = float("inf")
                    worst_name = name
                continue
            value = float((a.detach().cpu().float() - b.detach().cpu().float()).abs().max())
            if value > worst:
                worst = value
                worst_name = name
        return {"max_abs": worst, "worst": worst_name, "missing": missing, "exact": worst == 0.0 and not missing}

    import numpy as np

    task_state = task.state_dict()
    saved_task = payload["task"]
    saved_acc = (saved_task.get("accumulation") or {})
    live_acc = (task_state.get("accumulation") or {})
    report = {
        "rank": process.rank,
        "generator": exact(saved_task["generator"], task_state["generator"]),
        "discriminator": exact(saved_task["discriminator"], task_state["discriminator"]),
        "optim_g": exact(flat_tensors(saved_task["optim_g"]), flat_tensors(task_state["optim_g"])),
        "optim_d": exact(flat_tensors(saved_task["optim_d"]), flat_tensors(task_state["optim_d"])),
        "generator_grads": exact(saved_acc.get("generator_grads") or {}, live_acc.get("generator_grads") or {}),
        "discriminator_grads": exact(saved_acc.get("discriminator_grads") or {}, live_acc.get("discriminator_grads") or {}),
        "rng_python": current["python"] == state["python"],
        "rng_numpy": bool(np.array_equal(current["numpy"][1], state["numpy"][1]) and current["numpy"][2] == state["numpy"][2]),
        "rng_torch": bool(torch.equal(current["torch"].cpu(), state["torch"].cpu())),
        "rng_cuda": bool(torch.equal(current["cuda"].cpu(), state["cuda"].cpu())),
        "cuda_device": [state.get("cuda_device"), current.get("cuda_device")],
    }
    out = Path(os.environ["P10C_OUT_RANK"])
    _write(out, report)
    dist.destroy_process_group()


def _launch_restore(fix2b, config: dict, checkpoint: Path, destination: Path) -> dict:
    config_path = destination.parent / f"{destination.name}.yaml"
    fix2b._dump_config(config_path, config)
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
            "CUDA_VISIBLE_DEVICES": GPUS,
            "CUDA_LAUNCH_BLOCKING": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "P10C_CONFIG": str(config_path),
            "P10C_RESUME": str(checkpoint),
            "P10C_PHASE": "restore",
        }
    )
    log_path = OUT / "logs" / f"{destination.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / ".venvs" / "se" / "bin" / "python"),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(Path(__file__).resolve()),
        "restore",
        str(destination),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "returncode": completed.returncode,
        "rank0": json.loads((destination / "rank0.json").read_text()) if (destination / "rank0.json").is_file() else None,
        "rank1": json.loads((destination / "rank1.json").read_text()) if (destination / "rank1.json").is_file() else None,
    }


def _run_mode(fix2b, mode: str, results: dict) -> None:
    root = OUT / mode
    extra = {"P10C_DUMP_FORWARD": "1", "P10C_LOG_EPOCH": "1"}
    runs = {}
    for name in ("continuous-a", "continuous-b"):
        directory = root / name
        config = fix2b._base_config(
            directory,
            deterministic=(mode == "true"),
            verification_max_updates=3,
            checkpoint_interval=1,
        )
        print(f"phase {mode} {name}", flush=True)
        runs[name] = fix2b._run_train(f"{mode}-{name}", config, extra, timeout=1800)
        _write(OUT / "numerical.json", results)
    source = root / "continuous-a"
    kept = source / "continuous-kept"
    _copy_tree_files(source, kept, _live_names(source) + ["step_00000001.pt"])
    resumes = {}
    config = fix2b._base_config(
        source,
        deterministic=(mode == "true"),
        verification_max_updates=3,
        checkpoint_interval=1,
    )
    for name in ("resume-1", "resume-2"):
        for stale in ("step_00000002.pt", "step_00000003.pt", "interrupted.json", "complete.json"):
            path = source / stale
            if path.is_file():
                path.unlink()
        for pattern in ("forward-rank*.jsonl", "rank-delta-*.jsonl", "epoch-rank*.jsonl", "grad-rank*.jsonl"):
            for path in source.glob(pattern):
                path.unlink()
        if (kept / "metrics.jsonl").is_file():
            shutil.copy2(kept / "metrics.jsonl", source / "metrics.jsonl")
        print(f"phase {mode} {name}", flush=True)
        launch = fix2b._run_train(f"{mode}-{name}", config, extra, timeout=1800, resume=source / "step_00000001.pt")
        slot = source / name
        _copy_tree_files(source, slot, _live_names(source))
        resumes[name] = launch
        _write(OUT / "numerical.json", results)
    step = "step_00000003.pt"
    step2 = "step_00000002.pt"
    compare = fix2b.compare_checkpoints
    control = compare(root / "continuous-a" / "continuous-kept" / step, root / "continuous-b" / step)
    source_vs_resume = compare(root / "continuous-a" / "continuous-kept" / step, source / "resume-1" / step)
    resume_pair = compare(source / "resume-1" / step, source / "resume-2" / step)
    first = compare(root / "continuous-a" / "continuous-kept" / step2, source / "resume-1" / step2)

    def _jsonl(path: Path) -> list:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    forwards = {}
    for rank in (0, 1):
        continuous_rows = _jsonl(kept / f"forward-rank{rank}.jsonl")
        resume_rows = _jsonl(source / "resume-1" / f"forward-rank{rank}.jsonl")
        second = next((row for row in continuous_rows if row.get("update_ordinal") == 2), None)
        resumed = next((row for row in resume_rows if row.get("update_ordinal") == 1), None)
        forwards[str(rank)] = {
            "continuous_update_2": None if second is None else second.get("sha256"),
            "resume_update_1": None if resumed is None else resumed.get("sha256"),
            "equal": second is not None and resumed is not None and second.get("sha256") == resumed.get("sha256"),
        }
    results[mode] = {
        "runs": runs,
        "resumes": resumes,
        "continuous_vs_continuous_step3": control,
        "source_vs_resume1_step3": source_vs_resume,
        "resume1_vs_resume2_step3": resume_pair,
        "source_vs_resume1_step2": first,
        "judgment_step3": _judge_pair(source_vs_resume, resume_pair, PLAN["atol"]),
        "judgment_step2": _judge_pair(first, compare(source / "resume-1" / step2, source / "resume-2" / step2), PLAN["atol"]),
        "first_forward": forwards,
    }
    _write(OUT / "numerical.json", results)


def main() -> None:
    os.environ["P10C_OUT"] = str(OUT)
    os.environ["P10C_GPUS"] = GPUS
    import tests.training.p10c_fix2b_gpu as fix2b

    fix2b.OUT = OUT
    fix2b.GPUS = GPUS
    OUT.mkdir(parents=True, exist_ok=True)
    meta = _git_meta()
    _write(OUT / "plan.json", {"plan": PLAN, "git": meta, "gpus": GPUS})
    results: dict = {"plan": PLAN, "git": meta}
    _write(OUT / "numerical.json", results)
    for mode in ("false", "true"):
        try:
            _run_mode(fix2b, mode, results)
        except Exception as exc:
            results[mode] = {"error": str(exc)}
            _write(OUT / "numerical.json", results)
            print(f"mode {mode} failed: {exc}", flush=True)
    source = OUT / "false" / "continuous-a" / "step_00000001.pt"
    if source.is_file():
        config = fix2b._base_config(
            OUT / "false" / "continuous-a",
            deterministic=False,
            verification_max_updates=3,
            checkpoint_interval=1,
        )
        print("phase restore-only", flush=True)
        results["restore_only"] = _launch_restore(fix2b, config, source, OUT / "restore-only")
        _write(OUT / "numerical.json", results)
    print("phase ops", flush=True)
    fix2b.OUT = OUT / "ops"
    fix2b.GPUS = GPUS
    os.environ["P10C_OUT"] = str(OUT / "ops")
    fix2b.driver()
    print("done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        os.environ["P10C_OUT_RANK"] = str(Path(sys.argv[2]) / f"rank{os.environ.get('RANK', '0')}.json")
        _restore_worker()
    else:
        main()
