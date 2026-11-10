"""Launch one Phase 5 readiness config and record host GPU memory.

This is a thin wrapper around ``python -m src.experiments.cli``. It does not
change the training step. World size is the number of ``--devices`` entries.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch one experiment and sample GPU memory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--devices", required=True, help="comma-separated physical GPU indexes")
    parser.add_argument("--python", default=".venvs/se/bin/python")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--grace",
        type=float,
        default=None,
        help="seconds before the timeout to send SIGTERM. Default is min(120, timeout/4). SIGKILL waits until the timeout.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--signal-after", type=float, default=None, help="send SIGTERM after this many seconds")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("devices is empty")
    config_path = Path(args.config)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(document["output"]["directory"])
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env["AUDIO_DB_RESOURCE_PROFILE"] = "1"
    env.pop("WORLD_SIZE", None)
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    python = str(Path(args.python))
    if len(devices) == 1:
        command = [python, "-m", "src.experiments.cli", "--config", str(config_path)]
    else:
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(len(devices)),
            "-m",
            "src.experiments.cli",
            "--config",
            str(config_path),
        ]
    if args.preflight:
        command.append("--preflight")
    if args.resume:
        command.extend(["--resume", args.resume])
    grace = effective_grace(float(args.timeout), args.grace)
    started = time.perf_counter()
    process = subprocess.Popen(command, env=env, start_new_session=True)
    samples: list[dict[str, object]] = []
    peak: dict[str, int] = {device: 0 for device in devices}
    signaled = False
    killed = False
    timed_out = False
    checkpoints_at_kill: list[str] = []
    deadline = started + args.timeout
    while process.poll() is None:
        now = time.perf_counter()
        elapsed = now - started
        action = termination_action(
            elapsed,
            timeout=float(args.timeout),
            grace=grace,
            signal_after=args.signal_after,
            signaled=signaled,
        )
        if action == "kill":
            timed_out = True
            killed = True
            checkpoints_at_kill = _checkpoint_names(output)
            os.killpg(process.pid, signal.SIGKILL)
            break
        if action == "term":
            os.killpg(process.pid, signal.SIGTERM)
            signaled = True
        usage = _memory(devices)
        samples.append({"seconds": round(elapsed, 3), "memory_mib": usage})
        for device, used in usage.items():
            peak[device] = max(peak[device], used)
        time.sleep(2)
    code = process.wait()
    report = {
        "checkpoints_at_kill": checkpoints_at_kill,
        "command": command,
        "config": str(config_path),
        "devices": devices,
        "exit_code": code,
        "grace_seconds": grace,
        "killed": killed,
        "output": str(output),
        "peak_memory_mib": peak,
        "samples": samples,
        "signaled": signaled,
        "timed_out": timed_out,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("exit_code", "wall_seconds", "peak_memory_mib", "timed_out", "output")}))
    return 0 if code == 0 and not timed_out else 1


def effective_grace(timeout: float, grace: float | None) -> float:
    """Return the SIGTERM lead time. It is always shorter than the hard timeout."""

    if timeout <= 0:
        raise SystemExit("timeout must be positive")
    chosen = min(120.0, timeout / 4.0) if grace is None else float(grace)
    if chosen <= 0 or chosen >= timeout:
        raise SystemExit("grace must be positive and shorter than timeout")
    return chosen


def termination_action(
    elapsed: float,
    *,
    timeout: float,
    grace: float,
    signal_after: float | None,
    signaled: bool,
) -> str | None:
    """Choose SIGKILL, SIGTERM, or no signal.

    SIGTERM is sent before the hard timeout. SIGKILL is used only after the
    grace that follows that request. Finished checkpoints are not deleted.
    """

    if elapsed >= timeout:
        return "kill"
    term_at = timeout - grace
    if signal_after is not None:
        term_at = min(term_at, float(signal_after))
    if not signaled and elapsed >= term_at:
        return "term"
    return None


def _checkpoint_names(output: Path) -> list[str]:
    if not output.is_dir():
        return []
    return sorted(path.name for path in output.glob("step_*.pt") if path.is_file())


def _memory(devices: list[str]) -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    found: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        index, used = [item.strip() for item in line.split(",")]
        if index in devices:
            found[index] = int(used)
    return found


if __name__ == "__main__":
    raise SystemExit(main())
