"""Start one experiment, watch its processes, and record how it ended.

The same command runs on a lab GPU server, on a workstation, and inside a
TSUBAME ``qsub`` / ``qrsh`` job script. It does not submit jobs, choose GPUs,
retry, or resubmit. The caller decides ``CUDA_VISIBLE_DEVICES`` and the number
of processes.

- One process runs ``python -m src.experiments.cli``.
- ``--nproc N`` runs ``torchrun --standalone --max-restarts 0`` with ``N``
  ranks. torchrun ends every rank when one of them exits with an error, so a
  subset of ranks never trains on.

The training processes report an outcome and leave the verdict to the
launcher. ``COMPLETED`` and ``complete.json`` are written only after every
process has exited with code 0. A safe stop is ``INTERRUPTED``. Anything else
is ``FAILED``, with the exit code and the error each rank recorded.

``SIGTERM``, ``SIGINT``, ``SIGHUP``, and ``--stop-after`` ask the trainer to
stop at its next safe point, where it saves and exits. A request made while
the processes are still starting is delivered when training begins, so it
never kills a rank that is not listening yet. A second signal, or ``--grace``
seconds without an exit, kills every process and is ``FAILED``.

``launch.json`` and ``failure.rank<N>.json`` describe the last launch.
``status.json`` is rewritten only when the training of this launch started,
so a refused launch leaves the record of the earlier run as it was.
``gpu_memory.jsonl`` is the whole-device ``nvidia-smi`` series for this
launch. A failed sample does not change the verdict.

Exit code: 0 ``COMPLETED``, 3 ``INTERRUPTED``, 1 ``FAILED``, 2 usage.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.execution.status import (
    clear_failures,
    now,
    process_alive,
    process_identity,
    process_match,
    process_start,
    read_failures,
    read_status,
    replace_text,
    write_complete,
    write_status,
)

EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_INTERRUPTED = 3
LAUNCH_NAME = "launch.json"
LAUNCH_LOCK = ".launch.lock"
LAUNCH_FORMAT = "audio-db-launch-v1"
HEARTBEAT_SECONDS = 15.0
# A launch on another host is taken as alive while its heartbeat is newer than this.
STALE_SECONDS = 120.0
_POLL_SECONDS = 0.2
_SCAN_SECONDS = 1.0
_PR_SET_PDEATHSIG = 1
# Set by torchrun for each rank. A value inherited from the caller would describe another job.
_RANK_ENVIRONMENT = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "LOCAL_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.launcher",
        description="Run an audio-db experiment under supervision",
    )
    parser.add_argument("--config", required=True, help="audio-db-experiment/v1 YAML")
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint of this experiment to continue, or its output directory to continue from latest.pt",
    )
    parser.add_argument("--nproc", type=int, default=1, help="number of training processes (DDP ranks)")
    parser.add_argument(
        "--stop-after",
        type=float,
        default=None,
        help="seconds after which the trainer is asked to stop at its next safe point",
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=300.0,
        help="seconds a requested stop may take before every process is killed",
    )
    parser.add_argument(
        "--gpu-memory-interval",
        type=float,
        default=None,
        help=(
            "seconds between nvidia-smi whole-device memory samples "
            "(default 10, or AUDIO_DB_GPU_MEMORY_INTERVAL; 0 disables)"
        ),
    )
    args = parser.parse_args(argv)
    if args.nproc < 1:
        parser.error("--nproc must be >= 1")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    if args.grace <= 0:
        parser.error("--grace must be positive")
    from src.experiments.execution.gpu_memory import resolve_interval

    try:
        args.gpu_memory_interval = resolve_interval(args.gpu_memory_interval, os.environ)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return _launch(args)
    except _Refused as exc:
        print(f"launcher: {exc}", file=sys.stderr)
        return EXIT_FAILED


class _Refused(Exception):
    """The launch did not start. Nothing in the experiment directory was changed."""


class _Stop:
    """Stop request shared with the signal handlers. Two signals mean: do not wait for a safe point."""

    def __init__(self) -> None:
        self.reason: str | None = None
        self.requested_at: float | None = None
        self.signals = 0

    def request(self, reason: str, *, signalled: bool = False) -> None:
        if signalled:
            self.signals += 1
        if self.requested_at is None:
            self.reason = reason
            self.requested_at = time.monotonic()

    @property
    def force(self) -> bool:
        return self.signals >= 2


def _launch(args: argparse.Namespace) -> int:
    from src.experiments.api import output_directory
    from src.experiments.config import load_experiment
    from src.experiments.errors import ExperimentError
    from src.experiments.execution.checkpoint import DirectoryLock, assert_output_allowed

    try:
        directory = output_directory(load_experiment(args.config))
        assert_output_allowed(directory)
        directory.mkdir(parents=True, exist_ok=True)
    except (ExperimentError, OSError) as exc:
        raise _Refused(str(exc)) from exc
    try:
        own = DirectoryLock(directory, LAUNCH_LOCK)
    except ExperimentError as exc:
        raise _Refused(f"another launcher is supervising {directory}") from exc
    try:
        # A training process that outlived its launcher, or one started without it.
        try:
            DirectoryLock(directory).release()
        except ExperimentError as exc:
            raise _Refused(str(exc)) from exc
        clear_failures(directory)
        command = _command(args)
        record: dict[str, Any] = {
            "format": LAUNCH_FORMAT,
            "launch_id": uuid.uuid4().hex,
            "state": "RUNNING",
            **process_identity(),
            "world_size": args.nproc,
            "command": command,
            "started_at": now(),
            "heartbeat_at": now(),
        }
        _write_launch(directory, record)
        stop = _Stop()
        from src.experiments.execution.gpu_memory import GpuMemoryLog

        memory = GpuMemoryLog.start(directory, float(args.gpu_memory_interval), record["launch_id"])
        previous = {
            number: signal.signal(
                number, lambda signum, frame: stop.request(signal.Signals(signum).name, signalled=True)
            )
            for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        }
        try:
            code, killed = _supervise(command, args, stop, directory, record)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            memory.close()
        return _finish(directory, record, code, killed, stop)
    finally:
        own.release()


def _command(args: argparse.Namespace) -> list[str]:
    training = ["-m", "src.experiments.cli", "--config", str(args.config)]
    if args.resume is not None:
        training.extend(["--resume", str(args.resume)])
    if args.nproc == 1:
        return [sys.executable, *training]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--max-restarts",
        "0",
        "--nproc-per-node",
        str(args.nproc),
        *training,
    ]


def _supervise(
    command: list[str],
    args: argparse.Namespace,
    stop: _Stop,
    directory: Path,
    record: dict[str, Any],
) -> tuple[int, bool]:
    """Run the command until it exits. Return its exit code and whether the launcher killed anything."""

    from src.experiments.api import SUPERVISED_ENV

    env = os.environ.copy()
    env[SUPERVISED_ENV] = record["launch_id"]
    for name in _RANK_ENVIRONMENT:
        env.pop(name, None)
    started = time.monotonic()
    # Its own session: a terminal Ctrl-C reaches the launcher only, which asks for a safe stop.
    child = subprocess.Popen(command, env=env, start_new_session=True, preexec_fn=_die_with_parent)
    record["child_pid"] = child.pid
    # pid -> kernel start time. A pid alone could be another program after the rank has exited.
    # The direct child is recorded before the loop so a killed launcher still names torchrun or the rank.
    workers: dict[int, str | None] = {child.pid: process_start(child.pid)}
    _remember_processes(record, workers, child.pid)
    try:
        _write_launch(directory, record)
    except OSError:
        pass
    signalled: set[int] = set()
    heartbeat = started
    # The first pass records ranks immediately. Later passes are once a second.
    scanned = started - _SCAN_SECONDS
    listening = False
    killed = False
    while child.poll() is None:
        current = time.monotonic()
        if stop.requested_at is None and args.stop_after is not None and current - started >= args.stop_after:
            stop.request("stop_after")
        announce = stop.requested_at is not None and "stop_requested_at" not in record
        changed = False
        # Keep scanning until torchrun's ranks have been named. After that, once a second.
        expected = args.nproc if args.nproc == 1 else args.nproc + 1
        if announce or len(workers) < expected or current - scanned >= _SCAN_SECONDS:
            scanned = current
            # Ranks are children of torchrun. The direct child stays even when it is not its own child.
            found = [child.pid] if args.nproc == 1 else [child.pid, *_children(child.pid)]
            for pid in found:
                if pid not in workers or workers[pid] is None:
                    started_at = process_start(pid)
                    if workers.get(pid) != started_at:
                        workers[pid] = started_at
                        changed = True
        if changed:
            _remember_processes(record, workers, child.pid)
            try:
                _write_launch(directory, record)
            except OSError:
                pass
        if stop.requested_at is not None:
            if announce:
                record["stop_requested_at"] = now()
                record["stop_reason"] = stop.reason
                print(f"launcher: stop requested ({stop.reason}); waiting for a safe point", file=sys.stderr)
            if not listening:
                # Before RUNNING a rank may still have the default SIGTERM action, which would kill it.
                reported = read_status(directory) or {}
                listening = reported.get("status") == "RUNNING" and reported.get("launch_id") == record["launch_id"]
            if listening:
                # torchrun would kill its ranks on SIGTERM, so only the ranks are signalled.
                # ``workers`` also holds torchrun itself; that pid is not a rank when nproc > 1.
                ranks = [pid for pid in workers if args.nproc == 1 or pid != child.pid]
                for pid in sorted(set(ranks) - signalled):
                    _signal(pid, signal.SIGTERM)
                    signalled.add(pid)
            if stop.force or current - stop.requested_at >= args.grace:
                record["kill_reason"] = (
                    "second signal" if stop.force else f"no exit within {args.grace:g} s of the stop request"
                )
                killed = True
                break
        if announce or current - heartbeat >= HEARTBEAT_SECONDS:
            heartbeat = current
            record["heartbeat_at"] = now()
            _remember_processes(record, workers, child.pid)
            try:
                _write_launch(directory, record)
            except OSError:
                pass
        time.sleep(_POLL_SECONDS)
    if killed:
        if args.nproc > 1:
            # Stopped, torchrun starts no rank between this last look and its death.
            _signal(child.pid, signal.SIGSTOP)
            for pid in _children(child.pid):
                workers.setdefault(pid, process_start(pid))
        _kill_group(child.pid)
    code = child.wait()
    # torchrun has ended its ranks by now. One that is still there lost its parent; it must not train on.
    orphans = {pid: at for pid, at in workers.items() if pid != child.pid and process_alive(pid, at)}
    for pid in sorted(orphans):
        if _kill_group(pid):
            killed = True
            record.setdefault("kill_reason", "a rank outlived torchrun")
    # A killed rank holds the directory lock until the kernel has torn it down.
    deadline = time.monotonic() + 30.0
    while any(process_alive(pid, at) for pid, at in orphans.items()) and time.monotonic() < deadline:
        time.sleep(_POLL_SECONDS)
    return code, killed


def _finish(directory: Path, record: dict[str, Any], code: int, killed: bool, stop: _Stop) -> int:
    """Write the verdict. No process of this launch is running when this is called."""

    from src.experiments.errors import ExperimentError
    from src.experiments.execution.checkpoint import DirectoryLock

    reported = read_status(directory) or {}
    # Without this launch's id the training never started: the record belongs to an earlier run.
    started = reported.get("launch_id") == record["launch_id"]
    outcome = reported.get("outcome") if started and reported.get("status") == "RUNNING" else None
    step = reported.get("logical_step") if started else None
    failures = read_failures(directory)
    if killed:
        state, reason = "FAILED", f"killed by the launcher: {record.get('kill_reason')}"
    elif code == 0 and outcome == "completed" and isinstance(step, int):
        state, reason = "COMPLETED", None
    elif code == 0 and outcome == "interrupted":
        state, reason = "INTERRUPTED", stop.reason or "stop requested inside the experiment"
    elif code == 0:
        state, reason = "FAILED", "every process exited with code 0 but no outcome was reported"
    else:
        state, reason = "FAILED", _describe_exit(code, failures)
    if state == "FAILED" and not killed and stop.reason is not None:
        reason = f"{reason}; a stop had been requested ({stop.reason})"
    if not started and state == "FAILED":
        reason = f"training did not start: {reason}"
    details: dict[str, Any] = {"exit_code": code, "world_size": record["world_size"], "launch_id": record["launch_id"]}
    if isinstance(step, int):
        details["logical_step"] = step
    if reason is not None:
        details["reason"] = reason
    if state == "FAILED":
        if failures:
            details["failures"] = failures
        if started and isinstance(reported.get("error"), str):
            details["error"] = reported["error"]
    if started:
        try:
            lock = DirectoryLock(directory)
        except ExperimentError:
            # Another run took the directory. Its files are not this launch's to rewrite.
            state, reason = "FAILED", "the experiment directory is in use by another run"
        else:
            try:
                if state == "COMPLETED":
                    write_complete(directory, int(step))
                write_status(directory, state, **details)
            finally:
                lock.release()
    record.update({"state": state, "finished_at": now(), "heartbeat_at": now(), "exit_code": code})
    if reason is not None:
        record["reason"] = reason
    if failures:
        record["failures"] = failures
    _write_launch(directory, record)
    summary = f"launcher: {state}"
    if isinstance(step, int):
        summary += f" logical_step={step}"
    if reason is not None:
        summary += f" ({reason})"
    print(summary, file=sys.stderr)
    return {"COMPLETED": EXIT_COMPLETED, "INTERRUPTED": EXIT_INTERRUPTED}.get(state, EXIT_FAILED)


def _describe_exit(code: int, failures: list[dict[str, Any]]) -> str:
    if code < 0:
        try:
            text = f"terminated by {signal.Signals(-code).name}"
        except ValueError:
            text = f"terminated by signal {-code}"
    else:
        text = f"exit code {code}"
    if failures:
        first = failures[0]
        return f"{text}; rank {first.get('rank')} reported {first.get('kind')}: {first.get('error')}"
    return f"{text}; no rank recorded a Python error, so a process was killed or died outside Python"


def liveness(directory: Path) -> dict[str, str]:
    """Tell a live ``RUNNING`` from one left behind by a dead process.

    ``state`` is ``alive``, ``stale``, ``finished``, ``absent``, or
    ``unknown``. On this host a run is alive while any recorded process
    (the launcher, its child, a rank, or the process that wrote
    ``status.json``) is still that same process. One rank exiting does not
    make the launch stale while torchrun or another rank is running. A pid
    reused by a different process is not that launch. A launch on another
    host is alive while its heartbeat is recent. A run on another host
    without the launcher cannot be checked from here.
    """

    status = read_status(directory)
    if status is None:
        return {"state": "absent", "reason": "status.json is missing or unreadable"}
    if status.get("status") != "RUNNING":
        return {"state": "finished", "reason": f"status is {status.get('status')}"}
    launch = read_launch(directory)
    host = socket.gethostname()
    local = _local_identities(launch, status, host)
    if local:
        seen = [(row.get("pid"), process_match(row.get("pid"), row.get("process_start"))) for row in local]
        alive = [pid for pid, state in seen if state == "alive"]
        if alive:
            return {"state": "alive", "reason": f"pid {alive[0]} is running on {host}"}
        unknown = [pid for pid, state in seen if state == "unknown"]
        if unknown:
            return {
                "state": "unknown",
                "reason": f"pid {unknown[0]} on {host} exists but its start time could not be confirmed",
            }
        pids = ", ".join(str(pid) for pid, _state in seen)
        return {"state": "stale", "reason": f"pid {pids} is not running on {host}; the run ended without a record"}
    if launch is not None and launch.get("state") == "RUNNING":
        age = _age_seconds(launch.get("heartbeat_at"))
        if age is not None and age < STALE_SECONDS:
            return {"state": "alive", "reason": f"launcher heartbeat {age:.0f} s ago on {launch.get('host')}"}
        shown = "unreadable" if age is None else f"{age:.0f} s old"
        return {"state": "stale", "reason": f"launcher heartbeat on {launch.get('host')} is {shown}"}
    return {
        "state": "unknown",
        "reason": f"started on {status.get('host')} without the launcher; check pid {status.get('pid')} there",
    }


def _local_identities(launch: dict[str, Any] | None, status: dict[str, Any], host: str) -> list[dict[str, Any]]:
    """Process records on ``host``. A launch's ``processes`` list includes every supervised pid."""

    found: list[dict[str, Any]] = []
    if launch is not None and launch.get("state") == "RUNNING" and launch.get("host") == host:
        rows = launch.get("processes")
        listed = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        found.extend(listed or [launch])
    if status.get("host") == host:
        found.append(status)
    return found


def _remember_processes(record: dict[str, Any], workers: dict[int, str | None], child_pid: int) -> None:
    """Store the launcher, its direct child, and ranks. Rank 0's pid alone is not the whole launch."""

    rows: list[dict[str, Any]] = [
        {"role": "launcher", "pid": record.get("pid"), "process_start": record.get("process_start")}
    ]
    for pid, started in sorted(workers.items()):
        rows.append(
            {
                "role": "child" if pid == child_pid else "worker",
                "pid": pid,
                "process_start": started,
            }
        )
    record["processes"] = rows


def read_launch(directory: Path) -> dict[str, Any] | None:
    try:
        record = json.loads((directory / LAUNCH_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _write_launch(directory: Path, record: dict[str, Any]) -> None:
    replace_text(directory / LAUNCH_NAME, json.dumps(record, indent=2, sort_keys=True) + "\n")


def _age_seconds(stamp: object) -> float | None:
    if not isinstance(stamp, str):
        return None
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if then.tzinfo is None:
        return None
    return (datetime.now(timezone.utc) - then).total_seconds()


def _die_with_parent() -> None:
    """Linux: deliver ``SIGTERM`` to the child when the launcher dies, so nothing trains unsupervised."""

    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
    except (AttributeError, OSError):
        pass


def _children(pid: int) -> list[int]:
    """Direct children of ``pid``: the ranks torchrun started."""

    proc = Path("/proc")
    if (proc / "self" / "stat").is_file():
        found: list[int] = []
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # ``pid (comm) state ppid ...``; comm may itself contain spaces and parentheses.
            fields = stat.rpartition(")")[2].split()
            if len(fields) >= 2 and fields[0] != "Z" and fields[1] == str(pid):
                found.append(int(entry.name))
        return found
    listing = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], check=False, capture_output=True, text=True)
    pairs = (line.split() for line in listing.stdout.splitlines())
    return [int(pair[0]) for pair in pairs if len(pair) == 2 and pair[1] == str(pid)]


def _signal(pid: int, number: signal.Signals) -> None:
    try:
        os.kill(pid, number)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_group(pid: int) -> bool:
    """``SIGKILL`` the process group led by ``pid``. Return whether there was one to kill."""

    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
