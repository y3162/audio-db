"""Run the audio-db unittest suites with a bounded process pool.

CPU modules hide CUDA. GPU targets run only on devices the caller named or
on devices that are idle inside the parent's ``CUDA_VISIBLE_DEVICES``.
An empty value does not widen back to the machine. A CPU pass is not a GPU pass.

The runner first makes the production data read-only for itself and for every
test process (``tests.suite_guard``). Where that is not possible it refuses to
run unless ``AUDIO_DB_TEST_UNGUARDED=1``. No test writes below ``data/``, so
modules need no ordering between them.

``unittest discover`` remains the single-directory, single-process command.
``python -m tests.suite_guard COMMAND`` puts it behind the same guard.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import suite_guard  # noqa: E402
from tests.suite_resources import (  # noqa: E402
    available_cpus,
    gpu_inventory_required,
    gpu_test_environment,
    module_workers,
    parse_busy_uuids,
    parse_gpu_rows,
    process_threads,
    select_test_gpus,
    thread_environment,
)

SUITES = (
    "tests/asr",
    "tests/data",
    "tests/evaluation",
    "tests/experiments",
    "tests/se",
    "tests/setup",
    "tests/numerics",
)

# Real-model and CUDA paths. They also appear in the CPU lane, where CUDA is
# hidden and the CUDA-only cases are NOT_RUN. This lane runs them again with
# a visible device. One module per GPU slot; the default slot count is 1.
GPU_TARGETS = (
    "tests.se.test_tf_locoformer.TFLocoformerMatchTests.test_cuda_enhance_when_a_gpu_is_free",
    "tests.numerics.test_df_reference_spectral.DeepFilterNetReferenceSpectralTest.test_cuda_deterministic_fallback_remains_finite",
    "tests.numerics.test_moss_probe_accumulation.MossProbeAccumulationTest.test_cuda_stft_uses_official_cpu_window",
    "tests.numerics.test_model_tasks",
    "tests.experiments.test_cuda_amp",
    "tests.experiments.test_launcher.LauncherCudaTests",
)

# Long subprocess-heavy modules start first to avoid leaving one worker busy
# after the rest of the suite has finished. Discovery still determines coverage.
SLOW_CPU_MODULES = (
    "tests.experiments.test_launcher",
    "tests.experiments.test_phase6_mossformergan_safe",
    "tests.experiments.test_distributed_data",
    "tests.experiments.test_mossformergan",
    "tests.experiments.test_semambapp",
    "tests.numerics.test_four_process",
    "tests.data.reproduction.test_p9_semamba",
    "tests.experiments.test_ddp",
)

_PRINT = threading.Lock()
_PROCS: list[subprocess.Popen[str]] = []
_PROCS_LOCK = threading.Lock()


def _track(proc: subprocess.Popen[str]) -> None:
    with _PROCS_LOCK:
        _PROCS.append(proc)


def _untrack(proc: subprocess.Popen[str]) -> None:
    with _PROCS_LOCK:
        if proc in _PROCS:
            _PROCS.remove(proc)


def _stop_children(_signum: int, _frame: object) -> None:
    with _PROCS_LOCK:
        procs = list(_PROCS)
    for proc in procs:
        _kill(proc)


def discover(root: Path = ROOT) -> list[str]:
    names: list[str] = []
    for suite in SUITES:
        base = root / suite
        for path in sorted(base.rglob("test*.py")):
            names.append(".".join(path.relative_to(root).with_suffix("").parts))
    return names


def module_file(root: Path, name: str) -> Path | None:
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = root.joinpath(*parts[:end]).with_suffix(".py")
        if candidate.is_file():
            return candidate
    return None


def _positive_int(text: str, label: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise SystemExit(f"{label} is invalid: {text!r}") from exc
    if value < 1:
        raise SystemExit(f"{label} must be >= 1, got {value}")
    return value


def _timeout_seconds() -> float:
    raw = os.environ.get("AUDIO_DB_TEST_TIMEOUT", "").strip()
    if not raw:
        return 1800.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"AUDIO_DB_TEST_TIMEOUT is invalid: {raw!r}") from exc
    if value <= 0:
        raise SystemExit(f"AUDIO_DB_TEST_TIMEOUT must be positive, got {value}")
    return value


def _gpu_slots() -> int:
    raw = os.environ.get("AUDIO_DB_TEST_GPU_SLOTS", "").strip()
    if not raw:
        return 1
    return _positive_int(raw, "AUDIO_DB_TEST_GPU_SLOTS")


def _parent_visible() -> str | None:
    """Parent ``CUDA_VISIBLE_DEVICES``, or None when the variable is unset."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    return os.environ["CUDA_VISIBLE_DEVICES"]


def _parent_visible_label(visible: str | None) -> str:
    if visible is None:
        return "unset"
    if not visible.strip():
        return "empty"
    return visible


def _read_gpu_query() -> tuple[list[dict[str, object]] | None, set[str], str | None]:
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, set(), "nvidia-smi is not available"
    if gpus.returncode != 0:
        detail = gpus.stderr.strip() or gpus.stdout.strip()
        return None, set(), f"nvidia-smi failed: {detail}"
    if apps.returncode != 0:
        detail = apps.stderr.strip() or apps.stdout.strip()
        return None, set(), f"nvidia-smi compute-apps failed: {detail}"
    try:
        rows = parse_gpu_rows(gpus.stdout)
    except ValueError as exc:
        return None, set(), str(exc)
    return rows, parse_busy_uuids(apps.stdout), None


def _query_gpus() -> tuple[list[int], str]:
    visible = _parent_visible()
    requested_text = os.environ.get("AUDIO_DB_TEST_GPUS", "").strip()
    requested = requested_text or None
    rows: list[dict[str, object]] | None = None
    busy: set[str] = set()
    error: str | None = None
    if gpu_inventory_required(visible, requested):
        rows, busy, error = _read_gpu_query()
    try:
        return select_test_gpus(
            visible=visible,
            requested=requested,
            rows=rows,
            busy_uuids=busy,
            query_error=error,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _kind(case: dict[str, str]) -> str:
    if case["status"] == "skip" and "NOT_RUN" in case.get("message", ""):
        return "not_run"
    return case["status"]


def _tail(path: Path, limit: int = 40) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def _kill(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _run_named(name: str, result_path: str) -> None:
    import unittest

    payload: dict[str, object] = {"name": name, "cases": [], "load_error": None, "seconds": 0.0}
    cases: list[dict[str, object]] = []
    started = time.perf_counter()
    stream = open(os.devnull, "w", encoding="utf-8")
    try:
        suite = unittest.defaultTestLoader.loadTestsFromName(name)

        class Recording(unittest.TextTestResult):
            def startTest(self, test: unittest.TestCase) -> None:
                self._t0 = time.perf_counter()
                super().startTest(test)

            def _save(self, test: unittest.TestCase, status: str, message: str) -> None:
                cases.append(
                    {
                        "id": test.id(),
                        "status": status,
                        "seconds": time.perf_counter() - getattr(self, "_t0", started),
                        "message": message,
                    }
                )

            def addSuccess(self, test: unittest.TestCase) -> None:
                self._save(test, "ok", "")
                super().addSuccess(test)

            def addFailure(self, test: unittest.TestCase, err: tuple) -> None:
                self._save(test, "fail", "".join(traceback.format_exception(*err)))
                super().addFailure(test, err)

            def addError(self, test: unittest.TestCase, err: tuple) -> None:
                self._save(test, "error", "".join(traceback.format_exception(*err)))
                super().addError(test, err)

            def addSkip(self, test: unittest.TestCase, reason: str) -> None:
                self._save(test, "skip", str(reason))
                super().addSkip(test, reason)

            def addExpectedFailure(self, test: unittest.TestCase, err: tuple) -> None:
                self._save(test, "expected_failure", "".join(traceback.format_exception(*err)))
                super().addExpectedFailure(test, err)

            def addUnexpectedSuccess(self, test: unittest.TestCase) -> None:
                self._save(test, "unexpected_success", "")
                super().addUnexpectedSuccess(test)

            def addSubTest(self, test: unittest.TestCase, subtest: unittest.TestCase, err: tuple | None) -> None:
                if err is not None:
                    status = "fail" if issubclass(err[0], test.failureException) else "error"
                    cases.append(
                        {
                            "id": subtest.id(),
                            "status": status,
                            "seconds": 0.0,
                            "message": "".join(traceback.format_exception(*err)),
                        }
                    )
                super().addSubTest(test, subtest, err)

        suite.run(Recording(stream, descriptions=False, verbosity=1))
    except BaseException:
        payload["load_error"] = traceback.format_exc()
    finally:
        stream.close()
    payload["cases"] = cases
    payload["seconds"] = time.perf_counter() - started
    Path(result_path).write_text(json.dumps(payload), encoding="utf-8")


def _spawn(name: str, result_path: Path, log_path: Path, env: dict[str, str]) -> tuple[subprocess.Popen[str], object]:
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name, "--result", str(result_path)],
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _track(proc)
    return proc, log


def _read_job(job: dict[str, object], *, timed_out: bool, code: int | None, timeout: float) -> dict[str, object]:
    path = Path(job["result"])
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {
                "name": job["name"],
                "cases": [],
                "load_error": "result JSON was truncated\n" + _tail(Path(job["log_path"])),
                "seconds": time.perf_counter() - float(job["start"]),
            }
    else:
        detail = _tail(Path(job["log_path"]))
        reason = f"timeout after {timeout:.0f}s" if timed_out else f"exited {code} before writing a result"
        payload = {
            "name": job["name"],
            "cases": [],
            "load_error": reason + ("\n" + detail if detail else ""),
            "seconds": time.perf_counter() - float(job["start"]),
        }
    if timed_out:
        payload["load_error"] = f"timeout after {timeout:.0f}s\n" + str(payload.get("load_error") or "")
    payload["name"] = job["name"]
    return payload


def _counts(payload: dict[str, object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in payload.get("cases") or []:
        kind = _kind(case)
        counts[kind] = counts.get(kind, 0) + 1
    if payload.get("load_error"):
        counts["load_error"] = 1
    return counts


def _print_module(label: str, payload: dict[str, object]) -> None:
    counts = _counts(payload)
    detail = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    with _PRINT:
        print(
            f"[{label}] {payload['name']} {float(payload.get('seconds') or 0):.1f}s {detail}",
            flush=True,
        )


def run_lane(
    names: list[str],
    *,
    label: str,
    slots: int,
    env: dict[str, str],
    timeout: float,
) -> tuple[list[dict[str, object]], float]:
    started = time.perf_counter()
    priority = {name: index for index, name in enumerate(SLOW_CPU_MODULES)} if label == "cpu" else {}
    pending = sorted(names, key=lambda name: priority.get(name, len(priority)))
    running: list[dict[str, object]] = []
    finished: list[dict[str, object]] = []
    directory = tempfile.TemporaryDirectory(prefix=f"audio-db-{label}-")
    root = Path(directory.name)
    try:
        while pending or running:
            while pending and len(running) < slots:
                name = pending.pop(0)
                token = str(len(finished) + len(running))
                result_path = root / f"{token}.json"
                log_path = root / f"{token}.log"
                proc, log = _spawn(name, result_path, log_path, env)
                running.append(
                    {
                        "name": name,
                        "proc": proc,
                        "log": log,
                        "result": result_path,
                        "log_path": log_path,
                        "start": time.perf_counter(),
                    }
                )
            time.sleep(0.05)
            still: list[dict[str, object]] = []
            now = time.perf_counter()
            for job in running:
                proc = job["proc"]
                assert isinstance(proc, subprocess.Popen)
                code = proc.poll()
                timed_out = code is None and now - float(job["start"]) > timeout
                if code is None and not timed_out:
                    still.append(job)
                    continue
                if timed_out:
                    _kill(proc)
                    code = None
                else:
                    proc.wait()
                _untrack(proc)
                job["log"].close()
                payload = _read_job(job, timed_out=timed_out, code=code, timeout=timeout)
                finished.append(payload)
                _print_module(label, payload)
            running = still
    finally:
        for job in running:
            proc = job["proc"]
            assert isinstance(proc, subprocess.Popen)
            _kill(proc)
            _untrack(proc)
            job["log"].close()
        directory.cleanup()
    return finished, time.perf_counter() - started


def guard_production_data() -> tuple[str, list[str]]:
    """Make the production data read-only for this process and its children.

    Returns the state and the guarded paths. ``read-only`` is the guarded
    state, ``off`` is the caller's ``AUDIO_DB_TEST_UNGUARDED=1``, and ``none``
    means there is no ``data/`` to guard. Must run before any thread starts.
    """
    if suite_guard.unguarded():
        return "off", []
    paths = suite_guard.production_paths(ROOT)
    if not paths:
        return "none", []
    if not suite_guard.protected(paths):
        suite_guard.enter(paths)
    return "read-only", [str(path) for path in paths]


def _problems(rows: list[dict[str, object]]) -> list[str]:
    found: list[str] = []
    for payload in rows:
        if payload.get("load_error"):
            found.append(f"{payload['name']}\n{payload['load_error']}")
        for case in payload.get("cases") or []:
            if _kind(case) in {"fail", "error", "unexpected_success"}:
                found.append(f"{case['id']}\n{case.get('message', '')}")
    return found


def _tally(rows: list[dict[str, object]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for payload in rows:
        for key, value in _counts(payload).items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _lane_summary(rows: list[dict[str, object]], seconds: float, *, extra: dict[str, object]) -> dict[str, object]:
    return {
        "seconds": seconds,
        "counts": _tally(rows),
        "modules": [
            {
                "name": payload["name"],
                "seconds": payload.get("seconds"),
                "counts": _counts(payload),
                "load_error": payload.get("load_error"),
                "cases": [
                    {
                        "id": case["id"],
                        "status": _kind(case),
                        "seconds": case.get("seconds"),
                        **(
                            {}
                            if _kind(case) == "ok"
                            else {"message": str(case.get("message") or "")[:2000]}
                        ),
                    }
                    for case in payload.get("cases") or []
                ],
            }
            for payload in rows
        ],
        **extra,
    }


def _print_failures(problems: list[str]) -> None:
    for problem in problems:
        lines = problem.splitlines()
        clipped = "\n".join(lines[:50])
        with _PRINT:
            print(clipped, flush=True)
            if len(lines) > 50:
                print(f"... {len(lines) - 50} more lines", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run audio-db CPU and GPU unittest lanes")
    parser.add_argument("--cpu-only", action="store_true", help="hide CUDA and skip the GPU lane")
    parser.add_argument("--list", action="store_true", help="print CPU module names and exit")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    parser.add_argument("--summary")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--gpus", help="comma-separated physical GPU indexes")
    parser.add_argument("--timeout", type=float, help="seconds per module")
    args = parser.parse_args(argv)
    if args.worker:
        if not args.result:
            raise SystemExit("--result is required with --worker")
        _run_named(args.worker, args.result)
        return 0
    if args.workers is not None:
        os.environ["AUDIO_DB_TEST_WORKERS"] = str(_positive_int(str(args.workers), "--workers"))
    if args.threads is not None:
        os.environ["AUDIO_DB_TEST_THREADS"] = str(_positive_int(str(args.threads), "--threads"))
    if args.gpus is not None:
        os.environ["AUDIO_DB_TEST_GPUS"] = args.gpus
    if args.timeout is not None:
        os.environ["AUDIO_DB_TEST_TIMEOUT"] = str(args.timeout)

    names = discover(ROOT)
    if args.list:
        for name in names:
            print(name)
        return 0

    try:
        guard, guarded = guard_production_data()
    except suite_guard.GuardError as exc:
        print(
            f"production data cannot be made read-only: {exc}. "
            f"Set {suite_guard.UNGUARDED_ENV}=1 to run the tests without the guard.",
            file=sys.stderr,
        )
        return 2
    workers = module_workers()
    timeout = _timeout_seconds()
    gpu_devices: list[int] = []
    gpu_reason = "cpu-only"
    if not args.cpu_only:
        gpu_devices, gpu_reason = _query_gpus()
    slots = _gpu_slots()
    threads = process_threads(workers, slots if gpu_devices else 0)
    parent_visible = _parent_visible_label(_parent_visible())
    os.environ[suite_guard.STATE_ENV] = guard
    cpu_env = os.environ.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    cpu_env.update(thread_environment(threads))
    cpu_env["AUDIO_DB_TEST_WORKERS"] = str(workers)
    print(
        f"host={socket.gethostname()} host_cpus={os.cpu_count()} available_cpus={available_cpus()} "
        f"cpu_workers={workers} threads={threads} timeout_s={timeout:.0f}",
        flush=True,
    )
    print(f"data_guard={guard} guarded_paths={len(guarded)}", flush=True)
    print(f"cpu_modules={len(names)} cuda_visible_devices=hidden", flush=True)
    if gpu_devices:
        print(
            f"gpu_lane devices={','.join(map(str, gpu_devices))} slots={slots} "
            f"targets={len(GPU_TARGETS)} source={gpu_reason} "
            f"parent_cuda_visible_devices={parent_visible}",
            flush=True,
        )
    else:
        print(f"gpu_lane=NOT_RUN {gpu_reason} parent_cuda_visible_devices={parent_visible}", flush=True)

    cpu_box: dict[str, object] = {}
    gpu_box: dict[str, object] = {}

    def run_cpu() -> None:
        try:
            rows, seconds = run_lane(names, label="cpu", slots=workers, env=cpu_env, timeout=timeout)
        except BaseException:
            cpu_box["error"] = traceback.format_exc()
            return
        cpu_box["rows"] = rows
        cpu_box["seconds"] = seconds

    def run_gpu() -> None:
        env = os.environ.copy()
        env.update(thread_environment(threads))
        env.update(gpu_test_environment(gpu_devices))
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            rows, seconds = run_lane(list(GPU_TARGETS), label="gpu", slots=slots, env=env, timeout=timeout)
        except BaseException:
            gpu_box["error"] = traceback.format_exc()
            return
        gpu_box["rows"] = rows
        gpu_box["seconds"] = seconds

    previous_int = signal.signal(signal.SIGINT, _stop_children)
    previous_term = signal.signal(signal.SIGTERM, _stop_children)
    try:
        wall_start = time.perf_counter()
        cpu_thread = threading.Thread(target=run_cpu, name="audio-db-cpu-lane")
        cpu_thread.start()
        gpu_thread = None
        if gpu_devices:
            gpu_thread = threading.Thread(target=run_gpu, name="audio-db-gpu-lane")
            gpu_thread.start()
        cpu_thread.join()
        if gpu_thread is not None:
            gpu_thread.join()
        wall = time.perf_counter() - wall_start
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)

    cpu_rows = list(cpu_box.get("rows") or [])
    summary = {
        "host": socket.gethostname(),
        "cpus": os.cpu_count(),
        "available_cpus": available_cpus(),
        "cpu_workers": workers,
        "threads": threads,
        "wall_seconds": wall,
        "data_guard": {"state": guard, "paths": guarded},
        "cpu": _lane_summary(
            cpu_rows,
            float(cpu_box.get("seconds") or 0.0),
            extra={"modules_scheduled": len(names)},
        ),
        "gpu": {
            "status": "run" if gpu_devices else "not_run",
            "reason": None if gpu_devices else gpu_reason,
            "devices": gpu_devices,
            "slots": slots if gpu_devices else 0,
            "parent_cuda_visible_devices": parent_visible,
            "source": gpu_reason if gpu_devices else None,
        },
    }
    if gpu_devices:
        summary["gpu"] = {
            **summary["gpu"],
            **_lane_summary(list(gpu_box.get("rows") or []), float(gpu_box.get("seconds") or 0.0), extra={}),
        }
    problems = _problems(cpu_rows)
    if cpu_box.get("error"):
        problems.append(f"cpu lane\n{cpu_box['error']}")
    if gpu_devices:
        problems.extend(_problems(list(gpu_box.get("rows") or [])))
        if gpu_box.get("error"):
            problems.append(f"gpu lane\n{gpu_box['error']}")
    print(
        f"wall_s={wall:.1f} cpu_s={float(cpu_box.get('seconds') or 0):.1f} "
        f"cpu_counts={summary['cpu']['counts']}",
        flush=True,
    )
    if gpu_devices:
        print(
            f"gpu_s={float(gpu_box.get('seconds') or 0):.1f} gpu_counts={summary['gpu'].get('counts', {})}",
            flush=True,
        )
    else:
        print(f"gpu_not_run={gpu_reason}", flush=True)
    print("cpu_lane_does_not_certify_gpu_lane", flush=True)
    if problems:
        print(f"problems={len(problems)}", flush=True)
        _print_failures(problems)
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
