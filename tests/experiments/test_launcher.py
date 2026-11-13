"""The launcher: completion, failure of any rank, safe stop, and stale runs.

``LauncherTests`` runs on CPU with Gloo. ``LauncherCudaTests`` runs on two
idle GPUs with NCCL and is NOT_RUN when CUDA is hidden. Faults are injected
by ``tests.experiments.support.FaultTrainer``; the training math is the
standard loop, so a resumed run is compared with an uninterrupted one.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import yaml

from src.experiments.execution.checkpoint import DirectoryLock
from src.experiments.execution.status import process_identity, process_match, process_start
from src.experiments.launcher import liveness

ROOT = Path(__file__).resolve().parents[2]
_GPU0 = os.environ.get("AUDIO_DB_CUDA_GPU0", "0")
_GPU1 = os.environ.get("AUDIO_DB_CUDA_GPU1", "1")
STEPS = 4


class _Base(unittest.TestCase):
    """Shared checks. Subclasses set ``device`` and ``devices``."""

    device = "cpu"
    devices = ""
    _references: dict[int, dict[str, object]]
    _reference_root: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls._references = {}
        cls._reference_root = tempfile.TemporaryDirectory(prefix="audio-db-launcher-ref-")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._reference_root.cleanup()

    def directory(self) -> Path:
        return Path(self.enterContext(tempfile.TemporaryDirectory(prefix="audio-db-launcher-")))

    def yaml(self, directory: Path, **fault: object) -> Path:
        return _write_yaml(directory, device=self.device, **fault)  # type: ignore[arg-type]

    def launch(self, config: Path, *extra: str, nproc: int = 1, timeout: int = 240) -> subprocess.CompletedProcess[str]:
        return _launch(config, *extra, nproc=nproc, devices=self.devices, timeout=timeout)

    def start(self, config: Path, *extra: str, nproc: int = 1) -> _Running:
        running = _Running(config, *extra, nproc=nproc, devices=self.devices)
        self.addCleanup(running.cleanup)
        return running

    def reference(self, nproc: int) -> dict[str, object]:
        """Final checkpoint of an uninterrupted run with ``nproc`` ranks."""

        if nproc not in self._references:
            directory = Path(self._reference_root.name) / f"nproc{nproc}"
            directory.mkdir()
            done = self.launch(_write_yaml(directory, device=self.device, nproc=nproc), nproc=nproc)
            self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
            self._references[nproc] = _latest(directory / "run")
        return self._references[nproc]

    def assert_matches_reference(self, output: Path, nproc: int) -> None:
        expected = self.reference(nproc)
        actual = _latest(output)
        self.assertEqual(actual["logical_step"], STEPS)
        self.assertEqual(actual["optimizer_updates"], expected["optimizer_updates"])
        self.assertEqual(_max_abs(expected["states"], actual["states"]), 0.0)
        self.assertEqual(_max_abs(expected["rng_by_rank"], actual["rng_by_rank"]), 0.0)

    def assert_nothing_runs(self, directory: Path) -> None:
        """No rank of the launch is alive, and nothing holds the experiment directory."""

        for path in sorted(directory.glob("fault.pid*")):
            pid = int(path.read_text(encoding="utf-8"))
            self.assertIsNone(process_start(pid), f"{path.name}: pid {pid} is still running")
        DirectoryLock(directory / "run").release()

    def assert_failed(self, done: subprocess.CompletedProcess[str], directory: Path, *, world_size: int) -> dict:
        output = directory / "run"
        self.assertEqual(done.returncode, 1, done.stderr + done.stdout)
        status = _record(output / "status.json")
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["world_size"], world_size)
        self.assertFalse((output / "complete.json").exists())
        self.assertEqual(_record(output / "launch.json")["state"], "FAILED")
        self.assertEqual(liveness(output)["state"], "finished")
        self.assert_nothing_runs(directory)
        return status

    def resume_and_compare(self, config: Path, directory: Path, *, nproc: int) -> None:
        output = directory / "run"
        resumed = self.launch(config, "--resume", str(output), nproc=nproc)
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
        self.assertEqual(_record(output / "status.json")["status"], "COMPLETED")
        self.assertEqual(_record(output / "complete.json"), {"logical_step": STEPS})
        self.assertEqual(list(output.glob("failure.rank*.json")), [])
        self.assert_matches_reference(output, nproc)
        self.assert_nothing_runs(directory)


class LauncherTests(_Base):
    def test_single_process_completes_and_a_second_launch_changes_nothing(self) -> None:
        directory = self.directory()
        config = self.yaml(directory)
        done = self.launch(config)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        output = directory / "run"
        status = _record(output / "status.json")
        launch = _record(output / "launch.json")
        self.assertEqual((status["status"], status["logical_step"], status["world_size"]), ("COMPLETED", STEPS, 1))
        self.assertEqual(status["launch_id"], launch["launch_id"])
        self.assertEqual((launch["state"], launch["exit_code"]), ("COMPLETED", 0))
        self.assertEqual(_record(output / "complete.json"), {"logical_step": STEPS})
        self.assertEqual(liveness(output)["state"], "finished")
        self.assertFalse((output / "gpu_memory.jsonl").exists())
        self.assert_matches_reference(output, 1)

        # The directory holds a finished run. A launch that cannot start leaves its record as it was.
        before = _digests(output, "status.json", "complete.json", "latest.pt", "metrics.jsonl")
        again = self.launch(config)
        self.assertEqual(again.returncode, 1, again.stderr + again.stdout)
        self.assertIn("training did not start", again.stderr)
        self.assertEqual(_digests(output, "status.json", "complete.json", "latest.pt", "metrics.jsonl"), before)
        self.assertEqual(_record(output / "launch.json")["state"], "FAILED")

    def test_two_ranks_complete(self) -> None:
        directory = self.directory()
        done = self.launch(self.yaml(directory, nproc=2), nproc=2)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        output = directory / "run"
        status = _record(output / "status.json")
        self.assertEqual((status["status"], status["logical_step"], status["world_size"]), ("COMPLETED", STEPS, 2))
        self.assertEqual(_record(output / "complete.json"), {"logical_step": STEPS})
        self.assert_matches_reference(output, 2)
        self.assert_nothing_runs(directory)

    def test_completion_needs_every_rank_to_exit_cleanly(self) -> None:
        # Rank 0 trained to the end and reported it. Rank 1 then exits with an error code.
        directory = self.directory()
        config = self.yaml(directory, nproc=2, fault="exit_late", rank=1)
        done = self.launch(config, nproc=2)
        status = self.assert_failed(done, directory, world_size=2)
        self.assertIn("exit code", status["reason"])
        self.assertEqual(_latest(directory / "run")["logical_step"], STEPS)
        self.resume_and_compare(config, directory, nproc=2)

    def test_an_exception_on_one_rank_fails_the_job_and_resume_matches(self) -> None:
        directory = self.directory()
        config = self.yaml(directory, nproc=2, fault="exception", step=2, rank=1)
        started = time.monotonic()
        done = self.launch(config, nproc=2)
        self.assertLess(time.monotonic() - started, 120)
        status = self.assert_failed(done, directory, world_size=2)
        failures = {item["rank"]: item for item in status["failures"]}
        self.assertEqual(failures[1]["kind"], "exception")
        self.assertIn("injected failure on rank 1", failures[1]["error"])
        self.assertEqual(_latest(directory / "run")["logical_step"], 2)
        self.resume_and_compare(config, directory, nproc=2)

    def test_a_killed_rank_fails_the_job_and_resume_matches(self) -> None:
        for rank in (0, 1):
            with self.subTest(rank=rank):
                directory = self.directory()
                config = self.yaml(directory, nproc=2, fault="sigkill", step=2, rank=rank)
                started = time.monotonic()
                done = self.launch(config, nproc=2)
                self.assertLess(time.monotonic() - started, 120)
                status = self.assert_failed(done, directory, world_size=2)
                self.assertIn("exit code", status["reason"])
                self.assertEqual(_latest(directory / "run")["logical_step"], 2)
                self.resume_and_compare(config, directory, nproc=2)

    def test_a_killed_single_process_fails_and_resume_matches(self) -> None:
        directory = self.directory()
        config = self.yaml(directory, fault="sigkill", step=2)
        done = self.launch(config)
        status = self.assert_failed(done, directory, world_size=1)
        self.assertIn("SIGKILL", status["reason"])
        self.assertNotIn("failures", status)
        self.resume_and_compare(config, directory, nproc=1)

    def test_out_of_memory_is_recorded_as_cuda_oom(self) -> None:
        for nproc in (1, 2):
            with self.subTest(nproc=nproc):
                directory = self.directory()
                config = self.yaml(directory, nproc=nproc, fault="oom", step=1, rank=nproc - 1)
                done = self.launch(config, nproc=nproc)
                status = self.assert_failed(done, directory, world_size=nproc)
                failures = {item["rank"]: item for item in status["failures"]}
                self.assertEqual(failures[nproc - 1]["kind"], "cuda_oom")
                self.assertEqual(_latest(directory / "run")["logical_step"], 1)

    def test_a_stop_signal_saves_and_resume_matches(self) -> None:
        for nproc, number in ((1, signal.SIGINT), (2, signal.SIGTERM)):
            with self.subTest(nproc=nproc):
                directory = self.directory()
                config = self.yaml(directory, nproc=nproc, delay=2.0)
                output = directory / "run"
                running = self.start(config, nproc=nproc)
                _wait_for(lambda: (output / "latest.pt").is_file(), 120, "the first checkpoint", running)
                running.signal(number)
                self.assertEqual(running.wait(120), 3, running.output())
                status = _record(output / "status.json")
                self.assertEqual(status["status"], "INTERRUPTED")
                self.assertIn(status["logical_step"], range(1, STEPS))
                self.assertEqual(_latest(output)["logical_step"], status["logical_step"])
                self.assertFalse((output / "complete.json").exists())
                launch = _record(output / "launch.json")
                self.assertEqual((launch["state"], launch["stop_reason"]), ("INTERRUPTED", number.name))
                self.assertEqual(liveness(output)["state"], "finished")
                self.assert_nothing_runs(directory)
                self.resume_and_compare(config, directory, nproc=nproc)

    def test_a_stop_requested_during_startup_is_a_safe_stop(self) -> None:
        # The request arrives before the rank handles SIGTERM. It must wait, not kill the rank.
        for nproc in (1, 2):
            with self.subTest(nproc=nproc):
                directory = self.directory()
                config = self.yaml(directory, nproc=nproc, delay=1.0)
                output = directory / "run"
                done = self.launch(config, "--stop-after", "0.05", nproc=nproc)
                self.assertEqual(done.returncode, 3, done.stderr + done.stdout)
                status = _record(output / "status.json")
                self.assertEqual(status["status"], "INTERRUPTED")
                self.assertEqual(_record(output / "launch.json")["stop_reason"], "stop_after")
                self.assert_nothing_runs(directory)
                # The request is delivered when training begins: before the first step or during it.
                self.assertIn(status["logical_step"], (0, 1))
                if status["logical_step"] == 0:
                    # Nothing was trained, so there is nothing to resume: the experiment starts again.
                    self.assertFalse((output / "latest.pt").exists())
                    again = self.launch(config, nproc=nproc)
                    self.assertEqual(again.returncode, 0, again.stderr + again.stdout)
                    self.assert_matches_reference(output, nproc)
                else:
                    self.assertEqual(_latest(output)["logical_step"], 1)
                    self.resume_and_compare(config, directory, nproc=nproc)

    def test_a_stop_that_is_not_honoured_ends_in_a_kill(self) -> None:
        for nproc, how in ((2, "grace"), (1, "second signal")):
            with self.subTest(nproc=nproc, how=how):
                directory = self.directory()
                config = self.yaml(directory, nproc=nproc, fault="hang", step=2, rank=-1)
                output = directory / "run"
                running = self.start(config, "--grace", "3" if how == "grace" else "600", nproc=nproc)
                _wait_for(lambda: (directory / "fault").is_file(), 120, "the hang", running)
                running.signal(signal.SIGTERM)
                if how == "second signal":
                    _wait_for(
                        lambda: "stop_requested_at" in (_record(output / "launch.json") or {}),
                        30,
                        "the stop request",
                        running,
                    )
                    time.sleep(1.0)
                    self.assertIsNone(running.process.poll(), "the hung run ended on the first signal")
                    running.signal(signal.SIGTERM)
                self.assertEqual(running.wait(60), 1, running.output())
                status = _record(output / "status.json")
                self.assertEqual(status["status"], "FAILED")
                self.assertIn("killed by the launcher", status["reason"])
                self.assertIn("second signal" if how == "second signal" else "no exit within 3 s", status["reason"])
                self.assert_nothing_runs(directory)
                self.assertEqual(_latest(output)["logical_step"], 2)
                self.resume_and_compare(config, directory, nproc=nproc)

    def test_a_dead_launcher_leaves_a_running_record_that_is_reported_stale(self) -> None:
        for nproc in (1, 2):
            with self.subTest(nproc=nproc):
                directory = self.directory()
                config = self.yaml(directory, nproc=nproc, delay=2.0)
                output = directory / "run"
                running = self.start(config, nproc=nproc)
                _wait_for(lambda: (output / "latest.pt").is_file(), 120, "the first checkpoint", running)
                self.assertEqual(liveness(output)["state"], "alive")
                running.signal(signal.SIGKILL)
                # Exit is asynchronous. Sample twice and accept only a stable answer:
                # a descendant that is still the recorded process must be ``alive``,
                # and ``stale`` is required only once every descendant is gone.
                seen_live = False
                deadline = time.monotonic() + 60
                detail = ""
                while time.monotonic() < deadline:
                    before = _descendant_states(output)
                    info = liveness(output)
                    again = liveness(output)
                    after = _descendant_states(output)
                    detail = f"{info} / {again} states={before} -> {after}"
                    if not before or not after:
                        time.sleep(0.02)
                        continue
                    live = "alive" in before and "alive" in after
                    gone = "alive" not in before and "alive" not in after
                    gone = gone and "unknown" not in before and "unknown" not in after
                    if "alive" in before and info["state"] == "alive":
                        seen_live = True
                    if live:
                        self.assertEqual(info["state"], "alive", detail)
                        self.assertEqual(again["state"], "alive", detail)
                    elif gone and info["state"] == again["state"] == "stale":
                        break
                    elif gone and info["state"] == again["state"] == "alive":
                        self.fail(f"liveness stayed alive after the supervised processes exited\n{detail}")
                    time.sleep(0.02)
                else:
                    self.fail(f"liveness did not become stale\n{detail}")
                self.assertTrue(seen_live, f"no supervised process was observed alive\n{detail}")
                self.assertLess(running.wait(30), 0, running.output())
                self.assertEqual(liveness(output)["state"], "stale")
                self.assertEqual(_record(output / "status.json")["status"], "RUNNING")
                self.assertEqual(_record(output / "launch.json")["state"], "RUNNING")
                self.assertFalse((output / "complete.json").exists())
                self.assertIn(_latest(output)["logical_step"], range(1, STEPS + 1))
                self.resume_and_compare(config, directory, nproc=nproc)

    def test_a_second_launcher_is_refused_and_leaves_the_live_run_alone(self) -> None:
        directory = self.directory()
        config = self.yaml(directory, delay=2.0)
        output = directory / "run"
        running = self.start(config)
        _wait_for(lambda: (output / "latest.pt").is_file(), 120, "the first checkpoint", running)
        owner = _record(output / "launch.json")["launch_id"]
        second = self.launch(config, "--resume", str(output))
        self.assertEqual(second.returncode, 1, second.stderr + second.stdout)
        self.assertIn("another launcher", second.stderr)
        self.assertEqual(_record(output / "launch.json")["launch_id"], owner)
        self.assertEqual(_record(output / "status.json")["launch_id"], owner)
        self.assertEqual(liveness(output)["state"], "alive")
        self.assertEqual(running.wait(120), 0, running.output())
        self.assert_matches_reference(output, 1)

    def test_a_run_started_without_the_launcher_is_not_taken_over(self) -> None:
        directory = self.directory()
        config = self.yaml(directory, delay=2.0)
        output = directory / "run"
        direct = subprocess.Popen(
            [sys.executable, "-m", "src.experiments.cli", "--config", str(config)],
            cwd=ROOT,
            env=_environment(self.devices),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(direct.wait)
        self.addCleanup(direct.kill)
        _wait_for(lambda: (output / "latest.pt").is_file(), 120, "the first checkpoint", None)
        before = _digests(output, "status.json")
        refused = self.launch(config, "--resume", str(output))
        self.assertEqual(refused.returncode, 1, refused.stderr + refused.stdout)
        self.assertEqual(_digests(output, "status.json"), before)
        self.assertFalse((output / "launch.json").exists())
        self.assertEqual(liveness(output)["state"], "alive")
        direct.send_signal(signal.SIGTERM)
        self.assertEqual(direct.wait(120), 0)
        # Run directly, the process writes its own verdict.
        self.assertEqual(_record(output / "status.json")["status"], "INTERRUPTED")
        self.resume_and_compare(config, directory, nproc=1)

    def test_the_job_script_runs_the_launcher(self) -> None:
        script = ROOT / "scripts" / "experiments" / "job.sh"
        directory = self.directory()
        config = self.yaml(directory, nproc=2, delay=1.0)
        output = directory / "run"
        env = _environment(self.devices)
        env["AUDIO_DB_PYTHON"] = sys.executable
        usage = subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=False, capture_output=True, text=True)
        self.assertEqual(usage.returncode, 2, usage.stderr)
        self.assertFalse(output.exists())
        stopped = subprocess.run(
            ["bash", str(script), str(config), "2"],
            cwd=ROOT,
            env={**env, "AUDIO_DB_STOP_AFTER": "0.05", "AUDIO_DB_GRACE": "120"},
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        self.assertEqual(stopped.returncode, 3, stopped.stderr + stopped.stdout)
        self.assertEqual(_record(output / "status.json")["status"], "INTERRUPTED")
        # A stop before the first step saved nothing; after it, the run continues from its Latest.
        resume = {"AUDIO_DB_RESUME": str(output)} if (output / "latest.pt").exists() else {}
        done = subprocess.run(
            ["bash", str(script), str(config), "2"],
            cwd=ROOT,
            env={**env, **resume},
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        status = _record(output / "status.json")
        self.assertEqual((status["status"], status["world_size"]), ("COMPLETED", 2))
        self.assert_matches_reference(output, 2)

    def test_usage_errors_exit_2_before_anything_is_written(self) -> None:
        directory = self.directory()
        config = self.yaml(directory)
        for extra in (("--nproc", "0"), ("--stop-after", "0"), ("--grace", "-1"), ("--gpu-memory-interval", "-1")):
            with self.subTest(extra=extra):
                done = self.launch(config, *extra)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertFalse((directory / "run").exists())


class LivenessTests(unittest.TestCase):
    """``liveness`` on records written by hand. No process is started."""

    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="audio-db-liveness-")))
        self.me = process_identity()
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        self.dead = {"host": self.me["host"], "pid": dead.pid, "process_start": "1"}

    def write(self, name: str, record: dict[str, object]) -> None:
        (self.directory / name).write_text(json.dumps(record), encoding="utf-8")

    def state(self) -> str:
        return liveness(self.directory)["state"]

    def test_no_record_and_finished_records(self) -> None:
        self.assertEqual(self.state(), "absent")
        (self.directory / "status.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(self.state(), "absent")
        for status in ("COMPLETED", "INTERRUPTED", "FAILED"):
            self.write("status.json", {"status": status, **self.dead})
            self.assertEqual(self.state(), "finished")

    def test_running_on_this_host_follows_the_recorded_process(self) -> None:
        self.write("status.json", {"status": "RUNNING", **self.me})
        self.assertEqual(self.state(), "alive")
        self.write("status.json", {"status": "RUNNING", **self.dead})
        self.assertEqual(self.state(), "stale")
        # The pid exists but belongs to a later process.
        self.write("status.json", {"status": "RUNNING", **self.me, "process_start": "1"})
        self.assertEqual(self.state(), "stale")
        # A dead rank 0 under a live launcher is still a supervised run.
        self.write("status.json", {"status": "RUNNING", **self.dead})
        self.write("launch.json", {"state": "RUNNING", **self.me})
        self.assertEqual(self.state(), "alive")
        # A launch record that reached its verdict does not vouch for a later direct run.
        self.write("launch.json", {"state": "FAILED", **self.me})
        self.assertEqual(self.state(), "stale")

    def test_running_on_another_host_follows_the_launcher_heartbeat(self) -> None:
        elsewhere = {"host": socket.gethostname() + "-other", "pid": 1, "process_start": "1"}
        self.write("status.json", {"status": "RUNNING", **elsewhere})
        self.assertEqual(self.state(), "unknown")
        fresh = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.write("launch.json", {"state": "RUNNING", **elsewhere, "heartbeat_at": fresh})
        self.assertEqual(self.state(), "alive")
        old = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat(timespec="seconds")
        self.write("launch.json", {"state": "RUNNING", **elsewhere, "heartbeat_at": old})
        self.assertEqual(self.state(), "stale")
        self.write("launch.json", {"state": "RUNNING", **elsewhere, "heartbeat_at": "yesterday"})
        self.assertEqual(self.state(), "stale")

    def test_a_live_child_keeps_the_launch_alive_after_its_launcher_exits(self) -> None:
        # Rank 0 and the launcher are the records a reader used to check. A later rank
        # or torchrun is still this launch.
        self.write("status.json", {"status": "RUNNING", **self.dead})
        self.write(
            "launch.json",
            {
                "state": "RUNNING",
                "host": self.me["host"],
                "pid": self.dead["pid"],
                "process_start": self.dead["process_start"],
                "processes": [
                    {"role": "launcher", "pid": self.dead["pid"], "process_start": self.dead["process_start"]},
                    {"role": "child", "pid": self.me["pid"], "process_start": self.me["process_start"]},
                ],
            },
        )
        self.assertEqual(self.state(), "alive")
        self.write(
            "launch.json",
            {
                "state": "RUNNING",
                "host": self.me["host"],
                "pid": self.dead["pid"],
                "process_start": "1",
                "processes": [
                    {"role": "launcher", "pid": self.dead["pid"], "process_start": "1"},
                    {"role": "child", "pid": self.me["pid"], "process_start": "1"},
                ],
            },
        )
        self.assertEqual(self.state(), "stale")

    def test_a_live_pid_without_a_start_time_is_unknown(self) -> None:
        self.write("status.json", {"status": "RUNNING", "host": self.me["host"], "pid": self.me["pid"]})
        self.assertEqual(self.state(), "unknown")

    def test_a_zombie_is_not_a_running_launcher(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        start = process_start(proc.pid)
        self.assertIsNotNone(start)
        os.kill(proc.pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            zombie = _proc_state(proc.pid) in (None, "Z")
            if zombie:
                self.write(
                    "status.json",
                    {"status": "RUNNING", "host": self.me["host"], "pid": proc.pid, "process_start": start},
                )
                self.assertEqual(self.state(), "stale")
                return
            time.sleep(0.01)
        self.fail("the killed process did not become a zombie")


class LauncherCudaTests(_Base):
    """Two ranks on two GPUs with NCCL. NOT_RUN without two idle devices."""

    device = "cuda"
    devices = f"{_GPU0},{_GPU1}"

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("NOT_RUN: CUDA is not visible")
        if _GPU0 == _GPU1 or torch.cuda.device_count() < 2:
            raise unittest.SkipTest("NOT_RUN: need 2 visible CUDA devices")
        for index in range(2):
            free, total = torch.cuda.mem_get_info(index)
            used = (total - free) / (1024**2)
            if used > 1024:
                raise unittest.SkipTest(f"NOT_RUN: cuda:{index} is using {used:.0f} MiB")
        super().setUpClass()

    def test_two_gpu_run_completes_over_nccl(self) -> None:
        directory = self.directory()
        done = self.launch(self.yaml(directory, nproc=2), nproc=2)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        output = directory / "run"
        status = _record(output / "status.json")
        self.assertEqual((status["status"], status["logical_step"], status["world_size"]), ("COMPLETED", STEPS, 2))
        summary = json.loads((output / "gpu_memory.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(item["index"] for item in summary["peaks"]), [int(_GPU0), int(_GPU1)])
        self.assertGreaterEqual(summary["sample_count"], 1)
        self.assertIn("whole GPU", summary["metrics"]["memory_used_mib"])
        # A CUDA runtime with more than one rank refuses every backend but NCCL.
        runtime = _latest(output)["states"]["runtime"]
        self.assertEqual((str(runtime["device"])[:4], runtime["world_size"]), ("cuda", 2))
        self.assert_matches_reference(output, 2)
        self.assert_nothing_runs(directory)

    def test_a_failed_rank_fails_the_job_and_resume_matches(self) -> None:
        for fault, rank in (("exception", 1), ("sigkill", 1), ("sigkill", 0)):
            with self.subTest(fault=fault, rank=rank):
                directory = self.directory()
                config = self.yaml(directory, nproc=2, fault=fault, step=2, rank=rank)
                started = time.monotonic()
                done = self.launch(config, nproc=2)
                # torchrun ends the surviving rank; nothing waits on NCCL for good.
                self.assertLess(time.monotonic() - started, 150)
                status = self.assert_failed(done, directory, world_size=2)
                if fault == "exception":
                    failures = {item["rank"]: item for item in status["failures"]}
                    self.assertEqual(failures[rank]["kind"], "exception")
                self.assertEqual(_latest(directory / "run")["logical_step"], 2)
                self.resume_and_compare(config, directory, nproc=2)

    def test_a_real_cuda_out_of_memory_is_recorded(self) -> None:
        directory = self.directory()
        config = self.yaml(directory, nproc=2, fault="cuda_oom", step=2, rank=1)
        done = self.launch(config, nproc=2)
        status = self.assert_failed(done, directory, world_size=2)
        failures = {item["rank"]: item for item in status["failures"]}
        self.assertEqual(failures[1]["kind"], "cuda_oom")
        self.assertIn("out of memory", failures[1]["error"])
        self.resume_and_compare(config, directory, nproc=2)

    def test_a_stop_signal_saves_and_resume_matches(self) -> None:
        directory = self.directory()
        config = self.yaml(directory, nproc=2, delay=2.0)
        output = directory / "run"
        running = self.start(config, nproc=2)
        _wait_for(lambda: (output / "latest.pt").is_file(), 180, "the first checkpoint", running)
        running.signal(signal.SIGTERM)
        self.assertEqual(running.wait(180), 3, running.output())
        status = _record(output / "status.json")
        self.assertEqual(status["status"], "INTERRUPTED")
        self.assertEqual(_latest(output)["logical_step"], status["logical_step"])
        self.assert_nothing_runs(directory)
        self.resume_and_compare(config, directory, nproc=2)

    def test_hung_ranks_are_killed_after_the_grace_period(self) -> None:
        # NCCL collectives have no timeout. A stop that is not honoured must still end.
        directory = self.directory()
        config = self.yaml(directory, nproc=2, fault="hang", step=2, rank=1)
        output = directory / "run"
        running = self.start(config, "--grace", "5", nproc=2)
        _wait_for(lambda: (directory / "fault").is_file(), 180, "the hang", running)
        time.sleep(2.0)
        running.signal(signal.SIGTERM)
        self.assertEqual(running.wait(90), 1, running.output())
        status = _record(output / "status.json")
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("no exit within 5 s", status["reason"])
        self.assert_nothing_runs(directory)
        self.resume_and_compare(config, directory, nproc=2)


class _Running:
    """A launcher started in the background, with its output kept in files."""

    def __init__(self, config: Path, *extra: str, nproc: int, devices: str) -> None:
        self._log = tempfile.NamedTemporaryFile(prefix="audio-db-launcher-log-", mode="w+", encoding="utf-8")
        self.process = subprocess.Popen(
            _command(config, *extra, nproc=nproc),
            cwd=ROOT,
            env=_environment(devices),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            # Its own session, as under a job script: a signal reaches the launcher alone.
            start_new_session=True,
        )

    def signal(self, number: signal.Signals) -> None:
        self.process.send_signal(number)

    def wait(self, timeout: float) -> int:
        try:
            return self.process.wait(timeout)
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(f"the launcher did not exit within {timeout} s\n{self.output()}") from exc

    def output(self) -> str:
        self._log.flush()
        return Path(self._log.name).read_text(encoding="utf-8", errors="replace")

    def cleanup(self) -> None:
        if self.process.poll() is None:
            # Asks for a stop, then insists: the launcher kills every rank on the second signal.
            self.process.send_signal(signal.SIGTERM)
            time.sleep(0.5)
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._log.close()


def _write_yaml(
    directory: Path,
    *,
    device: str = "cpu",
    nproc: int = 1,
    fault: str = "none",
    step: int = 0,
    rank: int = 0,
    delay: float = 0.0,
) -> Path:
    distribute = "ddp" if nproc > 1 else "none"
    text = f"""schema: audio-db-experiment/v1
checkpoint:
  selector:
    class: src.experiments.selectors.StandardSelector
    kwargs:
      latest_every: 1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: {device}
    precision: fp32
    timeout_seconds: 60
output:
  directory: {directory / "run"}
components:
  model:
    role: module
    class: torch.nn.Linear
    kwargs:
      in_features: 8
      out_features: 8
    prepare: {{trainable: true, distribute: {distribute}}}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: 8
      examples: 32
      batch_size: 4
  optimizer:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: model
    kwargs: {{lr: 0.001}}
  trainer:
    role: trainer
    class: tests.experiments.support.FaultTrainer
    kwargs:
      max_logical_steps: {STEPS}
      accumulation: 1
      fault: {fault}
      step: {step}
      rank: {rank}
      delay: {delay}
      marker: {directory / "fault"}
    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""
    path = directory / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(text)
    return path


def _environment(devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.pop("AUDIO_DB_LAUNCH_ID", None)
    return env


def _command(config: Path, *extra: str, nproc: int) -> list[str]:
    return [sys.executable, "-m", "src.experiments.launcher", "--config", str(config), "--nproc", str(nproc), *extra]


def _launch(
    config: Path, *extra: str, nproc: int, devices: str, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(config, *extra, nproc=nproc),
        cwd=ROOT,
        env=_environment(devices),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _wait_for(predicate, timeout: float, what: str, running: _Running | None) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if running is not None and running.process.poll() is not None and not predicate():
            raise AssertionError(f"the launcher exited while waiting for {what}\n{running.output()}")
        time.sleep(0.05)
    raise AssertionError(f"timed out after {timeout} s waiting for {what}" + ("" if running is None else "\n" + running.output()))


def _proc_state(pid: int) -> str | None:
    """Process state from ``/proc``, or ``None`` when the pid is gone."""

    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return None
    try:
        stat = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "?"
    fields = stat.rpartition(")")[2].split()
    return fields[0] if fields else "?"


def _descendant_states(output: Path) -> list[str]:
    """Liveness of every supervised process except the launcher itself."""

    rows = _record(output / "launch.json").get("processes") or []
    return [
        process_match(row.get("pid"), row.get("process_start"))
        for row in rows
        if isinstance(row, dict) and row.get("role") != "launcher"
    ]


def _record(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _latest(output: Path) -> dict[str, object]:
    return torch.load(output / "latest.pt", map_location="cpu", weights_only=False)


def _digests(output: Path, *names: str) -> dict[str, str]:
    return {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names}


def _max_abs(left: object, right: object) -> float:
    """Largest tensor difference. Raises when the two trees do not have the same tensors."""

    worst = 0.0
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape:
            raise AssertionError("tensor layout differs")
        if left.numel():
            worst = float((left.detach().cpu().double() - right.detach().cpu().double()).abs().max())
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError("keys differ")
        for key, value in left.items():
            worst = max(worst, _max_abs(value, right[key]))
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise AssertionError("lengths differ")
        for lhs, rhs in zip(left, right):
            worst = max(worst, _max_abs(lhs, rhs))
    return worst


if __name__ == "__main__":
    unittest.main()
