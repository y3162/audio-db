"""Whole-device GPU memory recorded by the launcher.

The series is ``nvidia-smi`` memory for the device, not PyTorch allocator
stats. A failed query is stored and does not stop the recorder. CPU runs
that hide every GPU write nothing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src.experiments.execution.gpu_memory import (
    GpuMemoryLog,
    parse_gpu_query,
    query_nvidia_smi,
    resolve_interval,
    select_devices,
)
from src.experiments.execution.status import write_status

ROOT = Path(__file__).resolve().parents[2]


class ParseTests(unittest.TestCase):
    def test_rows_keep_index_uuid_and_whole_device_memory(self) -> None:
        rows = parse_gpu_query("0, GPU-aaa, 13, 81920\n1, GPU-bbb, [N/A], 40960\n")
        self.assertEqual(rows[0], {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 13, "memory_total_mib": 81920})
        self.assertEqual(rows[1]["memory_used_mib"], None)
        self.assertEqual(select_devices(rows, "0"), [rows[0]])
        self.assertEqual(select_devices(rows, "GPU-bbb"), [rows[1]])
        self.assertEqual(select_devices(rows, None), rows)
        with self.assertRaises(ValueError):
            parse_gpu_query("0, 13\n")

    def test_interval_uses_the_flag_then_the_environment(self) -> None:
        self.assertEqual(resolve_interval(None, {}), 10.0)
        self.assertEqual(resolve_interval(None, {"AUDIO_DB_GPU_MEMORY_INTERVAL": "2.5"}), 2.5)
        self.assertEqual(resolve_interval(0.0, {"AUDIO_DB_GPU_MEMORY_INTERVAL": "2.5"}), 0.0)
        with self.assertRaises(ValueError):
            resolve_interval(-1, {})
        with self.assertRaises(ValueError):
            resolve_interval(None, {"AUDIO_DB_GPU_MEMORY_INTERVAL": "soon"})
        with self.assertRaises(ValueError):
            resolve_interval(float("nan"), {})


class LogTests(unittest.TestCase):
    def directory(self) -> Path:
        return Path(self.enterContext(tempfile.TemporaryDirectory(prefix="audio-db-gpu-memory-")))

    def test_hidden_or_disabled_recording_does_not_query(self) -> None:
        def refuse() -> str:
            raise AssertionError("nvidia-smi was called")

        for environ, interval in (({"CUDA_VISIBLE_DEVICES": ""}, 0.01), ({"CUDA_VISIBLE_DEVICES": "0"}, 0.0)):
            with self.subTest(environ=environ, interval=interval):
                directory = self.directory()
                log = GpuMemoryLog.start(directory, interval, "launch", environ=environ, query=refuse)
                log.close()
                self.assertFalse((directory / "gpu_memory.jsonl").exists())

    def test_samples_name_the_gpu_and_keep_the_peak(self) -> None:
        directory = self.directory()
        values = iter(("0, GPU-aaa, 10, 100\n1, GPU-bbb, 80, 200\n", "0, GPU-aaa, 40, 100\n1, GPU-bbb, 70, 200\n"))

        def query() -> str:
            return next(values)

        write_status(directory, "RUNNING", launch_id="launch")
        log = GpuMemoryLog.start(
            directory, 0.05, "launch", environ={"CUDA_VISIBLE_DEVICES": "0,GPU-bbb"}, query=query
        )
        self._wait(directory)
        # A killed recorder leaves every complete line. Close is not required for that.
        lines = _lines(directory)
        self.assertGreaterEqual(len(lines), 2)
        for line in lines:
            self.assertIn(json.loads(line)["kind"], {"header", "sample"})
        log.close()
        summary = json.loads((directory / "gpu_memory.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["sample_count"] >= 2, True)
        self.assertEqual([item["index"] for item in summary["peaks"]], [0, 1])
        self.assertEqual(summary["peaks"][0]["memory_used_mib"], 40)
        self.assertEqual(summary["peaks"][1]["memory_used_mib"], 80)
        self.assertIn("whole GPU", summary["metrics"]["memory_used_mib"])
        samples = [json.loads(line) for line in _lines(directory) if json.loads(line)["kind"] == "sample"]
        self.assertEqual({row["index"] for row in samples[0]["devices"]}, {0, 1})
        self.assertTrue(samples[0]["recorded_at"])
        self.assertIsNone(samples[0]["error"])
        shown = _show(directory)
        self.assertIn("samples=", shown)
        self.assertIn("0:40MiB", shown)
        self.assertIn("1:80MiB", shown)

    def test_a_failed_query_is_recorded_and_a_foreign_launch_is_not(self) -> None:
        directory = self.directory()

        def fail() -> str:
            raise RuntimeError("nvidia-smi missing")

        write_status(directory, "COMPLETED", launch_id="earlier")
        foreign = GpuMemoryLog.start(directory, 0.05, "this", environ={"CUDA_VISIBLE_DEVICES": "0"}, query=fail)
        time.sleep(0.2)
        foreign.close()
        self.assertFalse((directory / "gpu_memory.jsonl").exists())

        write_status(directory, "RUNNING", launch_id="this")
        log = GpuMemoryLog.start(directory, 60.0, "this", environ={"CUDA_VISIBLE_DEVICES": "0"}, query=fail)
        log.close()
        sample = json.loads(_lines(directory)[-1])
        self.assertEqual(sample["kind"], "sample")
        self.assertIn("RuntimeError", sample["error"])
        self.assertEqual(sample["devices"], [])
        summary = json.loads((directory / "gpu_memory.json").read_text(encoding="utf-8"))
        self.assertIn("RuntimeError", summary["last_error"])

    def test_real_nvidia_smi_reports_whole_device_rows(self) -> None:
        try:
            text = query_nvidia_smi()
        except Exception as exc:
            self.skipTest(f"nvidia-smi unavailable: {exc}")
        rows = parse_gpu_query(text)
        self.assertTrue(rows)
        self.assertIsInstance(rows[0]["index"], int)
        self.assertTrue(rows[0]["uuid"])
        self.assertIsInstance(rows[0]["memory_total_mib"], int)

    def test_the_launcher_keeps_samples_for_one_gpu_after_it_is_killed(self) -> None:
        if not _has_gpu(0):
            self.skipTest("physical GPU 0 is not present")
        directory = self.directory()
        config = _cpu_yaml(directory, steps=30, delay=0.3)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["AUDIO_DB_GPU_MEMORY_INTERVAL"] = "0.2"
        process = subprocess.Popen(
            [sys.executable, "-m", "src.experiments.launcher", "--config", str(config)],
            cwd=ROOT,
            env=env,
            start_new_session=True,
        )
        try:
            self._wait(directory / "run")
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            _reap(directory)
        lines = _lines(directory / "run")
        self.assertGreaterEqual(len(lines), 2)
        indexes: set[int] = set()
        for line in lines:
            payload = json.loads(line)
            if payload["kind"] != "sample":
                self.assertEqual(payload["cuda_visible_devices"], "0")
                continue
            self.assertTrue(payload["recorded_at"])
            for device in payload["devices"]:
                indexes.add(device["index"])
                self.assertIn("memory_used_mib", device)
                self.assertIn("uuid", device)
        self.assertEqual(indexes, {0})

    def test_a_stopped_launcher_keeps_the_summary(self) -> None:
        if not _has_gpu(0):
            self.skipTest("physical GPU 0 is not present")
        directory = self.directory()
        config = _cpu_yaml(directory, steps=30, delay=0.2)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["AUDIO_DB_GPU_MEMORY_INTERVAL"] = "0.2"
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.experiments.launcher",
                "--config",
                str(config),
                "--stop-after",
                "0.6",
                "--grace",
                "60",
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(done.returncode, 3, done.stderr + done.stdout)
        summary = json.loads((directory / "run" / "gpu_memory.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(summary["sample_count"], 1)
        self.assertEqual([item["index"] for item in summary["peaks"]], [0])
        status = json.loads((directory / "run" / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "INTERRUPTED")
        if int(status["logical_step"]) > 0:
            self.assertTrue((directory / "run" / "latest.pt").is_file())
        self.assertIn("gpu_memory\tsamples=", _show(directory / "run"))

    def test_interval_zero_writes_nothing(self) -> None:
        if not _has_gpu(0):
            self.skipTest("physical GPU 0 is not present")
        directory = self.directory()
        config = _cpu_yaml(directory, steps=1, delay=0.0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["AUDIO_DB_GPU_MEMORY_INTERVAL"] = "0"
        done = subprocess.run(
            [sys.executable, "-m", "src.experiments.launcher", "--config", str(config)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertFalse((directory / "run" / "gpu_memory.jsonl").exists())

    def _wait(self, directory: Path) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (directory / "gpu_memory.jsonl").is_file() and len(_lines(directory)) >= 2:
                return
            time.sleep(0.05)
        self.fail(f"no gpu memory sample in {directory}")


def _lines(directory: Path) -> list[str]:
    return [line for line in (directory / "gpu_memory.jsonl").read_text(encoding="utf-8").splitlines() if line]


def _show(directory: Path) -> str:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts/experiments/show_status.py"), str(directory)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise AssertionError(done.stderr)
    return done.stdout


def _has_gpu(index: int) -> bool:
    try:
        rows = parse_gpu_query(query_nvidia_smi())
    except Exception:
        return False
    return any(row["index"] == index for row in rows)


def _cpu_yaml(directory: Path, *, steps: int, delay: float) -> Path:
    output = directory / "run"
    text = f"""schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
output:
  directory: "{output}"
components:
  model:
    role: module
    class: torch.nn.Linear
    kwargs: {{in_features: 4, out_features: 4}}
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: 4
      examples: 16
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
      max_logical_steps: {steps}
      accumulation: 1
      fault: none
      step: 0
      rank: 0
      delay: {delay}
      marker: "{directory / "fault"}"
    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""
    path = directory / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _reap(directory: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        alive = []
        for path in directory.glob("fault.pid*"):
            try:
                pid = int(path.read_text(encoding="utf-8"))
                os.kill(pid, 0)
            except (OSError, ValueError):
                continue
            alive.append(pid)
        if not alive:
            return
        time.sleep(0.2)
    for path in directory.glob("fault.pid*"):
        try:
            os.kill(int(path.read_text(encoding="utf-8")), signal.SIGKILL)
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    unittest.main()
