"""CUDA and NCCL coverage for research selection.

These tests are NOT_RUN when CUDA is hidden. A skip is not a pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from tests.experiments.test_research_training import _selection_text, _write


def _require_cuda(count: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        raise unittest.SkipTest(f"NOT_RUN: need {count} visible CUDA devices")
    for index in range(count):
        free, total = torch.cuda.mem_get_info(index)
        used = total - free
        if used > 2_000_000_000:
            raise unittest.SkipTest(f"NOT_RUN: cuda:{index} is using {used / 1_000_000:.0f} MiB")


class ResearchCudaTests(unittest.TestCase):
    def test_cuda_amp_research_selection(self) -> None:
        _require_cuda(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = _selection_text(root, mode="min", steps=1, updates=1)
            text = text.replace("device: cpu", "device: cuda", 1).replace("precision: fp32", "precision: amp_fp16", 1)
            completed = _launch(root, text, nproc=1)
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            choice = json.loads((root / "research_selection" / "best.json").read_text(encoding="utf-8"))
            self.assertEqual(choice["metric"], "validation.loss")
            self.assertEqual(choice["logical_step"], 1)

    def test_nccl_two_ranks_agree(self) -> None:
        _require_cuda(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = _selection_text(root, mode="min", steps=1, updates=1, distribute="ddp")
            text = text.replace("device: cpu", "device: cuda", 1).replace("precision: fp32", "precision: amp_fp16", 1)
            completed = _launch(root, text, nproc=2)
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            choice = json.loads((root / "research_selection" / "best.json").read_text(encoding="utf-8"))
            self.assertEqual(choice["metric"], "validation.loss")
            self.assertTrue((root / "latest.pt").is_file())


def _launch(directory: Path, text: str, *, nproc: int) -> subprocess.CompletedProcess[str]:
    path = _write(directory, text)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.launcher",
            "--config",
            str(path),
            "--nproc",
            str(nproc),
            "--gpu-memory-interval",
            "0",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )


if __name__ == "__main__":
    unittest.main()
