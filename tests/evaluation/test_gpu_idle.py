"""CUDA_VISIBLE_DEVICES must select the nvidia-smi index, not the visible index."""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest.mock import patch

from src.evaluation.errors import ConfigError
from src.evaluation.gpu import BUSY_MEMORY_MIB, assert_gpu_idle, physical_device_index
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.gpu import physical_index
from tests.suite_resources import DEFAULT_BUSY_MEMORY_MIB


class GpuIdleMappingTests(unittest.TestCase):
    def test_busy_cutoff_matches_the_runner(self) -> None:
        self.assertEqual(DEFAULT_BUSY_MEMORY_MIB, BUSY_MEMORY_MIB)

    def test_empty_visible_devices_keep_the_index(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(physical_device_index(3), 3)
            self.assertEqual(physical_index(3), 3)

    def test_visible_zero_checks_the_listed_physical_gpu(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, 30000, 100\n5, 10, 0\n",
            stderr="",
        )
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}), patch(
            "src.evaluation.gpu.subprocess.run", return_value=completed
        ) as queried:
            assert_gpu_idle("cuda:0")
            assert_gpu_idle("cuda")
        self.assertGreaterEqual(queried.call_count, 2)

    def test_busy_listed_gpu_is_the_one_refused(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, 10, 0\n5, 4000, 90\n",
            stderr="",
        )
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}), patch(
            "src.evaluation.gpu.subprocess.run", return_value=completed
        ):
            with self.assertRaises(ConfigError) as caught:
                assert_gpu_idle("cuda:0")
        self.assertIn("GPU 5", str(caught.exception))

    def test_uuid_list_is_refused_before_nvidia_smi(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-abc"}), patch(
            "src.evaluation.gpu.subprocess.run"
        ) as queried:
            with self.assertRaises(ConfigError):
                assert_gpu_idle("cuda")
            with self.assertRaises(TrainingError):
                physical_index(0)
        queried.assert_not_called()

    def test_cpu_and_allow_busy_do_not_query(self) -> None:
        with patch("src.evaluation.gpu.subprocess.run") as queried:
            assert_gpu_idle("cpu")
            assert_gpu_idle("cuda:0", allow_busy=True)
        queried.assert_not_called()

    def test_second_visible_device_maps_to_the_second_physical_index(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5, 6"}):
            self.assertEqual(physical_device_index(1), 6)
            self.assertEqual(physical_index(1), 6)
