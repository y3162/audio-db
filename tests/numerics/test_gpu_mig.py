"""Training runtime does not refuse a CUDA device that another process is using."""

import os
import unittest
from unittest.mock import patch

from src.experiments.errors import PrepareError
from src.experiments.runtime import ProcessRuntime


class TrainingGpuGateTests(unittest.TestCase):
    def test_busy_memory_does_not_refuse_launch(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}, clear=False), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.set_device") as set_device, \
             patch("subprocess.run") as smi:
            runtime = ProcessRuntime(device="cuda", precision="fp32")
            try:
                self.assertEqual(runtime.device.type, "cuda")
                self.assertEqual(runtime.device.index, 0)
                self.assertFalse(hasattr(runtime, "allow_busy_gpu"))
            finally:
                runtime.shutdown()
            set_device.assert_called()
            smi.assert_not_called()

    def test_missing_cuda_is_refused(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(PrepareError):
                ProcessRuntime(device="cuda", precision="fp32")

    def test_local_rank_outside_visible_devices_is_refused(self) -> None:
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "5"}), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.set_device"):
            with self.assertRaises(PrepareError):
                ProcessRuntime(device="cuda", precision="fp32", world_size=2)

    def test_world_size_cannot_share_one_gpu(self) -> None:
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.set_device"):
            with self.assertRaises(PrepareError):
                ProcessRuntime(device="cuda", precision="fp32", world_size=2)

    def test_removed_busy_option_is_not_accepted(self) -> None:
        with self.assertRaises(TypeError):
            ProcessRuntime(device="cpu", allow_busy_gpu=True)  # type: ignore[call-arg]
