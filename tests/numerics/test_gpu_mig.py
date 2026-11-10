import unittest
from unittest.mock import patch

from src.experiments.calc.gpu import assert_gpu_idle
from src.experiments.calc.errors import TrainingError


class MigIdleTests(unittest.TestCase):
    def test_checks_mig_memory_without_parent_query(self):
        with patch('torch.cuda.get_device_name', return_value='NVIDIA H100 MIG 4g.47gb'), \
             patch('torch.cuda.mem_get_info', return_value=(47 * 1024**3, 48 * 1024**3)), \
             patch('src.experiments.calc.gpu.subprocess.run') as smi:
            with self.assertRaises(TrainingError):
                assert_gpu_idle(0, allow_busy=False)
            smi.assert_not_called()

    def test_free_mig_succeeds(self):
        with patch('torch.cuda.get_device_name', return_value='NVIDIA H100 MIG 4g.47gb'), \
             patch('torch.cuda.mem_get_info', return_value=(48 * 1024**3 - 58 * 1024**2, 48 * 1024**3)):
            assert_gpu_idle(0, allow_busy=False)
