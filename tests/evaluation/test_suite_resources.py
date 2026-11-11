"""Budgets for the test runner. These do not start a suite."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.suite_resources import (
    distribution_workers,
    gpu_test_environment,
    intra_threads,
    module_workers,
    parse_busy_uuids,
    parse_gpu_rows,
    select_free_gpus,
)

ROOT = Path(__file__).resolve().parents[2]


class SuiteResourceTests(unittest.TestCase):
    def test_workers_follow_cpu_count_and_the_override(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_DB_TEST_WORKERS", None)
            self.assertEqual(module_workers(128), 8)
            self.assertEqual(module_workers(4), 2)
            self.assertEqual(module_workers(1), 1)
        with patch.dict(os.environ, {"AUDIO_DB_TEST_WORKERS": "3"}):
            self.assertEqual(module_workers(128), 3)

    def test_threads_shrink_as_modules_share_the_machine(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_DB_TEST_THREADS", None)
            self.assertEqual(intra_threads(8, 128), 8)
            self.assertEqual(intra_threads(8, 16), 2)
            self.assertEqual(intra_threads(1, 4), 4)
        with patch.dict(os.environ, {"AUDIO_DB_TEST_THREADS": "5"}):
            self.assertEqual(intra_threads(8, 128), 5)

    def test_distribution_workers_leave_room_for_other_modules(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_DB_DFN3_WORKERS", None)
            os.environ.pop("AUDIO_DB_TEST_WORKERS", None)
            self.assertEqual(distribution_workers(128), 32)
            self.assertEqual(distribution_workers(4), 4)
        with patch.dict(os.environ, {"AUDIO_DB_TEST_WORKERS": "8"}):
            os.environ.pop("AUDIO_DB_DFN3_WORKERS", None)
            self.assertEqual(distribution_workers(128), 16)
        with patch.dict(os.environ, {"AUDIO_DB_DFN3_WORKERS": "2", "AUDIO_DB_TEST_WORKERS": "8"}):
            self.assertEqual(distribution_workers(128), 2)

    def test_free_gpu_skips_compute_processes_and_large_allocations(self) -> None:
        rows = [
            {"index": 0, "uuid": "A", "memory_used_mib": 800},
            {"index": 5, "uuid": "B", "memory_used_mib": 14},
            {"index": 4, "uuid": "C", "memory_used_mib": 5000},
        ]
        self.assertEqual(select_free_gpus(rows, {"A"}), [5])
        parsed = parse_gpu_rows("0, GPU-a, [N/A]\n1, GPU-b, 10\n")
        self.assertEqual(select_free_gpus(parsed, set()), [1])
        self.assertEqual(parse_busy_uuids("GPU-a, 12\n\nGPU-b, 13\n"), {"GPU-a", "GPU-b"})

    def test_discovery_lists_the_suites_and_not_reference_scripts(self) -> None:
        from scripts.tests import run_suite

        names = run_suite.discover(ROOT)
        self.assertIn("tests.evaluation.test_real_smoke", names)
        self.assertIn("tests.numerics.test_four_process", names)
        self.assertIn("tests.experiments.test_cuda_amp", names)
        self.assertIn("tests.data.reproduction.test_p8_dfn3", names)
        self.assertNotIn("tests.numerics.reference.compare_mp_senet", names)
        self.assertFalse(any(".reference." in name for name in names))
        for target in run_suite.GPU_TARGETS:
            self.assertIsNotNone(run_suite.module_file(ROOT, target), target)

    def test_production_db_modules_do_not_overlap(self) -> None:
        from scripts.tests.run_suite import PRODUCTION_DB_MODULES, blocked_by_production_db

        self.assertIn("tests.data.reproduction.test_p9_semamba", PRODUCTION_DB_MODULES)
        running = {"tests.data.reproduction.test_p9_semamba"}
        self.assertTrue(blocked_by_production_db("tests.data.reproduction.test_p1", running))
        self.assertTrue(blocked_by_production_db("tests.experiments.test_phase4i_integration", running))
        self.assertFalse(blocked_by_production_db("tests.experiments.test_semambapp", running))
        self.assertFalse(blocked_by_production_db("tests.data.reproduction.test_p1", set()))

    def test_gpu_lane_keeps_child_processes_on_the_selected_devices(self) -> None:
        env = gpu_test_environment([1, 2])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,2")
        self.assertEqual(env["AUDIO_DB_CUDA_GPU0"], "1")
        self.assertEqual(env["AUDIO_DB_CUDA_GPU1"], "2")
        one = gpu_test_environment([5])
        self.assertEqual(one["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(one["AUDIO_DB_CUDA_GPU0"], "5")
        self.assertEqual(one["AUDIO_DB_CUDA_GPU1"], "5")
