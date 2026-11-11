"""Budgets for the test runner. These do not start a suite."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.suite_resources import (
    _parse_cfs_quota,
    _parse_cpu_max,
    _tightest_quota,
    available_cpus,
    distribution_workers,
    gpu_inventory_required,
    gpu_test_environment,
    intra_threads,
    module_workers,
    parse_busy_uuids,
    parse_gpu_rows,
    process_threads,
    select_free_gpus,
    select_test_gpus,
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
        self.assertTrue(blocked_by_production_db("tests.experiments.test_semambapp", running))
        self.assertTrue(
            blocked_by_production_db(
                "tests.data.reproduction.test_p9_semamba",
                {"tests.experiments.test_semambapp"},
            )
        )
        self.assertFalse(blocked_by_production_db("tests.experiments.test_semambapp", set()))
        self.assertFalse(
            blocked_by_production_db(
                "tests.data.reproduction.test_p1",
                {"tests.experiments.test_semambapp"},
            )
        )
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
        self.assertEqual(one["CUDA_VISIBLE_DEVICES"].split(","), ["5"])


_ROWS = [
    {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 10},
    {"index": 1, "uuid": "GPU-bbb", "memory_used_mib": 10},
    {"index": 4, "uuid": "GPU-ccc", "memory_used_mib": 5000},
]


class AvailableCpuTests(unittest.TestCase):
    def test_quota_parsers_treat_unlimited_as_no_cap(self) -> None:
        self.assertIsNone(_parse_cpu_max("max 100000"))
        self.assertEqual(_parse_cpu_max("50000 100000"), 1)
        self.assertEqual(_parse_cpu_max("250000 100000"), 2)
        self.assertIsNone(_parse_cfs_quota("-1", "100000"))
        self.assertEqual(_parse_cfs_quota("250000", "100000"), 2)

    def test_ancestor_quota_is_the_tighter_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            leaf = parent / "leaf"
            leaf.mkdir(parents=True)
            (parent / "cpu.max").write_text("100000 100000\n", encoding="utf-8")
            (leaf / "cpu.max").write_text("max 100000\n", encoding="utf-8")
            self.assertEqual(_tightest_quota(leaf, root), 1)
            other_parent = root / "cfs"
            other = other_parent / "job"
            other.mkdir(parents=True)
            (other_parent / "cpu.cfs_quota_us").write_text("250000", encoding="utf-8")
            (other_parent / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")
            (other / "cpu.cfs_quota_us").write_text("-1", encoding="utf-8")
            (other / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")
            self.assertEqual(_tightest_quota(other, root), 2)

    def test_defaults_follow_available_cpus_and_overrides_stay_literal(self) -> None:
        with patch("tests.suite_resources.available_cpus", return_value=4), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_DB_TEST_WORKERS", None)
            os.environ.pop("AUDIO_DB_TEST_THREADS", None)
            os.environ.pop("AUDIO_DB_DFN3_WORKERS", None)
            self.assertEqual(module_workers(), 2)
            self.assertEqual(distribution_workers(), 4)
            self.assertEqual(process_threads(2, 0), 2)
        with patch("tests.suite_resources.available_cpus", return_value=4), patch.dict(
            os.environ,
            {"AUDIO_DB_TEST_WORKERS": "7", "AUDIO_DB_TEST_THREADS": "6", "AUDIO_DB_DFN3_WORKERS": "9"},
        ):
            self.assertEqual(module_workers(), 7)
            self.assertEqual(process_threads(7, 1), 6)
            self.assertEqual(distribution_workers(), 9)

    def test_gpu_lane_shares_the_thread_budget(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_DB_TEST_THREADS", None)
            self.assertEqual(process_threads(8, 0, cpus=16), 2)
            self.assertEqual(process_threads(8, 1, cpus=16), 1)

    def test_available_cpus_do_not_exceed_affinity_or_the_host(self) -> None:
        host = os.cpu_count() or 1
        affinity = len(os.sched_getaffinity(0))
        got = available_cpus()
        self.assertGreaterEqual(got, 1)
        self.assertLessEqual(got, affinity)
        self.assertLessEqual(got, host)


class GpuSelectionTests(unittest.TestCase):
    def test_automatic_selection_stays_inside_the_parent_mask(self) -> None:
        devices, reason = select_test_gpus(visible=None, requested=None, rows=_ROWS, busy_uuids=set())
        self.assertEqual(devices, [0, 1])
        self.assertEqual(reason, "idle")
        narrowed, narrowed_reason = select_test_gpus(visible="1", requested=None, rows=_ROWS, busy_uuids=set())
        self.assertEqual(narrowed, [1])
        self.assertIn("CUDA_VISIBLE_DEVICES", narrowed_reason)
        ordered, _reason = select_test_gpus(visible="1,0", requested=None, rows=_ROWS, busy_uuids=set())
        self.assertEqual(ordered, [1, 0])

    def test_empty_uuid_and_mig_are_not_rewritten_as_another_gpu(self) -> None:
        empty, empty_reason = select_test_gpus(visible="", requested=None, rows=_ROWS, busy_uuids=set())
        self.assertEqual(empty, [])
        self.assertIn("empty", empty_reason)
        forced, forced_reason = select_test_gpus(visible="", requested="0", rows=None, busy_uuids=set())
        self.assertEqual(forced, [])
        self.assertIn("outside", forced_reason)
        uuid, _reason = select_test_gpus(visible="GPU-BBB", requested=None, rows=_ROWS, busy_uuids=set())
        self.assertEqual(uuid, [1])
        unknown, unknown_reason = select_test_gpus(
            visible="GPU-zzz", requested=None, rows=_ROWS, busy_uuids=set()
        )
        self.assertEqual(unknown, [])
        self.assertIn("did not match", unknown_reason)
        mig, mig_reason = select_test_gpus(visible="0,MIG-GPU-bbb", requested=None, rows=_ROWS, busy_uuids=set())
        self.assertEqual(mig, [])
        self.assertIn("not mapped", mig_reason)
        other, other_reason = select_test_gpus(visible="cuda:0", requested=None, rows=_ROWS, busy_uuids=set())
        self.assertEqual(other, [])
        self.assertIn("cuda:0", other_reason)

    def test_explicit_indexes_cannot_leave_the_mask_and_skip_the_idle_check(self) -> None:
        outside, outside_reason = select_test_gpus(visible="1", requested="0", rows=_ROWS, busy_uuids=set())
        self.assertEqual(outside, [])
        self.assertIn("outside", outside_reason)
        busy, busy_reason = select_test_gpus(visible="4", requested="4", rows=_ROWS, busy_uuids=set())
        self.assertEqual(busy, [4])
        self.assertEqual(busy_reason, "AUDIO_DB_TEST_GPUS")
        named, named_reason = select_test_gpus(visible=None, requested="4", rows=None, busy_uuids=set())
        self.assertEqual(named, [4])
        self.assertEqual(named_reason, "AUDIO_DB_TEST_GPUS")
        with self.assertRaises(ValueError):
            select_test_gpus(visible=None, requested="GPU-bbb", rows=_ROWS, busy_uuids=set())
        repeated, repeated_reason = select_test_gpus(visible="1,1", requested=None, rows=_ROWS, busy_uuids=set())
        self.assertEqual(repeated, [])
        self.assertIn("same GPU", repeated_reason)

    def test_missing_nvidia_smi_does_not_invent_an_idle_gpu(self) -> None:
        missing, missing_reason = select_test_gpus(
            visible=None,
            requested=None,
            rows=None,
            query_error="nvidia-smi is not available",
        )
        self.assertEqual(missing, [])
        self.assertIn("nvidia-smi is not available", missing_reason)
        masked, masked_reason = select_test_gpus(
            visible="0",
            requested=None,
            rows=None,
            query_error="nvidia-smi is not available",
        )
        self.assertEqual(masked, [])
        self.assertIn("nvidia-smi is not available", masked_reason)
        uuid, uuid_reason = select_test_gpus(
            visible="GPU-aaa",
            requested=None,
            rows=None,
            query_error="nvidia-smi failed: boom",
        )
        self.assertEqual(uuid, [])
        self.assertIn("boom", uuid_reason)
        self.assertIn("not mapped", uuid_reason)
        self.assertFalse(gpu_inventory_required("", None))
        self.assertFalse(gpu_inventory_required("MIG-x", None))
        self.assertFalse(gpu_inventory_required("0,1", "0"))
        self.assertFalse(gpu_inventory_required(None, "1"))
        self.assertTrue(gpu_inventory_required(None, None))
        self.assertTrue(gpu_inventory_required("GPU-aaa", "0"))

    def test_runner_query_uses_the_parent_mask(self) -> None:
        from scripts.tests import run_suite

        smi = "0, GPU-aaa, 10\n1, GPU-bbb, 10\n"
        apps = ""

        def completed(stdout: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            os.environ.pop("AUDIO_DB_TEST_GPUS", None)
            with patch("scripts.tests.run_suite.subprocess.run", side_effect=[completed(smi), completed(apps)]) as queried:
                devices, reason = run_suite._query_gpus()
        self.assertEqual(devices, [1])
        self.assertIn("CUDA_VISIBLE_DEVICES", reason)
        self.assertEqual(queried.call_count, 2)

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            os.environ.pop("AUDIO_DB_TEST_GPUS", None)
            with patch("scripts.tests.run_suite.subprocess.run") as queried:
                devices, reason = run_suite._query_gpus()
        self.assertEqual(devices, [])
        self.assertIn("empty", reason)
        queried.assert_not_called()

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "AUDIO_DB_TEST_GPUS": "1"}, clear=False):
            with patch("scripts.tests.run_suite.subprocess.run") as queried:
                devices, reason = run_suite._query_gpus()
        self.assertEqual(devices, [])
        self.assertIn("outside", reason)
        queried.assert_not_called()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("AUDIO_DB_TEST_GPUS", None)
            with patch("scripts.tests.run_suite.subprocess.run", side_effect=FileNotFoundError):
                devices, reason = run_suite._query_gpus()
        self.assertEqual(devices, [])
        self.assertIn("nvidia-smi is not available", reason)
