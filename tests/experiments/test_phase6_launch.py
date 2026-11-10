"""Phase 6 launcher: SIGTERM before the hard timeout, then SIGKILL after grace."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _launch():
    path = ROOT / "scripts/experiments/phase5_launch.py"
    spec = importlib.util.spec_from_file_location("phase5_launch", path)
    if spec is None or spec.loader is None:
        raise AssertionError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LaunchGraceTests(unittest.TestCase):
    def test_sigterm_precedes_sigkill_and_grace_is_positive(self) -> None:
        launch = _launch()
        self.assertEqual(launch.effective_grace(1200, None), 120.0)
        self.assertEqual(launch.effective_grace(8, None), 2.0)
        with self.assertRaises(SystemExit):
            launch.effective_grace(30, 30)
        with self.assertRaises(SystemExit):
            launch.effective_grace(30, 0)
        self.assertIsNone(
            launch.termination_action(10, timeout=120, grace=30, signal_after=None, signaled=False)
        )
        self.assertEqual(
            launch.termination_action(90, timeout=120, grace=30, signal_after=None, signaled=False),
            "term",
        )
        self.assertIsNone(
            launch.termination_action(90, timeout=120, grace=30, signal_after=None, signaled=True)
        )
        self.assertEqual(
            launch.termination_action(120, timeout=120, grace=30, signal_after=None, signaled=True),
            "kill",
        )
        self.assertEqual(
            launch.termination_action(5, timeout=120, grace=30, signal_after=4, signaled=False),
            "term",
        )
