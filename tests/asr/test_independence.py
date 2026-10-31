"""src.asr stays independent of the evaluation package and of model libraries."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class IndependenceTests(unittest.TestCase):
    def test_import_does_not_pull_evaluation_or_backends(self):
        code = """
import src.asr
import sys
names = set(sys.modules)
assert not any(name == "src.evaluation" or name.startswith("src.evaluation.") for name in names)
assert not any(name == "src.se" or name.startswith("src.se.") for name in names)
for blocked in ("nemo", "whisper", "transformers"):
    assert blocked not in names, blocked
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
