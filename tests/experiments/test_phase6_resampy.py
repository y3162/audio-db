"""SEMamba++ bandwidth resampling resolves resampy in the fused environment."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venvs" / "semambapp" / "bin" / "python"


class SemambaResampyTests(unittest.TestCase):
    def test_lock_pins_the_same_resampy_as_se(self) -> None:
        se_in = (ROOT / "requirements" / "se.in").read_text(encoding="utf-8")
        semamba_in = (ROOT / "requirements" / "semambapp.in").read_text(encoding="utf-8")
        lock = (ROOT / "requirements" / "semambapp.lock.txt").read_text(encoding="utf-8")
        self.assertIn("resampy==0.4.3", se_in)
        self.assertIn("-r se.in", semamba_in)
        self.assertIn("resampy==0.4.3", lock)

    def test_kaiser_resamplers_run_in_the_semambapp_environment(self) -> None:
        if not PYTHON.is_file():
            self.fail(f"semambapp interpreter is missing: {PYTHON}")
        script = textwrap.dedent(
            """\
            import resampy
            import numpy as np
            from src.reproduction.protocols.online_semamba import bandwidth_limitation

            assert resampy.__version__ == "0.4.3", resampy.__version__
            time = np.linspace(0, 0.05, 800, endpoint=False, dtype=np.float64)
            speech = np.sin(2 * np.pi * 440 * time)[None, :]
            for kind in ("kaiser_best", "kaiser_fast", "scipy", "polyphase"):
                limited, padded = bandwidth_limitation(speech, 16000, 4000, kind, "butter", 4)
                if limited.shape != speech.shape:
                    raise SystemExit(f"{kind} shape {limited.shape}")
                if not np.isfinite(limited).all():
                    raise SystemExit(f"{kind} is not finite")
                if np.allclose(limited, speech):
                    raise SystemExit(f"{kind} did not change the signal")
                if padded not in (True, False):
                    raise SystemExit(f"{kind} pad flag {padded}")
            print("resampy", resampy.__version__)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            [str(PYTHON), "-c", script],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn("resampy 0.4.3", completed.stdout)
