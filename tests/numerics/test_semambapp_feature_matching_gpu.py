"""Numerical regression for the shared CQT/MRD waveform junction."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'src/official/semambapp/model/discriminator.py'


@unittest.skipUnless(torch.cuda.is_available(), 'NOT_RUN: CUDA is required')
@unittest.skipUnless(OFFICIAL.is_file(), 'official SEMamba++ source is absent')
class FeatureMatchingJointBackwardTest(unittest.TestCase):
    def test_official_waveform_gradient_and_repeat(self):
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1')
        result = subprocess.run(
            [sys.executable, str(ROOT / 'tests/numerics/reference/probe_semambapp_joint_feature.py')],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=300, check=True,
        )
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        cases = [row for row in rows if row['case'] in {'q', 'r', 'joint'}]
        self.assertEqual(len(cases), 6, result.stdout)
        for row in cases:
            with self.subTest(seed=row['seed'], case=row['case']):
                self.assertEqual(row['maps_diff'], 0)
                self.assertEqual(row['audio_maps_diff'], 0)
                for name in ('audio', 'repeat'):
                    self.assertEqual(row[name]['mismatch'], 0, row)
                    self.assertEqual(row[name]['nonfinite'], 0, row)


if __name__ == '__main__':
    unittest.main()
