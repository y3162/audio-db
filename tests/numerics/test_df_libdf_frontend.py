"""DFStream matches libdf analysis and normalization on the same waveform.

The comparison imports libdf. It skips when that extension or the local
realfft 3.3.0 kernel is absent. Neither is required to import the model.
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "src/se/models/deepfilternet3/libdf_rfft.so"


def _libdf():
    path = os.environ.get("AUDIO_DB_LIBDF_PATH", str(ROOT / ".venvs/tsubame-libdf"))
    inserted = path not in sys.path
    sys.path.insert(0, path)
    try:
        import libdf
    finally:
        if inserted and path in sys.path:
            sys.path.remove(path)
    return libdf


class DeepFilterFrontendTest(unittest.TestCase):
    def setUp(self):
        if not KERNEL.is_file():
            self.skipTest("libdf_rfft.so is not built")
        try:
            self.libdf = _libdf()
        except ImportError as error:
            self.skipTest(f"libdf is not importable: {error}")
        from src.se.models.deepfilternet3.dsp import DFStream
        self.stream = DFStream(48000, 960, 480, 32, 96, 2, 1.0, torch.device("cpu"))

    def test_analysis_and_features_match_libdf(self):
        libdf = self.libdf
        reference = libdf.DF(48000, 960, 480, 32, 2)
        generator = np.random.default_rng(7)
        audio = generator.standard_normal((2, 4800)).astype(np.float32)
        # A second channel with a quiet band checks the log floor and unit norm.
        audio[1] *= np.float32(1e-4)
        expected = torch.from_numpy(np.array(reference.analysis(audio), copy=True))
        actual = self.stream.analyze(torch.from_numpy(audio))
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(float((actual - expected).abs().max()), 0.0)

        alpha = float(self.stream.alpha)
        widths = reference.erb_widths()
        spectrum = np.array(reference.analysis(audio), copy=True)
        erb = libdf.erb_norm(libdf.erb(spectrum, widths), alpha)
        unit = libdf.unit_norm(np.array(spectrum[..., :96], copy=True), alpha)
        spec, erb_hat, unit_hat = self.stream.features(torch.from_numpy(spectrum))
        self.assertEqual(float((erb_hat[:, 0] - torch.from_numpy(np.array(erb))).abs().max()), 0.0)
        unit_real = torch.view_as_real(torch.from_numpy(np.array(unit)))
        self.assertEqual(float((unit_hat[:, 0] - unit_real).abs().max()), 0.0)
        self.assertEqual(float((spec[:, 0] - torch.view_as_real(torch.from_numpy(spectrum))).abs().max()), 0.0)

    def test_padded_training_length_matches_libdf(self):
        libdf = self.libdf
        reference = libdf.DF(48000, 960, 480, 32, 2)
        waveform = torch.linspace(-0.2, 0.2, 4800).view(1, -1)
        padded = torch.nn.functional.pad(waveform, (0, 960))
        expected = torch.from_numpy(np.array(reference.analysis(padded.numpy()), copy=True))
        actual = self.stream.analyze(padded)
        self.assertEqual(float((actual - expected).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
