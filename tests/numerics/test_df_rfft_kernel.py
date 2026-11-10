"""The DeepFilterNet3 realfft kernel loads, rejects a bad ABI, and is unnormalized.

These checks do not import libdf. Bit-exact agreement with libdf is
``tests.numerics.test_df_libdf_frontend``.
"""
import os
import unittest
from pathlib import Path

import torch

from src.se.models.deepfilternet3.dsp import _rfft_kernel, rfft_library_path


class RealFftKernelTest(unittest.TestCase):
    def setUp(self):
        self._rfft_required = os.environ.get("DF_RFFT_REQUIRED")
        self._rfft_library = os.environ.get("DF_RFFT_LIBRARY")

    def tearDown(self):
        # Restore the process environment. Popping unconditionally made later
        # DeepFilterNet3 tests in the same suite miss DF_RFFT_LIBRARY.
        if self._rfft_required is None:
            os.environ.pop("DF_RFFT_REQUIRED", None)
        else:
            os.environ["DF_RFFT_REQUIRED"] = self._rfft_required
        if self._rfft_library is None:
            os.environ.pop("DF_RFFT_LIBRARY", None)
        else:
            os.environ["DF_RFFT_LIBRARY"] = self._rfft_library

    def test_unnormalized_fft_matches_sum_and_nyquist(self):
        kernel = _rfft_kernel()
        if kernel is None:
            self.skipTest(f"{rfft_library_path()} is not built")
        waveform = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        first = _transform(kernel, waveform)
        second = _transform(kernel, waveform)
        self.assertEqual(first, second)
        self.assertEqual(first[0], (10.0, 0.0))
        self.assertEqual(first[-1], (-2.0, 0.0))
        status = kernel.df_rfft_f32(None, 1, None, None)
        self.assertEqual(status, 1)
        self.assertEqual(kernel.df_rfft_abi(), 1)

    def test_missing_library_is_an_error_when_required(self):
        os.environ["DF_RFFT_REQUIRED"] = "1"
        os.environ["DF_RFFT_LIBRARY"] = str(Path("/tmp") / "audio-db-missing-libdf_rfft.so")
        with self.assertRaises(FileNotFoundError) as caught:
            _rfft_kernel()
        self.assertIn("make df-rfft", str(caught.exception))

    def test_missing_library_falls_back_when_optional(self):
        os.environ["DF_RFFT_LIBRARY"] = str(Path("/tmp") / "audio-db-missing-libdf_rfft.so")
        self.assertIsNone(_rfft_kernel())


def _transform(kernel, waveform):
    from src.se.models.deepfilternet3.dsp import _ptr

    bins = waveform.numel() // 2 + 1
    real = torch.empty(bins, dtype=torch.float32)
    imag = torch.empty(bins, dtype=torch.float32)
    status = kernel.df_rfft_f32(_ptr(waveform), waveform.numel(), _ptr(real), _ptr(imag))
    if status != 0:
        raise AssertionError(f"df_rfft_f32 status {status}")
    return [(float(real[index]), float(imag[index])) for index in range(bins)]


if __name__ == "__main__":
    unittest.main()
