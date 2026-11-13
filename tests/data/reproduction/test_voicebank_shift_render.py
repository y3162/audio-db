"""The SNR-shift renderer preserves the recipe's float reference and mixture."""

import tempfile
import hashlib
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.experiments.data.errors import ReproductionError
from src.experiments.data.render.voicebank_shift import render_voicebank_shift, render_voicebank_shift_files
from src.experiments.data.prepare.resample import resample_float32
from src.experiments.data.assets.store import AssetStore
from src.experiments.data.ids import digest
from src.experiments.data.prepare.voicebank_shift_recipe import canonical_shift_recipe, shift_case_id, snapshot_realization
from src.experiments.data.render.voicebank_shift_snapshot import render_shift_case


class VoiceBankShiftRenderTests(unittest.TestCase):
    def test_snapshot_realization_replays_recipe_and_verifies_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = np.linspace(-0.8, 0.7, 4803, dtype=np.float64)
            noise = np.linspace(0.3, -0.5, 5000, dtype=np.float64)
            for name, samples in (("clean.wav", clean), ("noise.wav", noise)):
                sf.write(root / name, samples, 48000, subtype="DOUBLE")
            shas = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("clean.wav", "noise.wav")}
            row = {"status": "ready", "clean_path": "data/corpora/clean.wav", "noise_path": "data/corpora/noise.wav",
                   "clean_sha256": shas["clean.wav"], "noise_sha256": shas["noise.wav"],
                   "noise_start_48k": 10, "noise_end_exclusive_48k": 4813,
                   "clean_gain_g": 0.97, "shifted_noise_gain": 2.1}
            canonical = canonical_shift_recipe(row)
            document = snapshot_realization({"case_id": shift_case_id(row), "recipe": canonical})
            case_id = digest(document)
            assets = [{"asset_id": "corpora/" + name, "rel_path": name, "sha256": shas[name]}
                      for name in ("clean.wav", "noise.wav")]
            store = AssetStore({"bindings": []}, assets, [], [], snapshot_hash="test", derived=root,
                               corpora=root, generated=root, docs=[{"case_id": case_id, "doc_json": document}])
            noisy, reference = render_shift_case(store, case_id, verify_on_read=True)
            original_noisy, original_clean = render_voicebank_shift_files(row, root / "clean.wav", root / "noise.wav")
            np.testing.assert_array_equal(noisy, resample_float32(original_noisy, 48000))
            np.testing.assert_array_equal(reference, resample_float32(original_clean, 48000))
            (root / "noise.wav").write_bytes(b"tampered")
            with self.assertRaisesRegex(ReproductionError, "sha256 mismatch"):
                render_shift_case(store, case_id, verify_on_read=True)

    def test_float_mixture_and_reference_are_not_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = np.array([0.5, -0.25, 0.75], dtype=np.float64)
            noise = np.array([0.1, 0.8, -0.4, 0.5, 0.2], dtype=np.float64)
            sf.write(root / "clean.wav", clean, 48000, subtype="DOUBLE")
            sf.write(root / "noise.wav", noise, 48000, subtype="DOUBLE")
            row = {"status": "ready", "clean_path": "clean.wav", "noise_path": "noise.wav",
                   "noise_start_48k": "1", "noise_end_exclusive_48k": "4",
                   "clean_gain_g": 0.5, "shifted_noise_gain": "2"}
            noisy, reference = render_voicebank_shift(row, root)
            np.testing.assert_allclose(reference, clean * 0.5)
            np.testing.assert_allclose(noisy, clean * 0.5 + noise[1:4] * 2)
            self.assertGreater(noisy.max(), 1)
            row["shifted_noise_gain"] = None
            with self.assertRaises(ReproductionError):
                render_voicebank_shift(row, root)

    def test_shift_uses_the_common_float_resampler(self):
        import soxr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = np.full(4803, 0.75, dtype=np.float64)
            noise = np.full(4803, 0.5, dtype=np.float64)
            sf.write(root / "clean.wav", clean, 48000, subtype="DOUBLE")
            sf.write(root / "noise.wav", noise, 48000, subtype="DOUBLE")
            row = {"status": "ready", "clean_path": "clean.wav", "noise_path": "noise.wav",
                   "noise_start_48k": 0, "noise_end_exclusive_48k": len(clean),
                   "clean_gain_g": 0.5, "shifted_noise_gain": 2.0}
            noisy, reference = render_voicebank_shift(row, root)
            for source in (noisy, reference):
                output = resample_float32(source, 48000)
                np.testing.assert_array_equal(
                    output, soxr.resample(source.astype(np.float32), 48000, 16000, quality="HQ")
                )
            self.assertGreater(float(resample_float32(noisy, 48000).max()), 1.0)
            row["noise_path"] = "../noise.wav"
            with self.assertRaises(ReproductionError):
                render_voicebank_shift(row, root)
            row["noise_path"] = "noise.wav"
            row["shifted_noise_gain"] = 2
            row["noise_end_exclusive_48k"] = 3
            with self.assertRaises(ReproductionError):
                render_voicebank_shift(row, root)
