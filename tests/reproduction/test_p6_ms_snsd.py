"""MS-SNSD mixture planning against the ESPnet recipe's audiolib."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpora" / "MS-SNSD"

from src.reproduction.build.ms_snsd import (
    AUDIO_SECONDS,
    SAMPLE_RATE,
    build_definition,
    list_wavs,
    measured_snr_db,
    nominal_snrs,
    ordered_paths,
    plan_clip,
    plan_realizations,
    read_frames,
    render_realization,
    snr_mixer,
    _pcm16_roundtrip,
)


def _official_mixer():
    sys.path.insert(0, str(CORPUS))
    from audiolib import snr_mixer as official

    return official


class FormulaTests(unittest.TestCase):
    def test_nominal_grid_and_half_snr(self):
        self.assertEqual(nominal_snrs(), (0.0, 10.0, 20.0, 30.0, 40.0))
        rng = np.random.default_rng(3)
        clean = rng.normal(size=16000)
        noise = rng.normal(size=16000)
        official = _official_mixer()
        for snr in nominal_snrs():
            got = snr_mixer(clean, noise, snr)
            ref = official(clean=clean, noise=noise, snr=snr)
            for left, right in zip(got, ref):
                self.assertEqual(left.dtype, np.float64)
                self.assertTrue(np.array_equal(left, right))
            self.assertAlmostEqual(measured_snr_db(got[0], got[1]), snr / 2.0, places=6)

    def test_long_source_is_not_cropped_and_short_sources_gain_silence(self):
        clean_names = ["a", "b", "c"]
        noise_names = ["n0", "n1"]
        clean_frames = [4, 3, 100]
        noise_frames = [2, 50]
        clip = plan_clip(
            clean_names, noise_names, clean_frames, noise_frames,
            seed=11, split="train", clip_index=0, audio_length=10, silence=2,
        )
        again = plan_clip(
            clean_names, noise_names, clean_frames, noise_frames,
            seed=11, split="train", clip_index=0, audio_length=10, silence=2,
        )
        self.assertEqual(clip, again)
        self.assertGreater(clip["length"], 10)
        self.assertEqual(clip["noise"][-1]["length"] + sum(
            piece.get("length", piece.get("silence", 0)) for piece in clip["noise"][:-1]
        ), clip["length"])
        other = plan_clip(
            clean_names, noise_names, clean_frames, noise_frames,
            seed=11, split="train", clip_index=1, audio_length=10, silence=2,
        )
        self.assertNotEqual(clip["clean"], other["clean"])


class RenderTests(unittest.TestCase):
    def test_one_real_clip_is_deterministic_and_resumable(self):
        if not (CORPUS / "clean_train").is_dir():
            self.skipTest("MS-SNSD corpus is absent")
        clean_paths = list_wavs(CORPUS / "clean_train")[:8]
        noise_paths = list_wavs(CORPUS / "noise_train")[:4]
        clean_names = ordered_paths(clean_paths, seed=17, split="train")
        noise_names = ordered_paths(noise_paths, seed=17, split="train")
        definition = build_definition(
            seed=17, split="train", clean_names=clean_names, noise_names=noise_names,
            total_hours=30, max_clips=1, shard_size=5,
        )
        documents = plan_realizations(
            definition,
            read_frames(CORPUS / "clean_train", clean_names),
            read_frames(CORPUS / "noise_train", noise_names),
        )
        self.assertEqual(len(documents), 5)
        self.assertEqual(
            [item["draws"]["nominal_snr_db"] for item in documents],
            [0.0, 10.0, 20.0, 30.0, 40.0],
        )
        self.assertGreater(documents[0]["output"]["length"], int(AUDIO_SECONDS * SAMPLE_RATE))
        roots = {"speech": CORPUS / "clean_train", "noise": CORPUS / "noise_train"}
        first = render_realization(documents[2], roots)
        second = render_realization(documents[2], roots)
        for key in ("clean", "noise", "noisy"):
            self.assertTrue(np.array_equal(first[key], second[key]))
        self.assertAlmostEqual(measured_snr_db(first["clean"], first["noise"]), 10.0, places=5)
        residual = first["noisy"] - first["clean"] - first["noise"]
        self.assertLess(float(np.max(np.abs(residual))), 1e-12)
        pcm = _pcm16_roundtrip(first["noisy"])
        self.assertTrue(np.array_equal(pcm, _pcm16_roundtrip(second["noisy"])))

    def test_concatenated_files_match_official_audioread(self):
        if not (CORPUS / "audiolib.py").is_file():
            self.skipTest("MS-SNSD audiolib.py is absent")
        official = _official_mixer()
        sys.path.insert(0, str(CORPUS))
        from audiolib import audioread

        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_a = 0.02 * rng.normal(size=4800)
            clean_b = 0.25 * rng.normal(size=6400)
            noise = 0.1 * rng.normal(size=20000)
            sf.write(root / "a.wav", clean_a, SAMPLE_RATE)
            sf.write(root / "b.wav", clean_b, SAMPLE_RATE)
            sf.write(root / "n.wav", noise, SAMPLE_RATE)
            first, rate = audioread(str(root / "a.wav"))
            second, _rate = audioread(str(root / "b.wav"))
            silence = np.zeros(int(rate * 0.2))
            clean = np.concatenate([first, silence, second])
            noise_wave, _rate = audioread(str(root / "n.wav"))
            noise_wave = noise_wave[: clean.shape[0]]
            reference = official(clean=clean, noise=noise_wave, snr=20.0)
            document = {
                "sources": {
                    "speech": [
                        {"name": "a.wav", "start": 0, "length": clean_a.shape[0]},
                        {"silence": int(0.2 * SAMPLE_RATE)},
                        {"name": "b.wav", "start": 0, "length": clean_b.shape[0]},
                    ],
                    "noise": [{"name": "n.wav", "start": 0, "length": int(clean.shape[0])}],
                },
                "draws": {"nominal_snr_db": 20.0, "gen_index": 0},
                "output": {"length": int(clean.shape[0])},
            }
            rendered = render_realization(document, {"speech": root, "noise": root})
        for got, ref in zip((rendered["clean"], rendered["noise"], rendered["noisy"]), reference):
            self.assertEqual(got.shape, np.asarray(ref).shape)
            self.assertTrue(np.array_equal(got, np.asarray(ref)))

    def test_stored_wav_writer_is_gone(self):
        import src.reproduction.build.ms_snsd as module

        self.assertFalse(hasattr(module, "write_build"))
        self.assertFalse(hasattr(module, "verify_build"))


if __name__ == "__main__":
    unittest.main()
