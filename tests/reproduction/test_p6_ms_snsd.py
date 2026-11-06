"""MS-SNSD mixture planning against the ESPnet recipe's audiolib."""

from __future__ import annotations

import json
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
    write_build,
)
from src.reproduction.errors import ReproductionError


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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_build(
                definition, documents,
                audio_root=root / "generated", staging_root=root / "staging",
                clean_dir=CORPUS / "clean_train", noise_dir=CORPUS / "noise_train",
            )
            manifest = json.loads((first["staging_dir"] / "shards" / "shard-0000.json").read_text(encoding="utf-8"))
            hashes = {item["rel_path"]: item["sha256"] for item in manifest["files"]}
            noisy = next(item for item in manifest["files"] if item["stream"] == "noisy" and item["gen_index"] == 2)
            audio, rate = sf.read(first["audio_dir"] / noisy["rel_path"], dtype="float64")
            self.assertEqual(rate, SAMPLE_RATE)
            self.assertEqual(audio.shape[0], noisy["frames"])
            waves = render_realization(documents[2], {"speech": CORPUS / "clean_train", "noise": CORPUS / "noise_train"})
            self.assertAlmostEqual(measured_snr_db(waves["clean"], waves["noise"]), 10.0, places=5)
            residual = waves["noisy"] - waves["clean"] - waves["noise"]
            self.assertLess(float(np.max(np.abs(residual))), 1e-12)
            second = write_build(
                definition, documents,
                audio_root=root / "generated", staging_root=root / "staging",
                clean_dir=CORPUS / "clean_train", noise_dir=CORPUS / "noise_train",
            )
            again = json.loads((second["staging_dir"] / "shards" / "shard-0000.json").read_text(encoding="utf-8"))
            self.assertEqual(hashes, {item["rel_path"]: item["sha256"] for item in again["files"]})
            shard = second["staging_dir"] / "shards" / "shard-0000.json"
            shard.unlink()
            target = first["audio_dir"] / noisy["rel_path"]
            target.write_bytes(b"broken")
            third = write_build(
                definition, documents,
                audio_root=root / "generated", staging_root=root / "staging",
                clean_dir=CORPUS / "clean_train", noise_dir=CORPUS / "noise_train",
            )
            restored = json.loads((third["staging_dir"] / "shards" / "shard-0000.json").read_text(encoding="utf-8"))
            self.assertEqual(hashes[noisy["rel_path"]], next(item["sha256"] for item in restored["files"] if item["rel_path"] == noisy["rel_path"]))
            self.assertTrue((third["staging_dir"] / "verified.json").is_file())

    def test_a_different_definition_is_refused(self):
        definition = build_definition(
            seed=1, split="train", clean_names=["a.wav"], noise_names=["n.wav"],
            total_hours=0.01, max_clips=1,
        )
        other = dict(definition)
        other["seed"] = 2
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "staging" / "builds" / definition["build_id"]
            stage.mkdir(parents=True)
            (stage / "build.json").write_text(json.dumps(other), encoding="utf-8")
            with self.assertRaises(ReproductionError):
                write_build(
                    definition,
                    [],
                    audio_root=Path(directory) / "generated",
                    staging_root=Path(directory) / "staging",
                    clean_dir=Path(directory),
                    noise_dir=Path(directory),
                )


if __name__ == "__main__":
    unittest.main()
