import unittest
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import validate_realization
from src.experiments.data.prepare.voicebank_shift_recipe import canonical_shift_recipe, read_staged_index, shift_case_id, snapshot_realization, write_staged_index


class VoiceBankShiftRecipeTests(unittest.TestCase):
    def test_case_identity_preserves_recorded_gain_precision(self):
        row = {"status": "ready", "clean_path": "data/corpora/voicebank/clean.wav",
               "noise_path": "data/corpora/noise/source.wav", "clean_sha256": "a" * 64,
               "noise_sha256": "b" * 64, "noise_start_48k": "5", "noise_end_exclusive_48k": "50",
               "clean_gain_g": 0.99999991, "shifted_noise_gain": "0.12345671"}
        recipe = canonical_shift_recipe(row)
        self.assertNotIn("case_id", recipe)
        self.assertEqual(float(recipe["shifted_noise_gain"]), float(row["shifted_noise_gain"]))
        original = shift_case_id(row)
        self.assertEqual(original, shift_case_id(dict(row)))
        realization = snapshot_realization({"case_id": original, "recipe": recipe})
        self.assertEqual(validate_realization(realization), realization)
        self.assertEqual(realization["draws"]["source_recipe_case_id"], original)
        self.assertEqual(realization["draws"]["shifted_noise_gain"], recipe["shifted_noise_gain"])
        self.assertEqual(realization["output"]["length"], 15)
        changed = dict(row, shifted_noise_gain="0.12345672")
        self.assertNotEqual(original, shift_case_id(changed))
        with self.assertRaises(ReproductionError):
            shift_case_id(dict(row, noise_start_48k=None))
        with self.assertRaises(ReproductionError):
            shift_case_id(dict(row, clean_path="data/corpora/../other.wav"))

    def test_staging_is_persistent_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw) / "staging"
            with patch("src.experiments.data.prepare.voicebank_shift_recipe.build_staged_index",
                       return_value=({"case_count": 1}, [{"case_id": "a" * 64}])):
                write_staged_index(Path("unused"), "unused", directory)
                before = (directory / "cases.jsonl").read_bytes()
                metadata = json.loads((directory / "metadata.json").read_text())
                self.assertEqual(metadata["case_count"], 1)
                with self.assertRaises(ReproductionError):
                    write_staged_index(Path("unused"), "unused", directory)
                self.assertEqual((directory / "cases.jsonl").read_bytes(), before)

    def test_staging_reader_detects_tampering_even_if_case_hash_is_rewritten(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            template = {"status": "ready", "clean_path": "data/corpora/clean.wav",
                        "noise_path": "data/corpora/noise.wav", "clean_sha256": "a" * 64,
                        "noise_sha256": "b" * 64, "clean_gain_g": 1.0, "shifted_noise_gain": 0.5}
            with source.open("w", encoding="utf-8") as f:
                for speaker, split, count in (("p226", "train", 10700), ("p286", "train", 872), ("p232", "test", 824)):
                    for n in range(count):
                        item = dict(template, utterance_id=f"{speaker}_{n}", split=split,
                                    clean_path=f"data/corpora/{speaker}_{n}.wav",
                                    noise_start_48k=n * 3, noise_end_exclusive_48k=n * 3 + 2)
                        f.write(json.dumps(item) + "\n")
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            destination = root / "staging"
            write_staged_index(source, sha, destination)
            _metadata, rows = read_staged_index(destination, source, sha)
            self.assertEqual(len(rows), 12396)
            path = destination / "cases.jsonl"
            data = path.read_bytes().replace(b'"speaker_id":"p226"', b'"speaker_id":"p227"', 1)
            path.write_bytes(data)
            metadata_path = destination / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["cases_sha256"] = hashlib.sha256(data).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(ReproductionError):
                read_staged_index(destination, source, sha)
