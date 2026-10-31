"""Manifest and read-only DuckDB selection."""

import json
import unittest
from pathlib import Path

import torch
import torchaudio

from src.data.db.schema import create_corpora_schema
from src.evaluation.datasets.duckdb import load_duckdb
from src.evaluation.datasets.manifest import load_manifest
from src.evaluation.errors import DatasetError


ROOT = Path(__file__).resolve().parents[2]


class ManifestTests(unittest.TestCase):
    def test_jsonl_resolves_relative_paths(self):
        with self._temporary_dir() as directory:
            directory = Path(directory)
            clean = directory / "clean.wav"
            _write_tone(clean)
            manifest = directory / "samples.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "utterance_id": "u1",
                        "dataset": "toy",
                        "split": "test",
                        "clean_path": "clean.wav",
                        "transcript": "hello world",
                        "speaker_id": "s1",
                        "room": "quiet",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples = load_manifest(manifest)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].utterance_id, "u1")
            self.assertEqual(samples[0].transcript, "hello world")
            self.assertEqual(samples[0].metadata["room"], "quiet")
            self.assertTrue(Path(samples[0].clean_path).is_file())

    def test_missing_manifest_is_an_error(self):
        with self.assertRaises(DatasetError):
            load_manifest("/tmp/audio-db-missing-manifest.jsonl")

    def test_blank_lines_are_ignored_and_bad_json_is_reported(self):
        with self._temporary_dir() as directory:
            directory = Path(directory)
            manifest = directory / "bad.jsonl"
            manifest.write_text("\n{not json}\n", encoding="utf-8")
            with self.assertRaises(DatasetError) as caught:
                load_manifest(manifest)
            self.assertIn("2", str(caught.exception))

    def _temporary_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()


class DuckDBTests(unittest.TestCase):
    def test_missing_database_does_not_create_one(self):
        missing = Path("/tmp/audio-db-missing-resource.duckdb")
        if missing.exists():
            missing.unlink()
        with self.assertRaises(DatasetError):
            load_duckdb(missing, filters={"corpus": "DEMAND"}, limit=1)
        self.assertFalse(missing.exists())

    def test_filters_select_a_row_from_a_temporary_database(self):
        import duckdb

        with self._temporary_dir() as directory:
            root = Path(directory)
            audio = root / "Toy" / "a.wav"
            _write_tone(audio)
            database = root / "toy.duckdb"
            connection = duckdb.connect(str(database))
            create_corpora_schema(connection, recreate=False)
            connection.execute(
                """
                INSERT INTO corpora.audio_files (
                    path, corpus, subset, file_name, sample_rate, frames, channels,
                    speaker_id, utterance_id, transcript
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ["Toy/a.wav", "Toy", "test", "a.wav", 16000, 1600, 1, "spk", "a", "hello"],
            )
            connection.close()
            samples = load_duckdb(
                database,
                filters={"corpus": "Toy", "speaker_id": "spk"},
                limit=1,
                role="clean",
                corpora_root=root,
            )
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].dataset, "Toy")
            self.assertEqual(samples[0].transcript, "hello")
            self.assertTrue(samples[0].clean_path.endswith("Toy/a.wav"))
            self.assertTrue(Path(samples[0].clean_path).is_file())

    def test_existing_resource_database_is_not_rewritten(self):
        database = ROOT / "data/db/resource.duckdb"
        if not database.is_file():
            self.skipTest("resource.duckdb is not present")
        import duckdb

        connection = duckdb.connect(str(database), read_only=True)
        before = connection.execute("SELECT count(*) FROM corpora.audio_files").fetchone()[0]
        connection.close()
        samples = load_duckdb(
            database,
            filters={"corpus": "DEMAND"},
            limit=1,
            role="noise",
            corpora_root=ROOT / "data/corpora",
        )
        connection = duckdb.connect(str(database), read_only=True)
        after = connection.execute("SELECT count(*) FROM corpora.audio_files").fetchone()[0]
        connection.close()
        self.assertEqual(before, after)
        self.assertEqual(samples[0].dataset, "DEMAND")
        self.assertTrue(samples[0].noise_path.endswith(".wav"))

    def _temporary_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()


def _write_tone(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = torch.arange(1600, dtype=torch.float32) / 16000
    torchaudio.save(str(path), (0.1 * torch.sin(2 * torch.pi * 220 * time)).reshape(1, -1), 16000)
