"""DuckDB evaluation sets: import, identity, and the shared dataset API."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio

from src.data.db.schema import create_corpora_schema
from src.data.recipe import case_content_hash
from src.evaluation.datasets.evaldb import EvaluationDB
from src.evaluation.errors import DatasetError
from src.evaluation.pipeline import execute
from src.data.init_eval import initialize
from tests.evaluation.support import experiment_config

ROOT = Path(__file__).resolve().parents[2]


class EvaluationDBTests(unittest.TestCase):
    def test_init_is_idempotent_and_leaves_corpus_rows_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "resource.duckdb"
            self._insert_corpus_row(database, root)
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            before = db.corpora_fingerprint()
            db.ensure_schema()
            db.ensure_schema()
            manifest = self._manifest(root, [("c1", "clean")])
            first = db.import_manifest(manifest, "once")
            second = db.import_manifest(manifest, "once")
            self.assertEqual(first.inserted, 1)
            self.assertEqual(second.inserted, 0)
            self.assertEqual(second.unchanged, 1)
            self.assertEqual(second.set_status, "unchanged")
            self.assertEqual(db.stats(), {"cases": 1, "sets": 1, "items": 1})
            self.assertEqual(db.corpora_fingerprint(), before)
            connection = _connect(database)
            try:
                count = connection.execute("SELECT count(*) FROM corpora.audio_files").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 1)

    def test_reregistration_keeps_migrated_legacy_case_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sets.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            manifest = self._manifest(root, [("c1", "clean")])
            db.import_manifest(manifest, "once", description="clean")
            connection = _connect(database)
            try:
                case_id, kind, dataset, split, rate, transcript, speaker, recipe, metadata = connection.execute(
                    """
                    SELECT case_id, kind, dataset, split, sample_rate, transcript, speaker_id,
                           recipe_json::VARCHAR, metadata::VARCHAR
                    FROM eval.cases
                    """
                ).fetchone()
                stored = json.loads(metadata)
                legacy = "corpus:old-id"
                stored["legacy_case_id"] = legacy
                stored["source_id"] = legacy
                body = {
                    "kind": kind,
                    "dataset": dataset,
                    "split": split,
                    "sample_rate": rate,
                    "transcript": transcript,
                    "speaker_id": speaker,
                    "recipe": json.loads(recipe),
                    "metadata": stored,
                }
                connection.execute(
                    """
                    UPDATE eval.cases
                    SET metadata = ?::JSON, content_hash = ?
                    WHERE case_id = ?
                    """,
                    [json.dumps(stored, sort_keys=True), case_content_hash(body), case_id],
                )
            finally:
                connection.close()
            again = db.import_manifest(manifest, "once", description="clean")
            self.assertEqual(again.inserted, 0)
            self.assertEqual(again.unchanged, 1)
            self.assertEqual(again.set_status, "unchanged")
            connection = _connect(database)
            try:
                kept = json.loads(
                    connection.execute("SELECT metadata::VARCHAR FROM eval.cases").fetchone()[0]
                )
            finally:
                connection.close()
            self.assertEqual(kept["legacy_case_id"], "corpus:old-id")
            self.assertEqual(kept["source_id"], "corpus:old-id")

    def test_manifest_round_trip_survives_deleting_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sets.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            manifest = self._manifest(
                root,
                [
                    ("b", "clean"),
                    ("a", "noisy"),
                    ("p", "paired"),
                    ("s", "synthetic"),
                ],
            )
            report = db.import_manifest(manifest, "kinds", description="four kinds")
            self.assertEqual(report.inserted, 4)
            self.assertEqual(report.set_status, "created")
            manifest.unlink()
            first = db.load_dataset("kinds")
            second = db.load_dataset("kinds")
            self.assertEqual([sample.metadata["source_id"] for sample in first], ["b", "a", "p", "s"])
            self.assertEqual(
                [sample.utterance_id for sample in second],
                [sample.utterance_id for sample in first],
            )
            self.assertTrue(all(len(sample.utterance_id) == 64 for sample in first))
            by_id = {sample.metadata["source_id"]: sample for sample in first}
            self.assertTrue(by_id["b"].clean_path.endswith("b-clean.wav"))
            self.assertIsNone(by_id["b"].clean_waveform)
            self.assertEqual(by_id["b"].metadata["reference"], "dry_clean")
            self.assertEqual(by_id["b"].metadata["recipe"]["operations"], [])
            self.assertTrue(by_id["a"].noisy_path.endswith("a-noisy.wav"))
            self.assertEqual(by_id["a"].metadata["reference"], "none")
            self.assertTrue(by_id["p"].clean_path.endswith("p-clean.wav"))
            self.assertTrue(by_id["p"].noisy_path.endswith("p-noisy.wav"))
            self.assertEqual(by_id["s"].metadata["eval_kind"], "synthetic")
            noise = by_id["s"].metadata["recipe"]["operations"][0]
            self.assertEqual(noise["type"], "additive_noise")
            self.assertIsInstance(noise["seed"], int)
            self.assertEqual(noise["noise_start_unit"], "sample")
            self.assertIsNone(by_id["s"].clean_waveform)
            self.assertEqual(by_id["s"].transcript, "text s")
            self.assertEqual(by_id["b"].speaker_id, "spk")

    def test_same_id_with_different_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sets.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            manifest = self._manifest(root, [("u1", "clean")])
            db.import_manifest(manifest, "fixed")
            changed = self._manifest(root, [("u1", "clean")], transcript="other")
            with self.assertRaises(DatasetError) as caught:
                db.import_manifest(changed, "fixed")
            self.assertIn("u1", str(caught.exception))
            self.assertEqual(db.load_dataset("fixed")[0].transcript, "text u1")
            self.assertEqual(db.stats()["cases"], 1)

    def test_invalid_manifest_and_missing_audio_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sets.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "missing-corpora")
            bad = root / "bad.jsonl"
            bad.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(DatasetError):
                db.import_manifest(bad, "bad")
            missing = root / "missing.jsonl"
            missing.write_text(
                json.dumps(
                    {
                        "utterance_id": "gone",
                        "dataset": "toy",
                        "clean_path": "no-such.wav",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetError) as caught:
                db.import_manifest(missing, "gone")
            self.assertIn("no-such.wav", str(caught.exception))
            outside = root / "outside.jsonl"
            audio = Path(tempfile.gettempdir()) / "audio-db-outside-eval.wav"
            _write_wav(audio, 1600, 220)
            outside.write_text(
                json.dumps(
                    {
                        "utterance_id": "out",
                        "dataset": "toy",
                        "clean_path": audio.as_posix(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetError) as caught:
                db.import_manifest(outside, "out")
            self.assertIn("outside", str(caught.exception))

    def test_json_array_manifest_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_wav(root / "clean.wav", 1600, 220)
            manifest = root / "rows.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "utterance_id": "j1",
                            "dataset": "toy",
                            "clean_path": "clean.wav",
                            "transcript": "json",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            db = EvaluationDB(root / "sets.duckdb", project_root=root, corpora_root=root / "corpora")
            report = db.import_manifest(manifest, "json_array")
            self.assertEqual(report.inserted, 1)
            self.assertEqual(db.load_dataset("json_array")[0].transcript, "json")

    def test_synthetic_mixture_matches_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sets.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            manifest = self._manifest(root, [("mix", "synthetic")])
            db.import_manifest(manifest, "mixes")
            sample = db.load_dataset("mixes")[0]
            left = _render_sample(sample)
            right = _render_sample(db.load_dataset("mixes")[0])
            self.assertTrue(torch.equal(left.observed, right.observed))
            stored = sample.metadata["recipe"]["operations"][0]["noise_start"]
            self.assertEqual(left.noise_start, stored)
            self.assertEqual(left.noise_start, right.noise_start)

    def test_pipeline_uses_the_stored_mixture_for_two_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sets.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            manifest = self._manifest(root, [("mix", "synthetic")])
            db.import_manifest(manifest, "mixes")
            output = root / "runs"
            hashes = []
            mix_seeds = []
            for seed, run_id in ((0, "seed0"), (1, "seed1")):
                config = experiment_config(output, run_id=run_id)
                config.experiment.seed = seed
                config.experiment.limit = None
                config.dataset.source = "duckdb"
                config.dataset.path = str(database)
                config.dataset.set_name = "mixes"
                config.dataset.corpora_root = str(root / "corpora")
                config.mixing.enabled = False
                config.asr.enabled = False
                config.metrics.wer = False
                config.metrics.pesq = False
                config.metrics.stoi = False
                result = execute(config)
                self.assertEqual(result["summary"]["num_ok"], 1)
                record = result["records"][0]
                hashes.append(record["identity"]["input_sha256"])
                mix_seeds.append(record["seeds"]["mix"])
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(mix_seeds[0], mix_seeds[1])

    def test_corpus_query_creates_a_set_without_a_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "resource.duckdb"
            self._insert_corpus_row(database, root, name="b.wav", transcript="second")
            self._insert_corpus_row(database, root, name="a.wav", transcript="first")
            _write_wav(root / "corpora" / "Toy" / "noise.wav", 8000, 50)
            connection = _connect(database)
            try:
                connection.execute(
                    """
                    INSERT INTO corpora.audio_files (
                        path, corpus, subset, file_name, sample_rate, frames, channels
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    ["Toy/noise.wav", "Noise", "all", "noise.wav", 16000, 8000, 1],
                )
            finally:
                connection.close()
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            report = db.create_set_from_corpus(
                "from_sql",
                role="clean",
                query="SELECT * FROM corpora.audio_files WHERE corpus = ? ORDER BY path",
                params=["Toy"],
            )
            self.assertEqual(report.inserted, 2)
            loaded = db.load_dataset("from_sql")
            self.assertEqual([sample.transcript for sample in loaded], ["first", "second"])
            synthetic = db.create_set_from_corpus(
                "from_filters",
                role="synthetic",
                filters={"corpus": "Toy"},
                noise_filters={"corpus": "Noise"},
                snr_db=5,
                seed=3,
                max_seconds=0.1,
            )
            self.assertEqual(synthetic.inserted, 2)
            mixed = db.load_dataset("from_filters")
            self.assertEqual(len(mixed), 2)
            self.assertEqual(mixed[0].metadata["eval_kind"], "synthetic")
            self.assertTrue(Path(mixed[0].clean_path).is_file())
            self.assertTrue(Path(mixed[0].noise_path).is_file())
            with self.assertRaises(DatasetError):
                db.create_set_from_corpus("bad_sql", query="DELETE FROM corpora.audio_files")
            _write_wav(root / "corpora" / "Toy" / "c.wav", 1600, 440)
            connection = _connect(database)
            try:
                connection.execute(
                    """
                    INSERT INTO corpora.audio_files (
                        path, corpus, subset, file_name, sample_rate, frames, channels,
                        speaker_id, utterance_id, transcript
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ["Toy/c.wav", "Toy", "test", "c.wav", 16000, 1600, 1, "spk", "c", "third"],
                )
            finally:
                connection.close()
            linked = root / "linked.jsonl"
            linked.write_text(
                json.dumps(
                    {
                        "utterance_id": "linked",
                        "dataset": "Toy",
                        "clean_path": "corpora/Toy/c.wav",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            db.import_manifest(linked, "linked")
            sample = db.load_dataset("linked")[0]
            self.assertEqual(sample.transcript, "third")
            self.assertEqual(sample.speaker_id, "spk")
            self.assertEqual(sample.metadata["corpus_path"], "Toy/c.wav")

    def test_initialize_skips_optional_inputs_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "empty.duckdb"
            config = root / "init.yaml"
            config.write_text(
                "\n".join(
                    [
                        f"database: {database.as_posix()}",
                        "corpora_root: corpora",
                        "manifests:",
                        "  - path: missing.jsonl",
                        "    set: example",
                        "    required: false",
                        "corpus_sets:",
                        "  - name: absent",
                        "    required: false",
                        "    role: clean",
                        "    filters:",
                        "      corpus: Missing",
                    ]
                ),
                encoding="utf-8",
            )
            # Paths in the config are used as written. Run with this directory as the project.
            original = Path.cwd()
            try:
                import os

                os.chdir(root)
                first = initialize(config)
                second = initialize(config)
            finally:
                os.chdir(original)
            self.assertIn("skipped", first)
            self.assertIn("not present", first)
            self.assertIn("skipped", second)
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            self.assertEqual(db.stats(), {"cases": 0, "sets": 0, "items": 0})
            self.assertIsNone(db.corpora_fingerprint())

    def _manifest(self, root: Path, rows: list[tuple[str, str]], transcript: str | None = None) -> Path:
        lines = []
        for utterance_id, kind in rows:
            record = {
                "utterance_id": utterance_id,
                "dataset": "toy",
                "split": "test",
                "speaker_id": "spk",
                "transcript": transcript or f"text {utterance_id}",
            }
            if kind == "clean":
                _write_wav(root / f"{utterance_id}-clean.wav", 1600, 220)
                record["clean_path"] = f"{utterance_id}-clean.wav"
            elif kind == "noisy":
                _write_wav(root / f"{utterance_id}-noisy.wav", 1600, 330)
                record["noisy_path"] = f"{utterance_id}-noisy.wav"
            elif kind == "paired":
                _write_wav(root / f"{utterance_id}-clean.wav", 1600, 220)
                _write_wav(root / f"{utterance_id}-noisy.wav", 1600, 330)
                record["clean_path"] = f"{utterance_id}-clean.wav"
                record["noisy_path"] = f"{utterance_id}-noisy.wav"
            elif kind == "synthetic":
                _write_wav(root / f"{utterance_id}-clean.wav", 3200, 220)
                _write_wav(root / f"{utterance_id}-noise.wav", 16000, 40)
                record["clean_path"] = f"{utterance_id}-clean.wav"
                record["noise_path"] = f"{utterance_id}-noise.wav"
                record["snr_db"] = 0
            lines.append(json.dumps(record))
        path = root / "samples.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _insert_corpus_row(self, database: Path, root: Path, name: str = "a.wav", transcript: str = "hello") -> None:
        audio = root / "corpora" / "Toy" / name
        _write_wav(audio, 1600, 220)
        connection = _connect(database)
        try:
            create_corpora_schema(connection, recreate=False)
            connection.execute(
                """
                INSERT INTO corpora.audio_files (
                    path, corpus, subset, file_name, sample_rate, frames, channels,
                    speaker_id, utterance_id, transcript
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [f"Toy/{name}", "Toy", "test", name, 16000, 1600, 1, "spk", Path(name).stem, transcript],
            )
        finally:
            connection.close()


def _connect(database: Path):
    import duckdb

    database.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(database))


def _write_wav(path: Path, frames: int, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = torch.arange(frames, dtype=torch.float32) / 16000
    wave = (0.1 * torch.sin(2 * torch.pi * frequency * time)).reshape(1, -1)
    torchaudio.save(str(path), wave, 16000)


def _render_sample(sample):
    from src.evaluation.audio.render import render_recipe

    return render_recipe(sample.metadata["recipe"], sample.metadata["audio_root"])


if __name__ == "__main__":
    unittest.main()
