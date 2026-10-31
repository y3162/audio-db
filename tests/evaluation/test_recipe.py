"""Recipe identity, rendering, and staging rebuilds."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio

from src.data.recipe import RecipeError, canonical_recipe, case_id_for, sha256_file
from src.data.staging import StagingError, export_database, rebuild_database, register_run, register_staging
from src.evaluation.audio.mixing import rms_snr_db
from src.evaluation.audio.render import render_recipe
from src.evaluation.datasets.evaldb import EvaluationDB
from src.evaluation.errors import DatasetError
from src.evaluation.pipeline import execute
from tests.evaluation.support import experiment_config


class RecipeIdentityTests(unittest.TestCase):
    def test_same_recipe_keeps_the_same_id_regardless_of_key_order(self):
        left = _recipe()
        right = {
            "output": dict(reversed(list(left["output"].items()))),
            "operations": left["operations"],
            "sources": left["sources"],
            "recipe_version": 1,
        }
        self.assertEqual(case_id_for(left), case_id_for(right))
        self.assertNotIn("case_id", canonical_recipe(left))

    def test_parameter_and_order_changes_change_the_id(self):
        base = case_id_for(_recipe())
        changed = _recipe()
        changed["operations"][0]["snr_db"] = -5
        self.assertNotEqual(base, case_id_for(changed))
        ordered = _recipe()
        ordered["operations"] = [
            {"type": "rir", "version": 1, "source": "rir", "channel": 0, "length": "trim_to_input"},
            {"type": "peak_normalize", "version": 1, "limit": 0.5},
        ]
        ordered["output"]["reference"] = "dry_clean"
        flipped = dict(ordered)
        flipped["operations"] = list(reversed(ordered["operations"]))
        self.assertNotEqual(case_id_for(ordered), case_id_for(flipped))

    def test_unknown_operation_version_and_shape_are_rejected(self):
        unknown = _recipe()
        unknown["operations"][0]["type"] = "reverb_sim"
        with self.assertRaises(RecipeError):
            case_id_for(unknown)
        bad_version = _recipe()
        bad_version["operations"][0]["version"] = 2
        with self.assertRaises(RecipeError):
            case_id_for(bad_version)
        missing_start = _recipe()
        del missing_start["operations"][0]["noise_start"]
        with self.assertRaises(RecipeError):
            case_id_for(missing_start)
        with self.assertRaises(RecipeError):
            canonical_recipe({"recipe_version": 1, "sources": {}, "operations": [], "output": {}})


class RenderTests(unittest.TestCase):
    def test_additive_noise_snr_start_rate_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = _tone(root / "clean.wav", 1600, 220, 16000)
            noise = _tone(root / "noise.wav", 800, 90, 8000)
            recipe = _noise_recipe(root, clean, noise, snr_db=-5, seed=4)
            first = render_recipe(recipe, root)
            second = render_recipe(recipe, root)
            self.assertTrue(torch.equal(first.observed, second.observed))
            self.assertEqual(first.noise_start, recipe["operations"][0]["noise_start"])
            self.assertEqual(first.sample_rate, 16000)
            self.assertEqual(first.observed.numel(), 1600)
            self.assertEqual(first.reference_name, "dry_clean")
            self.assertAlmostEqual(first.snr_db_achieved, -5.0, places=2)
            self.assertAlmostEqual(rms_snr_db(first.reference, first.observed - first.reference), -5.0, places=2)

    def test_rir_identity_known_kernel_and_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = _tone(root / "clean.wav", 8, 220, 16000)
            identity = _save(root / "identity.wav", torch.tensor([[1.0]]), 16000)
            known = _save(root / "known.wav", torch.tensor([[0.5, 0.25]]), 16000)
            stereo = _save(root / "stereo.wav", torch.tensor([[0.0, 1.0], [1.0, 0.0]]), 16000)
            original, _rate = torchaudio.load(str(clean))
            signal = original[0]
            rendered = render_recipe(_rir_recipe(root, clean, identity, reference="dry_clean"), root)
            self.assertTrue(torch.allclose(rendered.observed, signal, atol=1e-5))
            self.assertEqual(rendered.observed.numel(), 8)
            convolved = render_recipe(
                _rir_recipe(root, clean, known, reference="reverberant_clean"),
                root,
            )
            expected = torch.empty_like(signal)
            expected[0] = 0.5 * signal[0]
            expected[1:] = 0.5 * signal[1:] + 0.25 * signal[:-1]
            self.assertTrue(torch.allclose(convolved.reference, expected, atol=1e-5))
            self.assertTrue(torch.allclose(convolved.observed, expected, atol=1e-5))
            selected = render_recipe(
                _rir_recipe(root, clean, stereo, reference="dry_clean", channel=1),
                root,
            )
            self.assertTrue(torch.allclose(selected.observed, signal, atol=1e-4))
            delayed = render_recipe(
                _rir_recipe(root, clean, stereo, reference="reverberant_clean", channel=0),
                root,
            )
            self.assertAlmostEqual(float(delayed.reference[0]), 0.0, places=4)
            self.assertAlmostEqual(float(delayed.reference[1]), float(signal[0]), places=4)
            noise = _tone(root / "noise.wav", 32, 40, 16000)
            both = _recipe_files(
                root,
                clean,
                noise=noise,
                rir=known,
                operations=[
                    {"type": "rir", "version": 1, "source": "rir", "channel": 0, "length": "trim_to_input"},
                    {
                        "type": "additive_noise",
                        "version": 1,
                        "source": "noise",
                        "snr_db": 0,
                        "noise_start": 0,
                        "noise_start_unit": "sample",
                        "seed": 1,
                    },
                    {"type": "peak_normalize", "version": 1, "limit": 0.05},
                ],
                reference="reverberant_clean",
            )
            mixed = render_recipe(both, root)
            again = render_recipe(both, root)
            self.assertTrue(torch.equal(mixed.observed, again.observed))
            self.assertLessEqual(float(mixed.observed.abs().max()), 0.05 + 1e-5)
            self.assertEqual(mixed.reference_name, "reverberant_clean")
            self.assertEqual(mixed.observed.numel(), mixed.reference.numel())
            self.assertTrue(mixed.peak_scaled)

    def test_changed_source_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = _tone(root / "clean.wav", 160, 220, 16000)
            noise = _tone(root / "noise.wav", 160, 40, 16000)
            recipe = _noise_recipe(root, clean, noise, snr_db=0, seed=1)
            clean.write_bytes(clean.read_bytes() + b"\x00")
            with self.assertRaises(RecipeError) as caught:
                render_recipe(recipe, root)
            self.assertIn("changed", str(caught.exception))


class StagingTests(unittest.TestCase):
    def test_rebuild_is_idempotent_order_independent_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "source.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            _register_pair(db, root)
            staging = root / "staging"
            export_database(database, staging)
            rebuilt = root / "rebuilt.duckdb"
            first = rebuild_database(staging, rebuilt)
            self.assertGreaterEqual(first["cases"], 2)
            second = register_staging(staging, rebuilt)
            self.assertEqual(second["cases"], 0)
            self.assertEqual(second["sets"], 0)
            reversed_dir = root / "reversed"
            _reverse_cases(staging, reversed_dir)
            other = root / "other.duckdb"
            rebuild_database(reversed_dir, other)
            self.assertEqual(_snapshot(rebuilt), _snapshot(other))
            conflict = root / "conflict.duckdb"
            EvaluationDB(conflict, project_root=root, corpora_root=root / "corpora").import_manifest(
                _manifest(root, "b", transcript="other"),
                "kinds",
            )
            before = _snapshot(conflict)
            with self.assertRaises(StagingError):
                register_staging(staging, conflict)
            self.assertEqual(_snapshot(conflict), before)
            with self.assertRaises(StagingError):
                rebuild_database(staging, rebuilt)

    def test_incomplete_staging_is_rejected_before_a_database_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            (staging / "metadata.json").write_text(
                json.dumps({"staging_version": 1, "files": {"cases": "missing.parquet"}}),
                encoding="utf-8",
            )
            destination = root / "new.duckdb"
            with self.assertRaises(StagingError):
                rebuild_database(staging, destination)
            self.assertFalse(destination.exists())

    def test_results_round_trip_and_legacy_schema_is_left_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "source.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            _register_pair(db, root)
            output = root / "runs"
            config = experiment_config(output, run_id="recipe-run")
            config.dataset.source = "duckdb"
            config.dataset.path = str(database)
            config.dataset.set_name = "kinds"
            config.dataset.corpora_root = str(root / "corpora")
            config.mixing.enabled = False
            config.experiment.limit = 1
            config.asr.enabled = False
            config.metrics.wer = False
            config.metrics.pesq = False
            config.metrics.stoi = False
            result = execute(config)
            self.assertEqual(result["summary"]["num_ok"], 1)
            register_run(database, result["run_dir"])
            with self.assertRaises(StagingError):
                register_run(database, result["run_dir"] + "-missing")
            changed = dict(result["records"][0])
            changed["utterance_id"] = "different"
            # Same record id with different content must fail and keep the first row.
            run_dir = Path(result["run_dir"])
            original = (run_dir / "results.jsonl").read_text(encoding="utf-8")
            (run_dir / "results.jsonl").write_text(
                json.dumps(changed, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(StagingError):
                register_run(database, run_dir)
            (run_dir / "results.jsonl").write_text(original, encoding="utf-8")
            register_run(database, run_dir)
            staging = root / "staging"
            export_database(database, staging)
            rebuilt = root / "rebuilt.duckdb"
            rebuild_database(staging, rebuilt)
            self.assertEqual(_result_ids(database), _result_ids(rebuilt))
            legacy = root / "legacy.duckdb"
            connection = _duck(legacy)
            connection.execute("CREATE SCHEMA eval")
            connection.execute("CREATE TABLE eval.cases (case_id VARCHAR PRIMARY KEY, snr_db DOUBLE)")
            connection.execute("INSERT INTO eval.cases VALUES ('kept', 1.5)")
            connection.close()
            with self.assertRaises(DatasetError) as caught:
                EvaluationDB(legacy, project_root=root).ensure_schema()
            self.assertIn("previous", str(caught.exception))
            connection = _duck(legacy, read_only=True)
            row = connection.execute("SELECT case_id, snr_db FROM eval.cases").fetchone()
            columns = connection.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'eval' AND table_name = 'cases'
                """
            ).fetchall()
            connection.close()
            self.assertEqual(row, ("kept", 1.5))
            self.assertEqual([item[0] for item in columns], ["case_id", "snr_db"])


class RecipePipelineTests(unittest.TestCase):
    def test_named_set_renders_references_and_partial_wer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sets.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            manifest = _manifest(root, "whole")
            cropped = _manifest(root, "crop", max_seconds=0.01)
            noisy = root / "noisy.jsonl"
            _tone(root / "only-noisy.wav", 1600, 330, 16000)
            noisy.write_text(
                json.dumps(
                    {
                        "utterance_id": "noisy",
                        "dataset": "toy",
                        "noisy_path": "only-noisy.wav",
                        "transcript": "noise only",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            db.import_manifest(manifest, "whole")
            db.import_manifest(cropped, "crop")
            db.import_manifest(noisy, "noisy")
            whole = _run(database, root, "whole")
            self.assertEqual(whole["status"], "ok")
            self.assertEqual(whole["reference_name"], "dry_clean")
            self.assertFalse(whole["reference_audio"]["truncated"])
            self.assertEqual(whole["wer"]["status"], "ok")
            self.assertGreater(whole["wer"]["values"]["enhanced"]["normalized"]["N"], 0)
            crop = _run(database, root, "crop")
            self.assertEqual(crop["wer"]["status"], "na")
            self.assertEqual(crop["wer"]["reason"], "partial_audio")
            missing = _run(database, root, "noisy", pesq=True)
            self.assertEqual(missing["pesq"]["status"], "na")
            self.assertEqual(missing["pesq"]["reason"], "no_clean_reference")
            self.assertEqual(missing["reference_name"], "none")

    def test_generated_audio_reaches_a_real_enhancer_when_the_checkpoint_exists(self):
        checkpoint = Path(__file__).resolve().parents[2] / "data/checkpoints/mp-senet/g_best_vb"
        if not checkpoint.is_file():
            self.skipTest("MP-SENet checkpoint is not available")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sets.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root / "corpora")
            db.import_manifest(_manifest(root, "se", frames=16000), "se")
            config = experiment_config(root / "runs", run_id="mp-recipe")
            config.dataset.source = "duckdb"
            config.dataset.path = str(database)
            config.dataset.set_name = "se"
            config.dataset.corpora_root = str(root / "corpora")
            config.mixing.enabled = False
            config.experiment.limit = 1
            config.asr.enabled = False
            config.metrics.wer = False
            config.metrics.stoi = False
            config.models[0].name = "mp_senet"
            config.models[0].variant = "vb"
            config.models[0].checkpoint = str(checkpoint)
            config.runtime.device = "cpu"
            result = execute(config)
            record = result["records"][0]
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["model"]["name"], "mp_senet")
            self.assertEqual(record["pesq"]["values"]["enhanced"]["status"], "ok")
            self.assertIsInstance(record["pesq"]["values"]["enhanced"]["value"], float)


def _recipe() -> dict:
    return {
        "recipe_version": 1,
        "sources": {
            "clean": {"path": "clean.wav", "sha256": "a" * 64},
            "noise": {"path": "noise.wav", "sha256": "b" * 64},
            "rir": {"path": "rir.wav", "sha256": "c" * 64},
        },
        "operations": [
            {
                "type": "additive_noise",
                "version": 1,
                "source": "noise",
                "snr_db": 0,
                "noise_start": 3,
                "noise_start_unit": "sample",
                "seed": 9,
            },
            {
                "type": "rir",
                "version": 1,
                "source": "rir",
                "channel": 0,
                "length": "trim_to_input",
            },
        ],
        "output": {"sample_rate": 16000, "reference": "dry_clean"},
    }


def _noise_recipe(root: Path, clean: Path, noise: Path, *, snr_db: float, seed: int) -> dict:
    from src.evaluation.audio.render import prepare_noise_start

    start = prepare_noise_start(clean, noise, 16000, seed, None)
    return _recipe_files(
        root,
        clean,
        noise=noise,
        operations=[
            {
                "type": "additive_noise",
                "version": 1,
                "source": "noise",
                "snr_db": snr_db,
                "noise_start": start,
                "noise_start_unit": "sample",
                "seed": seed,
            }
        ],
        reference="dry_clean",
    )


def _rir_recipe(root: Path, clean: Path, rir: Path, *, reference: str, channel: int = 0) -> dict:
    return _recipe_files(
        root,
        clean,
        rir=rir,
        operations=[
            {
                "type": "rir",
                "version": 1,
                "source": "rir",
                "channel": channel,
                "length": "trim_to_input",
            }
        ],
        reference=reference,
    )


def _recipe_files(root, clean, *, noise=None, rir=None, operations, reference) -> dict:
    sources = {"clean": {"path": clean.name, "sha256": sha256_file(clean)}}
    if noise is not None:
        sources["noise"] = {"path": noise.name, "sha256": sha256_file(noise)}
    if rir is not None:
        sources["rir"] = {"path": rir.name, "sha256": sha256_file(rir)}
    return canonical_recipe(
        {
            "recipe_version": 1,
            "sources": sources,
            "operations": operations,
            "output": {"sample_rate": 16000, "reference": reference},
        }
    )


def _register_pair(db: EvaluationDB, root: Path) -> None:
    db.import_manifest(_manifest(root, "b"), "kinds")
    db.import_manifest(_manifest(root, "a", kind="noisy"), "other")


def _manifest(
    root: Path,
    utterance_id: str,
    *,
    kind: str = "synthetic",
    transcript: str = "alpha beta",
    max_seconds=None,
    frames: int = 1600,
) -> Path:
    record = {
        "utterance_id": utterance_id,
        "dataset": "toy",
        "split": "test",
        "speaker_id": "spk",
        "transcript": transcript,
    }
    if kind == "synthetic":
        _tone(root / f"{utterance_id}-clean.wav", frames, 220, 16000)
        _tone(root / f"{utterance_id}-noise.wav", frames * 2, 40, 16000)
        record["clean_path"] = f"{utterance_id}-clean.wav"
        record["noise_path"] = f"{utterance_id}-noise.wav"
        record["snr_db"] = 0
        if max_seconds is not None:
            record["max_seconds"] = max_seconds
    else:
        _tone(root / f"{utterance_id}-noisy.wav", 1600, 330, 16000)
        record["noisy_path"] = f"{utterance_id}-noisy.wav"
    path = root / f"{utterance_id}.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def _tone(path: Path, frames: int, frequency: float, rate: int) -> Path:
    time = torch.arange(frames, dtype=torch.float32) / rate
    _save(path, (0.2 * torch.sin(2 * torch.pi * frequency * time)).reshape(1, -1), rate)
    return path


def _save(path: Path, waveform: torch.Tensor, rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, rate)
    return path


def _reverse_cases(source: Path, destination: Path) -> None:
    import shutil

    shutil.copytree(source, destination)
    cases = destination / "evaluation" / "cases.parquet"
    connection = _duck(":memory:")
    reversed_path = destination / "evaluation" / "cases-reversed.parquet"
    connection.execute(
        "COPY (SELECT * FROM read_parquet(?) ORDER BY case_id DESC) TO ? (FORMAT PARQUET)",
        [str(cases), str(reversed_path)],
    )
    connection.close()
    reversed_path.replace(cases)


def _snapshot(database: Path):
    connection = _duck(database, read_only=True)
    try:
        cases = connection.execute(
            """
            SELECT case_id, kind, dataset, split, sample_rate, transcript, speaker_id, content_hash,
                   recipe_json::VARCHAR, metadata::VARCHAR
            FROM eval.cases ORDER BY case_id
            """
        ).fetchall()
        sets = connection.execute(
            "SELECT name, description, content_hash FROM eval.sets ORDER BY name"
        ).fetchall()
        items = connection.execute(
            "SELECT set_name, position, case_id FROM eval.set_items ORDER BY set_name, position"
        ).fetchall()
        corpora = connection.execute(
            """
            SELECT count(*) FROM information_schema.tables
            WHERE table_schema = 'corpora' AND table_name = 'audio_files'
            """
        ).fetchone()[0]
        corpus_rows = []
        if corpora:
            corpus_rows = connection.execute(
                "SELECT path, corpus, transcript FROM corpora.audio_files ORDER BY path"
            ).fetchall()
    finally:
        connection.close()
    return cases, sets, items, corpus_rows


def _result_ids(database: Path):
    connection = _duck(database, read_only=True)
    try:
        return connection.execute(
            "SELECT record_id, run_id, case_id, content_hash FROM eval.results ORDER BY record_id"
        ).fetchall()
    finally:
        connection.close()


def _run(database: Path, root: Path, set_name: str, *, pesq: bool = False):
    config = experiment_config(root / "runs" / set_name, run_id=set_name)
    config.dataset.source = "duckdb"
    config.dataset.path = str(database)
    config.dataset.set_name = set_name
    config.dataset.corpora_root = str(root / "corpora")
    config.mixing.enabled = False
    config.experiment.limit = 1
    config.metrics.pesq = pesq
    config.metrics.stoi = False
    result = execute(config)
    return result["records"][0]


def _duck(database, *, read_only: bool = False):
    import duckdb

    if database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(database), read_only=read_only)
