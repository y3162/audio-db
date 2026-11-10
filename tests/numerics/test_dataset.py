"""Recipe datasets, crop identity, and evaluation-set refusal."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import torch

from src.data.audio import read_audio_stream_info
from src.data.db.schema import create_corpora_schema
from src.data.recipe import canonical_recipe, case_id_for, sha256_file
from src.evaluation.audio.render import render_recipe
from src.evaluation.datasets.evaldb import EvaluationDB
from src.experiments.calc.rng import seed_model_init
from src.experiments.calc.dataset import load_recipe_dataset, resampled_frame_count
from src.experiments.calc.sampler import crop_start, epoch_indices
from src.experiments.calc.errors import TrainingError
from tests.numerics.support import paired_dataset, recorded_example, tone, write_wav

ROOT = Path(__file__).resolve().parents[2]
REBUILD = ROOT / "data/db/rebuild-a.duckdb"


class OrderAndCropTests(unittest.TestCase):
    def test_same_seed_reproduces_order_and_a_different_seed_changes_it(self):
        first = epoch_indices(8, seed=7, epoch=0, world_size=1, rank=0)
        second = epoch_indices(8, seed=7, epoch=0, world_size=1, rank=0)
        changed = epoch_indices(8, seed=8, epoch=0, world_size=1, rank=0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 8)

    def test_order_does_not_follow_model_initialization(self):
        before = epoch_indices(8, seed=3, epoch=1, world_size=1, rank=0)
        with seed_model_init(3):
            torch.randn(100)
        after = epoch_indices(8, seed=3, epoch=1, world_size=1, rank=0)
        self.assertEqual(before, after)

    def test_crop_depends_on_the_case_and_not_on_call_order(self):
        left = crop_start("case-a", 7, 0, 1000, 200)
        right = crop_start("case-b", 7, 0, 1000, 200)
        self.assertEqual(left, crop_start("case-a", 7, 0, 1000, 200))
        self.assertNotEqual(left, right)
        self.assertNotEqual(left, crop_start("case-a", 7, 1, 1000, 200))
        self.assertEqual(crop_start("case-a", 7, 0, 100, 100), 0)


class RecipeWaveformTests(unittest.TestCase):
    def test_same_recipe_and_seed_rebuild_the_same_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = paired_dataset(root, 2, frames=320, segment_size=100, seed=11)
            dataset.set_epoch(0)
            first = dataset.collate([0])
            second = dataset.collate([0])
            self.assertTrue(torch.equal(first["clean"], second["clean"]))
            self.assertTrue(torch.equal(first["noisy"], second["noisy"]))
            self.assertEqual(first["case_id"], [dataset.examples[0].case_id])
            self.assertEqual(first["case_id"][0], case_id_for(dataset.examples[0].recipe))

    def test_threaded_loads_match_the_serial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = paired_dataset(root, 4, frames=800, segment_size=200, seed=9)
            dataset.set_epoch(1)
            serial = dataset.collate([3, 1, 0, 2])
            dataset.load_workers = 4
            threaded = dataset.collate([3, 1, 0, 2])
            self.assertEqual(serial["case_id"], threaded["case_id"])
            self.assertTrue(torch.equal(serial["clean"], threaded["clean"]))
            self.assertTrue(torch.equal(serial["noisy"], threaded["noisy"]))

    def test_crop_does_not_depend_on_which_item_was_read_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = paired_dataset(root, 2, frames=500, segment_size=120, seed=5)
            dataset.set_epoch(2)
            forward = dataset.collate([0])
            backward = RecipeDataset_copy(dataset)
            backward.set_epoch(2)
            other_first = backward.collate([1])
            same = backward.collate([0])
            self.assertTrue(torch.equal(forward["clean"], same["clean"]))
            self.assertFalse(torch.equal(forward["clean"], other_first["clean"]))

    def test_dry_and_reverberant_teachers_stay_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.wav"
            impulse = root / "rir.wav"
            write_wav(clean, tone(400, 220))
            write_wav(impulse, torch.tensor([1.0, 0.0, 0.5, 0.0], dtype=torch.float32))
            dry = _rir_dataset(root, clean, impulse, "dry_clean")
            wet = _rir_dataset(root, clean, impulse, "reverberant_clean")
            dry.set_epoch(0)
            wet.set_epoch(0)
            dry_batch = dry.collate([0])
            wet_batch = wet.collate([0])
            self.assertFalse(torch.equal(dry_batch["clean"], wet_batch["clean"]))
            rendered = render_recipe(wet.examples[0].recipe, root)
            self.assertEqual(rendered.reference_name, "reverberant_clean")
            self.assertEqual(wet_batch["clean"].shape[-1], wet.segment_size)

    def test_official_mode_can_skip_utterance_rms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scaled = paired_dataset(root, 1, frames=480, segment_size=None, training=False)
            from src.experiments.calc.dataset import RecipeDataset

            raw = RecipeDataset(
                list(scaled.examples),
                root,
                name=scaled.name,
                role=scaled.role,
                seed=scaled.seed,
                sample_rate=scaled.sample_rate,
                segment_size=None,
                training=False,
                database=None,
                rms_normalize=False,
            )
            scaled_batch = scaled.collate([0])
            raw_batch = raw.collate([0])
            self.assertEqual(tuple(scaled_batch["noisy"].shape), tuple(raw_batch["noisy"].shape))
            self.assertFalse(torch.equal(scaled_batch["noisy"], raw_batch["noisy"]))
            rms = torch.sqrt(torch.mean(scaled_batch["noisy"] ** 2))
            self.assertTrue(torch.allclose(rms, torch.ones(())))
            rendered = render_recipe(raw.examples[0].recipe, root)
            observed = rendered.observed.detach().to(dtype=torch.float32).reshape(-1)
            self.assertTrue(torch.equal(raw_batch["noisy"][0], observed))
            self.assertNotIn("rms_normalize", scaled.identity())
            self.assertFalse(raw.identity()["rms_normalize"])

    def test_missing_teacher_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            noisy = root / "noisy.wav"
            write_wav(noisy, tone(200, 330))
            recipe = canonical_recipe(
                {
                    "recipe_version": 1,
                    "sources": {"noisy": {"path": noisy.name, "sha256": sha256_file(noisy)}},
                    "operations": [],
                    "output": {"sample_rate": 16000, "reference": "none"},
                }
            )
            from src.experiments.calc.dataset import RecipeDataset, TrainExample

            dataset = RecipeDataset(
                [TrainExample(case_id_for(recipe), recipe, "train", "toy", "none")],
                root,
                name="toy",
                role="train",
                seed=1,
                sample_rate=16000,
                segment_size=160,
                training=True,
                database=None,
            )
            with self.assertRaises(TrainingError):
                dataset.collate([0])


class DuckDBTests(unittest.TestCase):
    def test_named_set_uses_the_recipe_hash_and_matches_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            database = root / "train.duckdb"
            EvaluationDB(database, project_root=root).import_manifest(root / "train.jsonl", "toy_train")
            loaded = load_recipe_dataset(
                database,
                "toy_train",
                role="train",
                seed=4,
                sample_rate=16000,
                segment_size=160,
                training=True,
            )
            self.assertEqual(len(loaded), 2)
            for example in loaded.examples:
                self.assertEqual(example.case_id, case_id_for(example.recipe))
                self.assertEqual(len(example.case_id), 64)
                self.assertNotEqual(example.case_id, "1")
            loaded.set_epoch(0)
            batch = loaded.collate([0])
            rendered = render_recipe(loaded.examples[0].recipe, database.parent)
            self.assertEqual(batch["case_id"][0], loaded.examples[0].case_id)
            self.assertEqual(batch["clean"].shape[-1], 160)
            self.assertEqual(rendered.reference_name, "dry_clean")

    def test_none_reference_and_evaluation_names_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            noisy = root / "only-noisy.wav"
            write_wav(noisy, tone(200, 100))
            manifest = root / "noisy.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "utterance_id": "n1",
                        "dataset": "toy",
                        "split": "train",
                        "noisy_path": noisy.name,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            database = root / "sets.duckdb"
            EvaluationDB(database, project_root=root).import_manifest(manifest, "no_teacher")
            with self.assertRaises(TrainingError):
                load_recipe_dataset(
                    database,
                    "no_teacher",
                    role="train",
                    seed=1,
                    sample_rate=16000,
                    segment_size=160,
                    training=True,
                )
            EvaluationDB(database, project_root=root).import_manifest(manifest, "libri_testclean_full_smoke")
            with self.assertRaises(TrainingError) as caught:
                load_recipe_dataset(
                    database,
                    "libri_testclean_full_smoke",
                    role="train",
                    seed=1,
                    sample_rate=16000,
                    segment_size=160,
                    training=True,
                )
            self.assertIn("evaluation set", str(caught.exception))

    def test_production_database_name_is_refused(self):
        with self.assertRaises(TrainingError):
            load_recipe_dataset(
                "data/db/resource.duckdb",
                "toy",
                role="train",
                seed=1,
                sample_rate=16000,
                segment_size=160,
                training=True,
            )

    @unittest.skipUnless(REBUILD.is_file(), "rebuild database is absent")
    def test_rebuild_database_evaluation_sets_are_readable_and_refused(self):
        import duckdb

        connection = duckdb.connect(str(REBUILD), read_only=True)
        try:
            names = [row[0] for row in connection.execute("SELECT name FROM eval.sets ORDER BY name").fetchall()]
            case_count = connection.execute("SELECT count(*) FROM eval.cases").fetchone()[0]
        finally:
            connection.close()
        self.assertGreater(case_count, 0)
        self.assertIn("libri_testclean_full_smoke", names)
        with self.assertRaises(TrainingError) as caught:
            load_recipe_dataset(
                REBUILD,
                "libri_testclean_full_smoke",
                role="train",
                seed=1,
                sample_rate=16000,
                segment_size=160,
                training=True,
            )
        self.assertIn("evaluation set", str(caught.exception))
        if "chime3_et05_real_ch5" in names:
            with self.assertRaises(TrainingError) as chime:
                load_recipe_dataset(
                    REBUILD,
                    "chime3_et05_real_ch5",
                    role="validation",
                    seed=1,
                    sample_rate=16000,
                    segment_size=None,
                    training=False,
                )
            self.assertIn("chime3_et05", str(chime.exception))


class CatalogLengthTests(unittest.TestCase):
    def test_catalog_lengths_match_load_and_keep_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [100, 480, 250]
            examples = [
                recorded_example(root, f"item{index}", frames=count, clean_hz=180, noisy_hz=90)
                for index, count in enumerate(frames)
            ]
            catalog = _catalog(root, [(example, 16000) for example in examples])
            dataset = _catalog_dataset(examples, root, catalog, sample_rate=16000, training=True)
            state = random.getstate()
            torch_state = torch.get_rng_state().clone()
            with patch("duckdb.connect", wraps=duckdb.connect) as connect:
                lengths = dataset.waveform_lengths()
            self.assertEqual(connect.call_count, 1)
            self.assertEqual(lengths, frames)
            self.assertEqual(random.getstate(), state)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            for index, length in enumerate(lengths):
                loaded = dataset._load(index, crop=False)
                self.assertEqual(int(loaded[0].numel()), length)
                self.assertEqual(int(loaded[1].numel()), length)
            dataset.set_epoch(3)
            cropped = dataset.collate([2, 0])
            again = _catalog_dataset(examples, root, catalog, sample_rate=16000, training=True)
            again.set_epoch(3)
            self.assertTrue(torch.equal(cropped["clean"], again.collate([2, 0])["clean"]))
            self.assertEqual(cropped["case_id"], [examples[2].case_id, examples[0].case_id])

    def test_resampled_catalog_length_matches_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.wav"
            noisy = root / "noisy.wav"
            write_wav(clean, tone(48001, 220, rate=48000), rate=48000)
            write_wav(noisy, tone(48001, 110, rate=48000), rate=48000)
            recipe = canonical_recipe(
                {
                    "recipe_version": 1,
                    "sources": {
                        "clean": {"path": clean.name, "sha256": sha256_file(clean)},
                        "noisy": {"path": noisy.name, "sha256": sha256_file(noisy)},
                    },
                    "operations": [],
                    "output": {"sample_rate": 48000, "reference": "dry_clean", "max_seconds": 0.5},
                }
            )
            from src.experiments.calc.dataset import TrainExample

            example = TrainExample(case_id_for(recipe), recipe, "train", "toy", "dry_clean")
            catalog = _catalog(root, [(example, 48000)])
            dataset = _catalog_dataset([example], root, catalog, sample_rate=16000, training=False)
            expected = resampled_frame_count(min(48001, int(0.5 * 48000)), 48000, 16000)
            self.assertEqual(dataset.waveform_lengths(), [expected])
            self.assertEqual(int(dataset._load(0, crop=False)[0].numel()), expected)

    def test_operation_length_follows_clean_not_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.wav"
            noise = root / "noise.wav"
            write_wav(clean, tone(320, 200))
            write_wav(noise, tone(4000, 40))
            recipe = canonical_recipe(
                {
                    "recipe_version": 1,
                    "sources": {
                        "clean": {"path": clean.name, "sha256": sha256_file(clean)},
                        "noise": {"path": noise.name, "sha256": sha256_file(noise)},
                    },
                    "operations": [
                        {
                            "type": "additive_noise",
                            "version": 1,
                            "source": "noise",
                            "snr_db": 5,
                            "noise_start": 0,
                            "noise_start_unit": "sample",
                            "seed": 1,
                        }
                    ],
                    "output": {"sample_rate": 16000, "reference": "dry_clean"},
                }
            )
            from src.experiments.calc.dataset import TrainExample

            example = TrainExample(case_id_for(recipe), recipe, "train", "toy", "dry_clean")
            catalog = _catalog(root, [(example, 16000)])
            dataset = _catalog_dataset([example], root, catalog, sample_rate=16000, training=False)
            self.assertEqual(dataset.waveform_lengths(), [320])
            self.assertEqual(int(dataset._load(0, crop=False)[0].numel()), 320)

    def test_missing_catalog_row_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = recorded_example(root, "only", frames=80, clean_hz=100, noisy_hz=50)
            catalog = root / "empty.duckdb"
            connection = duckdb.connect(str(catalog))
            create_corpora_schema(connection)
            connection.close()
            dataset = _catalog_dataset([example], root, catalog, sample_rate=16000, training=False)
            with self.assertRaises(TrainingError):
                dataset.waveform_lengths()


def RecipeDataset_copy(dataset):
    from src.experiments.calc.dataset import RecipeDataset

    return RecipeDataset(
        list(dataset.examples),
        dataset.root,
        name=dataset.name,
        role=dataset.role,
        seed=dataset.seed,
        sample_rate=dataset.sample_rate,
        segment_size=dataset.segment_size,
        training=dataset.training,
        database=dataset.database,
    )


def _rir_dataset(root: Path, clean: Path, impulse: Path, reference: str):
    from src.experiments.calc.dataset import RecipeDataset, TrainExample

    recipe = canonical_recipe(
        {
            "recipe_version": 1,
            "sources": {
                "clean": {"path": clean.name, "sha256": sha256_file(clean)},
                "rir": {"path": impulse.name, "sha256": sha256_file(impulse)},
            },
            "operations": [
                {"type": "rir", "version": 1, "source": "rir", "channel": 0, "length": "trim_to_input"}
            ],
            "output": {"sample_rate": 16000, "reference": reference},
        }
    )
    return RecipeDataset(
        [TrainExample(case_id_for(recipe), recipe, "train", "toy", reference)],
        root,
        name=f"rir-{reference}",
        role="train",
        seed=2,
        sample_rate=16000,
        segment_size=200,
        training=True,
        database=None,
    )


def _catalog_dataset(examples, root: Path, catalog: Path, *, sample_rate: int, training: bool):
    from src.experiments.calc.dataset import RecipeDataset

    return RecipeDataset(
        examples,
        root,
        name="catalog",
        role="train" if training else "validation",
        seed=17,
        sample_rate=sample_rate,
        segment_size=64,
        training=training,
        database=None,
        audio_catalog=catalog,
        corpora_root=root,
    )


def _catalog(root: Path, rows: list[tuple]) -> Path:
    catalog = root / "catalog.duckdb"
    connection = duckdb.connect(str(catalog))
    create_corpora_schema(connection)
    inserted = []
    for example, rate in rows:
        for name, source in example.recipe["sources"].items():
            if name not in ("clean", "noisy", "noise"):
                continue
            path = source["path"]
            info_rate, frames, _channels = read_audio_stream_info(root / path)
            inserted.append((path, info_rate, frames))
            if info_rate != rate:
                raise AssertionError(f"{path} rate {info_rate} != {rate}")
    connection.executemany(
        """
        INSERT INTO corpora.audio_files
            (path, corpus, file_name, sample_rate, frames, channels)
        VALUES (?, 'toy', ?, ?, ?, 1)
        """,
        [(path, Path(path).name, rate, frames) for path, rate, frames in reversed(inserted)],
    )
    connection.close()
    return catalog


def _write_manifest(root: Path) -> None:
    rows = []
    for index, frequency in enumerate((180, 240), start=1):
        clean = root / f"c{index}.wav"
        noise = root / f"n{index}.wav"
        write_wav(clean, tone(400, frequency))
        write_wav(noise, tone(800, 40 + index))
        rows.append(
            {
                "utterance_id": f"u{index}",
                "dataset": "toy",
                "split": "train",
                "clean_path": clean.name,
                "noise_path": noise.name,
                "snr_db": 5,
                "sample_rate": 16000,
            }
        )
    (root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
