"""On-demand MS-SNSD conditions: no stored wavs, same mix twice, chunk plans."""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.reproduction.build.ms_snsd import (
    SAMPLE_RATE,
    build_definition,
    measured_snr_db,
    ordered_paths,
    plan_realizations,
    read_frames,
    render_realization,
    snr_mixer,
)
from src.reproduction.db.registry import initialize_database
from src.reproduction.db.snapshot import export_snapshot
from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.reproduction.pipeline.ondemand import OnDemandChunkData
from src.reproduction.pipeline.pregenerated import open_chunk_datasets
from src.reproduction.prepare.ms_snsd_run import run_config
from src.reproduction.snapshot_reader import load_snapshot

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpora" / "MS-SNSD"


def _tone(path: Path, frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    wave = rng.normal(scale=0.05, size=frames).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")


def _corpus(root: Path, *, clean: int, noise: int, clean_frames: int, noise_frames: int) -> None:
    for split, clean_count, noise_count, seed_base in (
        ("train", clean, noise, 1000),
        ("test", max(2, clean // 2), max(2, noise // 2), 4000),
    ):
        for index in range(clean_count):
            _tone(root / f"clean_{split}" / f"c{index}.wav", clean_frames, seed_base + index)
        for index in range(noise_count):
            _tone(root / f"noise_{split}" / f"n{index}.wav", noise_frames, seed_base + 500 + index)


class OnDemandTests(unittest.TestCase):
    def test_cli_refuses_stored_wavs_and_unbounded_scans(self):
        with self.assertRaises(ReproductionError):
            run_config(ROOT / "configs/data/builds/ms_snsd_tfgridnet.yaml", "plan")
        with self.assertRaises(ReproductionError):
            run_config(ROOT / "configs/data/builds/ms_snsd_tfgridnet_smoke.yaml", "register")
        with self.assertRaises(ReproductionError):
            run_config(ROOT / "configs/data/builds/ms_snsd_tfgridnet_ondemand.yaml", "generate")

    def test_register_snapshot_and_chunk_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "MS-SNSD"
            _corpus(corpus, clean=6, noise=3, clean_frames=48000, noise_frames=200000)
            config = root / "build.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    name: tf_gridnet.ms_snsd.ondemand_test
                    seed: 17
                    shard_size: 5
                    claim_label: official_recipe_scaled_down
                    delivery: on_demand
                    train:
                      total_hours: 30
                      max_clips: 5
                      clean_limit: 6
                      noise_limit: 3
                    test:
                      total_hours: 1
                      max_clips: 1
                      clean_limit: 4
                      noise_limit: 2
                    """
                ),
                encoding="utf-8",
            )
            database = root / "reproduction.duckdb"
            initialize_database(database)
            generated = root / "generated"
            summary = run_config(
                config,
                "register",
                corpus=corpus,
                audio_root=generated,
                staging=root / "staging",
                database=database,
                snapshots=root / "snapshots",
                resource_db=root / "missing-resource.duckdb",
            )
            self.assertFalse(summary["selection_uses_test"])
            self.assertEqual(summary["train_verify"]["wav_files"], 0)
            self.assertEqual(summary["test_verify"]["wav_files"], 0)
            self.assertEqual(list(generated.rglob("*.wav")), [])
            loaded = load_snapshot(root / "snapshots" / summary["dataset_id"])
            self.assertEqual(loaded.dataset["rule_json"]["delivery"], "on_demand")
            self.assertEqual(loaded.generated, [])
            self.assertEqual(len(loaded.docs), len(loaded.cases))
            frames = {row["case_id"]: int(row["frames"]) for row in loaded.cases}
            for doc in loaded.docs:
                self.assertEqual(int(doc["doc_json"]["output"]["length"]), frames[doc["case_id"]])
                self.assertIn("sources", doc["doc_json"])
                self.assertEqual(doc["doc_json"]["draws"]["build_id"], doc["doc_json"]["draws"]["build_id"])
            again = export_snapshot(
                database,
                "tf_gridnet.ms_snsd.ondemand_test@1",
                root / "snapshots",
                corpora=root,
                generated=generated,
                derived=root / "derived",
            )
            self.assertEqual(again, summary["snapshot_hash"])
            train, valid = open_chunk_datasets(
                dataset="tf_gridnet.ms_snsd.ondemand_test@1",
                snapshot_root=root / "snapshots",
                corpora_root=root,
                generated_root=generated,
                derived_root=root / "derived",
            )
            self.assertIsInstance(train, OnDemandChunkData)
            self.assertGreaterEqual(min(train.waveform_lengths()), 48000)
            self.assertTrue(set(train.case_ids()).isdisjoint(valid.case_ids()))
            samples = list(zip(train.case_ids(), train.waveform_lengths()))
            for world in (1, 2, 4):
                plan = plan_chunk_epoch(
                    samples,
                    espnet_epoch=1,
                    seed=17,
                    world_size=world,
                    num_samples=4,
                    chunk_length=48000,
                    global_batch=4,
                    num_cache_chunks=4,
                    shuffle=True,
                    shift_ratio=0.5,
                )
                self.assertEqual(plan.world_size, world)
                self.assertEqual(plan.per_rank_batch, 4 // world)
                self.assertGreater(plan.updates, 0)
                repeated = plan_chunk_epoch(
                    samples,
                    espnet_epoch=1,
                    seed=17,
                    world_size=world,
                    num_samples=4,
                    chunk_length=48000,
                    global_batch=4,
                    num_cache_chunks=4,
                    shuffle=True,
                    shift_ratio=0.5,
                )
                self.assertEqual(plan.plan_hash, repeated.plan_hash)
            chunks = plan.rank_batches[0][0]
            first = train.collate_chunks(chunks)
            second = train.collate_chunks(chunks)
            self.assertTrue(first["clean"].equal(second["clean"]))
            self.assertTrue(first["noisy"].equal(second["noisy"]))
            self.assertEqual(tuple(first["clean"].shape), (len(chunks), 48000))
            parts = train.provenance_parts()
            self.assertEqual(parts["pipeline"], "reproduction")
            self.assertEqual(parts["delivery"], "on_demand")
            self.assertFalse(parts["selection_uses_test"])
            self.assertEqual(parts["dataset_id"], summary["dataset_id"])
            self.assertGreater(parts["lineage_counts"]["source"], 0)
            self.assertTrue(any(row["sha256"] for row in parts["materials"]))

    def test_many_realizations_match_the_formula_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_dir = root / "clean"
            noise_dir = root / "noise"
            for index in range(40):
                _tone(clean_dir / f"c{index}.wav", 48000, index)
            for index in range(4):
                _tone(noise_dir / f"n{index}.wav", 200000, 500 + index)
            clean_names = ordered_paths(sorted(clean_dir.glob("*.wav")), seed=17, split="train")
            noise_names = ordered_paths(sorted(noise_dir.glob("*.wav")), seed=17, split="train")
            definition = build_definition(
                seed=17,
                split="train",
                clean_names=clean_names,
                noise_names=noise_names,
                total_hours=30,
                max_clips=200,
                shard_size=50,
            )
            documents = plan_realizations(
                definition,
                read_frames(clean_dir, clean_names),
                read_frames(noise_dir, noise_names),
            )
            self.assertEqual(len(documents), 1000)
            sys.path.insert(0, str(CORPUS))
            from audiolib import snr_mixer as official

            compared = False
            for document in documents:
                rendered = render_realization(document, {"speech": clean_dir, "noise": noise_dir})
                again = render_realization(document, {"speech": clean_dir, "noise": noise_dir})
                for key in ("clean", "noise", "noisy"):
                    self.assertTrue(np.array_equal(rendered[key], again[key]))
                self.assertEqual(int(rendered["noisy"].shape[0]), int(document["output"]["length"]))
                snr = float(document["draws"]["nominal_snr_db"])
                self.assertAlmostEqual(measured_snr_db(rendered["clean"], rendered["noise"]), snr / 2.0, places=4)
                if not compared:
                    reference = official(clean=rendered["clean"], noise=rendered["noise"], snr=snr)
                    direct = snr_mixer(rendered["clean"], rendered["noise"], snr)
                    for left, right in zip(direct, reference):
                        self.assertTrue(np.array_equal(left, right))
                    compared = True
                peak = max(float(np.max(np.abs(rendered[key]))) for key in rendered)
                self.assertTrue(np.isfinite(peak))


if __name__ == "__main__":
    unittest.main()
