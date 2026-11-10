"""Plan, generate, verify, register, snapshot, and chunk an MS-SNSD build."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.experiments.data.build.ms_snsd import (
    SAMPLE_RATE,
    build_definition,
    duration_kept,
    list_wavs,
    ordered_paths,
    plan_realizations,
    sort_like_kaldi,
    train_valid_counts,
)
from src.experiments.data.db.registry import initialize_database, register_staging
from src.experiments.data.db.snapshot import export_snapshot
from src.experiments.data.db.staging import write_graph
from src.experiments.data.errors import ReproductionError
from src.experiments.data.render.chunking import plan_chunk_epoch
from src.experiments.data.mix.pregenerated import open_chunk_datasets
from src.experiments.data.prepare.ms_snsd import (
    assemble_ondemand_graph,
    audit_conditions,
    collect_sources,
    material_set,
    write_shapes,
)
from src.experiments.data.snapshot import load_snapshot

CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpora" / "MS-SNSD"


def _tone(path: Path, frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    wave = rng.normal(scale=0.05, size=frames).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")


class PlanRuleTests(unittest.TestCase):
    def test_split_matches_the_shell_and_lexical_names(self):
        self.assertEqual(train_valid_counts(10), (8, 2))
        from src.experiments.data.build.ms_snsd import split_train_valid

        docs = []
        for clip in (1, 2, 3):
            for snr in (0.0, 10.0, 20.0, 30.0, 40.0):
                docs.append({
                    "draws": {"utt_id": f"noisy{clip}_SNRdb_{snr}_clnsp{clip}"},
                    "output": {"length": 16000 * 12},
                })
        train, valid, stats = split_train_valid(docs)
        train_clips = {item["draws"]["utt_id"].split("_clnsp")[-1] for item in train}
        valid_clips = {item["draws"]["utt_id"].split("_clnsp")[-1] for item in valid}
        self.assertTrue(train_clips.isdisjoint(valid_clips))
        self.assertGreater(stats["files_moved_to_keep_clip"], 0)
        self.assertEqual(train_valid_counts(5), (4, 1))
        self.assertTrue(duration_kept(16000 * 11))
        self.assertFalse(duration_kept(1600))
        self.assertFalse(duration_kept(31 * SAMPLE_RATE))
        documents = [
            {"draws": {"utt_id": "noisy2_SNRdb_0.0_clnsp2"}},
            {"draws": {"utt_id": "noisy10_SNRdb_0.0_clnsp10"}},
            {"draws": {"utt_id": "noisy1_SNRdb_40.0_clnsp1"}},
            {"draws": {"utt_id": "noisy1_SNRdb_0.0_clnsp1"}},
        ]
        ordered = [item["draws"]["utt_id"] for item in sort_like_kaldi(documents)]
        # '0' sorts before '_', so noisy10 precedes noisy1_. The five SNR
        # names of one clip still sit together.
        self.assertEqual(
            ordered,
            [
                "noisy10_SNRdb_0.0_clnsp10",
                "noisy1_SNRdb_0.0_clnsp1",
                "noisy1_SNRdb_40.0_clnsp1",
                "noisy2_SNRdb_0.0_clnsp2",
            ],
        )

    def test_plan_is_stable_and_space_is_required(self):
        definition = build_definition(
            seed=3, split="train", clean_names=["a.wav", "b.wav", "c.wav"],
            noise_names=["n.wav", "m.wav"], total_hours=1, max_clips=1, shard_size=5,
        )
        frames = {"a.wav": 100, "b.wav": 100, "c.wav": 100, "n.wav": 50, "m.wav": 80}
        first = plan_realizations(definition, frames, frames)
        second = plan_realizations(definition, frames, frames)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = list(root.rglob("*"))
            with self.assertRaises(ReproductionError):
                audit_conditions(definition, first, clean_dir=root / "clean", noise_dir=root / "noise")
            self.assertEqual(list(root.rglob("*")), before)


class PipelineTests(unittest.TestCase):
    def test_synthetic_build_registers_and_plans_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean_train"
            noise = root / "noise_train"
            test_clean_dir = root / "clean_test"
            test_noise_dir = root / "noise_test"
            for index in range(6):
                _tone(clean / f"c{index}.wav", SAMPLE_RATE * 3, 100 + index)
            for index in range(3):
                _tone(noise / f"n{index}.wav", SAMPLE_RATE * 3, 200 + index)
            for index in range(3):
                _tone(test_clean_dir / f"t{index}.wav", SAMPLE_RATE * 3, 300 + index)
            for index in range(2):
                _tone(test_noise_dir / f"m{index}.wav", SAMPLE_RATE * 3, 400 + index)
            clean_names = ordered_paths(list_wavs(clean), seed=17, split="train")
            noise_names = ordered_paths(list_wavs(noise), seed=17, split="train")
            test_clean = ordered_paths(list_wavs(test_clean_dir), seed=17, split="test")
            test_noise = ordered_paths(list_wavs(test_noise_dir), seed=17, split="test")
            train_clean, train_clean_frames = collect_sources(
                clean, clean_names, rel_prefix="MS-SNSD/clean_train", subset="clean_train",
            )
            train_noise, train_noise_frames = collect_sources(
                noise, noise_names, rel_prefix="MS-SNSD/noise_train", subset="noise_train",
            )
            test_clean_assets, test_clean_frames = collect_sources(
                test_clean_dir, test_clean, rel_prefix="MS-SNSD/clean_test", subset="clean_test",
            )
            test_noise_assets, test_noise_frames = collect_sources(
                test_noise_dir, test_noise, rel_prefix="MS-SNSD/noise_test", subset="noise_test",
            )
            train_clean_set = material_set(
                train_clean, name="ms_snsd.smoke.clean_train", role="speech",
                split_role="train", note="synthetic",
            )
            train_noise_set = material_set(
                train_noise, name="ms_snsd.smoke.noise_train", role="noise",
                split_role="train", note="synthetic",
            )
            test_clean_set = material_set(
                test_clean_assets, name="ms_snsd.smoke.clean_test", role="speech",
                split_role="test", note="synthetic",
            )
            test_noise_set = material_set(
                test_noise_assets, name="ms_snsd.smoke.noise_test", role="noise",
                split_role="test", note="synthetic",
            )
            train_definition = build_definition(
                seed=17, split="train", clean_names=clean_names, noise_names=noise_names,
                total_hours=30, max_clips=5, shard_size=5,
                inputs={"speech": train_clean_set["material_set_id"], "noise": train_noise_set["material_set_id"]},
            )
            test_definition = build_definition(
                seed=17, split="test", clean_names=test_clean, noise_names=test_noise,
                total_hours=1, max_clips=1, shard_size=5,
                inputs={"speech": test_clean_set["material_set_id"], "noise": test_noise_set["material_set_id"]},
            )
            self.assertNotEqual(train_definition["build_id"], test_definition["build_id"])
            train_documents = plan_realizations(train_definition, train_clean_frames, train_noise_frames)
            test_documents = plan_realizations(test_definition, test_clean_frames, test_noise_frames)
            self.assertEqual(len(train_documents), 25)
            self.assertEqual(len(test_documents), 5)
            train_report = audit_conditions(
                train_definition, train_documents, clean_dir=clean, noise_dir=noise,
            )
            test_report = audit_conditions(
                test_definition, test_documents, clean_dir=test_clean_dir, noise_dir=test_noise_dir,
            )
            self.assertEqual(train_report["delivery"], "on_demand")
            self.assertEqual(train_report["wav_files"], 0)
            self.assertEqual(train_report["rendered"], len(train_documents))
            graph_kwargs = dict(
                train_definition=train_definition,
                train_documents=train_documents,
                train_report=train_report,
                test_definition=test_definition,
                test_documents=test_documents,
                test_report=test_report,
                source_assets=train_clean + train_noise + test_clean_assets + test_noise_assets,
                train_clean=train_clean_set,
                train_noise=train_noise_set,
                test_clean=test_clean_set,
                test_noise=test_noise_set,
                dataset_name="tf_gridnet.ms_snsd.synthetic",
                resource_db=None,
            )
            graph = assemble_ondemand_graph(**graph_kwargs)
            self.assertTrue(all(row["storage_root"] == "corpora" for row in graph.assets))
            self.assertTrue(all(row["noisy_asset_id"] is None for row in graph.cases))
            refused = dict(train_report)
            refused["status"] = "FAIL"
            with self.assertRaises(ReproductionError):
                assemble_ondemand_graph(**{**graph_kwargs, "train_report": refused})
            staging = root / "staging"
            write_graph(staging, graph)
            database = root / "reproduction.duckdb"
            initialize_database(database)
            register_staging(staging, database, resource_db=None)
            dataset_id = graph.datasets[0]["dataset_id"]
            _link_sources(root / "corpora_mount", clean, noise, test_clean_dir, test_noise_dir)
            first_hash = export_snapshot(
                database, "tf_gridnet.ms_snsd.synthetic@1", root / "snapshots",
                corpora=root / "corpora_mount", generated=root / "generated", derived=root / "derived",
            )
            second_hash = export_snapshot(
                database, "tf_gridnet.ms_snsd.synthetic@1", root / "snapshots",
                corpora=root / "corpora_mount", generated=root / "generated", derived=root / "derived",
            )
            self.assertEqual(first_hash, second_hash)
            loaded = load_snapshot(root / "snapshots" / dataset_id)
            self.assertEqual(loaded.snapshot_hash, first_hash)
            train, valid = open_chunk_datasets(
                dataset="tf_gridnet.ms_snsd.synthetic@1",
                snapshot_root=root / "snapshots",
                generated_root=root / "generated",
                corpora_root=root / "corpora_mount",
                derived_root=root / "derived",
            )
            self.assertGreaterEqual(len(train.case_ids()), 1)
            self.assertGreaterEqual(len(valid.case_ids()), 1)
            self.assertEqual(len(train.case_ids()), len(train.waveform_lengths()))
            self.assertTrue(all(length > 48000 for length in train.waveform_lengths()))
            shape_dir = root / "shapes" / "train"
            # case set 0 is train because assemble appends train, valid, test.
            train_cases = [
                next(item for item in graph.cases if item["case_id"] == entry["case_id"])
                for entry in graph.case_sets[0]["items"]
            ]
            noisy_paths = {
                row["utterance_key"]: f"ondemand:{row['case_id']}"
                for row in train_cases
            }
            self.assertEqual([row["case_id"] for row in train_cases], train.case_ids())
            write_shapes(
                shape_dir,
                train_cases,
                {row["utterance_key"]: noisy_paths[row["utterance_key"]] for row in train_cases},
            )
            shape_rows = [
                line.split() for line in (shape_dir / "speech_mix_shape").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [int(frames) for _utt, frames in shape_rows],
                train.waveform_lengths(),
            )
            self.assertEqual(
                (shape_dir / "utt2num_samples").read_text(encoding="utf-8"),
                (shape_dir / "speech_ref1_shape").read_text(encoding="utf-8"),
            )
            samples = list(zip(train.case_ids(), train.waveform_lengths()))
            plan = plan_chunk_epoch(
                samples,
                espnet_epoch=1,
                seed=0,
                world_size=2,
                num_samples=5000,
                chunk_length=48000,
                global_batch=4,
                num_cache_chunks=1024,
                shuffle=True,
                shift_ratio=0.5,
            )
            self.assertEqual([len(rank) for rank in plan.selected], [5000, 5000])
            self.assertEqual(plan.per_rank_batch, 2)
            self.assertGreaterEqual(plan.updates, 1)
            chunk = plan.rank_batches[0][0][0]
            self.assertEqual(chunk.stop - chunk.start, 48000)
            batch = train.collate_chunks(plan.rank_batches[0][0])
            repeated = train.collate_chunks(plan.rank_batches[0][0])
            self.assertEqual(tuple(batch["noisy"].shape), (2, 48000))
            self.assertEqual(tuple(batch["clean"].shape), (2, 48000))
            self.assertTrue(batch["noisy"].equal(repeated["noisy"]))
            self.assertTrue(batch["clean"].equal(repeated["clean"]))
            self.assertIn("ondemand:", (shape_dir / "wav.scp").read_text(encoding="utf-8"))
            again = plan_chunk_epoch(
                samples, espnet_epoch=1, seed=0, world_size=2, num_samples=5000,
                chunk_length=48000, global_batch=4, num_cache_chunks=1024,
                shuffle=True, shift_ratio=0.5,
            )
            self.assertEqual(plan.plan_hash, again.plan_hash)


def _link_sources(mount: Path, clean: Path, noise: Path, test_clean: Path, test_noise: Path) -> None:
    for subset, source in (
        ("clean_train", clean),
        ("noise_train", noise),
        ("clean_test", test_clean),
        ("noise_test", test_noise),
    ):
        destination = mount / "MS-SNSD" / subset
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)


if __name__ == "__main__":
    unittest.main()
