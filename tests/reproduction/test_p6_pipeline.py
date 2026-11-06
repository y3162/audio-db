"""Plan, generate, verify, register, snapshot, and chunk an MS-SNSD build."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from src.reproduction.build.ms_snsd import (
    SAMPLE_RATE,
    build_definition,
    duration_kept,
    list_wavs,
    ordered_paths,
    plan_realizations,
    read_frames,
    sort_like_kaldi,
    train_valid_counts,
    verify_build,
    write_build,
)
from src.reproduction.db.registry import initialize_database, register_staging
from src.reproduction.db.snapshot import export_snapshot
from src.reproduction.db.staging import write_graph
from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.reproduction.pipeline.pregenerated import open_chunk_datasets
from src.reproduction.prepare.ms_snsd import (
    assemble_smoke_graph,
    collect_sources,
    material_set,
    write_shapes,
)
from src.reproduction.snapshot_reader import load_snapshot

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpora" / "MS-SNSD"


def _tone(path: Path, frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    wave = rng.normal(scale=0.05, size=frames).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")


class PlanRuleTests(unittest.TestCase):
    def test_split_matches_the_shell_and_lexical_names(self):
        self.assertEqual(train_valid_counts(10), (8, 2))
        from src.reproduction.build.ms_snsd import split_train_valid

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
            with mock.patch("src.reproduction.build.ms_snsd.shutil.disk_usage") as usage:
                usage.return_value = shutil.disk_usage(root)
                usage.return_value = mock.Mock(free=0)
                with self.assertRaises(ReproductionError):
                    write_build(
                        definition, first,
                        audio_root=root / "generated", staging_root=root / "staging",
                        clean_dir=root, noise_dir=root,
                    )


class PipelineTests(unittest.TestCase):
    def test_synthetic_build_registers_and_plans_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean"
            noise = root / "noise"
            for index in range(8):
                _tone(clean / f"c{index}.wav", SAMPLE_RATE * 3, 100 + index)
            for index in range(4):
                _tone(noise / f"n{index}.wav", SAMPLE_RATE * 3, 200 + index)
            clean_names = ordered_paths(list_wavs(clean), seed=17, split="train")
            noise_names = ordered_paths(list_wavs(noise), seed=17, split="train")
            test_clean = ordered_paths(list_wavs(clean), seed=17, split="test")
            test_noise = ordered_paths(list_wavs(noise), seed=17, split="test")
            train_clean, train_clean_frames = collect_sources(
                clean, clean_names, rel_prefix="MS-SNSD/clean_train", subset="clean_train",
            )
            train_noise, train_noise_frames = collect_sources(
                noise, noise_names, rel_prefix="MS-SNSD/noise_train", subset="noise_train",
            )
            test_clean_assets, test_clean_frames = collect_sources(
                clean, test_clean, rel_prefix="MS-SNSD/clean_test", subset="clean_test",
            )
            test_noise_assets, test_noise_frames = collect_sources(
                noise, test_noise, rel_prefix="MS-SNSD/noise_test", subset="noise_test",
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
            audio = root / "generated"
            workspace = root / "workspace"
            train_written = write_build(
                train_definition, train_documents,
                audio_root=audio, staging_root=workspace,
                clean_dir=clean, noise_dir=noise,
            )
            test_written = write_build(
                test_definition, test_documents,
                audio_root=audio, staging_root=workspace,
                clean_dir=clean, noise_dir=noise,
            )
            other_audio = (audio / "ms_snsd_synth" / train_definition["build_id"][:16]).resolve()
            self.assertTrue(other_audio.is_dir())
            train_report = verify_build(
                train_definition, train_documents,
                audio_dir=train_written["audio_dir"], staging_dir=train_written["staging_dir"],
                clean_dir=clean, noise_dir=noise,
            )
            broken = next((train_written["audio_dir"]).rglob("*.wav"))
            original = broken.read_bytes()
            broken.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
            with self.assertRaises(ReproductionError):
                verify_build(
                    train_definition, train_documents,
                    audio_dir=train_written["audio_dir"], staging_dir=train_written["staging_dir"],
                    clean_dir=clean, noise_dir=noise,
                )
            self.assertTrue(broken.is_file())
            broken.write_bytes(original)
            train_report = verify_build(
                train_definition, train_documents,
                audio_dir=train_written["audio_dir"], staging_dir=train_written["staging_dir"],
                clean_dir=clean, noise_dir=noise,
            )
            test_report = verify_build(
                test_definition, test_documents,
                audio_dir=test_written["audio_dir"], staging_dir=test_written["staging_dir"],
                clean_dir=clean, noise_dir=noise,
            )
            graph = assemble_smoke_graph(
                train_definition=train_definition,
                train_documents=train_documents,
                train_manifests=_manifests(train_written["staging_dir"]),
                train_verify=train_report,
                test_definition=test_definition,
                test_documents=test_documents,
                test_manifests=_manifests(test_written["staging_dir"]),
                test_verify=test_report,
                source_assets=train_clean + train_noise + test_clean_assets + test_noise_assets,
                train_clean=train_clean_set,
                train_noise=train_noise_set,
                test_clean=test_clean_set,
                test_noise=test_noise_set,
                dataset_name="tf_gridnet.ms_snsd.smoke",
                resource_db=None,
            )
            refused = dict(train_report)
            refused["status"] = "FAIL"
            with self.assertRaises(ReproductionError):
                assemble_smoke_graph(
                    train_definition=train_definition,
                    train_documents=train_documents,
                    train_manifests=_manifests(train_written["staging_dir"]),
                    train_verify=refused,
                    test_definition=test_definition,
                    test_documents=test_documents,
                    test_manifests=_manifests(test_written["staging_dir"]),
                    test_verify=test_report,
                    source_assets=train_clean + train_noise + test_clean_assets + test_noise_assets,
                    train_clean=train_clean_set,
                    train_noise=train_noise_set,
                    test_clean=test_clean_set,
                    test_noise=test_noise_set,
                    dataset_name="tf_gridnet.ms_snsd.smoke",
                    resource_db=None,
                )
            staging = root / "staging"
            write_graph(staging, graph)
            database = root / "reproduction.duckdb"
            initialize_database(database)
            register_staging(staging, database, resource_db=None)
            dataset_id = graph.datasets[0]["dataset_id"]
            _link_sources(root / "corpora_mount", clean, noise)
            first_hash = export_snapshot(
                database, f"tf_gridnet.ms_snsd.smoke@1", root / "snapshots",
                corpora=root / "corpora_mount", generated=audio, derived=root / "derived",
            )
            second_hash = export_snapshot(
                database, "tf_gridnet.ms_snsd.smoke@1", root / "snapshots",
                corpora=root / "corpora_mount", generated=audio, derived=root / "derived",
            )
            self.assertEqual(first_hash, second_hash)
            loaded = load_snapshot(root / "snapshots" / dataset_id)
            self.assertEqual(loaded.snapshot_hash, first_hash)
            train, valid = open_chunk_datasets(
                dataset="tf_gridnet.ms_snsd.smoke@1",
                snapshot_root=root / "snapshots",
                generated_root=audio,
                corpora_root=root / "corpora_mount",
                derived_root=root / "derived",
            )
            self.assertGreaterEqual(len(train.case_ids()), 1)
            self.assertGreaterEqual(len(valid.case_ids()), 1)
            self.assertEqual(len(train.case_ids()), len(train.waveform_lengths()))
            self.assertTrue(all(length > 48000 for length in train.waveform_lengths()))
            shape_dir = root / "shapes" / "train"
            noisy_paths = {}
            for case in graph.case_sets[0]["items"]:
                row = next(item for item in graph.cases if item["case_id"] == case["case_id"])
                noisy_paths[row["utterance_key"]] = row["noisy_asset_id"]
            # case set 0 is train because assemble appends train, valid, test.
            train_cases = [
                next(item for item in graph.cases if item["case_id"] == entry["case_id"])
                for entry in graph.case_sets[0]["items"]
            ]
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
            self.assertEqual(tuple(batch["noisy"].shape), (2, 48000))
            self.assertEqual(tuple(batch["clean"].shape), (2, 48000))
            again = plan_chunk_epoch(
                samples, espnet_epoch=1, seed=0, world_size=2, num_samples=5000,
                chunk_length=48000, global_batch=4, num_cache_chunks=1024,
                shuffle=True, shift_ratio=0.5,
            )
            self.assertEqual(plan.plan_hash, again.plan_hash)


def _manifests(staging_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((staging_dir / "shards").glob("shard-*.json"))
    ]


def _link_sources(mount: Path, clean: Path, noise: Path) -> None:
    for subset, source in (
        ("clean_train", clean),
        ("noise_train", noise),
        ("clean_test", clean),
        ("noise_test", noise),
    ):
        destination = mount / "MS-SNSD" / subset
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)


if __name__ == "__main__":
    unittest.main()
