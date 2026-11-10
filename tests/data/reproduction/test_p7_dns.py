"""DNS on-demand conditions, official formula, and chunk boundaries."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.experiments.data.build.dns2020 import (
    SAMPLE_RATE,
    activitydetector,
    build_definition,
    is_clipped,
    measured_snr_db,
    num_files_for_hours,
    official_uttid,
    order_sha256,
    pcm16_roundtrip,
    plan_case,
    plan_realizations,
    render_realization,
    shuffle_pools,
    snr_mixer,
    split_train_valid,
)
from src.experiments.data.db.registry import initialize_database
from src.experiments.data.db.validate import _claim
from src.experiments.data.errors import ReproductionError
from src.experiments.data.render.chunking import plan_chunk_epoch
from src.experiments.data.mix.dns_ondemand import DnsOnDemandChunkData
from src.experiments.data.mix.pregenerated import open_chunk_datasets
from src.experiments.data.prepare.dns_run import _check_config, run_config

ROOT = Path(__file__).resolve().parents[3]
DNS_ROOT = ROOT / "data" / "corpora" / "DNS2020"


def _tone(path: Path, frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    wave = np.clip(rng.normal(scale=0.05, size=frames), -0.5, 0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")


def _corpus(root: Path) -> None:
    base = root / "DNS2020" / "datasets"
    for index in range(4):
        _tone(
            base / "clean" / f"book_00001_chp_0001_reader_{index:05d}_0.wav",
            32000,
            100 + index,
        )
        _tone(base / "noise" / f"noise_{index}.wav", 8000, 200 + index)
    _tone(base / "test_set" / "synthetic" / "no_reverb" / "clean" / "clean_fileid_0.wav", 8000, 300)
    _tone(
        base / "test_set" / "synthetic" / "no_reverb" / "noisy" / "clnsp_snr0_tl-20_fileid_0.wav",
        8000,
        301,
    )


def _config(directory: Path) -> Path:
    path = directory / "build.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: dns.limited.recipe
            smoke_name: dns.limited.smoke
            smoke_num_files: 4
            num_files: 6
            seed: 17
            shard_size: 2
            delivery: on_demand
            source_scan: limited
            claim_label: official_recipe_scaled_down
            audio_seconds: 1
            audit_realizations: 2
            include_test: true
            clean_limit: 4
            noise_limit: 4
            workers: 2
            """
        ),
        encoding="utf-8",
    )
    return path


class DnsFormulaTests(unittest.TestCase):
    def test_bounds_split_and_uttid(self):
        self.assertEqual(num_files_for_hours(3000), 360000)
        documents = []
        clean, noise = shuffle_pools(
            [f"book_00001_chp_0001_reader_{index:05d}_0.wav" for index in range(3)],
            [f"noise_{index}.wav" for index in range(3)],
            17,
        )
        frames = {name: 496000 for name in clean}
        frames.update({name: 160000 for name in noise})
        order = order_sha256(clean, noise)
        definition = build_definition(
            seed=17,
            num_files=400,
            clean_count=len(clean),
            noise_count=len(noise),
            order_sha256=order,
            inputs={"speech": "a" * 64, "noise": "b" * 64},
            audio_seconds=30,
        )
        short = build_definition(
            seed=17,
            num_files=4,
            clean_count=len(clean),
            noise_count=len(noise),
            order_sha256=order,
            inputs={"speech": "a" * 64, "noise": "b" * 64},
            audio_seconds=30,
        )
        for gen_index in range(4):
            left = plan_case(definition, gen_index, clean, frames, noise, frames)
            right = plan_case(short, gen_index, clean, frames, noise, frames)
            self.assertEqual(left["sources"], right["sources"])
            self.assertEqual(left["draws"]["snr_db"], right["draws"]["snr_db"])
            self.assertNotEqual(left["draws"]["build_id"], right["draws"]["build_id"])
        documents = plan_realizations(definition, clean, frames, noise, frames)
        snr = {int(item["draws"]["snr_db"]) for item in documents}
        level = {int(item["draws"]["target_level_dbfs"]) for item in documents}
        self.assertTrue(snr <= set(range(-5, 15)))
        self.assertNotIn(15, snr)
        self.assertTrue(level <= set(range(-35, -15)))
        self.assertNotIn(-15, level)
        self.assertEqual(int(documents[0]["output"]["length"]), 480000)
        train, valid, stats = split_train_valid(documents)
        self.assertEqual(stats["train_kept"], 360)
        self.assertEqual(stats["valid_kept"], 40)
        self.assertEqual(360000 * 9 // 10, 324000)
        name = official_uttid(
            [{"name": "book_00993_chp_0017_reader_02995_8.wav"}],
            [{"name": "---1_cCGK4M.wav"}],
            -5,
            12,
        )
        self.assertEqual(name, "reader_02995_fileid_12")
        sed = subprocess.run(
            ["sed", "-e", r"s:.*reader_\([0-9]*\)_.*_\([0-9]*\).wav$:reader_\1_fileid_\2:i"],
            input="book_00993_chp_0017_reader_02995_8_---1_cCGK4M_snr-5_fileid_12.wav\n",
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(sed.stdout.strip(), name)

    def test_formula_matches_official_audiolib(self):
        audiolib_path = DNS_ROOT / "audiolib.py"
        if not audiolib_path.is_file():
            self.skipTest("DNS audiolib.py is not in the corpus checkout")
        state = np.random.get_state()
        spec = importlib.util.spec_from_file_location("dns_challenge_audiolib", audiolib_path)
        audiolib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audiolib)
        np.random.set_state(state)
        rng = np.random.default_rng(5)
        clean = rng.normal(scale=0.1, size=4800)
        noise = rng.normal(scale=0.1, size=3200)
        params = {"cfg": {}, "target_level_lower": -35, "target_level_upper": -15}
        np.random.seed(123)
        level = int(np.random.randint(-35, -15))
        np.random.seed(123)
        official = audiolib.snr_mixer(params, clean.copy(), noise.copy(), -3)
        ours = snr_mixer(clean, noise, -3, level)
        for left, right in zip(official, ours):
            self.assertTrue(np.allclose(np.asarray(left), np.asarray(right)))
        wave = rng.normal(scale=0.05, size=16000)
        self.assertEqual(float(activitydetector(wave)), float(audiolib.activitydetector(wave)))
        self.assertEqual(bool(is_clipped(np.array([1.0, 0.0]))), bool(audiolib.is_clipped(np.array([1.0, 0.0]))))
        quantized = pcm16_roundtrip(ours[2])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mix.wav"
            audiolib.audiowrite(str(path), ours[2], SAMPLE_RATE)
            written, rate = sf.read(path, dtype="float64")
        self.assertEqual(int(rate), SAMPLE_RATE)
        self.assertTrue(np.array_equal(quantized, written))
        snr = measured_snr_db(pcm16_roundtrip(ours[0]), pcm16_roundtrip(ours[1]))
        self.assertLess(abs(snr - (-3)), 0.2)

    def test_chunk_offset_is_zero_for_thirty_seconds(self):
        samples = [(f"utt-{index}", 480000) for index in range(16)]
        for world, updates in ((1, 14), (2, 28), (4, 56)):
            plan = plan_chunk_epoch(
                samples,
                espnet_epoch=1,
                seed=17,
                world_size=world,
                num_samples=4,
                chunk_length=64000,
                global_batch=4,
                num_cache_chunks=16,
                shuffle=True,
                shift_ratio=0.5,
            )
            again = plan_chunk_epoch(
                samples,
                espnet_epoch=1,
                seed=17,
                world_size=world,
                num_samples=4,
                chunk_length=64000,
                global_batch=4,
                num_cache_chunks=16,
                shuffle=True,
                shift_ratio=0.5,
            )
            self.assertEqual(plan.plan_hash, again.plan_hash)
            self.assertEqual(plan.updates, updates)
            self.assertEqual(list(plan.discarded_batches), [0] * world)
            self.assertEqual(list(plan.discarded_tail_chunks), [0] * world)
            seen: dict[str, set[int]] = {}
            for batches in plan.rank_batches:
                for batch in batches:
                    for chunk in batch:
                        self.assertEqual(chunk.stop - chunk.start, 64000)
                        self.assertEqual(chunk.start % 32000, 0)
                        seen.setdefault(chunk.case_id, set()).add(chunk.start)
            self.assertTrue(all(len(starts) == 14 for starts in seen.values()))

    def test_recipe_claim_accepts_the_dns_algorithm(self):
        dataset = {
            "claim_label": "espnet_public_se_recipe_reproduction",
            "profile": "official_faithful",
            "material_origin": "derived_from_official",
            "declared_status": "ready",
        }
        _claim(dataset, {"algorithm": "pregenerated.dns_tflocoformer", "deviation_class": "algorithm_change"}, {}, {}, {}, {})
        _claim(dataset, {"algorithm": "pregenerated.ms_snsd_tfgridnet", "deviation_class": "algorithm_change"}, {}, {}, {}, {})
        with self.assertRaises(ReproductionError):
            _claim(dataset, {"algorithm": "fixed_pair.mp_senet", "deviation_class": "algorithm_change"}, {}, {}, {}, {})


class DnsRegisterTests(unittest.TestCase):
    def test_refuses_stored_wavs_and_unbounded_scans(self):
        with self.assertRaises(ReproductionError):
            _check_config({"name": "a", "smoke_name": "b", "seed": 1, "shard_size": 1, "smoke_num_files": 2, "audit_realizations": 1, "source_scan": "limited"})
        with self.assertRaises(ReproductionError):
            _check_config({
                "name": "a", "smoke_name": "b", "seed": 1, "shard_size": 1, "smoke_num_files": 2,
                "audit_realizations": 1, "source_scan": "limited", "clean_limit": 1, "noise_limit": 1,
                "num_files": 2, "claim_label": "espnet_public_se_recipe_reproduction",
            })
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            text = config.read_text(encoding="utf-8").replace("delivery: on_demand", "delivery: stored")
            config.write_text(text, encoding="utf-8")
            with self.assertRaises(ReproductionError):
                run_config(config, "plan", corpus=Path(tmp), staging=Path(tmp) / "staging", database=Path(tmp) / "db.duckdb", resource_db=Path(tmp) / "missing.duckdb")
            with self.assertRaises(ReproductionError):
                run_config(config, "generate", corpus=Path(tmp))

    def test_register_snapshot_rebuilds_the_same_pcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root)
            config = _config(root)
            database = root / "reproduction.duckdb"
            initialize_database(database)
            generated = root / "generated"
            summary = run_config(
                config,
                "register",
                corpus=root,
                staging=root / "staging",
                database=database,
                snapshots=root / "snapshots",
                resource_db=root / "missing.duckdb",
                generated=generated,
            )
            self.assertEqual(summary["wav_files"], 0)
            self.assertEqual(list(generated.rglob("*.wav")), [])
            self.assertFalse(any(summary["selection_uses_test"].values()))
            self.assertEqual(summary["recipe_audit"]["wav_files"], 0)
            self.assertGreaterEqual(summary["test_pairs"], 1)
            train, valid = open_chunk_datasets(
                dataset="dns.limited.smoke@1",
                snapshot_root=root / "snapshots",
                corpora_root=root,
                generated_root=generated,
                derived_root=root / "derived",
            )
            self.assertIsInstance(train, DnsOnDemandChunkData)
            self.assertEqual(min(train.waveform_lengths()), 16000)
            self.assertEqual(valid.waveform_lengths(), [16000])
            case_id = train.case_ids()[0]
            first = train.collate_chunks([_chunk(case_id, 0, 1600)])
            second = train.collate_chunks([_chunk(case_id, 0, 1600)])
            self.assertTrue(torch_equal(first["clean"], second["clean"]))
            full = train._pair(case_id)[1]
            self.assertTrue(np.array_equal(first["noisy"][0].numpy(), full[:1600]))
            train.close()
            valid.close()


def _chunk(case_id: str, start: int, stop: int):
    from src.experiments.data.render.chunking import PlannedChunk

    return PlannedChunk(case_id, start, stop)


def torch_equal(left, right) -> bool:
    import torch

    return bool(torch.equal(left, right))
