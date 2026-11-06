"""P7.1 DNS acceptance, resampling, and checkpoint-average symlink."""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.reproduction.build.dns2020 import (
    SAMPLE_RATE,
    build_definition,
    is_clipped,
    order_sha256,
    plan_accepted_case,
    plan_case,
    render_realization,
    snr_mixer,
)
from src.reproduction.db.registry import initialize_database
from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.reproduction.pipeline.pregenerated import open_chunk_datasets
from src.reproduction.prepare.dns_run import run_config
from src.training.core.batch import plan_batches
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from src.training.se.tf_locoformer import TFLocoformerTask
from tests.se.test_tf_locoformer import TINY
from tests.training.support import cpu_process

ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, wave: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(wave, dtype=np.float64), rate, subtype="PCM_16")


def _active(frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(scale=0.08, size=frames), -0.4, 0.4)


def _corpus(root: Path) -> None:
    base = root / "DNS2020" / "datasets"
    _write(base / "clean" / "book_00000_chp_0000_reader_00000_0.wav", np.ones(32000), SAMPLE_RATE)
    _write(base / "clean" / "book_00000_chp_0000_reader_00001_0.wav", np.zeros(32000), SAMPLE_RATE)
    for index in range(2, 6):
        _write(
            base / "clean" / f"book_00000_chp_0000_reader_{index:05d}_0.wav",
            _active(32000, 100 + index),
            SAMPLE_RATE,
        )
    _write(base / "noise" / "noise_a.wav", _active(32000, 201), SAMPLE_RATE)
    _write(base / "noise" / "noise_b.wav", _active(32000, 202), SAMPLE_RATE)
    _write(base / "noise" / "noise_hi.wav", _active(8000, 203), 8000)
    _write(base / "test_set" / "synthetic" / "no_reverb" / "clean" / "clean_fileid_0.wav", _active(8000, 300), SAMPLE_RATE)
    _write(
        base / "test_set" / "synthetic" / "no_reverb" / "noisy" / "clnsp_snr0_tl-20_fileid_0.wav",
        _active(8000, 301),
        SAMPLE_RATE,
    )


def _config(directory: Path) -> Path:
    path = directory / "accept.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: dns.accept.limited
            seed: 17
            shard_size: 2
            delivery: on_demand
            source_scan: limited
            acceptance: deterministic_case_stream_v1
            claim_label: official_recipe_scaled_down
            clean_limit: 6
            noise_limit: 2
            include_non_16k_noise: 1
            num_files: 8
            audio_seconds: 1.0
            include_test: true
            audit_realizations: 4
            workers: 1
            """
        ),
        encoding="utf-8",
    )
    return path


class DnsAcceptanceTests(unittest.TestCase):
    def test_first_passing_candidate_matches_the_stored_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_name = "book_00000_chp_0000_reader_00002_0.wav"
            noise_name = "noise_a.wav"
            clean = root / clean_name
            noise = root / noise_name
            _write(clean, _active(32000, 1), SAMPLE_RATE)
            _write(noise, _active(32000, 2), SAMPLE_RATE)
            definition = build_definition(
                seed=17,
                num_files=2,
                clean_count=1,
                noise_count=1,
                order_sha256=order_sha256([clean_name], [noise_name]),
                inputs={"speech": "s", "noise": "n"},
                audio_seconds=1.0,
            )
            plain = plan_case(
                definition, 0, [clean_name], {clean_name: 32000}, [noise_name], {noise_name: 32000}
            )
            accepted, stats = plan_accepted_case(
                definition,
                0,
                [clean_name],
                {clean_name: clean},
                [noise_name],
                {noise_name: noise},
            )
        self.assertEqual(stats["retries"], 0)
        self.assertEqual(stats["clean_clip_skips"], 0)
        self.assertGreater(accepted["draws"]["clean_activity"], 0.6)
        for kind in ("speech", "noise"):
            self.assertEqual(
                [(piece["name"], piece["start"], piece["length"]) for piece in accepted["sources"][kind]],
                [(piece["name"], piece["start"], piece["length"]) for piece in plain["sources"][kind]],
            )
        self.assertEqual(accepted["draws"]["snr_db"], plain["draws"]["snr_db"])
        self.assertEqual(accepted["draws"]["target_level_dbfs"], plain["draws"]["target_level_dbfs"])

    def test_clipped_and_inactive_sources_are_not_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root)
            config = _config(root)
            database = root / "reproduction.duckdb"
            initialize_database(database)
            summary = run_config(
                config,
                "register",
                corpus=root,
                staging=root / "staging",
                database=database,
                snapshots=root / "snapshots",
                resource_db=root / "missing.duckdb",
                generated=root / "generated",
            )
            self.assertEqual(summary["wav_files"], 0)
            self.assertEqual(summary["non_16k_noise"], 1)
            self.assertEqual(summary["acceptance"]["failed_cases"], 0)
            self.assertGreater(summary["acceptance"]["clean_clip_skips"] + summary["acceptance"]["activity_rejects"], 0)
            self.assertEqual(summary["validation"], "sorted_uttid_tail_tenth_not_cv_synthetic_small")
            self.assertFalse(summary["selection_uses_test"])
            train, valid = open_chunk_datasets(
                dataset="dns.accept.limited@1",
                snapshot_root=root / "snapshots",
                corpora_root=root,
                generated_root=root / "generated",
                derived_root=root / "derived",
            )
            self.assertEqual(train.renderer, "dns_pcm16_accept_v2")
            self.assertEqual(train.retry_policy, "deterministic_case_stream_v1")
            self.assertIn("renderer", train.identity())
            lengths = train.waveform_lengths() + valid.waveform_lengths()
            self.assertTrue(all(length == SAMPLE_RATE for length in lengths))
            case_id = train.case_ids()[0]
            from src.reproduction.pipeline.chunking import PlannedChunk

            first = train.collate_chunks([PlannedChunk(case_id, 0, 1600)])
            second = train.collate_chunks([PlannedChunk(case_id, 0, 1600)])
            self.assertTrue(np.array_equal(first["noisy"][0].numpy(), second["noisy"][0].numpy()))
            document = train.store.docs[case_id]
            self.assertNotIn("book_00000_chp_0000_reader_00000_0.wav", json.dumps(document["sources"]))
            rendered = render_realization(document, {"speech": train.speech, "noise": train.noise})
            again = render_realization(document, {"speech": train.speech, "noise": train.noise})
            self.assertTrue(np.array_equal(rendered["noisy"], again["noisy"]))
            samples = list(zip(train.case_ids(), train.waveform_lengths()))
            hashes = []
            for world in (1, 2, 4):
                plan = plan_chunk_epoch(
                    samples,
                    espnet_epoch=1,
                    seed=17,
                    world_size=world,
                    num_samples=len(samples),
                    chunk_length=8000,
                    global_batch=world,
                    num_cache_chunks=8,
                    shuffle=True,
                    shift_ratio=0.5,
                )
                self.assertTrue(all(chunk.stop - chunk.start == 8000 for rank in plan.rank_batches for batch in rank for chunk in batch))
                hashes.append(plan.plan_hash)
                repeat = plan_chunk_epoch(
                    samples,
                    espnet_epoch=1,
                    seed=17,
                    world_size=world,
                    num_samples=len(samples),
                    chunk_length=8000,
                    global_batch=world,
                    num_cache_chunks=8,
                    shuffle=True,
                    shift_ratio=0.5,
                )
                self.assertEqual(plan.plan_hash, repeat.plan_hash)
            self.assertEqual(len(set(hashes)), 3)
            train.close()
            valid.close()

    def test_exhausted_attempts_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "clip.wav"
            _write(path, np.ones(8000), SAMPLE_RATE)
            definition = build_definition(
                seed=1,
                num_files=2,
                clean_count=1,
                noise_count=1,
                order_sha256=order_sha256(["clip.wav"], ["clip.wav"]),
                inputs={"speech": "s", "noise": "n"},
                audio_seconds=0.2,
                acceptance="deterministic_case_stream_v1",
            )
            with self.assertRaises(ReproductionError):
                plan_accepted_case(
                    definition,
                    0,
                    ["clip.wav"],
                    {"clip.wav": path},
                    ["clip.wav"],
                    {"clip.wav": path},
                )

    def test_post_mix_cancellation_is_rejected_by_the_official_predicate(self):
        tone = np.sin(np.linspace(0, 80, 1600))
        clean, noise, noisy, _level = snr_mixer(tone, -tone, 0, -15)
        self.assertTrue(is_clipped(clean) or is_clipped(noise) or is_clipped(noisy))

    def test_same_case_matches_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_name = "book_00000_chp_0000_reader_00002_0.wav"
            noise_name = "noise_hi.wav"
            clean = root / clean_name
            noise = root / noise_name
            _write(clean, _active(8000, 7), SAMPLE_RATE)
            _write(noise, _active(4000, 8), 8000)
            script = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                from src.reproduction.build.dns2020 import build_definition, order_sha256, plan_accepted_case, render_realization
                clean = Path({str(clean)!r})
                noise = Path({str(noise)!r})
                definition = build_definition(
                    seed=3, num_files=2, clean_count=1, noise_count=1,
                    order_sha256=order_sha256([{clean_name!r}], [{noise_name!r}]),
                    inputs={{"speech": "s", "noise": "n"}},
                    audio_seconds=0.2,
                    acceptance="deterministic_case_stream_v1",
                )
                document, stats = plan_accepted_case(
                    definition, 0, [{clean_name!r}], {{{clean_name!r}: clean}}, [{noise_name!r}], {{{noise_name!r}: noise}}
                )
                rendered = render_realization(document, {{"speech": {{{clean_name!r}: clean}}, "noise": {{{noise_name!r}: noise}}}})
                print(json.dumps({{"retries": stats["retries"], "snr": document["draws"]["snr_db"], "n": int(rendered["noisy"].shape[0])}}))
                """
            )
            first = subprocess.check_output([str(ROOT / ".venvs/se/bin/python"), "-c", script], cwd=ROOT, text=True)
            second = subprocess.check_output([str(ROOT / ".venvs/se/bin/python"), "-c", script], cwd=ROOT, text=True)
            self.assertEqual(json.loads(first), json.loads(second))

    def test_tiny_trainer_resumes_the_same_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root)
            database = root / "reproduction.duckdb"
            initialize_database(database)
            run_config(
                _config(root),
                "register",
                corpus=root,
                staging=root / "staging",
                database=database,
                snapshots=root / "snapshots",
                resource_db=root / "missing.duckdb",
                generated=root / "generated",
            )
            train, valid = open_chunk_datasets(
                dataset="dns.accept.limited@1",
                snapshot_root=root / "snapshots",
                corpora_root=root,
                generated_root=root / "generated",
                derived_root=root / "derived",
            )
            straight = _fit_tiny(root / "straight", train, valid, max_steps=2)
            paused = _fit_tiny(root / "paused", train, valid, max_steps=2, segment_steps=1)
            self.assertTrue((root / "paused" / "data_provenance.json").is_file())
            self.assertIn("dns_pcm16_accept_v2", (root / "paused" / "data_provenance.json").read_text())
            paused_log = (root / "paused" / "metrics.jsonl").read_text()
            self.assertIn('"validated": true', paused_log)
            mismatch = dict(paused["config"])
            mismatch["data"] = {**mismatch["data"], "virtual_world_size": 2}
            rejected = _trainer(root / "paused", train, mismatch, max_steps=2)
            with self.assertRaises(TrainingError):
                rejected.fit(train, valid, resume=paused["checkpoint"])
            resumed = _fit_tiny(
                root / "paused",
                train,
                valid,
                max_steps=2,
                resume=paused["checkpoint"],
                config=paused["config"],
            )
            self.assertEqual(paused["waves"] + resumed["waves"], straight["waves"])
            self.assertEqual(paused["case_ids"] + resumed["case_ids"], straight["case_ids"])
            self.assertGreaterEqual(len(straight["waves"]), 2)
            train.close()
            valid.close()


def _task_config() -> dict:
    return {
        "seed": 17,
        "deterministic": True,
        "loss": "official",
        "grad_clip": 5.0,
        "optimizer": {"name": "adamw", "lr": 1.0e-3, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "tf_locoformer",
            "variant": "dns",
            "stage": None,
            "sampling_rate": 16000,
            "architecture": dict(TINY),
            "stft": {},
            "components": {},
        },
        "chunk_iterator": {
            "num_samples_per_epoch": 1,
            "chunk_length": 15999,
            "chunk_shift_ratio": 0.5,
            "num_cache_chunks": 4,
            "shuffle": True,
            "max_epochs": 3,
            "early_stopping_patience": 5,
            "keep_nbest": 5,
        },
        "best_checkpoint_start_epoch": 0,
        "data": {"load_workers": 1},
    }


def _trainer(directory: Path, train, config: dict, *, max_steps: int) -> Trainer:
    return Trainer(
        TFLocoformerTask(config, cpu_process().device),
        plan_batches(1, 1, 1),
        cpu_process(),
        max_steps=max_steps,
        seed=17,
        checkpoint_dir=directory,
        checkpoint_interval=0,
        validation_interval=0,
        config=config,
    )


def _fit_tiny(directory, train, valid, *, max_steps, segment_steps=None, resume=None, config=None):
    config = config or _task_config()
    seen: list[bytes] = []
    cases: list[list[str]] = []
    trainer = _trainer(directory, train, config, max_steps=max_steps)

    def remember(batches, accumulation):
        noisy = batches[0]["noisy"].detach().cpu().numpy()
        seen.append(noisy.tobytes())
        cases.append(list(batches[0]["case_id"]))
        return TFLocoformerTask.train_update(trainer.task, batches, accumulation)

    trainer.task.train_update = remember
    trainer.fit(train, valid, resume, segment_steps=segment_steps)
    checkpoint = Path(trainer.last_checkpoint) if trainer.last_checkpoint else None
    return {"waves": seen, "case_ids": cases, "checkpoint": checkpoint, "config": config}


if __name__ == "__main__":
    unittest.main()
