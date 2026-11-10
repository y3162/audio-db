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
    _build_signal,
    build_definition,
    is_clipped,
    order_sha256,
    plan_accepted_case,
    plan_accepted_realizations,
    plan_case,
    read_signal_span,
    render_realization,
    snr_mixer,
)
from src.reproduction.db.registry import initialize_database
from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.reproduction.pipeline.pregenerated import open_chunk_datasets
from src.reproduction.prepare.dns_run import run_config

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

    def test_test_set_clean_keeps_a_file_longer_than_the_target(self):
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_00000_chp_0000_reader_00000_0.wav"
            _write(path, np.full(1000, 0.1), SAMPLE_RATE)
            rng = np.random.Generator(np.random.PCG64(1))
            pieces, _position, skips, ok = _build_signal(
                [path.name],
                {path.name: path},
                0,
                400,
                0,
                rng,
                is_clean=True,
                is_test_set=True,
                cache=None,
            )
            self.assertTrue(ok)
            self.assertEqual(skips, 0)
            self.assertEqual(pieces[0]["start"], 0)
            self.assertEqual(pieces[0]["length"], 1000)
            cropped, _position, _skips, cropped_ok = _build_signal(
                [path.name],
                {path.name: path},
                0,
                400,
                0,
                np.random.Generator(np.random.PCG64(1)),
                is_clean=True,
                is_test_set=False,
                cache=None,
            )
            self.assertTrue(cropped_ok)
            self.assertEqual(cropped[0]["length"], 400)
            official_path = ROOT / "data" / "corpora" / "DNS2020" / "noisyspeech_synthesizer_multiprocessing.py"
            if not official_path.is_file():
                self.skipTest("DNS synthesizer is not in the corpus checkout")
            # The official module seeds the global RNGs on import, so it runs in a child process.
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        f"""
                        import multiprocessing, sys
                        sys.path.insert(0, {str(official_path.parent)!r})
                        import noisyspeech_synthesizer_multiprocessing as official
                        official.clean_counter = multiprocessing.Value("i", 0)
                        audio, files, clipped = official.build_audio(
                            True,
                            {{
                                "fs": {SAMPLE_RATE},
                                "silence_length": 0.0,
                                "audio_length": {400 / SAMPLE_RATE},
                                "is_test_set": True,
                                "cleanfilenames": [{str(path)!r}],
                            }},
                            0,
                        )
                        assert clipped == []
                        assert len(files) == 1
                        assert len(audio) == 1000
                        """
                    ),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_worker_count_does_not_change_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_names = []
            noise_names = []
            clean_paths = {}
            noise_paths = {}
            for index in range(4):
                name = f"book_00000_chp_0000_reader_{index:05d}_0.wav"
                path = root / name
                _write(path, _active(16000, 10 + index), SAMPLE_RATE)
                clean_names.append(name)
                clean_paths[name] = path
            for index in range(3):
                name = f"noise_{index}.wav"
                path = root / name
                _write(path, _active(24000, 40 + index), SAMPLE_RATE)
                noise_names.append(name)
                noise_paths[name] = path
            definition = build_definition(
                seed=17,
                num_files=6,
                clean_count=4,
                noise_count=3,
                order_sha256=order_sha256(clean_names, noise_names),
                inputs={"speech": "s", "noise": "n"},
                audio_seconds=0.5,
                acceptance="deterministic_case_stream_v1",
            )
            one, stats_one = plan_accepted_realizations(
                definition, clean_names, clean_paths, noise_names, noise_paths, workers=1
            )
            two, stats_two = plan_accepted_realizations(
                definition, clean_names, clean_paths, noise_names, noise_paths, workers=2
            )
            self.assertEqual(one, two)
            self.assertEqual(stats_one["outer_attempts"], stats_two["outer_attempts"])
            full, _rate = sf.read(clean_paths[clean_names[0]], dtype="float64", always_2d=True)
            span, rate = read_signal_span(clean_paths[clean_names[0]], 100, 500)
            self.assertEqual(rate, SAMPLE_RATE)
            self.assertTrue(np.array_equal(span, np.asarray(full[100:600, 0], dtype=np.float64)))

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

if __name__ == "__main__":
    unittest.main()
