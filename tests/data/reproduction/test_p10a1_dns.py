"""P10-A.1 full DNS acceptance plan: spans, shards, and the recipe graph."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.recipe import sha256_file
from src.experiments.data.build.dns2020 import (
    SAMPLE_RATE,
    _stored,
    build_definition,
    order_sha256,
    plan_accepted_case,
    plan_accepted_span,
)
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import digest
from src.experiments.data.prepare.dns2020 import (
    accept_protocol_pair,
    assemble_accept_graph,
    asset_row,
    case_set_for,
    material_set,
    pair_test_files,
    protocol_pair,
    recorded_pair,
)
from src.experiments.data.prepare.dns_accept_store import (
    load_documents,
    missing_ranges,
    plan_identity,
    summarize_cases,
    write_shard,
)
from src.experiments.data.prepare.dns_run import (
    ACCEPT_RECIPE_NAME,
    _check_accept_recipe_config,
    load_build_config,
)

ROOT = Path(__file__).resolve().parents[3]


def _write(path: Path, wave: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(wave, dtype=np.float64), rate, subtype="PCM_16")


def _active(frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(scale=0.08, size=frames), -0.4, 0.4)


def _pools(root: Path) -> tuple[dict, dict, dict, dict]:
    clean_names = []
    noise_names = []
    clean_paths = {}
    noise_paths = {}
    for index in range(4):
        name = f"book_00000_chp_0000_reader_{index:05d}_0.wav"
        path = root / "DNS2020" / "datasets" / "clean" / name
        _write(path, _active(16000, 10 + index))
        clean_names.append(name)
        clean_paths[name] = path
    for index in range(3):
        name = f"noise_{index}.wav"
        path = root / "DNS2020" / "datasets" / "noise" / name
        _write(path, _active(24000, 40 + index))
        noise_names.append(name)
        noise_paths[name] = path
    return clean_names, noise_names, clean_paths, noise_paths


def _definition(clean_names, noise_names, speech_id: str = "s", noise_id: str = "n") -> dict:
    return build_definition(
        seed=17,
        num_files=4,
        clean_count=len(clean_names),
        noise_count=len(noise_names),
        order_sha256=order_sha256(clean_names, noise_names),
        inputs={"speech": speech_id, "noise": noise_id},
        audio_seconds=0.5,
        acceptance="deterministic_case_stream_v1",
    )


class AcceptSpanTests(unittest.TestCase):
    def test_span_matches_per_case_plans_and_worker_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_names, noise_names, clean_paths, noise_paths = _pools(root)
            definition = _definition(clean_names, noise_names)
            alone = [
                plan_accepted_case(
                    definition, index, clean_names, clean_paths, noise_names, noise_paths, cache=None
                )
                for index in range(4)
            ]
            serial, serial_totals = plan_accepted_span(
                definition, clean_names, clean_paths, noise_names, noise_paths, start=0, stop=4, workers=1
            )
            middle, middle_totals = plan_accepted_span(
                definition, clean_names, clean_paths, noise_names, noise_paths, start=1, stop=3, workers=1
            )
            parallel, parallel_totals = plan_accepted_span(
                definition, clean_names, clean_paths, noise_names, noise_paths, start=0, stop=4, workers=2
            )
        self.assertEqual(serial, [item[0] for item in alone])
        self.assertEqual(serial_totals["case_stats"], [item[1] for item in alone])
        self.assertEqual(middle, serial[1:3])
        self.assertEqual(middle_totals["case_stats"], serial_totals["case_stats"][1:3])
        self.assertEqual(
            [digest(_stored(item)) for item in parallel],
            [digest(_stored(item)) for item in serial],
        )
        self.assertEqual(parallel_totals["case_stats"], serial_totals["case_stats"])


class AcceptShardTests(unittest.TestCase):
    def test_resume_gap_duplicate_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_names, noise_names, clean_paths, noise_paths = _pools(root)
            definition = _definition(clean_names, noise_names)
            documents, totals = plan_accepted_span(
                definition, clean_names, clean_paths, noise_names, noise_paths, start=0, stop=4, workers=1
            )
            identity = plan_identity(
                definition,
                protocol_id="dns-accept-test",
                plan_shard=2,
                code_sha256="a" * 64,
            )
            shard_dir = root / "shards"
            rows = list(zip(documents, totals["case_stats"], strict=True))
            first = write_shard(shard_dir, identity, 0, 2, rows[:2])
            again = write_shard(shard_dir, identity, 0, 2, rows[:2])
            self.assertEqual(first, again)
            self.assertEqual(missing_ranges(shard_dir, identity, 4), [(2, 4)])
            write_shard(shard_dir, identity, 2, 4, rows[2:])
            self.assertEqual(missing_ranges(shard_dir, identity, 4), [])
            loaded, stats = load_documents(shard_dir, identity, 4)
            self.assertEqual(loaded, [_stored(item) for item in documents])
            self.assertEqual([int(row["retries"]) for row in stats], [int(item["retries"]) for item in totals["case_stats"]])
            changed = dict(identity)
            changed["generator_code_sha256"] = "b" * 64
            with self.assertRaises(ReproductionError):
                missing_ranges(shard_dir, changed, 4)
            shutil.copy(shard_dir / "0000002-0000004.jsonl", shard_dir / "copy.jsonl")
            with self.assertRaises(ReproductionError):
                missing_ranges(shard_dir, identity, 4)
            (shard_dir / "copy.jsonl").unlink()
            (shard_dir / "0000002-0000004.jsonl").unlink()
            with self.assertRaises(ReproductionError):
                load_documents(shard_dir, identity, 4)
            with self.assertRaises(ReproductionError):
                write_shard(
                    shard_dir,
                    identity,
                    0,
                    2,
                    [(documents[1], totals["case_stats"][1]), (documents[0], totals["case_stats"][0])],
                )


class AcceptMetadataTests(unittest.TestCase):
    def test_summary_accepts_only_full_length_cases(self):
        document, stats = _metadata_case(0)
        summary = summarize_cases([document], [stats], clean_names=["clean.wav"], noise_names=["noise.wav"])
        self.assertEqual(summary["cases"], 1)
        self.assertEqual(summary["non16_cases"], 1)
        self.assertEqual(summary["clean_clip_cases"], 1)
        self.assertGreater(summary["clean_activity_min"], 0.6)
        short = json_copy(document)
        short["output"]["length"] = 8000
        with self.assertRaises(ReproductionError):
            summarize_cases([short], [stats], clean_names=["clean.wav"], noise_names=["noise.wav"])
        inactive = json_copy(document)
        inactive["draws"]["clean_activity"] = 0.6
        with self.assertRaises(ReproductionError):
            summarize_cases([inactive], [stats], clean_names=["clean.wav"], noise_names=["noise.wav"])
        capped = json_copy(document)
        capped["draws"]["retries"] = 50
        with self.assertRaises(ReproductionError):
            summarize_cases([capped], [dict(stats)], clean_names=["clean.wav"], noise_names=["noise.wav"])


def json_copy(document: dict) -> dict:
    import copy

    return copy.deepcopy(document)


def _metadata_case(gen_index: int) -> tuple[dict, dict]:
    document = {
        "draws": {
            "gen_index": gen_index,
            "snr_db": -5,
            "target_level_dbfs": -35,
            "retries": 0,
            "utt_id": f"fileid_{gen_index}",
            "clean_activity": 0.61,
        },
        "output": {"length": 480000},
        "sources": {
            "speech": [{"name": "clean.wav", "start": 0, "length": 480000, "source_rate": 16000}],
            "noise": [{"name": "noise.wav", "start": 0, "length": 8000, "source_rate": 44100}],
        },
    }
    stats = {
        "retries": 0,
        "activity_rejects": 0,
        "post_clip_rejects": 0,
        "short_rejects": 0,
        "clean_clip_skips": 1,
        "noise_clip_skips": 0,
        "resampled_noise_pieces": 1,
    }
    return document, stats


class AcceptRecipeConfigTests(unittest.TestCase):
    def test_full_config_rejects_limits_and_scaled_claim(self):
        path = ROOT / "configs" / "data" / "builds" / "dns_tflocoformer_accept_recipe.yaml"
        config = load_build_config(path)
        _check_accept_recipe_config(config)
        self.assertEqual(config["name"], ACCEPT_RECIPE_NAME)
        self.assertIsNone(config.get("clean_limit"))
        self.assertIsNone(config.get("num_files"))
        for mutated in (
            {**config, "clean_limit": 8},
            {**config, "noise_limit": 8},
            {**config, "num_files": 256},
            {**config, "claim_label": "official_recipe_scaled_down"},
            {**config, "source_scan": "limited"},
            {**config, "include_non_16k_noise": 1},
            {**config, "name": "tf_locoformer.dns.recipe"},
        ):
            with self.assertRaises(ReproductionError):
                _check_accept_recipe_config(mutated)

    def test_smoke_config_still_requires_the_scaled_claim(self):
        text = (ROOT / "configs" / "data" / "builds" / "dns_tflocoformer_accept_smoke.yaml").read_text(encoding="utf-8")
        self.assertIn("official_recipe_scaled_down", text)
        self.assertNotIn("tf_locoformer.dns.accept_recipe", text)


class AcceptGraphTests(unittest.TestCase):
    def test_full_recipe_reuses_test_set_and_smoke_stays_scaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpora"
            clean_names, noise_names, clean_paths, noise_paths = _pools(corpus)
            clean_assets = [
                asset_row(
                    clean_paths[name],
                    corpus_root=corpus,
                    subset="clean",
                    rate=SAMPLE_RATE,
                    frames=16000,
                    channels=1,
                    sha256=sha256_file(clean_paths[name]),
                    speaker_id=f"reader_{index:05d}",
                )
                for index, name in enumerate(clean_names)
            ]
            noise_assets = [
                asset_row(
                    noise_paths[name],
                    corpus_root=corpus,
                    subset="noise",
                    rate=SAMPLE_RATE,
                    frames=24000,
                    channels=1,
                    sha256=sha256_file(noise_paths[name]),
                    speaker_id=None,
                )
                for name in noise_names
            ]
            speech = material_set(clean_assets, name="dns.tflocoformer.accept_recipe.speech", role="speech", note="unit")
            noise = material_set(noise_assets, name="dns.tflocoformer.accept_recipe.noise", role="noise", note="unit")
            definition = _definition(clean_names, noise_names, speech["material_set_id"], noise["material_set_id"])
            documents, _totals = plan_accepted_span(
                definition, clean_names, clean_paths, noise_names, noise_paths, start=0, stop=4, workers=1
            )
            test_clean = []
            test_noisy = []
            for index in range(150):
                clean_path = corpus / "DNS2020" / "datasets" / "test_set" / "synthetic" / "no_reverb" / "clean" / f"clean_fileid_{index}.wav"
                noisy_path = (
                    corpus
                    / "DNS2020"
                    / "datasets"
                    / "test_set"
                    / "synthetic"
                    / "no_reverb"
                    / "noisy"
                    / f"clnsp_snr0_tl-20_fileid_{index}.wav"
                )
                _write(clean_path, _active(4000, 1000 + index))
                _write(noisy_path, _active(4000, 2000 + index))
                test_clean.append(
                    asset_row(
                        clean_path,
                        corpus_root=corpus,
                        subset="test_no_reverb_clean",
                        rate=SAMPLE_RATE,
                        frames=4000,
                        channels=1,
                        sha256=sha256_file(clean_path),
                        speaker_id=None,
                    )
                )
                test_noisy.append(
                    asset_row(
                        noisy_path,
                        corpus_root=corpus,
                        subset="test_no_reverb_noisy",
                        rate=SAMPLE_RATE,
                        frames=4000,
                        channels=1,
                        sha256=sha256_file(noisy_path),
                        speaker_id=None,
                    )
                )
            _parent, ondemand = protocol_pair()
            pairs = pair_test_files(test_clean, test_noisy)
            test_cases = [recorded_pair(clean, noisy, ondemand["protocol_id"]) for clean, noisy in pairs]
            test_set = case_set_for(
                "dns.tflocoformer.no_reverb",
                "test",
                test_cases,
                {"rule": "explicit"},
                "official synthetic no_reverb pairs by fileid. Not used for selection. Not cv_synthetic_small.",
            )
            report = {"status": "PASS", "delivery": "on_demand", "wav_files": 0}
            graph = assemble_accept_graph(
                name=ACCEPT_RECIPE_NAME,
                claim="espnet_public_se_recipe_reproduction",
                definition=definition,
                documents=documents,
                report=report,
                speech=speech,
                noise=noise,
                source_assets=clean_assets + noise_assets + test_clean + test_noisy,
                test_cases=test_cases,
                resource_db=None,
                full_recipe=True,
                test_set=test_set,
                extra_protocols=[ondemand],
            )
            dataset = graph.datasets[0]
            deviation_ids = [item["id"] for item in dataset["deviations_json"]]
            self.assertIn("full_accept_recipe", deviation_ids)
            self.assertNotIn("scaled_down", deviation_ids)
            self.assertFalse(dataset["selection_uses_test"])
            self.assertEqual(dataset["name"], ACCEPT_RECIPE_NAME)
            roles = {row["split_role"]: row for row in graph.case_sets}
            self.assertEqual(roles["test"]["name"], "dns.tflocoformer.no_reverb")
            self.assertEqual(roles["test"]["member_count"], 150)
            self.assertEqual(roles["train"]["member_count"], 3)
            self.assertEqual(roles["valid"]["member_count"], 1)
            accept_id = dataset["protocol_id"]
            self.assertNotEqual(accept_id, ondemand["protocol_id"])
            by_id = {row["case_id"]: row for row in graph.cases}
            for item in roles["train"]["items"]:
                self.assertEqual(by_id[item["case_id"]]["protocol_id"], accept_id)
            for item in roles["test"]["items"]:
                self.assertEqual(by_id[item["case_id"]]["protocol_id"], ondemand["protocol_id"])
            _formula, accept_child = accept_protocol_pair()
            smoke_case = recorded_pair(test_clean[0], test_noisy[0], accept_child["protocol_id"])
            smoke = assemble_accept_graph(
                name="tf_locoformer.dns.accept_smoke",
                claim="official_recipe_scaled_down",
                definition=definition,
                documents=documents,
                report=report,
                speech=speech,
                noise=noise,
                source_assets=clean_assets + noise_assets + test_clean[:1] + test_noisy[:1],
                test_cases=[smoke_case],
                resource_db=None,
            )
            smoke_ids = [item["id"] for item in smoke.datasets[0]["deviations_json"]]
            self.assertIn("scaled_down", smoke_ids)
            self.assertNotIn("full_accept_recipe", smoke_ids)
            self.assertEqual(
                next(row["name"] for row in smoke.case_sets if row["split_role"] == "test"),
                "dns.tflocoformer.accept.no_reverb",
            )


if __name__ == "__main__":
    unittest.main()
