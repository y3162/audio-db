"""P2 fixed-pair preparation tests.

Audio here is synthetic. The pinned official lists are parsed, but the real
VoiceBank corpus is not opened.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from src.experiments.data.cli import main
from src.experiments.data.db.registry import initialize_database, read_graph, register_staging
from src.experiments.data.db.snapshot import export_snapshot
from src.experiments.data.errors import PreparationBlocked, ReproductionError
from src.experiments.data.paths import ROOT
from src.experiments.data.prepare.assemble import (
    build_voicebank_graph,
    load_dataset_specs,
    load_protocols,
)
from src.experiments.data.prepare.derive import derive_all
from src.experiments.data.prepare.resample import derivation_row, resample_float32
from src.experiments.data.prepare.lists import (
    holdout_split,
    load_manifest,
    parse_mossformergan_scp,
    parse_mp_senet,
)
from src.experiments.data.prepare.run import PreparePaths, execute
from src.experiments.data.snapshot import load_snapshot

CONFIG = ROOT / "configs" / "data"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pcm(path: Path, *, frames: int, rate: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    value = seed
    for index in range(frames):
        value = (value * 17 + index + seed) % 20001 - 10000
        samples.append(int(value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<" + "h" * frames, *samples))


class ListTests(unittest.TestCase):
    def test_eval_overlap_ignores_test_but_not_training_or_validation(self) -> None:
        from unittest.mock import patch
        from src.experiments.data.db.validate import _eval_overlap

        assets = {name: {"sha256": name} for name in ("train", "valid", "test")}
        groups = {name: {name} for name in assets}
        with patch("src.experiments.data.db.validate._read_eval_hashes", return_value={"test"}):
            self.assertEqual(_eval_overlap(groups, assets, "unused")["status"], "PASS")
        for name in ("train", "valid"):
            with patch("src.experiments.data.db.validate._read_eval_hashes", return_value={name}):
                self.assertEqual(_eval_overlap(groups, assets, "unused")["status"], "FAIL")
        self.assertEqual(_eval_overlap(groups, assets, None)["status"], "UNKNOWN")

    def test_pinned_lists_match_and_keep_order(self) -> None:
        loaded = load_manifest(CONFIG / "lists" / "manifest.yaml", ROOT)
        self.assertEqual(len(loaded["mp_senet"]["train"]["ids"]), 11572)
        self.assertEqual(len(loaded["mp_senet"]["test"]["ids"]), 824)
        self.assertEqual(loaded["mp_senet"]["train"]["ids"], loaded["mossformergan"]["train"]["ids"])
        self.assertEqual(loaded["mp_senet"]["test"]["ids"], loaded["mossformergan"]["test"]["ids"])
        self.assertEqual(loaded["mp_senet"]["train"]["sha256"], "00b32c062186df61fd980bde040be28d71bb3c3a1d0f25021b7fee49ff9ea363")
        self.assertEqual(loaded["mossformergan"]["test"]["sha256"], "4fc7acfe06a5b8f6cfafe2e0a975e04640db7f2a595b94669e4ac537e9c4d92b")
        protocols = {row["name"]: row for row in load_protocols(CONFIG)}
        self.assertEqual(protocols["mp_senet.fixed_pair"]["rule_json"]["normalize"], "noisy_rms_unit_v1")
        self.assertEqual(protocols["mossformergan_se_16k.fixed_pair"]["rule_json"]["normalize"], "independent_audio_norm_v1")
        ondemand = protocols["mp_senet.fixed_pair.ondemand"]["rule_json"]
        self.assertEqual(ondemand["delivery"], "on_demand")
        self.assertEqual(ondemand["resample"]["length"], "source_frames_plus_one_div_3")
        self.assertNotIn("resample", protocols["mp_senet.fixed_pair"]["rule_json"])
        specs = {row["name"]: row for row in load_dataset_specs(CONFIG)}
        self.assertEqual(specs["mp_senet.vb28.ondemand.official"]["expect"], {"train": 11572, "valid": 824, "test": 824})
        self.assertEqual(specs["mossformergan.vb28.ondemand.safe"]["profile"], "leakage_safe")

    def test_parser_rejects_duplicates_and_pair_mismatches(self) -> None:
        with self.assertRaises(ReproductionError):
            parse_mp_senet("p226_001|a\np226_001|b\n", where="training.txt")
        with self.assertRaises(ReproductionError):
            parse_mossformergan_scp("noisy/p226_001.wav clean/p226_002.wav\n", where="train.scp")
        ids = parse_mossformergan_scp(
            "/nas/noisy/p226_010.wav /nas/target/p226_010.wav\n",
            where="train.scp",
        )
        self.assertEqual(ids, ["p226_010"])
        self.assertEqual(parse_mp_senet("p226_010|ignored/path.wav\n", where="training.txt"), ["p226_010"])

    def test_training_launch_requires_on_demand(self) -> None:
        from src.experiments.data.mix.pregenerated import require_on_demand

        with self.assertRaises(ReproductionError):
            require_on_demand({"name": "mp_senet.vb28.official", "version": 1, "rule_json": {}})
        with self.assertRaises(ReproductionError):
            require_on_demand({"name": "tf_gridnet.ms_snsd.smoke", "version": 1, "rule_json": {"delivery": "stored"}})
        require_on_demand({"name": "mp_senet.vb28.ondemand.official", "version": 1, "rule_json": {"delivery": "on_demand"}})

    def test_holdout_preserves_order_without_a_seed(self) -> None:
        train, valid = holdout_split(
            ["p226_002", "p287_001", "p226_001", "p286_009"],
            ("p286", "p287"),
        )
        again, valid_again = holdout_split(
            ["p226_002", "p287_001", "p226_001", "p286_009"],
            ("p286", "p287"),
        )
        self.assertEqual(train, ["p226_002", "p226_001"])
        self.assertEqual(valid, ["p287_001", "p286_009"])
        self.assertEqual((train, valid), (again, valid_again))


class StoredDerivativeTests(unittest.TestCase):
    def test_stored_pcm_derivatives_and_float_resampling_for_each_remainder(self) -> None:
        database = ROOT / "data" / "db" / "reproduction.duckdb"
        if not database.is_file():
            self.skipTest("reproduction database is absent")
        import duckdb
        import numpy as np
        import soundfile as sf

        connection = duckdb.connect(str(database), read_only=True)
        try:
            rows = connection.execute(
                """
                WITH pairs AS (
                    SELECT c.rel_path AS source_rel, c.frames AS source_frames,
                           d.rel_path AS derived_rel, d.frames AS derived_frames,
                           d.sample_rate AS derived_rate, c.frames % 3 AS remainder
                    FROM repro.assets c
                    JOIN repro.assets d ON d.derived_from = c.asset_id
                    WHERE c.storage_root = 'corpora'
                      AND c.corpus = 'VoiceBank-DEMAND'
                      AND d.storage_root = 'derived'
                ),
                picked AS (
                    SELECT * FROM pairs
                    QUALIFY row_number() OVER (PARTITION BY remainder ORDER BY source_frames) = 1
                    UNION ALL
                    SELECT * FROM pairs
                    QUALIFY row_number() OVER (ORDER BY source_frames) = 1
                    UNION ALL
                    SELECT * FROM pairs
                    QUALIFY row_number() OVER (ORDER BY source_frames DESC) = 1
                )
                SELECT * FROM picked
                """
            ).fetchall()
        finally:
            connection.close()
        if not rows:
            self.skipTest("no stored VoiceBank derivatives are registered")
        seen = set()
        for source_rel, source_frames, derived_rel, derived_frames, derived_rate, remainder in rows:
            key = (source_rel, derived_rel)
            if key in seen:
                continue
            seen.add(key)
            source = ROOT / "data" / "corpora" / source_rel
            derived = ROOT / "data" / "derived" / derived_rel
            if not source.is_file() or not derived.is_file():
                self.skipTest(f"audio missing for {source_rel}")
            fresh = resample_float32(*_read_mono(source))
            stored, stored_rate = sf.read(derived, dtype="float32", always_2d=False)
            info = sf.info(derived)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertEqual(int(stored_rate), 16000)
            self.assertEqual(int(derived_rate), 16000)
            self.assertEqual(int(derived_frames), (int(source_frames) + 1) // 3)
            self.assertEqual(fresh.shape[0], int(derived_frames))
            self.assertEqual(int(remainder), int(source_frames) % 3)
            import soxr

            expected = soxr.resample(_read_mono(source)[0], 48000, 16000, quality="HQ")
            np.testing.assert_array_equal(fresh, expected)
            self.assertEqual(stored.shape, fresh.shape)
            self.assertTrue(np.isfinite(stored).all())


def _read_mono(path: Path):
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(rate)


class PrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_audio_creates_nothing(self) -> None:
        corpora = self.root / "corpora"
        corpora.mkdir()
        derived = self.root / "derived"
        staging = self.root / "staging"
        database = self.root / "reproduction.duckdb"
        code = main(
            [
                "prepare",
                "--step",
                "resolve",
                "--database",
                str(database),
                "--staging",
                str(staging),
                "--snapshots",
                str(self.root / "snapshots"),
                "--resource-db",
                str(self.root / "missing-resource.duckdb"),
                "--corpora",
                str(corpora),
                "--derived",
                str(derived),
                "--workers",
                "1",
            ]
        )
        self.assertEqual(code, 3)
        self.assertFalse(derived.exists())
        self.assertFalse(staging.exists())
        self.assertFalse(database.exists())

    def test_graph_distinguishes_official_and_safe(self) -> None:
        graph = _synthetic_graph()
        by_name = {row["name"]: row for row in graph.datasets}
        self.assertTrue(by_name["mp_senet.vb28.ondemand.official"]["selection_uses_test"])
        self.assertFalse(by_name["mp_senet.vb28.ondemand.safe"]["selection_uses_test"])
        self.assertFalse(by_name["mossformergan.vb28.ondemand.safe"]["selection_uses_test"])
        mp_protocol = next(row["protocol_id"] for row in graph.protocols if row["name"] == "mp_senet.fixed_pair.ondemand")
        moss_protocol = next(
            row["protocol_id"] for row in graph.protocols if row["name"] == "mossformergan_se_16k.fixed_pair.ondemand"
        )
        self.assertNotEqual(mp_protocol, moss_protocol)
        self.assertTrue(all(row["protocol_id"] == mp_protocol for row in graph.cases))
        self.assertEqual(by_name["mossformergan.vb28.ondemand.official"]["protocol_id"], moss_protocol)
        self.assertTrue(all(row["storage_root"] == "corpora" for row in graph.assets))
        counts = _member_counts(graph)
        self.assertEqual(counts["mp_senet.vb28.ondemand.train.official"], 3)
        self.assertEqual(counts["mp_senet.vb28.ondemand.valid.safe"], 2)
        self.assertEqual(counts["mp_senet.vb28.ondemand.train.safe"], 1)
        self.assertEqual(len({row["case_id"] for row in graph.cases}), 4)
        with self.assertRaises(ReproductionError):
            _synthetic_graph(safe_uses_test=True)

    def test_resample_is_deterministic_and_derive_writes_nothing(self) -> None:
        import soundfile as sf

        source = self.root / "source.wav"
        _pcm(source, frames=4803, rate=48000, seed=3)
        audio, rate = sf.read(source, dtype="float32", always_2d=False)
        first = resample_float32(audio, rate)
        second = resample_float32(audio, rate)
        self.assertEqual(first.shape, second.shape)
        self.assertTrue((first == second).all())
        self.assertEqual(first.shape[0], (4803 + 1) // 3)
        with self.assertRaises(ReproductionError):
            derive_all([{"source": source, "dest": self.root / "derived" / "out.wav"}], workers=1)
        self.assertEqual(list(self.root.rglob("*.wav")), [source])

    def test_resample_preserves_float_values_and_rejects_nonfinite(self) -> None:
        import numpy as np
        import soxr

        audio = np.full(4803, 1.234567, dtype=np.float32)
        audio[1000:2000] += np.float32(0.1234567)
        first = resample_float32(audio, 48000)
        expected = soxr.resample(audio, 48000, 16000, quality="HQ")
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(first, resample_float32(audio, 48000))
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(len(first), (len(audio) + 1) // 3)
        self.assertGreater(float(first.max()), 1.0)
        self.assertTrue(np.any(first * 32768 != np.rint(first * 32768)))
        for invalid in (np.nan, np.inf, -np.inf):
            broken = audio.copy()
            broken[0] = invalid
            with self.assertRaises(ReproductionError):
                resample_float32(broken, 48000)

    def test_old_quantized_snapshot_rule_is_rejected(self) -> None:
        from src.experiments.data.snapshot import FORMAT

        rule = dict(next(p for p in load_protocols(CONFIG) if p["name"] == "mp_senet.fixed_pair.ondemand")["rule_json"])
        rule["resample"] = dict(rule["resample"], quantization="pcm16")
        (self.root / "manifest.json").write_text(json.dumps({"provenance": {"format": FORMAT}, "schema_version": "1"}))
        (self.root / "dataset.json").write_text(json.dumps({"algorithm": "fixed_pair.mp_senet", "rule_json": rule}))
        with self.assertRaisesRegex(ReproductionError, "incompatible fixed-pair snapshot"):
            load_snapshot(self.root)

    def test_tiny_corpus_round_trip(self) -> None:
        paths = _tiny_paths(self.root)
        _write_tiny_corpus(paths.corpora)
        _write_tiny_config(paths.repo_root)
        execute("all", paths)
        stored = read_graph(paths.database)
        names = sorted(row["name"] for row in stored.datasets)
        self.assertEqual(
            names,
            [
                "mossformergan.vb28.ondemand.official",
                "mossformergan.vb28.ondemand.safe",
                "mp_senet.vb28.ondemand.official",
                "mp_senet.vb28.ondemand.safe",
            ],
        )
        by_name = {row["name"]: row for row in stored.datasets}
        official = by_name["mp_senet.vb28.ondemand.official"]
        safe = by_name["mp_senet.vb28.ondemand.safe"]
        self.assertTrue(official["selection_uses_test"])
        self.assertFalse(safe["selection_uses_test"])
        audits = {
            (row["subject_id"], row["audit_kind"]): row
            for row in stored.audits
            if row["subject_kind"] == "dataset"
        }
        self.assertEqual(audits[(official["dataset_id"], "count_check")]["status"], "PASS")
        self.assertEqual(audits[(official["dataset_id"], "leak_check")]["status"], "PASS")
        self.assertEqual(audits[(safe["dataset_id"], "leak_check")]["status"], "UNKNOWN")
        self.assertNotEqual(audits[(safe["dataset_id"], "leak_check")]["status"], "PASS")
        loaded = load_snapshot(paths.snapshots / safe["dataset_id"])
        self.assertTrue(loaded.manifest["provenance"]["allow_unknown_leak_check"])
        official_loaded = load_snapshot(paths.snapshots / official["dataset_id"])
        self.assertFalse(official_loaded.manifest["provenance"]["allow_unknown_leak_check"])
        self.assertEqual(official_loaded.dataset["algorithm"], "fixed_pair.mp_senet")
        moss = load_snapshot(paths.snapshots / by_name["mossformergan.vb28.ondemand.official"]["dataset_id"])
        self.assertEqual(moss.dataset["algorithm"], "fixed_pair.mossformergan_se_16k")
        self.assertTrue(all(row["protocol_id"] == official["protocol_id"] for row in moss.cases))
        again = export_snapshot(
            paths.database,
            "mp_senet.vb28.ondemand.safe@1",
            paths.snapshots,
            allow_unknown_leak_check=True,
            corpora=paths.corpora,
            derived=paths.derived,
        )
        self.assertEqual(again, loaded.snapshot_hash)
        with self.assertRaises(ReproductionError):
            export_snapshot(
                paths.database,
                "mp_senet.vb28.ondemand.safe@1",
                self.root / "other-snapshots",
                allow_unknown_leak_check=False,
                corpora=paths.corpora,
                derived=paths.derived,
            )
        empty = self.root / "empty-resource.duckdb"
        import duckdb

        duckdb.connect(str(empty)).close()
        other = self.root / "with-eval-absent.duckdb"
        initialize_database(other)
        register_staging(paths.staging, other, resource_db=str(empty))
        reasons = [
            row
            for row in read_graph(other).audits
            if row["audit_kind"] == "leak_check" and row["subject_id"] == safe["dataset_id"]
        ]
        self.assertTrue(reasons)
        self.assertTrue(all(row["status"] == "UNKNOWN" for row in reasons))
        self.assertTrue(any("eval schema is absent" in json.dumps(row["report_json"]) for row in reasons))
        self.assertEqual(list(paths.derived.rglob("*.wav")), [])
        with self.assertRaises(ReproductionError):
            execute("derive", paths)
        execute("register", paths)
        self.assertEqual(list(paths.derived.rglob("*.wav")), [])


def _synthetic_graph(*, safe_uses_test: bool = False):
    train = ["p226_001", "p286_001", "p287_001"]
    test = ["p232_001"]
    utterances = {utt: _fake_utterance(utt) for utt in train + test}
    list_meta = {
        "mp_senet": _list_meta("mp", train, test),
        "mossformergan": _list_meta("mfg", train, test),
    }
    specs = []
    for spec in load_dataset_specs(CONFIG):
        copied = dict(spec)
        copied["bindings"] = dict(spec["bindings"])
        if spec["profile"] == "official_faithful":
            copied["expect"] = {"train": 3, "valid": 1, "test": 1}
        else:
            copied["expect"] = {"train": 1, "valid": 2, "test": 1}
        if safe_uses_test and spec["profile"] == "leakage_safe":
            copied["bindings"] = {"train": "safe_train", "valid": "official_test", "test": "official_test"}
            copied["expect"] = {"train": 1, "valid": 1, "test": 1}
        specs.append(copied)
    return build_voicebank_graph(
        protocols=load_protocols(CONFIG),
        datasets=specs,
        list_meta=list_meta,
        train_ids=train,
        test_ids=test,
        holdout=("p286", "p287"),
        utterances=utterances,
        catalog_source_kind="staging_parquet",
        source_db_sha256=None,
        resource_db=None,
    )


def _fake_utterance(utterance_id: str) -> dict:
    from src.experiments.data.prepare.assemble import make_corpus_asset, make_derived_asset

    speaker = utterance_id.split("_", 1)[0]
    derivation = derivation_row()
    sides = {}
    for role, seed in (("clean", "a"), ("noisy", "b")):
        corpus = make_corpus_asset(
            rel_path=f"VoiceBank-DEMAND/subset/{role}/{utterance_id}.wav",
            subset=f"{role}_subset",
            speaker_id=speaker,
            utterance_id=utterance_id,
            sample_rate=48000,
            frames=4800,
            channels=1,
            size_bytes=1000,
            sha256=_sha(f"corpus-{role}-{seed}-{utterance_id}"),
            role=role,
        )
        derived = make_derived_asset(
            rel_path=f"VoiceBank-DEMAND/soxr_hq_16000/{role}/{utterance_id}.wav",
            subset=f"{role}_subset",
            speaker_id=speaker,
            utterance_id=utterance_id,
            frames=1600,
            size_bytes=400,
            sha256=_sha(f"derived-{role}-{seed}-{utterance_id}"),
            derivation_id=derivation["derivation_id"],
            derived_from=corpus["asset_id"],
            role=role,
            source_frames=4800,
            source_rate=48000,
            clipped_samples=0,
        )
        sides[role] = {"corpus": corpus, "derived": derived}
    return {
        "utterance_id": utterance_id,
        "speaker_id": speaker,
        "transcript": "hello" if utterance_id == "p226_001" else None,
        "derivation": derivation,
        "clean": sides["clean"],
        "noisy": sides["noisy"],
    }


def _list_meta(prefix: str, train: list[str], test: list[str]) -> dict:
    return {
        "source": prefix,
        "repo": f"https://example.invalid/{prefix}",
        "commit": "abc123",
        "format": "mp_senet",
        "train": {
            "path": f"{prefix}/training.txt",
            "file": f"configs/data/lists/{prefix}/training.txt",
            "sha256": _sha(prefix + "train"),
            "ids": list(train),
        },
        "test": {
            "path": f"{prefix}/test.txt",
            "file": f"configs/data/lists/{prefix}/test.txt",
            "sha256": _sha(prefix + "test"),
            "ids": list(test),
        },
    }


def _member_counts(graph) -> dict[str, int]:
    return {row["name"]: row["member_count"] for row in graph.case_sets}


def _tiny_paths(root: Path) -> PreparePaths:
    return PreparePaths(
        database=root / "reproduction.duckdb",
        staging=root / "staging",
        snapshots=root / "snapshots",
        resource_db=None,
        corpora=root / "corpora",
        derived=root / "derived",
        repo_root=root,
        config_root=root / "configs" / "data",
        workers=1,
    )


def _write_tiny_corpus(corpora: Path, extra_train: tuple[str, ...] = ()) -> None:
    pairs = {
        "clean_trainset_28spk_wav": ["p226_001", "p286_001", "p287_001"],
        "noisy_trainset_28spk_wav": ["p226_001", "p286_001", "p287_001"],
        "clean_testset_wav": ["p232_001"],
        "noisy_testset_wav": ["p232_001"],
    }
    for subset in ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"):
        pairs[subset].extend(extra_train)
    seed = 1
    for subset, names in pairs.items():
        for name in names:
            _pcm(corpora / "VoiceBank-DEMAND" / subset / f"{name}.wav", frames=4800, rate=48000, seed=seed)
            seed += 1
    text = corpora / "VoiceBank-DEMAND" / "trainset_28spk_txt" / "p226_001.txt"
    text.parent.mkdir(parents=True, exist_ok=True)
    text.write_text("Please call Stella.\n", encoding="utf-8")
    (corpora / "VoiceBank-DEMAND" / "testset_txt").mkdir(parents=True, exist_ok=True)


def _write_tiny_config(repo: Path, extra_train: tuple[str, ...] = ()) -> None:
    config = repo / "configs" / "data"
    config.mkdir(parents=True)
    shutil.copy(CONFIG / "voicebank_vb28.yaml", config / "voicebank_vb28.yaml")
    (config / "protocols").mkdir(parents=True)
    for name in (
        "mp_senet.fixed_pair.yaml",
        "mossformergan_se_16k.fixed_pair.yaml",
        "mp_senet.fixed_pair.ondemand.yaml",
        "mossformergan_se_16k.fixed_pair.ondemand.yaml",
    ):
        shutil.copy(CONFIG / "protocols" / name, config / "protocols" / name)
    train = ["p226_001", *extra_train, "p286_001", "p287_001"]
    test = ["p232_001"]
    mp_train = "".join(f"{item}|unused/{item}.wav\n" for item in train)
    mp_test = "".join(f"{item}|unused/{item}.wav\n" for item in test)
    mfg_train = "".join(f"/nas/noisy/{item}.wav /nas/target/{item}.wav\n" for item in train)
    mfg_test = "".join(f"/nas/noisy/{item}.wav /nas/target/{item}.wav\n" for item in test)
    files = {
        "configs/data/lists/mp_senet/training.txt": mp_train,
        "configs/data/lists/mp_senet/test.txt": mp_test,
        "configs/data/lists/mossformergan/tr_demand_28_spks_16k.scp": mfg_train,
        "configs/data/lists/mossformergan/cv_demand_testset_16k.scp": mfg_test,
    }
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    manifest = {
        "mp_senet": {
            "source": "synthetic-mp",
            "repo": "https://example.invalid/mp",
            "commit": "abc123",
            "format": "mp_senet",
            "train": _side("configs/data/lists/mp_senet/training.txt", "training.txt", mp_train),
            "test": _side("configs/data/lists/mp_senet/test.txt", "test.txt", mp_test),
        },
        "mossformergan": {
            "source": "synthetic-mfg",
            "repo": "https://example.invalid/mfg",
            "commit": "def456",
            "format": "mossformergan_scp",
            "train": _side(
                "configs/data/lists/mossformergan/tr_demand_28_spks_16k.scp",
                "tr.scp",
                mfg_train,
            ),
            "test": _side(
                "configs/data/lists/mossformergan/cv_demand_testset_16k.scp",
                "cv.scp",
                mfg_test,
            ),
        },
    }
    import yaml

    (config / "lists").mkdir(parents=True, exist_ok=True)
    (config / "lists" / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (config / "datasets").mkdir(parents=True)
    for name, profile, claim, train_role, valid_role, note in (
        ("mp_senet.vb28.ondemand.official", "official_faithful", "official_reproduction", "official_train", "official_test", "uses test"),
        ("mp_senet.vb28.ondemand.safe", "leakage_safe", "leakage_safe_comparison", "safe_train", "safe_valid", "holds out speakers"),
        (
            "mossformergan.vb28.ondemand.official",
            "official_faithful",
            "official_reproduction",
            "official_train",
            "official_test",
            "uses test",
        ),
        (
            "mossformergan.vb28.ondemand.safe",
            "leakage_safe",
            "leakage_safe_comparison",
            "safe_train",
            "safe_valid",
            "holds out speakers",
        ),
    ):
        expect = {"train": len(train), "valid": 1, "test": 1} if profile == "official_faithful" else {"train": len(train) - 2, "valid": 2, "test": 1}
        model = "mp_senet" if name.startswith("mp_senet") else "mossformergan"
        protocol = (
            "mp_senet.fixed_pair.ondemand" if model == "mp_senet" else "mossformergan_se_16k.fixed_pair.ondemand"
        )
        document = {
            "name": name,
            "version": 1,
            "protocol": protocol,
            "profile": profile,
            "material_origin": "derived_from_official",
            "claim_label": claim,
            "list_model": model,
            "selection_note": note,
            "bindings": {"train": train_role, "valid": valid_role, "test": "official_test"},
            "expect": expect,
            "deviations": [
                {
                    "id": "vb28-16k-converter-unknown",
                    "class": "official_effective",
                    "text": "converter unknown",
                    "ref": "UN-3",
                }
            ],
        }
        (config / "datasets" / f"{name}.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")


def _side(file_name: str, upstream: str, text: str) -> dict:
    return {"path": upstream, "file": file_name, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
