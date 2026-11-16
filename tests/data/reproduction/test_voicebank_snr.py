"""VoiceBank SNR-shift recipe. Audio and databases here are synthetic."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.experiments.data.adapters import open_pair_adapter
from src.experiments.data.db.graph import index_rows
from src.experiments.data.db.registry import initialize_database, register_staging
from src.experiments.data.db.snapshot import export_snapshot
from src.experiments.data.db.staging import write_graph
from src.experiments.data.db.validate import leak_report, validate_graph
from src.experiments.data.errors import ReproductionError
from src.experiments.data.fixed_pair import SnapshotSource
from src.experiments.data.ids import validate_realization
from src.experiments.data.mix.voicebank_snr import (
    component_snr_db,
    decimal_string,
    noise_scale,
    p56_snr_db,
    pcm16_round_nearest_even,
    read_span,
    render_case,
    render_components,
)
from src.data.recipe import sha256_file
from src.experiments.data.build import voicebank_snr as voicebank_builder
from src.experiments.data.build.voicebank_snr import (
    EXPECTED_COUNTS,
    RECIPE_PATH,
    assert_population,
    build_voicebank_snr_graph,
    dataset_name,
    dump_recipe_line,
    load_recipes,
    realization_document,
    records_from_recipes,
    stub_headers,
)
from src.experiments.data.paths import ROOT

FORBIDDEN = (
    "voicebank-snr-shift-v1",
    "voicebank-snr-shift-v2",
    "reproduction-snr-shift",
    "snapshots-snr-shift",
    "voicebank.snr_shift.mp_senet",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tone(path: Path, frequency: float, amplitude: float, frames: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(frames, dtype=np.float64)
    pcm = np.rint(amplitude * np.sin(2.0 * np.pi * frequency * time / 48000.0)).astype(np.int16)
    sf.write(str(path), pcm, 48000, subtype="PCM_16")
    info = sf.info(str(path))
    return {"pcm": pcm, "frames": int(info.frames), "size": path.stat().st_size, "sha256": _sha(path)}


class VoicebankSnrTests(unittest.TestCase):
    def test_quantization_and_shift_math(self) -> None:
        samples = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 1.1 * 32768, -1.1 * 32768], dtype=np.float64) / 32768.0
        quantized = pcm16_round_nearest_even(samples)
        self.assertEqual(quantized.tolist(), [0, 2, 2, 0, -2, 32767, -32768])
        self.assertEqual(noise_scale(0), 1.0)
        self.assertAlmostEqual(noise_scale(-15), 10 ** (15 / 20), places=12)
        self.assertAlmostEqual(noise_scale(-20), 10.0, places=12)
        self.assertAlmostEqual(noise_scale(-6), 10 ** (6 / 20), places=12)
        self.assertEqual(decimal_string(1.0), "1")

    def test_p56_shift_is_exact_and_distinct_from_power_ratio(self) -> None:
        time = np.arange(48000, dtype=np.float64)
        clean = 0.2 * np.sin(2.0 * np.pi * 440.0 * time / 48000.0)
        noise = 0.05 * np.sin(2.0 * np.pi * 1000.0 * time / 48000.0)
        base = p56_snr_db(clean, noise)
        shifted = p56_snr_db(clean, noise * noise_scale(-15))
        lower = p56_snr_db(clean, noise * noise_scale(-20))
        self.assertTrue(np.isfinite(base) and np.isfinite(shifted) and np.isfinite(lower))
        self.assertAlmostEqual(shifted - base, -15.0, places=6)
        self.assertAlmostEqual(lower - base, -20.0, places=6)
        power = component_snr_db(clean, noise)
        self.assertNotAlmostEqual(power, base, places=2)

    def test_g_scales_clean_and_not_the_noise_shift(self) -> None:
        speech = np.linspace(-0.2, 0.4, 64, dtype=np.float64)
        noise = np.linspace(0.05, -0.05, 64, dtype=np.float64)
        gain = decimal_string(1.0 / 1.32)
        beta = decimal_string(0.25)
        after = decimal_string(float(gain) * float(beta))
        document = _document(
            utterance="p277_019",
            speaker="p277",
            split="train",
            speech_asset="corpora/VCTK/wav48/p277/p277_019.wav",
            noise_asset="corpora/DEMAND48/STRAFFIC/ch01.wav",
            noise_kind="demand",
            noise_name="traffic",
            scale=decimal_string(0.5),
            gain=gain,
            beta=beta,
            after=after,
            steps=2,
            length=64,
            snr=15,
        )
        validate_realization(document)
        clean, noisy = render_components(document, speech, noise, snr_shift_db=0)
        shifted_clean, shifted = render_components(document, speech, noise, snr_shift_db=-15)
        np.testing.assert_allclose(clean, speech * 0.5 * float(gain))
        np.testing.assert_array_equal(clean, shifted_clean)
        ratio = np.max(np.abs(shifted - shifted_clean)) / np.max(np.abs(noisy - clean))
        self.assertAlmostEqual(ratio, noise_scale(-15), places=12)

    def test_shift_names_do_not_collide(self) -> None:
        self.assertEqual(dataset_name(0), "voicebank.snr_shift.no")
        self.assertEqual(dataset_name(-0.0), "voicebank.snr_shift.no")
        self.assertEqual(dataset_name(-15), "voicebank.snr_shift.minus_15db")
        self.assertEqual(dataset_name(-15.0), "voicebank.snr_shift.minus_15db")
        self.assertEqual(dataset_name(-20), "voicebank.snr_shift.minus_20db")
        self.assertEqual(dataset_name(6), "voicebank.snr_shift.plus_6db")
        self.assertEqual(dataset_name(-6.5), "voicebank.snr_shift.minus_6p5db")
        self.assertEqual(dataset_name(6.5), "voicebank.snr_shift.plus_6p5db")
        names = [dataset_name(shift) for shift in (0, -15, 15, -6.5, 6.5, -20)]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(dataset_name(0), dataset_name(-0.0))

    def test_recipe_rejects_bad_rows(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="voicebank-recipe-"))
        try:
            path = root / "recipes.jsonl"
            path.write_text(dump_recipe_line(_recipe()) + "\n", encoding="utf-8")
            loaded = load_recipes(path, check_population=False)
            self.assertEqual(loaded[0]["split"], "valid")
            self.assertEqual(loaded[0]["clean_gain_g"], "1")
            bad = _recipe()
            bad["clean"]["gain_g"] = "1.0"
            bad["noise"]["gain_after_peak"] = "0.5"
            path.write_text(dump_recipe_line(bad) + "\n", encoding="utf-8")
            with self.assertRaises(ReproductionError):
                load_recipes(path, check_population=False)
            shifted = _recipe()
            shifted["snr_shift_db"] = -15
            path.write_text(dump_recipe_line(shifted) + "\n", encoding="utf-8")
            with self.assertRaises(ReproductionError):
                load_recipes(path, check_population=False)
            train_holdout = _recipe()
            train_holdout["split"] = "train"
            path.write_text(dump_recipe_line(train_holdout) + "\n", encoding="utf-8")
            with self.assertRaises(ReproductionError):
                load_recipes(path, check_population=False)
            test_speaker = _recipe(utterance="p232_001", speaker="p232", split="train")
            path.write_text(dump_recipe_line(test_speaker) + "\n", encoding="utf-8")
            with self.assertRaises(ReproductionError):
                load_recipes(path, check_population=False)
        finally:
            shutil.rmtree(root)

    def test_build_id_follows_jsonl_bytes(self) -> None:
        self.assertFalse(hasattr(voicebank_builder, "MEASURED_SOURCE_SHA256"))
        rows = load_recipes(RECIPE_PATH)
        recipe_sha = sha256_file(RECIPE_PATH)
        records = records_from_recipes(rows, stub_headers(rows))
        graph = build_voicebank_snr_graph(records, recipe_sha256=recipe_sha)
        again = build_voicebank_snr_graph(records, recipe_sha256=recipe_sha)
        params = graph.builds[0]["params_json"]
        self.assertEqual(params["recipe_sha256"], recipe_sha)
        self.assertNotIn("source_recipes_sha256", params)
        self.assertEqual(graph.builds[0]["build_id"], again.builds[0]["build_id"])
        self.assertEqual(len(graph.builds[0]["build_id"]), 64)
        protocols = {case["protocol_id"] for case in graph.cases}
        self.assertEqual(protocols, {"08fe93a6f16ad272a5309b4dd663422c185df1cf3f7b3732b8cf18a40ff629f0"})
        case_ids = [case["case_id"] for case in graph.cases]
        self.assertEqual(len(set(case_ids)), 12396)
        for dataset in graph.datasets:
            self.assertFalse(dataset["selection_uses_test"])
            if dataset["declared_status"] != "ready":
                continue
            bound = [item["case_set_id"] for item in dataset["bindings"] if item["case_set_id"]]
            self.assertEqual(bound, [row["case_set_id"] for row in graph.case_sets])
        root = Path(tempfile.mkdtemp(prefix="voicebank-recipe-hash-"))
        try:
            copied = root / "nested" / "other-name.jsonl"
            copied.parent.mkdir(parents=True)
            copied.write_bytes(RECIPE_PATH.read_bytes())
            copied_rows = load_recipes(copied)
            copied_graph = build_voicebank_snr_graph(
                records_from_recipes(copied_rows, stub_headers(copied_rows)),
                recipe_sha256=sha256_file(copied),
            )
            self.assertEqual(copied_graph.builds[0]["build_id"], graph.builds[0]["build_id"])
            public = [json.loads(line) for line in copied.read_text(encoding="utf-8").splitlines()]
            public[0]["official_snr_db"] = 17 if public[0]["official_snr_db"] != 17 else 16
            changed = root / "changed.jsonl"
            changed.write_text("".join(dump_recipe_line(row) + "\n" for row in public), encoding="utf-8")
            self.assertNotEqual(sha256_file(changed), recipe_sha)
            changed_loaded = load_recipes(changed)
            changed_graph = build_voicebank_snr_graph(
                records_from_recipes(changed_loaded, stub_headers(changed_loaded)),
                recipe_sha256=sha256_file(changed),
            )
            self.assertNotEqual(changed_graph.builds[0]["build_id"], graph.builds[0]["build_id"])
            self.assertNotIn("source_recipes_sha256", changed_graph.builds[0]["params_json"])
        finally:
            shutil.rmtree(root)

    def test_population_counts(self) -> None:
        rows = []
        index = 0
        for split, count in EXPECTED_COUNTS.items():
            for _ in range(count):
                rows.append({"utterance_id": f"u{index}", "split": split})
                index += 1
        assert_population(rows)
        rows[0]["utterance_id"] = rows[1]["utterance_id"]
        with self.assertRaises(ReproductionError):
            assert_population(rows)
        with self.assertRaises(ReproductionError):
            assert_population(rows[:3])

    def test_dataset_round_trip_and_leak(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="voicebank-snr-"))
        try:
            corpora = root / "corpora"
            disjoint = _graph_from_wavs(corpora, shared=False)
            validate_graph(disjoint)
            self.assertEqual(leak_status(disjoint, "voicebank.snr_shift.no"), "PASS")
            shared = _graph_from_wavs(corpora / "shared", shared=True)
            validate_graph(shared)
            report = leak_of(shared, "voicebank.snr_shift.no")
            self.assertEqual(report["status"], "WARN")
            noise = next(item for item in report["components"] if item["check"] == "noise_interval_independence")
            self.assertFalse(noise["validation_noise_independent"])
            self.assertGreater(noise["pairs"]["train_valid"]["overlap_samples"], 0)
            self.assertEqual(noise["pairs"]["train_test"]["overlap_samples"], 0)
            crossed = _graph_from_wavs(corpora / "crossed", shared="test")
            with self.assertRaises(ReproductionError):
                validate_graph(crossed)
            staging = root / "staging"
            database = root / "reproduction.duckdb"
            snapshots = root / "snapshots"
            write_graph(staging, disjoint)
            initialize_database(database)
            register_staging(staging, database)
            no_hash = export_snapshot(database, "voicebank.snr_shift.no@1", snapshots, corpora=corpora)
            low_hash = export_snapshot(database, "voicebank.snr_shift.minus_15db@1", snapshots, corpora=corpora)
            lower_hash = export_snapshot(database, "voicebank.snr_shift.minus_20db@1", snapshots, corpora=corpora)
            self.assertEqual(len({no_hash, low_hash, lower_hash}), 3)
            no_source = SnapshotSource("voicebank.snr_shift.no@1", snapshots, "train", corpora=corpora, derived=root / "derived", generated=root / "generated")
            low_source = SnapshotSource("voicebank.snr_shift.minus_15db@1", snapshots, "train", corpora=corpora, derived=root / "derived", generated=root / "generated")
            no_source.open()
            low_source.open()
            case_ids = no_source.case_ids("train.cases")
            self.assertEqual(case_ids, low_source.case_ids("train.cases"))
            self.assertEqual(len(no_source.case_ids("valid.cases")), 1)
            self.assertEqual(len(no_source.case_ids("test.cases")), 1)
            case_id = case_ids[0]
            document = no_source.store.docs[case_id]
            self.assertNotIn("VoiceBank-DEMAND", str(document))
            clean, noisy = render_case(no_source.store, case_id, snr_shift_db=0)
            again_clean, again_noisy = render_case(no_source.store, case_id, snr_shift_db=0)
            low_clean, low_noisy = render_case(low_source.store, case_id, snr_shift_db=-15)
            np.testing.assert_array_equal(clean, again_clean)
            np.testing.assert_array_equal(noisy, again_noisy)
            np.testing.assert_array_equal(clean, low_clean)
            base = noisy - clean
            shifted = low_noisy - low_clean
            ratio = np.sqrt(np.mean(shifted ** 2) / np.mean(base ** 2))
            self.assertAlmostEqual(float(ratio), noise_scale(-15), places=9)
            lower_source = SnapshotSource("voicebank.snr_shift.minus_20db@1", snapshots, "train", corpora=corpora, derived=root / "derived", generated=root / "generated")
            lower_source.open()
            lower_clean, lower_noisy = render_case(lower_source.store, case_id, snr_shift_db=-20)
            np.testing.assert_array_equal(clean, lower_clean)
            lower_ratio = np.sqrt(np.mean((lower_noisy - lower_clean) ** 2) / np.mean(base ** 2))
            self.assertAlmostEqual(float(lower_ratio), noise_scale(-20), places=9)
            _, arbitrary = render_case(no_source.store, case_id, snr_shift_db=-6)
            arbitrary_ratio = np.sqrt(np.mean((arbitrary - clean) ** 2) / np.mean(base ** 2))
            self.assertAlmostEqual(float(arbitrary_ratio), noise_scale(-6), places=9)
            adapter = open_pair_adapter(no_source, verify_on_read=False, sample_rate=48000, resample=None)
            adapter.prepare()
            adapted_noisy, adapted_clean = adapter.render_full(case_id)
            self.assertEqual(adapted_noisy.dtype, np.float32)
            self.assertEqual(adapted_clean.dtype, np.float32)
            self.assertTrue(np.isfinite(adapted_noisy).all())
            self.assertLess(float(np.max(np.abs(adapted_noisy))), 2.0)
        finally:
            shutil.rmtree(root)

    def test_published_pairs_are_not_a_generation_input(self) -> None:
        with self.assertRaises(ReproductionError):
            read_span(Path("/tmp/VoiceBank-DEMAND/clean.wav"), channel=0, start=0, end=1)

    def test_old_snr_shift_names_are_gone_from_code(self) -> None:
        roots = [ROOT / "src", ROOT / "configs"]
        hits = []
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".md", ".json"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for needle in FORBIDDEN:
                    if needle in text:
                        hits.append(f"{path}:{needle}")
        self.assertEqual(hits, [])

    def test_register_is_idempotent_and_rejects_a_reused_name(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="voicebank-register-"))
        try:
            corpora = root / "corpora"
            graph = _graph_from_wavs(corpora, shared=False)
            staging = root / "staging"
            database = root / "db.duckdb"
            write_graph(staging, graph)
            initialize_database(database)
            register_staging(staging, database)
            again = register_staging(staging, database)
            self.assertTrue(all(count == 0 for count in again.values()))
            renamed = _graph_from_wavs(corpora, shared=False)
            collided = next(row for row in renamed.datasets if row["name"] == "voicebank.snr_shift.minus_15db")
            collided["name"] = "voicebank.snr_shift.no"
            other = root / "other-staging"
            write_graph(other, renamed)
            with self.assertRaises(ReproductionError):
                register_staging(other, database)
            arbitrary = _graph_from_wavs(root / "arbitrary", shared=False, shifts=(-6,))
            names = {row["name"] for row in arbitrary.datasets}
            self.assertEqual(names, {"voicebank.snr_shift.published", "voicebank.snr_shift.minus_6db"})
            self.assertIn("voicebank.snr_shift.no", {row["name"] for row in arbitrary.protocols})
            validate_graph(arbitrary)
            write_graph(root / "arbitrary-staging", arbitrary)
            initialize_database(root / "arbitrary.duckdb")
            register_staging(root / "arbitrary-staging", root / "arbitrary.duckdb")
            export_snapshot(
                root / "arbitrary.duckdb",
                "voicebank.snr_shift.minus_6db@1",
                root / "arbitrary-snapshots",
                corpora=root / "arbitrary",
            )
        finally:
            shutil.rmtree(root)


def leak_status(graph, name: str) -> str:
    return leak_of(graph, name)["status"]


def leak_of(graph, name: str) -> dict:
    dataset = next(row for row in graph.datasets if row["name"] == name)
    return leak_report(
        dataset,
        index_rows(graph.case_sets, "case_set_id", "case_sets"),
        index_rows(graph.materials, "material_set_id", "materials"),
        index_rows(graph.cases, "case_id", "cases"),
        index_rows(graph.assets, "asset_id", "assets"),
        resource_db=None,
        protocols=index_rows(graph.protocols, "protocol_id", "protocols"),
    )


def _graph_from_wavs(corpora: Path, *, shared: bool | str, shifts: tuple = (0, -15, -20)):
    corpora.mkdir(parents=True, exist_ok=True)
    specs = [
        ("train", "p226", "p226_001", 15, "babble", "Babble/train.wav", 220.0),
        ("valid", "p286", "p286_001", 10, "ssn", "SSN/valid.wav", 330.0),
        ("test", "p232", "p232_001", 2.5, "demand", "DEMAND48/bus.wav", 440.0),
    ]
    if shared is True:
        specs[1] = ("valid", "p286", "p286_001", 10, "babble", "Babble/train.wav", 330.0)
    elif shared == "test":
        specs[2] = ("test", "p232", "p232_001", 2.5, "babble", "Babble/train.wav", 440.0)
    records = []
    noise_cache: dict[str, dict] = {}
    for split, speaker, utterance, snr, kind, noise_rel, frequency in specs:
        speech_rel = f"VCTK/wav48/{speaker}/{utterance}.wav"
        speech_path = corpora / speech_rel
        speech = _tone(speech_path, frequency, 8000.0, 48000)
        if noise_rel not in noise_cache:
            noise_cache[noise_rel] = _tone(corpora / noise_rel, frequency + 500.0, 1000.0, 96000)
        noise = noise_cache[noise_rel]
        start = 0 if split == "train" else 24000
        audio, _rate = sf.read(str(speech_path), dtype="float64", always_2d=False)
        gain = "1"
        beta = decimal_string(0.2)
        steps = 0
        if split == "valid":
            gain = decimal_string(1.0 / 1.32)
            beta = decimal_string(0.25)
            steps = 2
        after = decimal_string(float(gain) * float(beta))
        records.append(
            {
                "utterance_id": utterance,
                "speaker_id": speaker,
                "split": split,
                "official_snr_db": snr,
                "clean_scale": _scale(audio),
                "clean_gain_g": gain,
                "noise_gain_before_peak_beta": beta,
                "noise_gain_after_peak": after,
                "peak_steps": steps,
                "silence": {"source_frames": speech["frames"], "leading_removed": 0, "trailing_removed": 0},
                "speech": _asset(speech_rel, speech, channel=0, start=0, end=speech["frames"], speaker=speaker, key=utterance),
                "noise": _asset(noise_rel, noise, channel=0, start=start, end=start + speech["frames"], speaker=None, key=Path(noise_rel).stem, kind=kind, name=kind),
            }
        )
    recipe_sha = hashlib.sha256(b"synthetic-voicebank-snr-test").hexdigest()
    return build_voicebank_snr_graph(records, shifts, recipe_sha256=recipe_sha)


def _recipe(utterance: str = "p286_001", speaker: str = "p286", split: str = "valid") -> dict:
    return {
        "recipe_version": 1,
        "utterance_id": utterance,
        "speaker_id": speaker,
        "split": split,
        "position": 0,
        "sample_rate": 48000,
        "official_snr_db": 15,
        "peak_steps": 0,
        "target_peak_pcm": 16384,
        "output_length": 10,
        "clean": {
            "corpus": "VCTK",
            "rel_path": f"VCTK/wav48/{speaker}/{utterance}.wav",
            "sha256": "a" * 64,
            "channel": 0,
            "start": 0,
            "end": 10,
            "scale": "1",
            "gain_g": "1",
            "silence": {
                "mode": "contiguous_slice",
                "source_frames": 10,
                "leading_removed": 0,
                "trailing_removed": 0,
            },
        },
        "noise": {
            "corpus": "DEMAND48",
            "rel_path": "DEMAND48/PCAFETER/ch01.wav",
            "sha256": "b" * 64,
            "channel": 0,
            "start": 0,
            "end": 10,
            "kind": "demand",
            "name": "cafeteria",
            "gain_before_peak_beta": "0.5",
            "gain_after_peak": "0.5",
        },
    }


def _scale(samples: np.ndarray) -> str:
    from src.experiments.data.mix.voicebank_snr import clean_scale_for

    return clean_scale_for(samples)


def _asset(rel: str, info: dict, *, channel: int, start: int, end: int, speaker: str | None, key: str, kind: str | None = None, name: str | None = None) -> dict:
    corpus = rel.split("/", 1)[0]
    row = {
        "asset_id": f"corpora/{rel}",
        "rel_path": rel,
        "sha256": info["sha256"],
        "frames": info["frames"],
        "channels": 1,
        "size_bytes": info["size"],
        "sample_format": "PCM_16",
        "corpus": corpus,
        "subset": rel.split("/")[1],
        "channel": channel,
        "start": start,
        "end": end,
        "speaker_id": speaker,
        "utterance_key": key,
    }
    if kind is not None:
        row["kind"] = kind
        row["name"] = name
    return row


def _document(**kwargs) -> dict:
    record = {
        "utterance_id": kwargs["utterance"],
        "speaker_id": kwargs["speaker"],
        "split": kwargs["split"],
        "official_snr_db": kwargs["snr"],
        "clean_scale": kwargs["scale"],
        "clean_gain_g": kwargs["gain"],
        "noise_gain_before_peak_beta": kwargs["beta"],
        "noise_gain_after_peak": kwargs["after"],
        "peak_steps": kwargs["steps"],
        "silence": {
            "source_frames": kwargs["length"],
            "leading_removed": 0,
            "trailing_removed": 0,
        },
        "speech": {
            "asset_id": kwargs["speech_asset"],
            "sha256": "a" * 64,
            "channel": 0,
            "start": 0,
            "end": kwargs["length"],
        },
        "noise": {
            "asset_id": kwargs["noise_asset"],
            "sha256": "b" * 64,
            "channel": 0,
            "start": 0,
            "end": kwargs["length"],
            "kind": kwargs["noise_kind"],
            "name": kwargs["noise_name"],
        },
    }
    return realization_document(record, build_id="d" * 64, gen_index=0)


if __name__ == "__main__":
    unittest.main()
