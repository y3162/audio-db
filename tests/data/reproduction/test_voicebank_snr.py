"""VoiceBank SNR-shift recipe. Audio and databases here are synthetic."""

from __future__ import annotations

import hashlib
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
from src.experiments.data.paths import ROOT
from src.experiments.data.prepare.voicebank_snr import (
    EXPECTED_COUNTS,
    assert_population,
    build_voicebank_snr_graph,
    normalize_recipe_row,
    realization_document,
)

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
        self.assertAlmostEqual(noise_scale(-6), 10 ** (6 / 20), places=12)
        self.assertEqual(decimal_string(1.0), "1")

    def test_p56_shift_is_exact_and_distinct_from_power_ratio(self) -> None:
        time = np.arange(48000, dtype=np.float64)
        clean = 0.2 * np.sin(2.0 * np.pi * 440.0 * time / 48000.0)
        noise = 0.05 * np.sin(2.0 * np.pi * 1000.0 * time / 48000.0)
        base = p56_snr_db(clean, noise)
        shifted = p56_snr_db(clean, noise * noise_scale(-15))
        self.assertTrue(np.isfinite(base) and np.isfinite(shifted))
        self.assertAlmostEqual(shifted - base, -15.0, places=6)
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

    def test_recipe_row_assigns_validation_and_canonical_gains(self) -> None:
        row = normalize_recipe_row(
            {
                "utterance_id": "p286_001",
                "split": "train",
                "noise_type": "cafeteria",
                "source_vctk": "data/corpora/VCTK/wav48/p286/p286_001.wav",
                "source_vctk_sha256": "a" * 64,
                "noise_path": "data/corpora/DEMAND48/PCAFETER/ch01.wav",
                "noise_sha256": "b" * 64,
                "noise_channel_1based": "1",
                "noise_start_48k": "10",
                "noise_end_exclusive_48k": "20",
                "vctk_alignment_offset": "5",
                "official_log_snr_db": 15.0,
                "clean_gain_g": "1.0",
                "noise_gain_before_peak_beta": "0.5",
                "noise_gain_after_peak": "0.5",
                "peak_steps": "0",
            }
        )
        self.assertEqual(row["split"], "valid")
        self.assertEqual(row["noise_kind"], "demand")
        self.assertEqual(row["clean_gain_g"], "1")
        self.assertEqual(row["noise_channel"], 0)
        with self.assertRaises(ReproductionError):
            normalize_recipe_row(
                {
                    "utterance_id": "p232_001",
                    "split": "train",
                    "noise_type": "babble",
                    "source_vctk": "data/corpora/VCTK/wav48/p232/p232_001.wav",
                    "source_vctk_sha256": "a" * 64,
                    "noise_path": "data/corpora/Babble/babble.wav",
                    "noise_sha256": "b" * 64,
                    "noise_channel_1based": 1,
                    "noise_start_48k": 0,
                    "noise_end_exclusive_48k": 8,
                    "vctk_alignment_offset": 0,
                    "official_log_snr_db": 2.5,
                    "clean_gain_g": 1.0,
                    "noise_gain_before_peak_beta": 0.2,
                    "noise_gain_after_peak": 0.2,
                    "peak_steps": 0,
                }
            )

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
            self.assertNotEqual(no_hash, low_hash)
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


def _graph_from_wavs(corpora: Path, *, shared: bool | str):
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
    return build_voicebank_snr_graph(
        records,
        verification_report={"status": "PASS", "bit_exact": False, "utterances": 3, "synthetic": True},
        source_recipes_sha256="c" * 64,
    )


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
