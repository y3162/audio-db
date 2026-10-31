"""CHiME3 annotations and the MP-SENet VoiceBank file list."""

import json
import os
import unittest
import wave
from pathlib import Path

from src.data.corpora.chime3_annotations import load_chime3_samples
from src.data.corpora.mpsenet_filelist import load_mpsenet_filelist
from src.data.init_eval import initialize
from src.evaluation.datasets.evaldb import EvaluationDB
from src.evaluation.errors import DatasetError

CHIME_ROOT = Path("data/corpora/CHiME3")


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


class Chime3AnnotationTests(unittest.TestCase):
    def test_channels_transcripts_and_idempotent_registration(self):
        with self._temporary() as root:
            self._chime_fixture(root)
            real = load_chime3_samples(root, split="et05", condition="real", channel="5")
            close = load_chime3_samples(root, split="et05", condition="real", channel="CH0")
            simu = load_chime3_samples(root, split="et05", condition="simu", channel="CH5")
            original = load_chime3_samples(root, split="tr05", condition="org")
            self.assertEqual(real[0].transcript, "At N. E. C., the need will keep rising.")
            self.assertEqual(real[0].metadata["reference_role"], "tablet_microphone")
            self.assertEqual(real[0].metadata["mic_channel"], "CH5")
            self.assertIsNone(real[0].clean_path)
            self.assertTrue(real[0].noisy_path.endswith("F05_440C0202_BUS.CH5.wav"))
            self.assertEqual(close[0].metadata["reference_role"], "close_talk_in_noise")
            self.assertFalse(close[0].metadata["clean_reference"])
            self.assertIsNone(close[0].clean_path)
            self.assertEqual(simu[0].metadata["reference_role"], "simulated_noisy")
            self.assertTrue(original[0].clean_path.endswith("011_011C0201_ORG.wav"))
            self.assertIsNone(original[0].noisy_path)
            self.assertNotEqual(real[0].utterance_id, close[0].utterance_id)
            with self.assertRaises(ValueError):
                load_chime3_samples(root, split="et05", condition="real", channel="CH7")
            with self.assertRaises(ValueError):
                load_chime3_samples(root, split="tr05", condition="org", channel="CH0")
            database = root / "annotations.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root)
            before = db.corpora_fingerprint()
            first = db.register_samples(real, "chime3_et05_real_ch5", description="tablet")
            second = db.register_samples(real, "chime3_et05_real_ch5", description="tablet")
            self.assertEqual(first.inserted, 1)
            self.assertEqual(second.unchanged, 1)
            self.assertEqual(second.set_status, "unchanged")
            loaded = db.load_dataset("chime3_et05_real_ch5")
            self.assertEqual(loaded[0].transcript, real[0].transcript)
            self.assertEqual(loaded[0].metadata["mic_channel"], "CH5")
            self.assertEqual(loaded[0].metadata["eval_kind"], "noisy")
            changed = load_chime3_samples(root, split="et05", condition="real", channel="CH5")
            changed[0].transcript = "different"
            with self.assertRaises(DatasetError):
                db.register_samples(changed, "chime3_et05_real_ch5")
            self.assertEqual(db.corpora_fingerprint(), before)

    def test_missing_audio_does_not_register_a_partial_set(self):
        with self._temporary() as root:
            self._chime_fixture(root, with_audio=False)
            with self.assertRaises(ValueError):
                load_chime3_samples(root, split="et05", condition="real", channel="CH5")
            database = root / "annotations.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root)
            db.ensure_schema()
            self.assertEqual(db.stats()["cases"], 0)

    def test_initialize_skips_absent_audio_and_registers_a_fixture(self):
        with self._temporary() as root:
            self._chime_fixture(root)
            database = root / "resource.duckdb"
            config = root / "init.yaml"
            config.write_text(
                "\n".join(
                    [
                        f"database: {database.as_posix()}",
                        f"corpora_root: {root.as_posix()}",
                        "annotation_sets:",
                        "  - importer: chime3",
                        "    set: chime3_et05_real_ch5",
                        "    required: false",
                        "    split: et05",
                        "    condition: real",
                        "    channel: CH5",
                        "    root: " + root.as_posix(),
                        "  - importer: missing",
                        "    set: bad",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(DatasetError):
                initialize(config)
            original = Path.cwd()
            config.write_text(
                "\n".join(
                    [
                        f"database: {database.as_posix()}",
                        f"corpora_root: {(root / 'missing-corpora').as_posix()}",
                        "annotation_sets:",
                        "  - importer: chime3",
                        "    set: chime3_et05_real_ch5",
                        "    required: false",
                        "    split: et05",
                        "    condition: real",
                        "    channel: CH5",
                        "    root: " + (root / "absent").as_posix(),
                        "  - importer: chime3",
                        "    set: present",
                        "    required: false",
                        "    split: et05",
                        "    condition: real",
                        "    channel: CH5",
                        "    root: " + root.as_posix(),
                    ]
                ),
                encoding="utf-8",
            )
            try:
                os.chdir(root)
                text = initialize(config)
                again = initialize(config)
            finally:
                os.chdir(original)
            self.assertIn("skipped", text)
            self.assertIn("present", text)
            db = EvaluationDB(database, project_root=root, corpora_root=root)
            self.assertEqual(len(db.load_dataset("present")), 1)
            self.assertIn("unchanged", again)
            self.assertEqual(len(db.load_dataset("present")), 1)

    @unittest.skipUnless(
        (CHIME_ROOT / "data/annotations/et05_real.json").is_file(),
        "CHiME3 annotations are not available",
    )
    def test_official_files_match_the_annotation(self):
        real = load_chime3_samples(CHIME_ROOT, split="et05", condition="real", channel="CH5")
        close = load_chime3_samples(CHIME_ROOT, split="et05", condition="real", channel="CH0")
        simu = load_chime3_samples(CHIME_ROOT, split="et05", condition="simu", channel="CH5")
        self.assertGreater(len(real), 0)
        self.assertEqual(len(real), len(close))
        self.assertEqual(real[0].metadata["reference_role"], "tablet_microphone")
        self.assertEqual(close[0].metadata["reference_role"], "close_talk_in_noise")
        self.assertEqual(simu[0].metadata["reference_role"], "simulated_noisy")
        self.assertTrue(Path(real[0].noisy_path).is_file())
        self.assertTrue(real[0].transcript)

    def _chime_fixture(self, root: Path, with_audio: bool = True) -> None:
        real = {
            "speaker": "F05",
            "environment": "BUS",
            "wsj_name": "440C0202",
            "prompt": "At N. E. C., the need will keep rising.",
            "dot": "At N\\. E\\. C\\. the need will keep rising",
            "wavfile": "F05_150422_030_BUS",
            "start": 1.0,
            "end": 2.5,
        }
        (root / "data/annotations").mkdir(parents=True)
        (root / "data/annotations/et05_real.json").write_text(json.dumps([real]), encoding="utf-8")
        simu = dict(real)
        simu["wavfile"] = "F05_150520_040_BTH"
        (root / "data/annotations/et05_simu.json").write_text(json.dumps([simu]), encoding="utf-8")
        original = {
            "speaker": "011",
            "environment": "ORG",
            "wsj_name": "011C0201",
            "prompt": "The sale of the hotels is part of the strategy.",
        }
        (root / "data/annotations/tr05_org.json").write_text(json.dumps([original]), encoding="utf-8")
        if not with_audio:
            return
        isolated = root / "data/audio/16kHz/isolated"
        _wav(isolated / "et05_bus_real/F05_440C0202_BUS.CH5.wav")
        _wav(isolated / "et05_bus_real/F05_440C0202_BUS.CH0.wav")
        _wav(isolated / "et05_bus_simu/F05_440C0202_BUS.CH5.wav")
        _wav(isolated / "tr05_org/011_011C0201_ORG.wav")

    def _temporary(self):
        import tempfile

        class _Dir:
            def __enter__(self):
                self._raw = tempfile.TemporaryDirectory()
                return Path(self._raw.__enter__())

            def __exit__(self, exc_type, exc, traceback):
                return self._raw.__exit__(exc_type, exc, traceback)

        return _Dir()


class VoiceBankFileListTests(unittest.TestCase):
    def test_pairs_ids_and_refuses_partial_audio(self):
        with self._temporary() as directory:
            root = Path(directory)
            listing = root / "test.txt"
            listing.write_text(
                "p232_001|VoiceBank+DEMAND/wav_clean/p232_001.wav\np232_002|VoiceBank+DEMAND/wav_clean/p232_002.wav\n",
                encoding="utf-8",
            )
            _wav(root / "clean/p232_001.wav")
            _wav(root / "noisy/p232_001.wav")
            with self.assertRaises(ValueError):
                load_mpsenet_filelist(listing, clean_dir=root / "clean", noisy_dir=root / "noisy")
            _wav(root / "clean/p232_002.wav")
            _wav(root / "noisy/p232_002.wav")
            samples = load_mpsenet_filelist(listing, clean_dir=root / "clean", noisy_dir=root / "noisy")
            self.assertEqual(samples[0].speaker_id, "p232")
            self.assertIsNone(samples[0].transcript)
            self.assertIsNone(samples[0].snr_db)
            self.assertTrue(samples[0].clean_path.endswith("p232_001.wav"))
            self.assertEqual(samples[0].metadata["listed_path"], "VoiceBank+DEMAND/wav_clean/p232_001.wav")
            database = root / "annotations.duckdb"
            db = EvaluationDB(database, project_root=root, corpora_root=root)
            report = db.register_samples(samples, "voicebank_demand_test")
            self.assertEqual(report.inserted, 2)
            loaded = db.load_dataset("voicebank_demand_test")
            self.assertEqual(loaded[0].metadata["eval_kind"], "paired")
            self.assertEqual(len(loaded), 2)

    def _temporary(self):
        import tempfile

        class _Dir:
            def __enter__(self):
                self._raw = tempfile.TemporaryDirectory()
                return Path(self._raw.__enter__())

            def __exit__(self, exc_type, exc, traceback):
                return self._raw.__exit__(exc_type, exc, traceback)

        return _Dir()
