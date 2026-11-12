import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.data.corpora.demand import iter_rows
from src.data.corpora.generic import iter_generic


def _pcm_wav(path: Path, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x01\x00" * 8)


class NoiseCorpusTests(unittest.TestCase):
    def test_demand_and_demand48_keep_header_rates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _pcm_wav(root / "DEMAND" / "DKITCHEN" / "ch01.wav", 16000)
            _pcm_wav(root / "DEMAND48" / "SCAFE" / "ch01.wav", 48000)
            _pcm_wav(root / "Babble" / "babble.wav", 48000)
            _pcm_wav(root / "SSN" / "ssn.wav", 48000)
            with (
                patch("src.data.corpora.demand.CORPORA_ROOT", root),
                patch("src.data.corpora.generic.CORPORA_ROOT", root),
                patch("src.data.corpora.paths.CORPORA_ROOT", root),
            ):
                demand = list(iter_rows())
                demand48 = list(iter_rows("DEMAND48"))
                babble = list(iter_generic("Babble"))
                ssn = list(iter_generic("SSN"))
            self.assertEqual([(row.path, row.sample_rate, row.corpus) for row in demand], [
                ("DEMAND/DKITCHEN/ch01.wav", 16000, "DEMAND"),
            ])
            self.assertEqual([(row.path, row.sample_rate, row.scene_code) for row in demand48], [
                ("DEMAND48/SCAFE/ch01.wav", 48000, "SCAFE"),
            ])
            self.assertEqual(demand48[0].channel_label, "ch01")
            self.assertEqual([(row.corpus, row.subset, row.sample_rate, row.file_name) for row in babble], [
                ("Babble", None, 48000, "babble.wav"),
            ])
            self.assertEqual([(row.corpus, row.subset, row.sample_rate, row.file_name) for row in ssn], [
                ("SSN", None, 48000, "ssn.wav"),
            ])
