"""One real MP-SENet utterance. This is a pipeline smoke, not a benchmark."""

import unittest
from pathlib import Path

from src.evaluation.pipeline import execute
from tests.evaluation.support import experiment_config

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "data/checkpoints/mp-senet/g_best_vb"
DATABASE = ROOT / "data/db/resource.duckdb"
CORPORA = ROOT / "data/corpora"


class RealMpSenetSmoke(unittest.TestCase):
    @unittest.skipUnless(CHECKPOINT.is_file(), "MP-SENet checkpoint is not available")
    @unittest.skipUnless(DATABASE.is_file(), "resource.duckdb is not available")
    def test_one_librispeech_utterance_on_cpu(self):
        with self._temporary_dir() as directory:
            config = experiment_config(directory, run_id="mp-smoke")
            config.experiment.limit = 1
            config.dataset.source = "duckdb"
            config.dataset.path = str(DATABASE)
            config.dataset.corpora_root = str(CORPORA)
            config.dataset.role = "clean"
            config.dataset.filters = {"corpus": "LibriSpeech", "subset": "test-clean"}
            config.dataset.limit = 1
            config.dataset.noise_filters = {"corpus": "DEMAND"}
            config.dataset.noise_limit = 1
            config.dataset.pair = "hash"
            config.dataset.max_seconds = 2.0
            config.mixing.enabled = True
            config.mixing.snr_db = [0.0]
            config.mixing.prevent_clip = True
            config.models[0].name = "mp_senet"
            config.models[0].variant = "vb"
            config.models[0].checkpoint = str(CHECKPOINT)
            config.asr.enabled = False
            config.metrics.wer = False
            config.metrics.pesq = True
            config.metrics.stoi = True
            config.runtime.device = "cpu"
            result = execute(config)
            self.assertEqual(result["summary"]["num_ok"], 1)
            record = result["records"][0]
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["model"]["name"], "mp_senet")
            self.assertEqual(record["model"]["variant"], "vb")
            self.assertEqual(len(record["model"]["checkpoint_sha256"]["checkpoint"]), 64)
            self.assertEqual(record["sample_rate"], 16000)
            self.assertTrue(Path(record["paths"]["clean"]).is_file())
            self.assertTrue(Path(record["paths"]["noise"]).is_file())
            enhanced_pesq = record["pesq"]["values"]["enhanced"]
            noisy_pesq = record["pesq"]["values"]["noisy"]
            self.assertEqual(enhanced_pesq["status"], "ok")
            self.assertIsInstance(enhanced_pesq["value"], float)
            self.assertEqual(noisy_pesq["status"], "ok")
            self.assertNotEqual(enhanced_pesq["value"], 0.0)
            self.assertEqual(record["stoi"]["values"]["enhanced"]["status"], "ok")
            self.assertAlmostEqual(record["mix"]["snr_db_achieved"], 0.0, places=2)
            record_path = Path(result["run_dir"]) / "records" / f"{record['record_id']}.json"
            before = record_path.read_bytes()
            config.output.resume = True
            resumed = execute(config)
            self.assertEqual(resumed["reused"], 1)
            self.assertEqual(resumed["written"], 0)
            self.assertEqual(record_path.read_bytes(), before)

    def _temporary_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()
