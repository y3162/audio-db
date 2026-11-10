import tempfile
import unittest
import shutil
from pathlib import Path

import duckdb
import torch
import torchaudio

from src.data.corpora_catalog import CATALOG
from src.data.db.schema import create_corpora_schema
from src.data.link_corpora import link_corpora
from src.evaluation.datasets.evaldb import EvaluationDB
from src.experiments.calc.dataset import load_recipe_dataset


class PortabilityTests(unittest.TestCase):
    def test_two_dataset_roots_produce_identical_recipes_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            results = []
            first_audio = None
            for name in ("a", "b"):
                physical = base / ("source_" + name)
                project = base / ("project_" + name)
                for spec in CATALOG:
                    (physical / spec.source_name).mkdir(parents=True)
                audio = physical / "DEMAND" / "sample.wav"
                if first_audio is None:
                    torchaudio.save(str(audio), torch.sin(torch.arange(8000).float() / 20).unsqueeze(0), 8000)
                    first_audio = audio
                else:
                    shutil.copyfile(first_audio, audio)
                corpora = project / "data" / "corpora"
                link_corpora(physical, corpora)
                database = project / "data" / "db" / "training.duckdb"
                database.parent.mkdir(parents=True)
                con = duckdb.connect(str(database))
                create_corpora_schema(con)
                con.execute("INSERT INTO corpora.audio_files (path, corpus, file_name, sample_rate, frames, channels) VALUES ('DEMAND/sample.wav', 'DEMAND', 'sample.wav', 8000, 8000, 1)")
                con.close()
                db = EvaluationDB(database, project_root=project, corpora_root=corpora)
                db.create_set_from_corpus("toy_train", role="clean", filters={"corpus": "DEMAND"})
                dataset = load_recipe_dataset(database, "toy_train", role="train", seed=1, sample_rate=16000, segment_size=None, training=False)
                batch = dataset.collate([0])
                results.append((dataset.examples[0].case_id, dataset.examples[0].recipe, batch["clean"]))
            self.assertEqual(results[0][0:2], results[1][0:2])
            self.assertTrue(torch.equal(results[0][2], results[1][2]))
