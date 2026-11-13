"""Build and export a tiny shared-noise SNR-shift reproduction graph."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.experiments.data.db.registry import initialize_database, register_staging, read_graph
from src.experiments.data.db.snapshot import export_snapshot
from src.experiments.data.db.staging import write_graph
from src.experiments.data.assets.store import AssetStore
from src.experiments.data.errors import ReproductionError
from src.experiments.data.prepare.run import execute
from src.experiments.data.prepare.voicebank_shift_audit import audit_recipes
from src.experiments.data.prepare.voicebank_shift_graph import build_shift_graph
from src.experiments.data.prepare.voicebank_shift_recipe import canonical_shift_recipe, shift_case_id
from src.experiments.data.snapshot import load_snapshot
from src.experiments.data.snapshot_provenance import snapshot_provenance_parts
from src.experiments.data.render.voicebank_shift_snapshot import render_shift_case
from src.experiments.data.fixed_pair import FixedPairProtocol, MicroBatcher, OrderSampler, SnapshotDataModule, SnapshotSource
from src.experiments.data.chunk import ChunkProtocol, ChunkSampler
from tests.data.reproduction.test_p2 import _tiny_paths, _write_tiny_config, _write_tiny_corpus


class VoiceBankShiftGraphTests(unittest.TestCase):
    def test_graph_registers_and_exports_all_shared_sources(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = _tiny_paths(root / "base")
            _write_tiny_corpus(paths.corpora)
            _write_tiny_config(paths.repo_root)
            execute("all", paths)
            reference = read_graph(paths.database)
            noise_dir = paths.corpora / "Noise"
            noise_dir.mkdir()
            for name, offset in (("train.wav", 0.0), ("test.wav", 0.2)):
                sf.write(noise_dir / name, np.sin(np.arange(6000) / 17.0) * 0.2 + offset, 48000, subtype="FLOAT")
            noise_hash = {name: hashlib.sha256((noise_dir / name).read_bytes()).hexdigest()
                          for name in ("train.wav", "test.wav")}
            staged, raw_rows = [], []
            for utterance, source_split, split in (("p226_001", "train", "train"),
                                                   ("p286_001", "train", "valid"),
                                                   ("p287_001", "train", "valid"),
                                                   ("p232_001", "test", "test")):
                subset = "clean_trainset_28spk_wav" if source_split == "train" else "clean_testset_wav"
                clean = paths.corpora / "VoiceBank-DEMAND" / subset / f"{utterance}.wav"
                noise = "train.wav" if source_split == "train" else "test.wav"
                row = {"status": "ready", "utterance_id": utterance, "split": source_split,
                       "clean_path": f"data/corpora/VoiceBank-DEMAND/{subset}/{utterance}.wav",
                       "noise_path": f"data/corpora/Noise/{noise}",
                       "clean_sha256": hashlib.sha256(clean.read_bytes()).hexdigest(), "noise_sha256": noise_hash[noise],
                       "noise_start_48k": 0, "noise_end_exclusive_48k": 4800,
                       "clean_gain_g": 0.9, "shifted_noise_gain": 0.7}
                raw_rows.append(row)
                staged.append({"case_id": shift_case_id(row), "recipe": canonical_shift_recipe(row),
                               "utterance_id": utterance, "speaker_id": utterance.split("_")[0], "split": split})
            metadata = {"source_sha256": "a" * 64, "noise_overlap_audit": audit_recipes(raw_rows)}
            graph = build_shift_graph(metadata, staged, reference, paths.corpora, resource_db=None)
            self.assertEqual(len(graph.datasets), 4)
            self.assertTrue(all(d["selection_uses_test"] is False for d in graph.datasets))
            self.assertEqual(len({tuple(b["case_set_id"] for b in d["bindings"]) for d in graph.datasets}), 1)
            bad = dict(raw_rows[-1], noise_path="data/corpora/Noise/train.wav", noise_sha256=noise_hash["train.wav"])
            bad_item = dict(staged[-1], case_id=shift_case_id(bad), recipe=canonical_shift_recipe(bad))
            bad_metadata = {"source_sha256": "a" * 64, "noise_overlap_audit": audit_recipes([*raw_rows[:-1], bad])}
            with self.assertRaisesRegex(ReproductionError, "failed leak_check|selection uses test"):
                build_shift_graph(bad_metadata, [*staged[:-1], bad_item], reference, paths.corpora, resource_db=None)
            staging = root / "shift-staging"
            database = root / "shift.duckdb"
            initialize_database(database)
            write_graph(staging, graph)
            register_staging(staging, database)
            snapshots = root / "snapshots"
            for dataset in graph.datasets:
                export_snapshot(database, dataset["dataset_id"], snapshots, allow_unknown_leak_check=True,
                                corpora=paths.corpora, derived=paths.derived)
                loaded = load_snapshot(snapshots / dataset["dataset_id"])
                self.assertEqual(len(loaded.cases), 4)
                self.assertEqual(len(loaded.assets), 6)
                self.assertFalse(loaded.dataset["selection_uses_test"])
                store = AssetStore.from_snapshot(loaded, corpora=paths.corpora, derived=paths.derived,
                                                 generated=root / "generated")
                provenance = snapshot_provenance_parts(store, workers=1)
                self.assertTrue(any("train and validation noise overlaps" in note for note in provenance["limitations"]))
                self.assertEqual(provenance["lineage_counts"]["unknown"], 0)
                self.assertFalse(provenance["selection_uses_test"])
                for slot in ("train.cases", "valid.cases", "test.cases"):
                    noisy, clean = render_shift_case(store, store.case_ids(slot)[0], verify_on_read=True)
                    self.assertEqual(noisy.shape, clean.shape)
                    self.assertEqual(len(noisy), 1600)
                source = SnapshotSource(dataset["dataset_id"], snapshots, "train", corpora=paths.corpora)
                if dataset["name"].endswith(("tf_gridnet", "tf_locoformer")):
                    protocol = ChunkProtocol(source, verify_on_read=True)
                    sampler = ChunkSampler(source, seed=1, num_samples_per_epoch=1,
                                           chunk_length=2000, num_cache_chunks=2, shuffle=False)
                else:
                    protocol = FixedPairProtocol(source, verify_on_read=True)
                    sampler = OrderSampler(source, protocol, seed=1)
                module = SnapshotDataModule(source, sampler, protocol, MicroBatcher(1, 1))
                class Runtime:
                    device = torch.device("cpu")
                    world_size = 1
                    rank = 0
                try:
                    module.setup(Runtime())
                    view = module.train_epoch(0, start_update=0)
                    try:
                        batch = view.batch(0)[0]
                        self.assertEqual(batch.inputs.shape, batch.targets.shape)
                        self.assertTrue(torch.isfinite(batch.inputs).all())
                        if dataset["name"].endswith(("tf_gridnet", "tf_locoformer")):
                            self.assertEqual(batch.inputs.shape, (1, 2000))
                            self.assertEqual(batch.lengths.tolist(), [1600])
                            self.assertTrue(torch.all(batch.inputs[:, 1600:] == 0))
                    finally:
                        view.close()
                    valid = list(module.validation_epoch(0))
                    valid_ids = [case_id for batch in valid for case_id in batch.sample_ids]
                    self.assertTrue(set(valid_ids).isdisjoint(set(source.case_ids("test.cases"))))
                    if dataset["name"].endswith(("tf_gridnet", "tf_locoformer")):
                        # 1600 samples are shorter than the 2000-sample chunk, so the
                        # original ESPnet rule keeps no validation chunk.
                        self.assertEqual(valid, [])
                    else:
                        self.assertEqual(set(valid_ids), set(source.case_ids("valid.cases")))
                        self.assertEqual(len(valid_ids), len(set(valid_ids)))
                        self.assertTrue(all(batch.inputs.shape == (1, 1600) for batch in valid))
                finally:
                    module.close()
