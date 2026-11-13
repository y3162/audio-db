"""Real snapshot adapters emit one common audio pair."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.experiments.data.chunk import ChunkProtocol, ChunkSampler
from src.experiments.data.fixed_pair import (
    FixedPairProtocol,
    MicroBatcher,
    OrderSampler,
    SnapshotDataModule,
    SnapshotSource,
)
from src.experiments.data.online import Dfn3IndexSampler, Dfn3Protocol, SemambaProtocol
from src.experiments.data.pair import require_audio_contract
from src.experiments.errors import RuntimeFailure

ROOT = Path(__file__).resolve().parents[2]


class _Runtime:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.world_size = 1
        self.rank = 0


def _open(module: SnapshotDataModule):
    module.setup(_Runtime())
    return module


class SnapshotConnectionTest(unittest.TestCase):
    def test_saved_and_ondemand_pairs_share_the_16k_contract(self) -> None:
        vb = ROOT / "data/db/snapshots-voicebank-float-v1"
        if not vb.is_dir():
            self.skipTest("NOT_RUN: VoiceBank float snapshots are not present")
        modules = []
        try:
            for name in ("mp_senet.vb28.ondemand.official@1", "mossformergan.vb28.ondemand.official@1"):
                source = SnapshotSource(name, vb, "train")
                protocol = FixedPairProtocol(source)
                module = _open(
                    SnapshotDataModule(
                        source,
                        OrderSampler(source, protocol, seed=0),
                        protocol,
                        MicroBatcher(1, 1),
                    )
                )
                modules.append(module)
                for _model in ("mp_senet", "mossformergan", "semambapp", "tf_gridnet", "tf_locoformer"):
                    require_audio_contract(module, sample_rate=16000, teachers=("dry_clean",))
                with self.assertRaises(RuntimeFailure):
                    require_audio_contract(module, sample_rate=48000, teachers=("official_mixed_target",))
                batch = module.train_epoch(0, start_update=0).batch(0)[0]
                valid = next(iter(module.validation_epoch(0)))
                for item in (batch, valid):
                    self.assertEqual(item.sample_rate, 16000)
                    self.assertEqual(item.teacher_kind, "dry_clean")
                    self.assertEqual(tuple(item.inputs.shape), tuple(item.targets.shape))
                    self.assertTrue(torch.isfinite(item.inputs).all())
                    self.assertTrue(torch.isfinite(item.targets).all())
                    self.assertLessEqual(int(item.lengths.max()), int(item.inputs.shape[-1]))
            self.assertEqual(modules[0].identity()["normalize"], "noisy_rms_unit_v1")
            self.assertEqual(modules[1].identity()["normalize"], "independent_audio_norm_v1")
            self.assertNotEqual(modules[0].identity()["algorithm"], modules[1].identity()["algorithm"])
        finally:
            for module in modules:
                module.close()

    def test_chunk_adapters_emit_dry_clean_pairs(self) -> None:
        root = ROOT / "data/db/snapshots"
        specs = (
            ("tf_gridnet.ms_snsd.espnet_recipe@1", 8000),
            ("tf_locoformer.dns.accept_recipe@1", 8000),
        )
        if not root.is_dir():
            self.skipTest("NOT_RUN: reproduction snapshots are not present")
        modules = []
        try:
            for name, chunk_length in specs:
                source = SnapshotSource(name, root, "train")
                protocol = ChunkProtocol(source)
                sampler = ChunkSampler(
                    source,
                    seed=0,
                    num_samples_per_epoch=2,
                    chunk_length=chunk_length,
                    shift_ratio=0.5,
                    num_cache_chunks=4,
                    shuffle=False,
                    global_batch=1,
                )
                module = _open(SnapshotDataModule(source, sampler, protocol, MicroBatcher(1, 1)))
                modules.append(module)
                require_audio_contract(module, sample_rate=16000, teachers=("dry_clean",))
                batch = module.train_epoch(0, start_update=0).batch(0)[0]
                self.assertEqual(batch.sample_rate, 16000)
                self.assertEqual(batch.teacher_kind, "dry_clean")
                self.assertEqual(int(batch.inputs.shape[-1]), chunk_length)
                self.assertTrue(torch.isfinite(batch.inputs).all())
                self.assertTrue(torch.isfinite(batch.targets).all())
        finally:
            for module in modules:
                module.close()

    def test_online_adapters_keep_their_teacher_and_rate(self) -> None:
        root = ROOT / "data/db/snapshots"
        if not root.is_dir():
            self.skipTest("NOT_RUN: reproduction snapshots are not present")
        dfn_source = SnapshotSource("dfn3.online.smoke@1", root, "train")
        sem_source = SnapshotSource("semambapp.gsr.smoke@1", root, "train")
        dfn = SnapshotDataModule(
            dfn_source,
            Dfn3IndexSampler(dfn_source, seed=0),
            Dfn3Protocol(dfn_source),
            MicroBatcher(1, 1),
        )
        sem_protocol = SemambaProtocol(sem_source)
        sem = SnapshotDataModule(
            sem_source,
            OrderSampler(sem_source, sem_protocol, seed=0, global_batch=1),
            sem_protocol,
            MicroBatcher(1, 1),
        )
        opened = []
        try:
            for module, rate, teacher in (
                (dfn, 48000, "official_mixed_target"),
                (sem, 16000, "dry_clean"),
            ):
                _open(module)
                opened.append(module)
                require_audio_contract(module, sample_rate=rate, teachers=(teacher,))
                batch = module.train_epoch(0, start_update=0).batch(0)[0]
                self.assertEqual(batch.sample_rate, rate)
                self.assertEqual(batch.teacher_kind, teacher)
                self.assertEqual(tuple(batch.inputs.shape), tuple(batch.targets.shape))
                self.assertTrue(torch.isfinite(batch.inputs).all())
                self.assertTrue(torch.isfinite(batch.targets).all())
            with self.assertRaises(RuntimeFailure):
                require_audio_contract(dfn, sample_rate=16000, teachers=("dry_clean",))
            require_audio_contract(sem, sample_rate=16000, teachers=("dry_clean",))
        finally:
            for module in opened:
                module.close()


if __name__ == "__main__":
    unittest.main()
