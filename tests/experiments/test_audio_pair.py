"""Audio pairs are addressed by role, rate, and teacher, not by dataset name."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.experiments.data.chunk import ChunkSampler
from src.experiments.data.errors import ReproductionError
from src.experiments.data.fixed_pair import validation_scope
from src.experiments.data.ids import validate_rule
from src.experiments.data.pair import AudioPairData, check_waveforms, require_audio_contract
from src.experiments.data.render.chunking import plan_chunk_epoch
from src.experiments.errors import DataError, RuntimeFailure

ROOT = Path(__file__).resolve().parents[2]
_TRAINERS = (
    "src/experiments/official/mp_senet/trainer.py",
    "src/experiments/official/mossformergan/trainer.py",
    "src/experiments/official/semambapp/trainer.py",
    "src/experiments/official/tf_gridnet/trainer.py",
    "src/experiments/official/tf_locoformer/trainer.py",
    "src/experiments/official/deepfilternet3/trainer.py",
)


def _pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    noisy = np.linspace(-0.2, 0.2, 16, dtype=np.float32)
    target = np.linspace(0.1, -0.1, 16, dtype=np.float32)
    short_noisy = np.array([0.3, -0.3, 0.1], dtype=np.float32)
    short_target = np.array([0.2, -0.2, 0.0], dtype=np.float32)
    return [(noisy, target), (short_noisy, short_target)]


class _Identity:
    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def identity(self) -> dict[str, object]:
        return dict(self._fields)


class AudioPairContractTest(unittest.TestCase):
    def test_supply_has_no_dataset_name_and_replays_the_same_batch(self) -> None:
        data = AudioPairData(_pairs(), sample_rate=16000, teacher="dry_clean", batch_size=1)
        self.assertNotIn("dataset_id", data.identity())
        self.assertNotIn("algorithm", data.identity())
        self.assertEqual(data.identity()["sample_rate"], 16000)
        self.assertEqual(data.identity()["teacher"], "dry_clean")
        first = data.train_epoch(0, start_update=0).batch(0)[0]
        data.epoch = 2
        data.update_index = 1
        state = data.state_dict()
        data.epoch = 0
        data.update_index = 0
        data.load_state_dict(state)
        replay = data.train_epoch(0, start_update=1).batch(1)[0]
        self.assertEqual(data.epoch, 2)
        self.assertEqual(data.update_index, 1)
        self.assertEqual(int(torch.max(torch.abs(first.inputs - data.train_epoch(0, start_update=0).batch(0)[0].inputs))), 0)
        self.assertEqual(tuple(replay.sample_ids), ("pair-1",))
        self.assertEqual(int(replay.lengths[0]), 3)
        self.assertEqual(replay.teacher_kind, "dry_clean")
        self.assertEqual(replay.sample_rate, 16000)
        again = AudioPairData(_pairs(), sample_rate=16000, teacher="dry_clean", batch_size=1)
        self.assertEqual(again.plan_hash, data.plan_hash)

    def test_rate_and_teacher_mismatches_are_rejected(self) -> None:
        with self.assertRaises(DataError):
            check_waveforms(np.array([1.0, np.nan], dtype=np.float32), np.ones(2, dtype=np.float32), sample_rate=16000, teacher="dry_clean")
        with self.assertRaises(DataError):
            check_waveforms(np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32), sample_rate=16000, teacher="dry_clean")
        with self.assertRaises(DataError):
            check_waveforms(np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32), sample_rate=16000, teacher="clean")
        sixteen = _Identity(sample_rate=16000, teacher="dry_clean")
        for teachers in (("dry_clean",),):
            require_audio_contract(sixteen, sample_rate=16000, teachers=teachers)
        with self.assertRaises(RuntimeFailure):
            require_audio_contract(sixteen, sample_rate=48000, teachers=("dry_clean",))
        with self.assertRaises(RuntimeFailure):
            require_audio_contract(_Identity(sample_rate=16000, teacher="reverberant_clean"), sample_rate=16000, teachers=("dry_clean",))
        with self.assertRaises(RuntimeFailure):
            require_audio_contract(_Identity(sample_rate=16000, teacher="dry_clean"), sample_rate=16000, teachers=("official_mixed_target",))
        require_audio_contract(_Identity(**{"class": "src.experiments.data.TensorPairData", "features": 4}), sample_rate=16000, teachers=("dry_clean",))
        with self.assertRaises(RuntimeFailure):
            require_audio_contract(_Identity(features=4), sample_rate=16000, teachers=("dry_clean",))

    def test_the_same_pair_satisfies_every_16k_dry_clean_model(self) -> None:
        data = AudioPairData(_pairs(), sample_rate=16000, teacher="dry_clean")
        for _model in ("mp_senet", "mossformergan", "semambapp", "tf_gridnet", "tf_locoformer"):
            require_audio_contract(data, sample_rate=16000, teachers=("dry_clean",))
        with self.assertRaises(RuntimeFailure):
            require_audio_contract(data, sample_rate=48000, teachers=("official_mixed_target",))
        reverberant = AudioPairData(_pairs(), sample_rate=16000, teacher="reverberant_clean")
        with self.assertRaises(RuntimeFailure):
            require_audio_contract(reverberant, sample_rate=16000, teachers=("dry_clean",))

    def test_chunk_plan_and_validation_scope_ignore_dataset_names(self) -> None:
        samples = [("a", 40), ("b", 10)]
        left = plan_chunk_epoch(samples, espnet_epoch=1, seed=3, world_size=1, num_samples=4, chunk_length=16, global_batch=2, num_cache_chunks=4, shuffle=False, shift_ratio=0.5)
        right = plan_chunk_epoch(list(samples), espnet_epoch=1, seed=3, world_size=1, num_samples=4, chunk_length=16, global_batch=2, num_cache_chunks=4, shuffle=False, shift_ratio=0.5)
        self.assertEqual(left.plan_hash, right.plan_hash)

        class Boom:
            def dataset(self) -> dict[str, str]:
                raise AssertionError("chunk sampler read a dataset name")

        parameters = ChunkSampler(Boom(), seed=1, num_samples_per_epoch=4, chunk_length=16, shift_ratio=0.5, num_cache_chunks=4, shuffle=True).order_parameters()
        self.assertNotIn("short_utterances", parameters)
        self.assertEqual(parameters["chunk_length"], 16)
        self.assertEqual(validation_scope("rank_mod_fixed_v1"), "per_rank")
        self.assertEqual(validation_scope("fixed_single_shuffle_v1"), "rank0")
        self.assertEqual(validation_scope("torch_distributed_sampler_v1"), "rank0")

    def test_fixed_pair_normalize_is_a_pipeline_field(self) -> None:
        base = {"sample_rate": 16000, "teacher": "dry_clean"}
        for algorithm in ("fixed_pair.mp_senet", "fixed_pair.mossformergan_se_16k"):
            for normalize in ("noisy_rms_unit_v1", "independent_audio_norm_v1"):
                rule = validate_rule(algorithm, {**base, "normalize": normalize})
                self.assertEqual(rule["normalize"], normalize)
        with self.assertRaises(ReproductionError):
            validate_rule("fixed_pair.mp_senet", {**base, "normalize": "independent_peak_v1"})

    def test_trainers_do_not_select_a_dataset_algorithm(self) -> None:
        for relative in _TRAINERS:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("identity.get(\"algorithm\")", text, relative)
            self.assertNotIn("fixed_pair.", text, relative)
            self.assertNotIn("online.", text, relative)
            self.assertNotIn("pregenerated.", text, relative)


if __name__ == "__main__":
    unittest.main()
