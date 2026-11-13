"""SNR-shift training may pad short chunks. Validation uses the ESPnet drop."""

from __future__ import annotations

import unittest

from src.experiments.data.chunk import ChunkSampler
from src.experiments.data.fixed_pair import MicroBatcher
from src.experiments.data.render.chunking import plan_chunk_epoch
from src.experiments.data.sampling.planner import chunk_case_length


class _Store:
    def __init__(self, lengths: dict[str, int]) -> None:
        self.docs = {case_id: {"output": {"length": length}} for case_id, length in lengths.items()}
        self._cases = {case_id: {"frames": length} for case_id, length in lengths.items()}

    def case(self, case_id: str) -> dict[str, int]:
        return self._cases[case_id]


class _Source:
    split = "train"

    def __init__(self, train: list[str], valid: list[str], lengths: dict[str, int]) -> None:
        self._slots = {"train.cases": train, "valid.cases": valid}
        self.store = _Store(lengths)

    def dataset(self) -> dict[str, str]:
        return {"algorithm": "fixed_pair.voicebank_snr_shift"}

    def case_ids(self, slot: str) -> list[str]:
        if slot == "test.cases":
            raise AssertionError("test split was read")
        return list(self._slots[slot])


class SNRShiftChunkValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = ["short", "long"]
        self.valid = ["vshort", "vlong", "vmid"]
        self.lengths = {"short": 10, "long": 40, "vshort": 9, "vlong": 36, "vmid": 32}
        self.source = _Source(self.train, self.valid, self.lengths)
        self.sampler = ChunkSampler(
            self.source,
            seed=3,
            num_samples_per_epoch=2,
            chunk_length=16,
            shift_ratio=0.5,
            num_cache_chunks=4,
            shuffle=False,
            global_batch=2,
        )
        self.sampler.bind_batch(MicroBatcher(1, 2), 1)

    def test_training_pads_and_validation_drops_short_utterances(self) -> None:
        train = self.sampler.sequence(0, 0, 1)
        valid = self.sampler.validation_ranks(0, 1)[0]
        self.assertTrue(any(sample.case_id == "short" for sample in train))
        self.assertTrue(all(sample.chunk is not None and sample.chunk.stop - sample.chunk.start == 16 for sample in train))
        self.assertNotIn("vshort", [sample.case_id for sample in valid])
        self.assertIn("vlong", [sample.case_id for sample in valid])
        self.assertTrue(all(sample.case_id in self.valid for sample in valid))
        self.assertTrue(all(sample.case_id in self.train for sample in train))

    def test_validation_matches_unpadded_plan_and_is_stable_across_epochs(self) -> None:
        samples = [(case_id, chunk_case_length(self.source.store, case_id)) for case_id in self.valid]
        expected = plan_chunk_epoch(
            samples,
            espnet_epoch=1,
            seed=3,
            world_size=1,
            num_samples=None,
            chunk_length=16,
            global_batch=2,
            num_cache_chunks=4,
            shuffle=False,
            shift_ratio=0.5,
            pad_short=False,
        )
        first = self.sampler.validation_ranks(0, 1)[0]
        again = self.sampler.validation_ranks(4, 1)[0]
        spans = [(sample.case_id, sample.chunk.start, sample.chunk.stop) for sample in first]  # type: ignore[union-attr]
        planned = [
            (chunk.case_id, chunk.start, chunk.stop)
            for batch in expected.rank_batches[0]
            for chunk in batch
        ]
        self.assertEqual(spans, planned)
        self.assertEqual(list(first), list(again))
        self.assertGreater(expected.updates, 0)

    def test_padded_training_plan_differs_from_the_original_drop(self) -> None:
        samples = [(case_id, self.lengths[case_id]) for case_id in self.train]
        padded = plan_chunk_epoch(
            samples, espnet_epoch=1, seed=3, world_size=1, num_samples=2,
            chunk_length=16, global_batch=2, num_cache_chunks=4, shuffle=False, pad_short=True,
        )
        dropped = plan_chunk_epoch(
            samples, espnet_epoch=1, seed=3, world_size=1, num_samples=2,
            chunk_length=16, global_batch=2, num_cache_chunks=4, shuffle=False, pad_short=False,
        )
        padded_ids = {chunk.case_id for batch in padded.rank_batches[0] for chunk in batch}
        dropped_ids = {chunk.case_id for batch in dropped.rank_batches[0] for chunk in batch}
        self.assertIn("short", padded_ids)
        self.assertNotIn("short", dropped_ids)
        self.assertNotEqual(padded.plan_hash, dropped.plan_hash)
