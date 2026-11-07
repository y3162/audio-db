"""CPU comparison with the pinned ESPnet iterator (optional source checkout)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.training.data.chunk_plan import chunk_batches, select_utterances

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'build/clean-20261104/sources/espnet'


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, lengths): self.lengths = lengths
    def __len__(self): return len(self.lengths)
    def __getitem__(self, index):
        return str(index), {'speech': torch.arange(self.lengths[index], dtype=torch.float32)}


def collate(batch):
    ids, rows = zip(*batch)
    return list(ids), {'speech': torch.stack([r['speech'] for r in rows]),
                       'speech_lengths': torch.tensor([len(r['speech']) for r in rows])}


def _import_factories():
    inserted = str(SOURCE) not in sys.path
    sys.path.insert(0, str(SOURCE))
    try:
        from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
        from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest(f'pinned ESPnet import is unavailable in this environment: {exc}') from exc
    finally:
        if inserted:
            while sys.path and sys.path[0] == str(SOURCE):
                sys.path.pop(0)
    return ChunkIterFactory, SequenceIterFactory


@unittest.skipUnless(SOURCE.is_dir(), 'pinned ESPnet source unavailable')
class OfficialChunkTest(unittest.TestCase):
    def test_chunk_values_and_batch_order(self):
        ChunkIterFactory, _sequence = _import_factories()
        lengths = [8, 13, 17, 25, 10, 29]
        ds = AudioDataset(lengths)
        for shuffle in (False, True):
            for epoch in (1, 2):
                factory = ChunkIterFactory(ds, batch_size=2, batches=[[i] for i in range(len(ds))],
                    chunk_length=8, num_cache_chunks=3, seed=17, shuffle=shuffle,
                    num_workers=0, collate_fn=collate)
                # SequenceIterFactory shuffles the utterance stream independently.
                indices = list(range(len(ds)))
                if shuffle:
                    np.random.RandomState(epoch + 17).shuffle(indices)
                planned = list(chunk_batches([(str(i), lengths[i]) for i in indices],
                    epoch=epoch, seed=17, chunk_length=8, batch_size=2,
                    cache_size=3, shuffle=shuffle))
                actual = list(factory.build_iter(epoch))
                self.assertEqual(len(actual), len(planned))
                for (ids, batch), chunks in zip(actual, planned):
                    self.assertEqual(ids, [c.case_id for c in chunks])
                    self.assertTrue(torch.equal(batch['speech'], torch.stack([
                        torch.arange(c.start, c.stop, dtype=torch.float32) for c in chunks])))

    def test_rank_shards_match_official_factories(self):
        """Each rank's factory sees batches[rank::world], then iterator_stop keeps the minimum."""
        ChunkIterFactory, _sequence = _import_factories()
        lengths = [30, 18, 40, 12, 25, 50, 9, 33, 21, 44, 16, 28]
        dataset = AudioDataset(lengths)
        global_batch = 4
        cache = 8
        num_samples = 5
        seed = 17
        for world in (1, 2, 4):
            per_rank = global_batch // world
            for epoch in (1, 2, 3):
                for shuffle in (False, True):
                    plan = plan_chunk_epoch(
                        [(str(i), lengths[i]) for i in range(len(lengths))],
                        espnet_epoch=epoch,
                        seed=seed,
                        world_size=world,
                        num_samples=num_samples,
                        chunk_length=8,
                        global_batch=global_batch,
                        num_cache_chunks=cache,
                        shuffle=shuffle,
                    )
                    produced = []
                    for rank in range(world):
                        factory = ChunkIterFactory(
                            dataset,
                            batch_size=per_rank,
                            batches=[[i] for i in range(len(lengths))][rank::world],
                            chunk_length=8,
                            num_cache_chunks=cache // world,
                            num_samples_per_epoch=num_samples,
                            seed=seed,
                            shuffle=shuffle,
                            num_workers=0,
                            collate_fn=collate,
                        )
                        actual = list(factory.build_iter(epoch))
                        produced.append(len(actual))
                        self.assertEqual(len(actual), len(plan.raw_rank_batches[rank]))
                        for (ids, batch), chunks in zip(actual, plan.raw_rank_batches[rank]):
                            self.assertEqual(ids, [chunk.case_id for chunk in chunks])
                            self.assertTrue(torch.equal(batch['speech'], torch.stack([
                                torch.arange(chunk.start, chunk.stop, dtype=torch.float32) for chunk in chunks
                            ])))
                    self.assertEqual(plan.updates, min(produced))
                    self.assertTrue(all(len(batches) == plan.updates for batches in plan.rank_batches))

    def test_validation_shards_match_official_factories(self):
        """Validation uses every utterance once, no shuffle, and the same minimum stop."""
        ChunkIterFactory, _sequence = _import_factories()
        lengths = [30, 18, 40, 12, 25, 50, 9, 33]
        dataset = AudioDataset(lengths)
        for world, global_batch, cache in ((2, 4, 8), (4, 4, 8)):
            plan = plan_chunk_epoch(
                [(str(i), lengths[i]) for i in range(len(lengths))],
                espnet_epoch=1,
                seed=17,
                world_size=world,
                num_samples=None,
                chunk_length=8,
                global_batch=global_batch,
                num_cache_chunks=cache,
                shuffle=False,
            )
            counts = []
            for rank in range(world):
                factory = ChunkIterFactory(
                    dataset,
                    batch_size=global_batch // world,
                    batches=[[i] for i in range(len(lengths))][rank::world],
                    chunk_length=8,
                    num_cache_chunks=cache // world,
                    num_samples_per_epoch=None,
                    seed=17,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate,
                )
                actual = list(factory.build_iter(1))
                counts.append(len(actual))
                self.assertEqual(
                    [ids for ids, _batch in actual],
                    [[chunk.case_id for chunk in chunks] for chunks in plan.raw_rank_batches[rank]],
                )
            self.assertEqual(plan.updates, min(counts))
            covered = []
            for rank in range(world):
                covered.extend(case_id for case_id, _length in [(str(i), lengths[i]) for i in range(len(lengths))][rank::world])
            self.assertEqual(sorted(covered), [str(i) for i in range(len(lengths))])

    def test_selected_utterances(self):
        _chunk, SequenceIterFactory = _import_factories()
        ds = AudioDataset([9] * 7)
        for n in (3, 7, 10, 20):
            for epoch in (1, 2, 3, 4):
                for shuffle in (False, True):
                    factory = SequenceIterFactory(ds, [[i] for i in range(7)],
                        num_iters_per_epoch=n, seed=17, shuffle=shuffle, collate_fn=collate)
                    expected = [ids[0] for ids, _ in factory.build_iter(epoch)]
                    self.assertEqual(expected, select_utterances([str(i) for i in range(7)],
                        epoch=epoch, seed=17, num_samples=n, shuffle=shuffle))
