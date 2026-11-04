"""CPU comparison with the pinned ESPnet iterator (optional source checkout)."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
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


@unittest.skipUnless(SOURCE.is_dir(), 'pinned ESPnet source unavailable')
class OfficialChunkTest(unittest.TestCase):
    def test_chunk_values_and_batch_order(self):
        sys.path.insert(0, str(SOURCE))
        from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
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

    def test_selected_utterances(self):
        sys.path.insert(0, str(SOURCE))
        from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
        ds = AudioDataset([9] * 7)
        for n in (3, 7, 10, 20):
            for epoch in (1, 2, 3, 4):
                for shuffle in (False, True):
                    factory = SequenceIterFactory(ds, [[i] for i in range(7)],
                        num_iters_per_epoch=n, seed=17, shuffle=shuffle, collate_fn=collate)
                    expected = [ids[0] for ids, _ in factory.build_iter(epoch)]
                    self.assertEqual(expected, select_utterances([str(i) for i in range(7)],
                        epoch=epoch, seed=17, num_samples=n, shuffle=shuffle))
