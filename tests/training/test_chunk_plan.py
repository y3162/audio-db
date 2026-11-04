import unittest
from src.training.data.chunk_plan import chunk_batches


class ChunkPlanTest(unittest.TestCase):
    def test_boundaries_and_leftovers(self):
        batches = list(chunk_batches([('short', 8), ('a', 13), ('b', 17)],
            epoch=1, seed=0, chunk_length=8, batch_size=2, cache_size=2, shuffle=False))
        self.assertEqual([[(c.case_id, c.start, c.stop) for c in b] for b in batches],
                         [[('a', 0, 8), ('a', 4, 12)], [('b', 0, 8), ('b', 4, 12)]])

    def test_seed_reproducibility(self):
        def plan():
            return list(chunk_batches([('a', 21), ('b', 19)], epoch=1, seed=17,
                        chunk_length=8, batch_size=2, cache_size=2))
        self.assertEqual(plan(), plan())
