import unittest
from unittest.mock import patch

from tests.training.four_process.shared_fork import remap_optimizer


class SharedForkTest(unittest.TestCase):
    def test_optimizer_moments_follow_names(self):
        state = {'param_groups': [{'params': [0, 1, 2]}],
                 'state': {0: {'moment': 10}, 2: {'moment': 30}}}
        remap_optimizer(state, ['a', 'b', 'c'], ['c', 'a', 'b'])
        self.assertEqual(state['state'], {0: {'moment': 30}, 1: {'moment': 10}})
        self.assertEqual(state['param_groups'][0]['params'], [0, 1, 2])
        with self.assertRaises(ValueError):
            remap_optimizer(state, ['a', 'a', 'c'], ['a', 'b', 'c'])

    def test_df_frontend_must_match_recorded_probe(self):
        from tests.training.four_process.real_models import DeepFilterNet3
        with patch.dict('os.environ', {'DF_PROBE_OFFICIAL_LIBDF': '1'}):
            adapter = DeepFilterNet3()
            adapter.configure({})
            self.assertEqual(adapter.probe['frontend'], 'official libdf vs independent DFStream')
            with self.assertRaises(ValueError):
                adapter.configure({'probe': {'frontend': 'independent DFStream; libdf analysis is not executed'}})
        with patch.dict('os.environ', {'DF_PROBE_OFFICIAL_LIBDF': 'shared'}):
            adapter.configure({})
            self.assertEqual(adapter.probe['frontend'], 'official libdf on both sides')
