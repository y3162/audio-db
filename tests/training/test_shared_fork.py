import unittest
from unittest.mock import patch

import torch

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

    def test_semamba_resume_keeps_moments_on_the_same_names(self):
        from tests.training.four_process.real_models import SEMambaPP
        adapter = SEMambaPP()
        adapter.configure({"scan_backend": "reference", "scheduler_step": "epoch"})
        device = torch.device("cpu")
        first = adapter.build("independent", device)["generator"]
        second = adapter.build("independent", device)["generator"]
        names = [name for name, _ in first.named_parameters()]
        self.assertEqual(names, [name for name, _ in second.named_parameters()])
        second.load_state_dict(first.state_dict(), strict=True)
        source = torch.optim.AdamW(first.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
        torch.manual_seed(4)
        for parameter in first.parameters():
            parameter.grad = torch.randn_like(parameter)
        source.step()
        restored = torch.optim.AdamW(second.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
        restored.load_state_dict(source.state_dict())
        for left, right in zip(first.parameters(), second.parameters(), strict=True):
            self.assertTrue(torch.equal(source.state[left]["exp_avg"], restored.state[right]["exp_avg"]))
            self.assertTrue(torch.equal(source.state[left]["exp_avg_sq"], restored.state[right]["exp_avg_sq"]))

    def test_semamba_cross_implementation_optimizer_needs_name_remap(self):
        import sys
        stub = sys.modules.get("librosa")
        if stub is not None and not hasattr(stub, "__path__"):
            del sys.modules["librosa"]
        from tests.training.four_process.real_models import SEMambaPP
        adapter = SEMambaPP()
        adapter.configure({"scan_backend": "reference", "scheduler_step": "epoch"})
        device = torch.device("cpu")
        try:
            official = adapter.build("official", device)["generator"]
        except Exception as error:
            self.skipTest(f"official SEMamba++ is not importable: {error}")
        independent = adapter.build("independent", device)["generator"]
        official_names = [name for name, _ in official.named_parameters()]
        independent_names = [name for name, _ in independent.named_parameters()]
        self.assertEqual(set(official_names), set(independent_names))
        self.assertNotEqual(official_names, independent_names)
        independent.load_state_dict(official.state_dict(), strict=True)
        source = torch.optim.AdamW(official.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
        torch.manual_seed(5)
        for parameter in official.parameters():
            parameter.grad = torch.randn_like(parameter)
        source.step()
        positional = torch.optim.AdamW(independent.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
        positional.load_state_dict(source.state_dict())
        by_name = {name: parameter for name, parameter in independent.named_parameters()}
        official_moments = {
            name: source.state[parameter]["exp_avg"].detach().clone()
            for name, parameter in official.named_parameters()
        }
        mismatched = 0
        for name, moment in official_moments.items():
            if not torch.equal(positional.state[by_name[name]]["exp_avg"], moment):
                mismatched += 1
        self.assertGreater(mismatched, 0)
        state = source.state_dict()
        remap_optimizer(state, official_names, independent_names)
        named = torch.optim.AdamW(independent.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
        named.load_state_dict(state)
        for name, moment in official_moments.items():
            self.assertTrue(torch.equal(named.state[by_name[name]]["exp_avg"], moment), name)
