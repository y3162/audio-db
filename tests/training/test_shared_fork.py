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

    def test_unmapped_optimizer_entries_fail(self):
        state = {'param_groups': [{'params': [0, 1]}], 'state': {0: {'moment': 1}, 7: {'moment': 2}}}
        with self.assertRaises(ValueError):
            remap_optimizer(state, ['a', 'b'], ['b', 'a'])

    def test_mp_senet_import_leaves_a_real_librosa(self):
        import sys
        from tests.training.test_mp_senet_official_step import OfficialStepTests
        OfficialStepTests.setUpClass()
        try:
            from dataset import mag_pha_stft
            self.assertTrue(callable(mag_pha_stft))
        finally:
            OfficialStepTests.tearDownClass()
        stub = sys.modules.get("librosa")
        self.assertIsNotNone(stub)
        self.assertTrue(hasattr(stub, "__path__"), "empty librosa module leaked from the MP-SENet import")
        from librosa.filters import mel
        self.assertTrue(callable(mel))

    def _semamba_pair(self):
        import sys
        stub = sys.modules.get("librosa")
        if stub is not None and not hasattr(stub, "__path__"):
            self.fail("empty librosa module leaked into this process")
        from tests.training.four_process.real_models import SEMambaPP
        adapter = SEMambaPP()
        adapter.configure({"scan_backend": "reference", "scheduler_step": "epoch"})
        device = torch.device("cpu")
        try:
            official = adapter.build("official", device)
        except (ModuleNotFoundError, RuntimeError) as error:
            text = str(error)
            if "mamba" in text or "selective_scan" in text or "active drivers" in text:
                self.skipTest(f"official SEMamba++ is not importable: {error}")
            raise
        independent = adapter.build("independent", device)
        for name, module in official.items():
            independent[name].load_state_dict(module.state_dict(), strict=True)
            official_params = dict(module.named_parameters())
            independent_params = dict(independent[name].named_parameters())
            self.assertEqual(set(official_params), set(independent_params))
            for key, parameter in official_params.items():
                other = independent_params[key]
                self.assertEqual(tuple(parameter.shape), tuple(other.shape), key)
                self.assertEqual(parameter.dtype, other.dtype, key)
                self.assertTrue(torch.equal(parameter, other), key)
        return adapter, official, independent

    def test_semamba_cross_implementation_optimizer_needs_name_remap(self):
        _, official_modules, independent_modules = self._semamba_pair()
        official = official_modules["generator"]
        independent = independent_modules["generator"]
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
            self.assertTrue(torch.equal(
                named.state[by_name[name]]["exp_avg_sq"], source.state[dict(official.named_parameters())[name]]["exp_avg_sq"]
            ), name)

    def test_semamba_checkpoint_state_transfers_both_ways(self):
        from src.training.core.rng import capture_rng, restore_rng
        adapter, official, independent = self._semamba_pair()
        official_opt = adapter.optimizers(official, adapter.lr)
        independent_opt = adapter.optimizers(independent, adapter.lr)
        official_sched = adapter.schedulers(official_opt)
        independent_sched = adapter.schedulers(independent_opt)
        torch.manual_seed(9)
        for name in official:
            by_official = dict(official[name].named_parameters())
            by_independent = dict(independent[name].named_parameters())
            for key, parameter in by_official.items():
                gradient = torch.randn_like(parameter)
                parameter.grad = gradient
                by_independent[key].grad = gradient.clone()
        for opt in (official_opt, independent_opt):
            for optimizer in opt.values():
                optimizer.step()
        for schedulers in (official_sched, independent_sched):
            for scheduler in schedulers.values():
                scheduler.step()
        groups = {
            "generator": ("generator",),
            "discriminator": ("discriminator_band", "discriminator_mrd"),
        }

        def names(modules, keys):
            return [f"{module}.{name}" for module in keys for name, _ in modules[module].named_parameters()]

        def moments(modules, optimizers):
            found = {}
            for module_name, module in modules.items():
                optimizer = optimizers["generator" if module_name == "generator" else "discriminator"]
                for name, parameter in module.named_parameters():
                    state = optimizer.state[parameter]
                    found[f"{module_name}.{name}"] = (
                        state["exp_avg"].detach().clone(),
                        state["exp_avg_sq"].detach().clone(),
                        int(state["step"]),
                    )
            return found

        official_moments = moments(official, official_opt)
        independent_moments = moments(independent, independent_opt)
        self.assertEqual(set(official_moments), set(independent_moments))
        for name, (avg, square, step) in official_moments.items():
            other_avg, other_square, other_step = independent_moments[name]
            self.assertTrue(torch.equal(avg, other_avg), name)
            self.assertTrue(torch.equal(square, other_square), name)
            self.assertEqual(step, other_step, name)

        def transplant(source_modules, source_opt, dest_modules):
            rebuilt = adapter.optimizers(dest_modules, adapter.lr)
            for opt_name, module_names in groups.items():
                state = source_opt[opt_name].state_dict()
                remap_optimizer(state, names(source_modules, module_names), names(dest_modules, module_names))
                rebuilt[opt_name].load_state_dict(state)
            return rebuilt

        fresh_official = adapter.build("official", torch.device("cpu"))
        fresh_independent = adapter.build("independent", torch.device("cpu"))
        for name, module in official.items():
            fresh_official[name].load_state_dict(module.state_dict(), strict=True)
            fresh_independent[name].load_state_dict(module.state_dict(), strict=True)
        transfers = {
            "official_to_official": transplant(official, official_opt, fresh_official),
            "independent_to_independent": transplant(independent, independent_opt, fresh_independent),
            "official_to_independent": transplant(official, official_opt, fresh_independent),
            "independent_to_official": transplant(independent, independent_opt, fresh_official),
        }
        expected = {
            "official_to_official": official_moments,
            "independent_to_independent": independent_moments,
            "official_to_independent": official_moments,
            "independent_to_official": independent_moments,
        }
        hosts = {
            "official_to_official": fresh_official,
            "independent_to_independent": fresh_independent,
            "official_to_independent": fresh_independent,
            "independent_to_official": fresh_official,
        }
        for label, restored in transfers.items():
            got = moments(hosts[label], restored)
            for name, (avg, square, step) in expected[label].items():
                other_avg, other_square, other_step = got[name]
                self.assertTrue(torch.equal(avg, other_avg), f"{label} {name}")
                self.assertTrue(torch.equal(square, other_square), f"{label} {name}")
                self.assertEqual(step, other_step, f"{label} {name}")
            for opt_name, scheduler in adapter.schedulers(restored).items():
                scheduler.load_state_dict(official_sched[opt_name].state_dict())
                self.assertEqual(scheduler.state_dict()["last_epoch"], official_sched[opt_name].state_dict()["last_epoch"])
                self.assertEqual(scheduler.get_last_lr(), official_sched[opt_name].get_last_lr())
        before = capture_rng(cuda=False)
        torch.manual_seed(123)
        random_draw = torch.rand(4)
        restore_rng(before, cuda=False)
        self.assertTrue(torch.equal(capture_rng(cuda=False)["torch"], before["torch"]))
        self.assertFalse(torch.equal(random_draw, torch.rand(4)))
