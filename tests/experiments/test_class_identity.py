"""Checkpoint policy identities and their public import paths."""
import unittest
from unittest import mock

from src.experiments.core.identity import policy_identity
from src.experiments.errors import CheckpointError
from src.experiments.execution.selector import StandardSelector as Implementation
from src.experiments.selectors import StandardSelector
from src.experiments.stopping import PatienceStopPolicy, TrainingControl


class IdentityTests(unittest.TestCase):
    def test_public_class_is_the_implementation_and_id_is_stable(self):
        self.assertIs(StandardSelector, Implementation)
        instance = StandardSelector()
        before = policy_identity(instance)
        with mock.patch.object(StandardSelector, '__module__', 'moved.internal'):
            self.assertEqual(policy_identity(instance), before)
        self.assertEqual(before, 'src.experiments.selectors.StandardSelector')

    def test_selector_public_exports_are_explicit(self):
        import src.experiments.selectors as public
        self.assertEqual(set(public.__all__), {
            'DEFAULT_LATEST_EVERY', 'EVENTS', 'Decision', 'PeriodicArchiveSelector',
            'SavePoint', 'Selector', 'StandardSelector', 'TopKSelector',
            'build_selector', 'scalar_metrics', 'selector_identity',
        })
        self.assertFalse(hasattr(public, 'math'))

    def test_duplicate_and_inherited_and_invalid_ids_are_rejected(self):
        class One:
            checkpoint_class_id = 'tests.unique.identity'
        class Two:
            checkpoint_class_id = 'tests.unique.identity'
        class Derived(One):
            pass
        class Bad:
            checkpoint_class_id = 'invalid'
        class BadPath:
            checkpoint_class_id = 'invalid..path'
        policy_identity(One())
        for obj in (Two(), Derived(), Bad(), BadPath()):
            with self.assertRaises(CheckpointError):
                policy_identity(obj)

    def test_rank_disagreement_on_selector_settings_is_rejected(self):
        from src.experiments.execution.checkpoint import _assert_replicated_states

        class States:
            def ids(self):
                return ()

        class Runtime:
            world_size = 2
            def gather_objects(self, local):
                other = dict(local)
                other['checkpoint.selector'] = dict(local['checkpoint.selector'], kwargs={'latest_every': 2})
                return [local, other]

        with self.assertRaisesRegex(CheckpointError, 'ranks disagree'):
            _assert_replicated_states(States(), StandardSelector(), Runtime(), {'latest_every': 1})

    def test_stop_control_rejects_another_identity(self):
        control = TrainingControl(PatienceStopPolicy('validation.loss', 'min', 2), object())
        state = control.state_dict()
        state['policy_class'] = 'tests.other.Policy'
        with self.assertRaisesRegex(ValueError, 'class'):
            control.load_state_dict(state)


class ConfigClassTests(unittest.TestCase):
    def test_experiment_config_paths_resolve_to_classes(self):
        from pathlib import Path
        from src.experiments.config import load_experiment
        from src.experiments.core.factory import ComponentFactory

        root = Path(__file__).resolve().parents[2]
        paths = []
        for directory in ('configs/reproduction/experiments', 'tests/fixtures/configs/experiments'):
            paths.extend((root / directory).rglob('*.yaml'))
        self.assertTrue(paths)
        resolver = ComponentFactory.resolve
        for path in paths:
            with self.subTest(path=str(path)):
                config = load_experiment(path)
                classes = [config.runtime_class, config.selector_class]
                classes.extend(spec.class_path for spec in config.components.values() if spec.class_path)
                for name in classes:
                    self.assertIsInstance(resolver(name), type, name)

    def test_canonical_config_hash_is_stable(self):
        from pathlib import Path
        from src.experiments.config import config_sha256, load_experiment

        path = Path(__file__).resolve().parents[2] / 'tests/fixtures/configs/experiments/linear_supervised.yaml'
        self.assertEqual(config_sha256(load_experiment(path)),
                         '9a5e82f736c6e6f56e2b9181fd9ea434c5a7c689b43fa7d9b4decef02e6193d9')
