import math
import unittest

from src.experiments.stopping import MetricEvent, PatienceStopPolicy


class PatienceTests(unittest.TestCase):
    def event(self, value, step=1, source="validation"):
        return MetricEvent(source, step, {"loss": value})

    def test_min_delta_patience_and_restore(self):
        policy = PatienceStopPolicy("validation.loss", "min", 2, min_delta=0.5, start_after=1)
        policy.observe(self.event(100))
        policy.observe(self.event(3, 2))
        policy.observe(self.event(2.5, 3))  # equality to delta is not improvement
        saved = policy.state_dict()
        resumed = PatienceStopPolicy("validation.loss", "min", 2, min_delta=0.5, start_after=1)
        resumed.load_state_dict(saved)
        self.assertTrue(resumed.observe(self.event(3, 4)).stop)
        self.assertEqual(resumed.best, 3)
        self.assertEqual(resumed.best_step, 2)
        self.assertEqual(resumed.observations, 4)

    def test_max_and_reset(self):
        policy = PatienceStopPolicy("validation.loss", "max", 2)
        policy.observe(self.event(1))
        policy.observe(self.event(1, 2))
        policy.observe(self.event(2, 3))
        self.assertEqual(policy.bad_observations, 0)
        self.assertFalse(policy.observe(self.event(1, 4)).stop)
        self.assertTrue(policy.observe(self.event(1, 5)).stop)

    def test_unrelated_and_invalid(self):
        policy = PatienceStopPolicy("validation.loss", "min", 1)
        policy.observe(self.event(0, source="train"))
        policy.observe(MetricEvent("validation", 1, {"pesq": 3}, expected_metrics=frozenset({"pesq"})))
        self.assertEqual(policy.observations, 0)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                policy.observe(self.event(value))
        with self.assertRaises(ValueError):
            policy.observe(MetricEvent("validation", 1, {}))
        for kwargs in ({"metric": "loss", "mode": "min", "patience": 1},
                       {"metric": "validation.loss", "mode": "bad", "patience": 1},
                       {"metric": "validation.loss", "mode": "min", "patience": 0},
                       {"metric": "validation.loss", "mode": "min", "patience": 1, "min_delta": math.nan}):
            with self.assertRaises(ValueError):
                PatienceStopPolicy(**kwargs)


class StoppingIntegrationTests(unittest.TestCase):
    def _config(self, directory, *, stop_after=None, distribute="none"):
        from tests.experiments.test_tf_gridnet_training import _scripted_yaml
        import yaml
        path = _scripted_yaml(directory, steps=5, updates=1, stop_after=stop_after, distribute=distribute)
        doc = yaml.safe_load(path.read_text())
        doc['components']['early_stop'] = {
            'role': 'stopping', 'class': 'src.experiments.stopping.PatienceStopPolicy',
            'kwargs': {'metric': 'validation.loss', 'mode': 'min', 'patience': 2, 'min_delta': 1e8}}
        doc['checkpoint'] = {'selector': {'class': 'src.experiments.selectors.TopKSelector',
                                          'kwargs': {'metric': 'loss', 'mode': 'min', 'k': 2}}}
        path.write_text(yaml.safe_dump(doc))
        return path

    def test_real_validation_resume_and_selector(self):
        import json
        import tempfile
        from pathlib import Path
        import torch
        import yaml
        from src.experiments.api import build_experiment
        from src.experiments.config import load_experiment
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._config(root / 'full')
            partial = self._config(root / 'resumed', stop_after=1)
            result = build_experiment(load_experiment(full)).run()
            self.assertEqual((result.stop_reason, result.logical_step), ('early_stopping', 3))
            interrupted = build_experiment(load_experiment(partial)).run()
            self.assertEqual(interrupted.stop_reason, 'interrupted')
            doc = yaml.safe_load(partial.read_text())
            del doc['components']['trainer']['kwargs']['stop_after_logical_steps']
            partial.write_text(yaml.safe_dump(doc))
            resumed = build_experiment(load_experiment(partial)).run(resume=root / 'resumed')
            self.assertEqual((resumed.stop_reason, resumed.logical_step), ('early_stopping', 3))
            a = torch.load(root / 'full' / 'latest.pt', map_location='cpu', weights_only=False)
            b = torch.load(root / 'resumed' / 'latest.pt', map_location='cpu', weights_only=False)
            for name in ('early_stop', 'model', 'optimizer', 'scheduler', 'data'):
                self._compare(a['states'][name], b['states'][name])
            self.assertEqual(a['states']['early_stop']['policy']['observations'], 3)
            self.assertTrue(list((root / 'resumed' / 'archive').glob('*.pt')))
            complete = json.loads((root / 'resumed' / 'complete.json').read_text())
            self.assertEqual(complete['reason'], 'early_stopping')
            self.assertEqual(complete['stopping']['policy']['bad_observations'], 2)

    def _compare(self, a, b):
        import torch
        import numpy as np
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self._compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self._compare(left, right)
        else:
            self.assertEqual(a, b)

    def test_gloo_launcher_and_resume(self):
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        import torch
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._config(root / "full", distribute="ddp")
            partial = self._config(root / "resumed", stop_after=1, distribute="ddp")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
            def launch(path, *args):
                return subprocess.run([sys.executable, "-m", "src.experiments.launcher",
                    "--config", str(path), "--nproc", "2", "--gpu-memory-interval", "0", *args],
                    env=env, capture_output=True, text=True, timeout=180)
            first = launch(full)
            self.assertEqual(first.returncode, 0, first.stderr)
            stopped = launch(partial)
            self.assertEqual(stopped.returncode, 3, stopped.stderr)
            doc = yaml.safe_load(partial.read_text())
            del doc["components"]["trainer"]["kwargs"]["stop_after_logical_steps"]
            partial.write_text(yaml.safe_dump(doc))
            resumed = launch(partial, "--resume", str(root / "resumed"))
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            a = torch.load(root / "full" / "latest.pt", map_location="cpu", weights_only=False)
            b = torch.load(root / "resumed" / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(a["logical_step"], 3)
            self.assertEqual(b["logical_step"], 3)
            for name in ("early_stop", "model", "optimizer", "scheduler", "data"):
                self._compare(a["states"][name], b["states"][name])
            self._compare(a["rng_by_rank"], b["rng_by_rank"])
            status = json.loads((root / "resumed" / "status.json").read_text())
            self.assertEqual((status["status"], status["reason"]), ("COMPLETED", "early_stopping"))

    def test_resume_after_completed_boundary_and_changed_config(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        import yaml
        from src.experiments.api import build_experiment
        from src.experiments.config import load_experiment
        from src.experiments.errors import CheckpointError
        from src.experiments.official.tf_gridnet.trainer import TFGridNetTrainer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._config(root / 'full')
            partial = self._config(root / 'resumed')
            build_experiment(load_experiment(full)).run()
            def interrupt_after_boundary(trainer, runtime):
                return trainer.logical_step == 1 and trainer.phase == 'in_epoch'
            with patch.object(TFGridNetTrainer, '_stop_requested', interrupt_after_boundary):
                result = build_experiment(load_experiment(partial)).run()
            self.assertEqual((result.stop_reason, result.logical_step), ('interrupted', 1))
            pending = torch.load(root / 'resumed' / 'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(pending['states']['early_stop']['policy']['observations'], 1)
            doc = yaml.safe_load(partial.read_text())
            doc['components']['early_stop']['kwargs']['patience'] = 3
            partial.write_text(yaml.safe_dump(doc))
            with self.assertRaises(CheckpointError):
                build_experiment(load_experiment(partial)).run(resume=root / 'resumed')
            doc['components']['early_stop']['kwargs']['patience'] = 2
            partial.write_text(yaml.safe_dump(doc))
            result = build_experiment(load_experiment(partial)).run(resume=root / 'resumed')
            self.assertEqual((result.stop_reason, result.logical_step), ('early_stopping', 3))
            a = torch.load(root / 'full' / 'latest.pt', map_location='cpu', weights_only=False)
            b = torch.load(root / 'resumed' / 'latest.pt', map_location='cpu', weights_only=False)
            for name in ('early_stop', 'model', 'optimizer', 'scheduler', 'data'):
                self._compare(a['states'][name], b['states'][name])
            self._compare(a['rng_by_rank'], b['rng_by_rank'])

    def test_rejects_adding_policy_to_old_checkpoint_and_multiple_policies(self):
        import tempfile
        from pathlib import Path
        import yaml
        from src.experiments.api import build_experiment
        from src.experiments.config import load_experiment
        from src.experiments.errors import BuildError, StateError
        from tests.experiments.test_tf_gridnet_training import _scripted_yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _scripted_yaml(root / 'old', steps=5, updates=1, stop_after=1)
            self.assertEqual(build_experiment(load_experiment(path)).run().stop_reason, 'interrupted')
            doc = yaml.safe_load(path.read_text())
            policy = {'role': 'stopping', 'class': 'src.experiments.stopping.PatienceStopPolicy',
                      'kwargs': {'metric': 'validation.loss', 'mode': 'min', 'patience': 2}}
            doc['components']['early_stop'] = policy
            path.write_text(yaml.safe_dump(doc))
            with self.assertRaises(StateError):
                build_experiment(load_experiment(path)).run(resume=root / 'old')
            doc['components']['second_stop'] = policy
            path.write_text(yaml.safe_dump(doc))
            with self.assertRaises(BuildError):
                build_experiment(load_experiment(path))

    def test_rejects_unsupported_metric_source(self):
        from src.experiments.stopping import TrainingControl
        class Runtime:
            world_size = 1
        with self.assertRaisesRegex(ValueError, 'only validation'):
            TrainingControl(PatienceStopPolicy('train.loss', 'min', 2), Runtime())

    def test_rank_disagreement_fails(self):
        from src.experiments.stopping import TrainingControl
        class Runtime:
            world_size = 2
            def gather_objects(self, item):
                other = (item[0], {**item[1], 'stopped': True})
                return [item, other]
        control = TrainingControl(PatienceStopPolicy('validation.loss', 'min', 2), Runtime())
        with self.assertRaisesRegex(RuntimeError, 'ranks disagree'):
            control.observe(MetricEvent('validation', 1, {'loss': 2.0}))

    def test_standard_trainer_monitors_training_metric(self):
        import tempfile
        from pathlib import Path
        import json
        import yaml
        from src.experiments.api import build_experiment
        from src.experiments.config import load_experiment
        from tests.experiments.test_minimal import _config_text
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _config_text(root, 5)
            doc = yaml.safe_load(path.read_text())
            doc['components']['early_stop'] = {
                'role': 'stopping', 'class': 'src.experiments.stopping.PatienceStopPolicy',
                'kwargs': {'metric': 'train.mean_rank_loss', 'mode': 'min', 'patience': 1, 'min_delta': 1e8}}
            path.write_text(yaml.safe_dump(doc))
            result = build_experiment(load_experiment(path)).run()
            self.assertEqual((result.stop_reason, result.logical_step), ('early_stopping', 2))
            record = json.loads((root / 'out' / 'complete.json').read_text())
            self.assertEqual(record['stopping']['last_event']['source'], 'train')

    def test_official_claim_is_refused(self):
        import tempfile
        from pathlib import Path
        import yaml
        from src.experiments.api import build_experiment
        from src.experiments.config import load_experiment
        from src.experiments.errors import BuildError
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(Path(tmp) / 'claim')
            doc = yaml.safe_load(path.read_text())
            doc['components']['trainer']['kwargs']['experiment_claim'] = 'official_faithful'
            path.write_text(yaml.safe_dump(doc))
            with self.assertRaisesRegex(BuildError, 'official_faithful'):
                build_experiment(load_experiment(path))


    def test_locoformer_validation_policy(self):
        import tempfile
        from pathlib import Path
        import yaml
        from src.experiments.api import build_experiment
        from src.experiments.config import load_experiment
        from tests.experiments.test_tf_locoformer_training import _scripted_yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'loco'
            path = _scripted_yaml(root, steps=4, updates=1)
            doc = yaml.safe_load(path.read_text())
            doc['components']['early_stop'] = {
                'role': 'stopping', 'class': 'src.experiments.stopping.PatienceStopPolicy',
                'kwargs': {'metric': 'validation.loss', 'mode': 'min', 'patience': 1, 'min_delta': 1e8}}
            path.write_text(yaml.safe_dump(doc))
            result = build_experiment(load_experiment(path)).run()
            self.assertEqual((result.stop_reason, result.logical_step), ('early_stopping', 2))


if __name__ == "__main__":
    unittest.main()
