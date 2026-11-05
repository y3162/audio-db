"""CPU two-update checks for monitoring side effects on real training tasks."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.core.batch import plan_batches
from src.training.core.rng import capture_rng
from src.training.core.trainer import Trainer
from tests.training.support import cpu_process, paired_dataset
from tests.training.tensorboard_models_ddp_worker import case
from tests.training.test_mock_trainer import _assert_exact_state


@unittest.skipUnless(importlib.util.find_spec('tensorboard'), 'tensorboard not installed')
class TensorBoardEquivalenceTest(unittest.TestCase):
    def test_real_models_two_updates_and_validation(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        names = ('mp_senet', 'mp_senet_official', 'tf_gridnet', 'tf_locoformer',
                 'semambapp', 'deepfilternet3', 'mossformergan')
        with tempfile.TemporaryDirectory(prefix='tb-equivalence-') as tmp:
            root = Path(tmp)
            for name in names:
                with self.subTest(model=name):
                    cls, config, length = case(name, root / name)
                    config.update(global_batch_size=1, micro_batch_size_per_gpu=1, max_steps=2)
                    train = paired_dataset(root / name / 'train', 2, frames=length, segment_size=length, seed=7)
                    valid = paired_dataset(root / name / 'valid', 1, frames=length, segment_size=length, seed=8)
                    trials = []
                    for enabled in (False, True):
                        conf = copy.deepcopy(config)
                        conf['logging'] = {'tensorboard': {'enabled': enabled, 'scalar_interval': 1,
                            'audio': {'enabled': True, 'max_cases': 1, 'max_seconds': 1}, 'spectrogram': True}}
                        task = cls(conf, torch.device('cpu'))
                        initial = copy.deepcopy(task.state_dict())
                        snapshots = []
                        original = task.train_update
                        def update(*args, **kwargs):
                            result = original(*args, **kwargs)
                            snapshots.append((copy.deepcopy(task.state_dict()), copy.deepcopy(capture_rng(cuda=False))))
                            return result
                        task.train_update = update
                        trainer = Trainer(task, plan_batches(1, 1, 1), cpu_process(), max_steps=2,
                            seed=7, checkpoint_dir=root / name / str(enabled), checkpoint_interval=0,
                            validation_interval=2, config=conf)
                        trainer.fit(train, valid)
                        snapshots.append((copy.deepcopy(task.state_dict()), copy.deepcopy(capture_rng(cuda=False))))
                        records = [json.loads(x) for x in (root / name / str(enabled) / 'metrics.jsonl').read_text().splitlines()]
                        trials.append((snapshots, records, initial))
                    _assert_exact_state(trials[0][2], trials[1][2], name + '.initial')
                    for left, right in zip(trials[0][0], trials[1][0]):
                        _assert_exact_state(left[0], right[0], name)
                        self.assertEqual(left[1]['python'], right[1]['python'])
                        self.assertEqual(left[1]['numpy'][0], right[1]['numpy'][0])
                        self.assertTrue((left[1]['numpy'][1] == right[1]['numpy'][1]).all())
                        self.assertEqual(left[1]['numpy'][2:], right[1]['numpy'][2:])
                        self.assertTrue(torch.equal(left[1]['torch'], right[1]['torch']))
                    for kind in ('train', 'validation'):
                        a = [r['losses'] for r in trials[0][1] if r['type'] == kind]
                        b = [r['losses'] for r in trials[1][1] if r['type'] == kind]
                        self.assertEqual(a, b)
                    self.assertEqual(
                        [r['case_ids'] for r in trials[0][1] if r['type'] == 'train'],
                        [r['case_ids'] for r in trials[1][1] if r['type'] == 'train'])
                    for metric in ('pesq', 'stoi', 'si_sdr', 'si_snr'):
                        self.assertEqual(
                            [r.get(metric) for r in trials[0][1] if r['type'] == 'validation'],
                            [r.get(metric) for r in trials[1][1] if r['type'] == 'validation'])
                    events = EventAccumulator(str(root / name / 'True' / 'tensorboard')).Reload()
                    self.assertFalse([t for t in events.Tags()['tensors'] if t.startswith('preview_errors/')])
                    self.assertEqual(len(events.Tags()['audio']), 3)
                    self.assertEqual(len(events.Tags()['images']), 1)
                    print(f'PASS equivalence {name}', flush=True)
