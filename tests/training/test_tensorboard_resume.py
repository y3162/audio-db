"""Small real Trainer checks for TensorBoard continuation and scalar rebuild."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import torch
from src.training.core.batch import plan_batches
from src.training.core.rng import capture_rng
from src.training.core.tensorboard import scalars
from src.training.core.trainer import Trainer
from tests.training.mock_task import LinearPairTask
from tests.training.support import cpu_process, paired_dataset, tiny_config
from tests.training.test_mock_trainer import _assert_exact_state


@unittest.skipUnless(importlib.util.find_spec('tensorboard'), 'tensorboard not installed')
class ResumeTest(unittest.TestCase):
    def test_preview_failure_preserves_training_state(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory(prefix='tb-preview-failure-') as tmp:
            root = Path(tmp)
            data = paired_dataset(root / 'train', 8, frames=320, segment_size=80, seed=7)
            valid = paired_dataset(root / 'valid', 1, frames=320, segment_size=80, seed=8)
            results = []
            for enabled in (False, True):
                config = tiny_config(root / str(enabled), max_steps=2, seed=7, global_batch_size=4,
                                     micro_batch_size_per_gpu=2, loss='mock')
                config['logging'] = {'tensorboard': {'enabled': enabled, 'scalar_interval': 1,
                    'audio': {'enabled': True, 'max_cases': 1, 'max_seconds': 1}, 'spectrogram': True}}
                task = LinearPairTask(config, torch.device('cpu'))
                metric = {'status': 'ok', 'value': 0.5, 'reason': None}
                task.validation_report = lambda batch: {'losses': {'generator': metric}, 'pesq': metric, 'stoi': metric}
                trainer = Trainer(task, plan_batches(4, 2, 1), cpu_process(), max_steps=2, seed=7,
                                  checkpoint_dir=root / str(enabled), checkpoint_interval=0,
                                  validation_interval=2, config=config)
                trainer.fit(data, valid)
                results.append((task.state_dict(), capture_rng(cuda=False)))
            _assert_exact_state(results[0][0], results[1][0], 'task')
            self.assertEqual(results[0][1]['python'], results[1][1]['python'])
            self.assertTrue(torch.equal(results[0][1]['torch'], results[1][1]['torch']))
            self.assertTrue((results[0][1]['numpy'][1] == results[1][1]['numpy'][1]).all())
            self.assertEqual(results[0][1]['numpy'][2:], results[1][1]['numpy'][2:])
            events = EventAccumulator(str(root / 'True' / 'tensorboard')).Reload()
            errors = [tag for tag in events.Tags()['tensors'] if tag.startswith('preview_errors/')]
            self.assertEqual(len(errors), 1)
            self.assertIn(b"spec", events.Tensors(errors[0])[0].tensor_proto.string_val[0])

    def test_continue_rewind_and_rebuild(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory(prefix='tb-resume-') as tmp:
            root = Path(tmp)
            train = paired_dataset(root / 'train', 8, frames=320, segment_size=80, seed=7)
            valid = paired_dataset(root / 'valid', 1, frames=320, segment_size=80, seed=8)
            def make(name):
                directory = root / name
                config = tiny_config(directory, max_steps=4, seed=7, global_batch_size=4,
                                     micro_batch_size_per_gpu=2, loss='mock')
                config['logging'] = {'tensorboard': {'enabled': True, 'scalar_interval': 1,
                    'audio': {'enabled': False}, 'spectrogram': False}}
                task = LinearPairTask(config, torch.device('cpu'))
                metric = {'status': 'ok', 'value': 0.5, 'reason': None}
                task.validation_report = lambda batch: {'losses': {'generator': metric}, 'pesq': metric, 'stoi': metric}
                trainer = Trainer(task, plan_batches(4, 2, 1), cpu_process(), max_steps=4, seed=7,
                                  checkpoint_dir=directory, checkpoint_interval=1, validation_interval=1, config=config)
                return task, trainer
            full_task, full = make('full')
            full.fit(train, valid)
            full_rng = capture_rng(cuda=False)
            _, first = make('split')
            first.fit(train, valid, segment_steps=2)
            checkpoint = Path(first.last_checkpoint)
            _, second = make('split')
            second.fit(train, valid, resume=checkpoint, segment_steps=1)
            # Roll back from the still unfinished step 3 to step 2.
            task, third = make('split')
            third.fit(train, valid, resume=checkpoint)
            _assert_exact_state(full_task.state_dict(), task.state_dict(), 'task')
            state = capture_rng(cuda=False)
            self.assertEqual(full_rng['python'], state['python'])
            self.assertTrue(torch.equal(full_rng['torch'], state['torch']))
            self.assertTrue((full_rng['numpy'][1] == state['numpy'][1]).all())
            self.assertEqual(full_rng['numpy'][2:], state['numpy'][2:])
            source = root / 'split' / 'metrics.jsonl'
            records = [json.loads(line) for line in source.read_text().splitlines()]
            expected = defaultdict(list)
            for record in records:
                for tag, value, step in scalars(record):
                    expected[tag].append((step, value))
            for kind in ('train', 'validation'):
                self.assertEqual([r['global_step'] for r in records if r['type'] == kind], [1, 2, 3, 4])
            def check(directory):
                events = EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()
                for tag, values in expected.items():
                    actual = events.Scalars(tag)
                    self.assertEqual([v.step for v in actual], [s for s, _ in values], tag)
                    for event, (_, value) in zip(actual, values):
                        self.assertAlmostEqual(event.value, value, delta=max(1e-7, abs(value)*1e-6), msg=tag)
            check(root / 'split' / 'tensorboard')
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, checkpoint)}
            destination = root / 'rebuild'
            command = [sys.executable, '-m', 'src.training.core.tensorboard', str(source), str(destination)]
            subprocess.run(command, check=True, capture_output=True)
            check(destination)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})

    def test_interruption_at_checkpoint_and_jsonl_boundaries(self):
        from unittest.mock import patch
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from src.training.core import tensorboard as tb
        for boundary in ('after_checkpoint', 'after_jsonl', 'after_event_flush'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(prefix='tb-fault-') as tmp:
                root = Path(tmp)
                data = paired_dataset(root / 'wav', 8, frames=320, segment_size=80, seed=7)
                config = tiny_config(root / 'ckpt', max_steps=3, seed=7, global_batch_size=4,
                                     micro_batch_size_per_gpu=2, loss='mock')
                config['logging'] = {'tensorboard': {'enabled': True, 'scalar_interval': 1,
                    'audio': {'enabled': False}, 'spectrogram': False}}
                def make():
                    task = LinearPairTask(config, torch.device('cpu'))
                    return Trainer(task, plan_batches(4, 2, 1), cpu_process(), max_steps=3, seed=7,
                        checkpoint_dir=root / 'ckpt', checkpoint_interval=1, validation_interval=0, config=config)
                first = make()
                first.fit(data, segment_steps=1)
                interrupted = make()
                if boundary == 'after_checkpoint':
                    original = interrupted._save
                    def save_then_interrupt(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError('injected interruption')
                    patcher = patch.object(interrupted, '_save', side_effect=save_then_interrupt)
                    checkpoint = root / 'ckpt' / 'step_00000002.pt'
                elif boundary == 'after_jsonl':
                    patcher = patch.object(tb, 'write_scalars', side_effect=RuntimeError('injected interruption'))
                    checkpoint = root / 'ckpt' / 'step_00000001.pt'
                else:
                    patcher = patch.object(interrupted, '_save', side_effect=RuntimeError('injected interruption'))
                    checkpoint = root / 'ckpt' / 'step_00000001.pt'
                with patcher, self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                    interrupted.fit(data, resume=Path(first.last_checkpoint))
                self.assertTrue(checkpoint.is_file())
                if boundary != 'after_checkpoint':
                    self.assertFalse((root / 'ckpt' / 'step_00000002.pt').exists())
                make().fit(data, resume=checkpoint)
                records = [json.loads(x) for x in (root / 'ckpt' / 'metrics.jsonl').read_text().splitlines()]
                self.assertEqual([r['global_step'] for r in records if r['type'] == 'train'], [1, 2, 3])
                events = EventAccumulator(str(root / 'ckpt' / 'tensorboard')).Reload()
                self.assertEqual([e.step for e in events.Scalars('train/loss/generator')], [1, 2, 3])
