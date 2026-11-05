"""Small real-model Trainer -> JSONL -> TensorBoard integration checks."""
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None

from src.training.core.batch import plan_batches
from src.training.core.distributed import ProcessInfo
from src.training.core.trainer import Trainer
from src.training.se.deepfilternet import DeepFilterNetTask
from src.training.se.mossformergan import MossFormerGANTask
from src.training.se.semambapp import SEMambaTask
from src.training.se.tf_gridnet import TFGridNetTask
from src.training.se.tf_locoformer import TFLocoformerTask
from tests.se.test_tf_locoformer import TINY
from tests.training.support import paired_dataset
from tests.training.test_model_tasks import _base


def model_cases():
    grid = dict(n_fft=32, stride=16, n_layers=1, lstm_hidden_units=8, attn_n_head=1,
                attn_approx_qk_dim=8, emb_dim=8, emb_ks=1, emb_hs=1)
    semamba = dict(hid_feature=8, num_tfmamba=1, d_state=4, d_conv=2, expand=2, n_fft=400)
    df = dict(sr=16000, fft_size=128, hop_size=64, nb_erb=16, nb_df=16, conv_ch=16,
              emb_hidden_dim=32, df_hidden_dim=32, enc_lin_groups=8, lin_groups=8,
              emb_num_layers=2, df_num_layers=1, df_order=5, conv_lookahead=0, df_lookahead=0)
    return {
        'tf_gridnet': (TFGridNetTask, dict(variant='ms-snsd', architecture=grid), 800),
        'tf_locoformer': (TFLocoformerTask, dict(variant='dns', architecture=dict(TINY)), 512),
        'semambapp': (SEMambaTask, dict(architecture=semamba, stft={'hop': 100, 'win': 400, 'disc_windows': [256, 512]}), 2048),
        'deepfilternet3': (DeepFilterNetTask, dict(architecture=df, grad_clip=1.0), 1024),
        'mossformergan': (MossFormerGANTask, dict(variant='se_16k', architecture={'n_fft': 400, 'win_len': 400, 'hop': 100, 'num_channel': 64}, grad_clip=10.0), 8000),
    }


def check_model(name, device_name):
    cls, params, length = model_cases()[name]
    with tempfile.TemporaryDirectory(prefix='audio-db-tb-test-') as tmp:
        root = Path(tmp)
        config = _base(root, name, segment_size=length, **params)
        config['logging'] = {'tensorboard': {'enabled': True, 'scalar_interval': 1,
            'audio': {'enabled': True, 'max_cases': 1, 'max_seconds': 1}, 'spectrogram': True}}
        train = paired_dataset(root / 'train', 1, frames=length, segment_size=length, seed=config['seed'])
        valid = paired_dataset(root / 'valid', 1, frames=length, segment_size=length, seed=config['seed'] + 1)
        device = torch.device(device_name)
        process = ProcessInfo(0, 1, 0, device, False)
        task = cls(config, device)
        trainer = Trainer(task, plan_batches(1, 1, 1), process, max_steps=1, seed=config['seed'],
                          checkpoint_dir=root / 'ckpt', checkpoint_interval=0, validation_interval=1, config=config)
        trainer.fit(train, valid)
        records = [json.loads(line) for line in (root / 'ckpt' / 'metrics.jsonl').read_text().splitlines()]
        train_record = next(r for r in records if r['type'] == 'train')
        valid_record = next(r for r in records if r['type'] == 'validation')
        events = EventAccumulator(str(root / 'ckpt' / 'tensorboard')).Reload()
        for key, value in train_record['losses'].items():
            logged = events.Scalars('train/loss/' + key)
            assert len(logged) == 1 and logged[0].step == 1 and math.isclose(logged[0].value, value, rel_tol=1e-6, abs_tol=1e-7), (name, key, logged, value)
        for key, value in valid_record['losses'].items():
            tag = 'validation/loss/' + key
            if value['status'] == 'ok' and value['value'] is not None and math.isfinite(value['value']):
                logged = events.Scalars(tag)
                assert len(logged) == 1 and logged[0].step == 1 and math.isclose(logged[0].value, value['value'], rel_tol=1e-6, abs_tol=1e-7), (name, key)
            else:
                assert tag not in events.Tags()['scalars'], (name, key)
        for key in ('pesq', 'stoi', 'si_sdr', 'si_snr'):
            value = valid_record.get(key)
            tag = 'validation/' + key
            if value and value.get('status') == 'ok' and value.get('value') is not None and math.isfinite(value['value']):
                logged = events.Scalars(tag)
                assert len(logged) == 1 and logged[0].step == 1 and math.isclose(logged[0].value, value['value'], rel_tol=1e-6, abs_tol=1e-7), (name, key)
            else:
                assert tag not in events.Tags()['scalars'], (name, key)
        for key, value in train_record['learning_rate'].items():
            logged = events.Scalars('learning_rate/after_update/' + key)
            assert len(logged) == 1 and logged[0].step == 1 and math.isclose(logged[0].value, value, rel_tol=1e-6, abs_tol=1e-8), (name, key)
        assert len(events.Tags()['audio']) == 3, (name, events.Tags())
        assert len(events.Tags()['images']) == 1, (name, events.Tags())
        errors = [x for x in events.Tags()['tensors'] if x.startswith('preview_errors/')]
        assert not errors, [(x, events.Tensors(x)[0].tensor_proto.string_val) for x in errors]
        if device.type == 'cuda':
            assert events.Scalars('performance/cuda_allocated_bytes_rank0')[0].value > 0
        print(f'PASS {name} device={device_name} losses={list(train_record["losses"])}', flush=True)


@unittest.skipIf(EventAccumulator is None, "tensorboard not installed")
class TensorBoardModelIntegration(unittest.TestCase):
    def test_models(self):
        names = os.environ.get('AUDIO_DB_TB_MODELS', ','.join(model_cases())).split(',')
        device = os.environ.get('AUDIO_DB_TB_DEVICE', 'cpu')
        for name in names:
            with self.subTest(model=name):
                check_model(name, device)


@unittest.skipIf(EventAccumulator is None, "tensorboard not installed")
class TensorBoardDDPIntegration(unittest.TestCase):
    def test_two_ranks_write_once(self):
        import sys
        from unittest.mock import patch
        from tests.training.support import run_distributed_workers
        root_dir = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='audio-db-tb-ddp-') as tmp:
            root = Path(tmp)
            paired_dataset(root / 'wav', 4, frames=320, segment_size=80, seed=7)
            # Preserve the caller's package paths without overriding the venv's packages.
            code = 'import sys,runpy; sys.path.extend(' + repr(sys.path) + '); runpy.run_path(' + repr(str(root_dir / 'tests/training/mock_ddp_worker.py')) + ', run_name="__main__")'
            def arguments(port):
                return [[sys.executable, '-c', code, str(rank), '2', str(port), str(root), str(root / 'state.pt'), '1'] for rank in range(2)]
            with patch.dict(os.environ, {'AUDIO_DB_TB_DDP_TEST': '1'}):
                run_distributed_workers(arguments, cwd=root_dir, timeout=120)
            events_dir = root / 'ckpt-0' / 'tensorboard'
            self.assertEqual(len(list(events_dir.glob('events.out.tfevents.*'))), 1)
            self.assertFalse((root / 'ckpt-1' / 'tensorboard').exists())
            events = EventAccumulator(str(events_dir)).Reload()
            for tag in ('train/loss/generator', 'train/loss/discriminator', 'learning_rate/after_update/generator', 'learning_rate/after_update/discriminator'):
                self.assertEqual([e.step for e in events.Scalars(tag)], [1])
            self.assertEqual(events.Scalars('progress/world_size')[0].value, 2)
            if os.environ.get('AUDIO_DB_TB_DDP_DEVICE') == 'cuda':
                self.assertGreater(events.Scalars('performance/cuda_allocated_bytes_rank0')[0].value, 0)


    def test_other_real_models_two_ranks_validation(self):
        import sys
        from tests.training.support import run_distributed_workers
        from tests.training.tensorboard_models_ddp_worker import case
        root_dir = Path(__file__).resolve().parents[2]
        names = os.environ.get('AUDIO_DB_TB_DDP_MODELS', 'mp_senet,mp_senet_official,tf_locoformer,semambapp,deepfilternet3,mossformergan').split(',')
        for name in names:
            with self.subTest(model=name), tempfile.TemporaryDirectory(prefix='audio-db-tb-model-ddp-') as tmp:
                root = Path(tmp)
                _, _, length = case(name, root)
                global_batch = int(os.environ.get('AUDIO_DB_TB_DDP_GLOBAL_BATCH', '2'))
                paired_dataset(root / 'train', 2 * global_batch, frames=length, segment_size=length, seed=7)
                paired_dataset(root / 'valid', 1, frames=length, segment_size=length, seed=8)
                code = 'import sys,runpy; sys.path.extend(' + repr(sys.path) + '); runpy.run_path(' + repr(str(root_dir / 'tests/training/tensorboard_models_ddp_worker.py')) + ', run_name="__main__")'
                def arguments(port):
                    return [[sys.executable, '-c', code, str(rank), str(port), str(root), name] for rank in range(2)]
                run_distributed_workers(arguments, cwd=root_dir, timeout=300)
                left = torch.load(root / 'parameters-0.pt', map_location='cpu', weights_only=True)
                right = torch.load(root / 'parameters-1.pt', map_location='cpu', weights_only=True)
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    self.assertTrue(torch.allclose(left[key], right[key], rtol=1e-6, atol=1e-7), (name, key, (left[key] - right[key]).abs().max().item()))
                directory = root / 'ckpt' / 'tensorboard'
                self.assertEqual(len(list(directory.glob('events.out.tfevents.*'))), 1)
                events = EventAccumulator(str(directory)).Reload()
                records = [json.loads(line) for line in (root / 'ckpt' / 'metrics.jsonl').read_text().splitlines()]
                train = [r for r in records if r['type'] == 'train']
                valid = next(r for r in records if r['type'] == 'validation')
                self.assertEqual([r['global_step'] for r in train], [1, 2])
                for key in train[0]['losses']:
                    values = events.Scalars('train/loss/' + key)
                    self.assertEqual([v.step for v in values], [1, 2], (name, key))
                    for event, record in zip(values, train):
                        self.assertAlmostEqual(event.value, record['losses'][key], delta=max(1e-7, abs(record['losses'][key])*1e-6))
                for key, metric in valid['losses'].items():
                    tag = 'validation/loss/' + key
                    if metric['status'] == 'ok' and metric['value'] is not None and math.isfinite(metric['value']):
                        values = events.Scalars(tag)
                        self.assertEqual([v.step for v in values], [2])
                        self.assertAlmostEqual(values[0].value, metric['value'], delta=max(1e-7, abs(metric['value'])*1e-6))
                    else:
                        self.assertNotIn(tag, events.Tags()['scalars'])
                for key in train[0]['learning_rate']:
                    values = events.Scalars('learning_rate/after_update/' + key)
                    self.assertEqual([v.step for v in values], [1, 2])
                    for event, record in zip(values, train):
                        self.assertAlmostEqual(event.value, record['learning_rate'][key], delta=max(1e-8, abs(record['learning_rate'][key])*1e-6))
                self.assertEqual(len(events.Tags()['audio']), 3)
                self.assertEqual(len(events.Tags()['images']), 1)
                self.assertFalse([tag for tag in events.Tags()['tensors'] if tag.startswith('preview_errors/')])
                print(f'PASS DDP {name} device={os.environ.get("AUDIO_DB_TB_DDP_DEVICE", "cpu")}', flush=True)

    def test_real_gridnet_two_ranks_validation(self):
        import sys
        from tests.training.support import run_distributed_workers
        root_dir = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='audio-db-tb-grid-ddp-') as tmp:
            root = Path(tmp)
            paired_dataset(root / 'train', 4, frames=800, segment_size=800, seed=7)
            paired_dataset(root / 'valid', 1, frames=800, segment_size=800, seed=8)
            code = 'import sys,runpy; sys.path.extend(' + repr(sys.path) + '); runpy.run_path(' + repr(str(root_dir / 'tests/training/tensorboard_gridnet_ddp_worker.py')) + ', run_name="__main__")'
            def arguments(port):
                return [[sys.executable, '-c', code, str(rank), str(port), str(root)] for rank in range(2)]
            run_distributed_workers(arguments, cwd=root_dir, timeout=120)
            directory = root / 'ckpt' / 'tensorboard'
            self.assertEqual(len(list(directory.glob('events.out.tfevents.*'))), 1)
            events = EventAccumulator(str(directory)).Reload()
            for tag in ('train/loss/generator', 'validation/loss/generator', 'learning_rate/after_update/generator'):
                self.assertEqual([e.step for e in events.Scalars(tag)], [1])
            self.assertEqual(len(events.Tags()['audio']), 3)
            self.assertEqual(len(events.Tags()['images']), 1)
            self.assertFalse([tag for tag in events.Tags()['tensors'] if tag.startswith('preview_errors/')])


if __name__ == '__main__':
    unittest.main()
