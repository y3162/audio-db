import unittest
import importlib.util

from src.training.core.tensorboard import scalars


class ScalarConversionTest(unittest.TestCase):
    def test_train_and_missing_validation(self):
        train = {'type': 'train', 'global_step': 4, 'losses': {'generator': 1.5, 'bad': float('nan')},
                 'learning_rate': {'generator': 0.01, 'discriminator': 0.02}, 'epoch': 2}
        result = list(scalars(train))
        self.assertIn(('train/loss/generator', 1.5, 4), result)
        self.assertIn(('learning_rate/after_update/discriminator', 0.02, 4), result)
        self.assertFalse(any('bad' in tag for tag, _, _ in result))
        validation = {'type': 'validation', 'global_step': 4,
                      'losses': {'generator': {'status': 'ok', 'value': 2}},
                      'pesq': {'status': 'unavailable', 'value': None},
                      'stoi': {'status': 'ok', 'value': 0.8}}
        self.assertEqual(list(scalars(validation)), [('validation/loss/generator', 2.0, 4),
                                                       ('validation/stoi', 0.8, 4)])
        self.assertEqual(list(scalars({'type': 'validation_case', 'global_step': 4})), [])

class EventFileTest(unittest.TestCase):
    def test_rebuild_and_resume(self):
        import json
        import tempfile
        from pathlib import Path
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            from src.training.core.tensorboard import rebuild, open_writer
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'metrics.jsonl'
            source.write_text('\n'.join(json.dumps(x) for x in [
                {'type': 'train', 'global_step': 1, 'losses': {'generator': 3}},
                {'type': 'validation', 'global_step': 1, 'stoi': {'status': 'ok', 'value': 0.75}},
            ]))
            rebuild(source, root / 'converted')
            events = EventAccumulator(str(root / 'converted')).Reload()
            self.assertEqual(events.Scalars('train/loss/generator')[0].step, 1)
            self.assertAlmostEqual(events.Scalars('validation/stoi')[0].value, .75)
            with self.assertRaises(FileExistsError):
                rebuild(source, root / 'converted')
            log = root / 'resume'
            writer = SummaryWriter(str(log))
            writer.add_scalar('train/loss/generator', 1, 1)
            writer.add_scalar('train/loss/generator', 99, 2)
            writer.close()
            writer = open_writer(log, 1)
            writer.add_scalar('train/loss/generator', 2, 2)
            writer.close()
            events = EventAccumulator(str(log)).Reload().Scalars('train/loss/generator')
            self.assertEqual([(e.step, e.value) for e in events], [(1, 1), (2, 2)])

@unittest.skipUnless(importlib.util.find_spec("tensorboard"), "tensorboard not installed")
class TrainingEquivalenceTest(unittest.TestCase):
    def test_enabled_does_not_change_mock_training(self):
        import tempfile
        from pathlib import Path
        import torch
        from src.training.core.batch import plan_batches
        from src.training.core.rng import capture_rng
        from src.training.core.trainer import Trainer
        from tests.training.mock_task import LinearPairTask
        from tests.training.support import cpu_process, paired_dataset, tiny_config
        from tests.training.test_mock_trainer import _assert_exact, _assert_rng
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = paired_dataset(root / 'wav', 8, frames=320, segment_size=80, seed=7)
            results = []
            for enabled in (False, True):
                config = tiny_config(root / str(enabled), max_steps=2, seed=7,
                                     global_batch_size=4, micro_batch_size_per_gpu=2, loss='mock')
                if enabled:
                    config['logging'] = {'tensorboard': {'enabled': True, 'scalar_interval': 1,
                        'audio': {'enabled': False, 'max_cases': 3, 'max_seconds': 5}, 'spectrogram': False}}
                task = LinearPairTask(config, torch.device('cpu'))
                trainer = Trainer(task, plan_batches(4, 2, 1), cpu_process(), max_steps=2, seed=7,
                                  checkpoint_dir=root / str(enabled), checkpoint_interval=0,
                                  validation_interval=0, config=config)
                trainer.fit(dataset)
                results.append((task.state_dict(), capture_rng(cuda=False)))
            _assert_exact(results[0][0], results[1][0])
            _assert_rng(results[0][1], results[1][1])

@unittest.skipUnless(importlib.util.find_spec("tensorboard"), "tensorboard not installed")
class RealModelPreviewTest(unittest.TestCase):
    def test_mp_senet_preview_and_training_equivalence(self):
        import tempfile
        from pathlib import Path
        import torch
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from src.training.core.batch import plan_batches
        from src.training.core.rng import capture_rng
        from src.training.core.trainer import Trainer
        from src.training.se.mp_senet import MPSENetTask
        from tests.training.support import cpu_process, paired_dataset, tiny_config
        from tests.training.test_mock_trainer import _assert_exact, _assert_rng
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = paired_dataset(root / 'train', 2, frames=512, segment_size=512, seed=7)
            valid = paired_dataset(root / 'valid', 1, frames=512, segment_size=512, seed=8)
            results = []
            for enabled in (False, True):
                config = tiny_config(root / str(enabled), max_steps=1, seed=7, segment_size=512,
                                     global_batch_size=1, micro_batch_size_per_gpu=1)
                if enabled:
                    config['logging'] = {'tensorboard': {'enabled': True, 'scalar_interval': 1,
                        'audio': {'enabled': True, 'max_cases': 1, 'max_seconds': 1}, 'spectrogram': True}}
                task = MPSENetTask(config, torch.device('cpu'))
                trainer = Trainer(task, plan_batches(1, 1, 1), cpu_process(), max_steps=1, seed=7,
                                  checkpoint_dir=root / str(enabled), checkpoint_interval=0,
                                  validation_interval=1, config=config)
                trainer.fit(train, valid)
                results.append((task.state_dict(), capture_rng(cuda=False)))
            self.assertEqual(results[0][0].keys(), results[1][0].keys())
            from tests.training.test_mock_trainer import _assert_exact_state
            _assert_exact_state(results[0][0], results[1][0], "task")
            _assert_rng(results[0][1], results[1][1])
            event = EventAccumulator(str(root / 'True' / 'tensorboard')).Reload()
            self.assertEqual(len(event.Tags()['audio']), 3)
            errors = [tag for tag in event.Tags()['tensors'] if tag.startswith('preview_errors/')]
            self.assertFalse(errors, [(tag, event.Tensors(tag)[0].tensor_proto.string_val) for tag in errors])
            self.assertEqual(len(event.Tags()['images']), 1)
            self.assertTrue(event.Scalars('validation/loss/generator'))

class OtherModelPreviewTest(unittest.TestCase):
    def test_small_models_forward(self):
        import tempfile
        from pathlib import Path
        import torch
        from src.training.se.preview import predict
        from src.training.se.tf_gridnet import TFGridNetTask
        from src.training.se.tf_locoformer import TFLocoformerTask
        from src.training.se.semambapp import SEMambaTask
        from src.training.se.deepfilternet import DeepFilterNetTask
        from src.training.se.mossformergan import MossFormerGANTask
        from tests.training.test_model_tasks import _base
        from tests.se.test_tf_locoformer import TINY
        grid = dict(n_fft=32, stride=16, n_layers=1, lstm_hidden_units=8, attn_n_head=1,
                    attn_approx_qk_dim=8, emb_dim=8, emb_ks=1, emb_hs=1)
        semamba = dict(hid_feature=8, num_tfmamba=1, d_state=4, d_conv=2, expand=2, n_fft=400)
        df = dict(sr=16000, fft_size=128, hop_size=64, nb_erb=16, nb_df=16, conv_ch=16,
                  emb_hidden_dim=32, df_hidden_dim=32, enc_lin_groups=8, lin_groups=8,
                  emb_num_layers=2, df_num_layers=1, df_order=5, conv_lookahead=0, df_lookahead=0)
        cases = [
            ('tf_gridnet', TFGridNetTask, dict(variant='ms-snsd', architecture=grid), 800),
            ('tf_locoformer', TFLocoformerTask, dict(variant='dns', architecture=dict(TINY)), 512),
            ('semambapp', SEMambaTask, dict(architecture=semamba, stft={'hop': 100, 'win': 400, 'disc_windows': [256, 512]}), 2048),
            ('deepfilternet3', DeepFilterNetTask, dict(architecture=df, grad_clip=1.0), 1024),
            ('mossformergan', MossFormerGANTask, dict(variant='se_16k', architecture={'n_fft': 400, 'win_len': 400, 'hop': 100, 'num_channel': 64}, grad_clip=10.0), 8000),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, cls, params, length in cases:
                with self.subTest(model=name):
                    config = _base(Path(tmp), name, segment_size=length, **params)
                    task = cls(config, torch.device('cpu'))
                    task.eval_mode()
                    batch = {'clean': torch.randn(1, length), 'noisy': torch.randn(1, length)}
                    with torch.inference_mode():
                        waves = predict(task, batch, name)
                    self.assertEqual(len(waves), 3)
                    self.assertTrue(all(torch.isfinite(w).all() and len(w) > 0 for w in waves))

class MonitoringConfigTest(unittest.TestCase):
    def test_configuration_validation_and_training_identity(self):
        import tempfile
        from pathlib import Path
        import yaml
        from src.training.errors import TrainingError
        from src.training.se.config import load_training_config
        from src.training.se.job_chain import training_config_digest
        raw = yaml.safe_load(Path('configs/training/mp_senet.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text(yaml.safe_dump(raw))
            original = load_training_config(path)
            raw['logging'] = {'tensorboard': {'enabled': True, 'scalar_interval': 1}}
            path.write_text(yaml.safe_dump(raw))
            monitored = load_training_config(path)
            self.assertEqual(training_config_digest(original), training_config_digest(monitored))
            raw['seed'] += 1
            path.write_text(yaml.safe_dump(raw))
            self.assertNotEqual(training_config_digest(original), training_config_digest(load_training_config(path)))
            raw['logging']['tensorboard']['scalar_interval'] = 0
            path.write_text(yaml.safe_dump(raw))
            with self.assertRaises(TrainingError):
                load_training_config(path)


if __name__ == '__main__':
    unittest.main()
