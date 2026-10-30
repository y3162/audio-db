import tempfile
from pathlib import Path
import unittest
import torch
from src.se import build_model, load_enhancer, model_metadata
from src.se.core.checkpoint import checkpoint_paths
from src.se.core.config import load_config
from src.se.core.registry import ModelRegistry, ModelSpec

ROOT = Path(__file__).resolve().parents[2]

class CoreTests(unittest.TestCase):
    def test_registry(self):
        r = ModelRegistry()
        s = ModelSpec('test', lambda x: x, lambda *x: x, 16000, ('alias',))
        r.register(s)
        self.assertIs(r.get('alias'), s)
        with self.assertRaises(ValueError): r.register(s)
        with self.assertRaises(ValueError): r.get('absent')
        self.assertEqual(model_metadata('mp-senet').sample_rate, 16000)
        with self.assertRaises(ValueError): build_model('tf_gridnet', variant='wsj0-2mix')

    def test_config_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text('{"n_layers": 1}')
            self.assertEqual(load_config(path), {'n_layers': 1})
        with self.assertRaises(ValueError): checkpoint_paths('a', ('one', 'two'))
        self.assertEqual(set(checkpoint_paths({'one':'a', 'two':'b'}, ('one','two'))), {'one', 'two'})

    @unittest.skipUnless((ROOT / 'data/checkpoints/mp-senet/g_best_vb').exists(), 'checkpoint unavailable')
    def test_batch(self):
        torch.set_num_threads(1)
        e = load_enhancer('mp_senet', {'generator': ROOT / 'data/checkpoints/mp-senet/g_best_vb'}, variant='vb')
        x = torch.randn(2, 600)
        y, lengths = e.enhance_batch(x, 16000, [600, 400])
        torch.testing.assert_close(y[1,:400], e.enhance(x[1,:400], 16000), atol=0, rtol=0)
        self.assertEqual(lengths.tolist(), [600, 400])
        self.assertTrue(torch.all(y[1,400:] == 0))
        with self.assertRaises(ValueError): e.enhance_batch(x, 16000, [600, 0])
