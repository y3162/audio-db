import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'data/checkpoints/mp-senet/g_best_vb'


class IndependenceTest(unittest.TestCase):
    @unittest.skipUnless(CHECKPOINT.exists(), 'checkpoint unavailable')
    def test_package_without_official_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / 'src/se'
            package.parent.mkdir()
            shutil.copytree(ROOT / 'src/se', package, ignore=shutil.ignore_patterns('official', '__pycache__'))
            env = os.environ.copy()
            env['PYTHONPATH'] = tmp
            code = '''import torch
from src.se import load_enhancer
torch.set_num_threads(1)
e = load_enhancer("mp_senet", r"{}")
y = e.enhance(torch.randn(1000), 16000)
assert y.shape == (1000,) and torch.isfinite(y).all()
'''.format(CHECKPOINT)
            subprocess.run([sys.executable, '-c', code], cwd=tmp, env=env, check=True)

    @unittest.skipUnless((ROOT / 'data/checkpoints/tf-gridnet/urgent24/exp/enh_train_enh_tfgridnet_raw/21epoch.pth').exists(), 'checkpoint unavailable')
    def test_gridnet_without_official_tree(self):
        paths = [ROOT / 'data/checkpoints/tf-gridnet/urgent24/exp/enh_train_enh_tfgridnet_raw/21epoch.pth',
                 ROOT / 'data/checkpoints/tf-gridnet/ms-snsd/valid.loss.best.pth']
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / 'src/se'
            package.parent.mkdir()
            shutil.copytree(ROOT / 'src/se', package, ignore=shutil.ignore_patterns('official', '__pycache__'))
            env = os.environ.copy(); env['PYTHONPATH'] = tmp
            code = """import torch
from src.se import load_enhancer
torch.set_num_threads(1)
for variant, path, sr in [("urgent24", %r, 8000), ("ms-snsd", %r, 16000)]:
    e = load_enhancer("tf_gridnet", path, variant=variant)
    y = e.enhance(torch.randn(1000), sr)
    assert y.shape == (1000,) and torch.isfinite(y).all()
""" % tuple(str(p) for p in paths)
            subprocess.run([sys.executable, '-c', code], cwd=tmp, env=env, check=True)
