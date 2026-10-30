"""Compare DeepFilterNet3 to official libdf enhance."""
import os
import sys
from pathlib import Path

os.environ['DEVICE'] = 'cpu'

import torch

ROOT = Path('/net/spring/work/seki/exp/audio-db')
sys.path.insert(0, str(ROOT))
from src.se import load_enhancer
from tests.se.reference.metrics import compare, format_row
sys.path.insert(0, str(ROOT / 'src/se/official/deepfilternet/DeepFilterNet'))
from df.enhance import enhance, init_df

BASE = ROOT / 'data/checkpoints/deepfilternet3/DeepFilterNet3'


def main():
    torch.set_num_threads(4)
    model, state, _, epoch = init_df(str(BASE), post_filter=False, log_file=None, epoch='best')
    print('official epoch', epoch, flush=True)
    enhancer = load_enhancer(
        'deepfilternet3',
        BASE / 'checkpoints/model_120.ckpt.best',
        device='cpu',
        config=BASE / 'config.ini',
    )
    cases = {
        '100ms': torch.randn(4800),
        '1s': torch.randn(48000),
        'sine': torch.sin(2 * torch.pi * 440 * torch.arange(48000) / 48000) * 0.2,
        'small': torch.randn(48000) * 1e-4,
    }
    torch.manual_seed(12)
    cases['100ms'] = torch.randn(4800)
    cases['1s'] = torch.randn(48000)
    for name, wave in cases.items():
        official = enhance(model, state, wave.float().contiguous()[None], pad=True, atten_lim_db=None)
        actual = enhancer.enhance(wave.float(), 48000)
        print(format_row(name, compare(official[0], actual)), flush=True)
        enhancer.reset_state()


if __name__ == '__main__':
    main()
