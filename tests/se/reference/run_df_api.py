"""DeepFilterNet3 common-API resampling versus libdf at 48 kHz.

The API turns 16 kHz into 48 kHz with torchaudio, runs the model, then
resamples back. libdf itself is only compared at 48 kHz; the difference
below is that round trip, not a second resampler inside libdf.
"""
import sys
from pathlib import Path

import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parents[3]
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
    torch.manual_seed(12)
    wave = torch.randn(16000)
    up = AF.resample(wave, 16000, 48000)
    official = enhance(model, state, up.float().contiguous()[None], pad=True, atten_lim_db=None)[0]
    down = AF.resample(official.cpu().float(), 48000, 16000)
    if down.numel() < wave.numel():
        down = torch.nn.functional.pad(down, (0, wave.numel() - down.numel()))
    down = down[:wave.numel()]
    actual = enhancer.enhance(wave, 16000)
    print(format_row('df_api_16k_roundtrip', compare(down, actual)), flush=True)
    print('lengths', wave.numel(), actual.numel(), down.numel(), official.numel(), flush=True)


if __name__ == '__main__':
    main()
