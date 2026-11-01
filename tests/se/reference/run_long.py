"""Longer official-vs-independent comparisons.

Run with ``.venvs/ref`` for MossFormerGAN or ``.venvs/lab``
for the independent API alone. Official imports are not used by the library.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.se.reference.metrics import compare, format_row

CKPT = ROOT / 'data/checkpoints'


def section(title):
    print(f'\n== {title} ==', flush=True)


def report(name, reference, actual):
    print(format_row(name, compare(reference, actual)), flush=True)


def make_wave(seconds, sample_rate, seed, scale=0.05):
    generator = torch.Generator().manual_seed(seed)
    length = int(round(seconds * sample_rate))
    return torch.randn(1, length, generator=generator) * scale








def _timed(label, fn):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak = float('nan')
    elapsed = time.perf_counter() - start
    print(f'{label}: {elapsed:.2f}s peak_vram_mib={peak:.1f}', flush=True)
    return value




















def compare_mossformer(device, seconds, seed):
    section(f'MossFormerGAN {seconds}s seed {seed} {device}')
    root = ROOT / 'src/se/official/clearervoice-studio/clearvoice'
    sys.path.insert(0, str(root))
    from clearvoice.models.mossformer_gan_se.generator import MossFormerGAN_SE_16K
    from clearvoice.utils.decode import decode_one_audio_mossformergan_se_16k
    from src.se import load_enhancer

    class Args:
        mode = 'inference'
        network = 'MossFormerGAN_SE_16K'
        sampling_rate = 16000
        one_time_decode_length = 10
        decode_window = 10
        win_type = 'hamming'
        win_len = 400
        win_inc = 100
        fft_len = 400

    args = Args()
    generator = MossFormerGAN_SE_16K(args).model.to(device).eval()
    state = torch.load(CKPT / 'mossformergan/last_best_checkpoint.pt', map_location='cpu', weights_only=False)
    generator.load_state_dict(state['model'], strict=True)
    enhancer = load_enhancer('mossformergan', CKPT / 'mossformergan/last_best_checkpoint.pt', device=str(device))
    wave = make_wave(seconds, 16000, seed).squeeze(0)
    with torch.inference_mode():
        official = _timed('official', lambda: decode_one_audio_mossformergan_se_16k(
            generator, device, wave.numpy()[None], args))
        independent = _timed('independent', lambda: enhancer.enhance(wave.to(device), 16000))
    report(f'mossformer_{seconds}s', torch.as_tensor(official).reshape(-1).float(), independent.cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=('mossformer',))
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    try:
        compare_mossformer(device, args.seconds, args.seed)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
