"""Longer official-vs-independent comparisons.

Run with ``.venvs/ref`` (UniPASE, StuPASE, MossFormerGAN) or ``.venvs/lab``
for the independent API alone. Official imports are not used by the library.
"""
import argparse
import sys
import time
import traceback
import types
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


def _capture_postnet(module, box):
    original = module.infer

    def wrapped(x, sr_in=16000, sr_out=None):
        y = original(x, sr_in=sr_in, sr_out=sr_out)
        box['in'] = x.detach()
        box['out'] = y.detach()
        return y

    module.infer = wrapped


def peak_official(enhanced, scale):
    """Official file inference: ``y / (max(|y|) + 1e-8) * input_peak``."""
    return enhanced / (enhanced.abs().max() + 1e-8) * scale


def _stub_espnet():
    from torch import nn
    for name in (
        'espnet2', 'espnet2.enh', 'espnet2.enh.separator',
        'espnet2.enh.separator.abs_separator', 'espnet2.torch_utils',
        'espnet2.torch_utils.get_layer_from_string',
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['espnet2.enh.separator.abs_separator'].AbsSeparator = nn.Module

    def get_layer(l_name, library=nn):
        match = [x for x in dir(library) if l_name.lower() == x.lower()]
        return getattr(library, match[0])

    sys.modules['espnet2.torch_utils.get_layer_from_string'].get_layer = get_layer


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


def unipase_pair(device):
    _stub_espnet()
    sys.path.insert(0, str(ROOT / 'src/se/official/unipase'))
    from models.unipase import UniPASE as Official
    from src.se.models.unipase.unipase import UniPASE as Independent
    paths = {
        'dewavlm': CKPT / 'unipase/DeWavLM-Omni.pt',
        'adapter': CKPT / 'unipase/Adapter.pt',
        'vocoder': CKPT / 'unipase/Vocoder_DWO-L1.pt',
        'postnet': CKPT / 'unipase/PostNet.pt',
    }
    kwargs = {name + '_ckpt_path': str(path) for name, path in paths.items()}
    official = Official(**kwargs).to(device).eval()
    independent = Independent(**kwargs).to(device).eval()
    return official, independent


def compare_unipase(device, seconds, sample_rate, seed):
    section(f'UniPASE forward {sample_rate} Hz {seconds}s seed {seed} {device}')
    official, independent = unipase_pair(device)
    wave = make_wave(seconds, sample_rate, seed).to(device)
    scale = wave.abs().max()
    captured = {'official': {}, 'independent': {}}
    if sample_rate > 16000:
        _capture_postnet(official.postnet, captured['official'])
        _capture_postnet(independent.postnet, captured['independent'])
    with torch.inference_mode():
        y_off = _timed('official', lambda: official(wave, sr_in=sample_rate, sr_out=sample_rate, enable_plc=True))
        y_ind = _timed('independent', lambda: independent(wave, sr_in=sample_rate, sr_out=sample_rate, enable_plc=True))
    tag = f'unipase_{sample_rate}_{seconds}s'
    report(tag + '_raw', y_off, y_ind)
    report(tag + '_peak', peak_official(y_off, scale), peak_official(y_ind, scale))
    if captured['official']:
        report(tag + '_postnet_in', captured['official']['in'], captured['independent']['in'])
        report(tag + '_postnet_out', captured['official']['out'], captured['independent']['out'])
    from src.se.models.unipase.inference import UniPASEEnhancer
    api = UniPASEEnhancer(independent, device).enhance(wave.squeeze(0), sample_rate)
    report(tag + '_api', peak_official(y_off, scale), api)
    del official, independent, y_off, y_ind
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def long_patches(audio, sample_rate):
    """Official ``inference_long`` segmentation. ``audio`` is (1, T)."""
    seg_len = sample_rate * 8
    hop = sample_rate * 4
    patches = []
    start = 0
    end_tmp = 0
    while start < audio.shape[-1]:
        end = start + seg_len
        if end > audio.shape[-1]:
            if end - audio.shape[-1] < 2 * sample_rate:
                patches.append(audio[:, start:])
            else:
                patches[-1] = torch.cat((patches[-1], audio[:, end_tmp:]), dim=1)
            break
        patches.append(audio[:, start:end])
        end_tmp = end
        start += hop
    return patches


def stitch_long(patches, sample_rate):
    """Official overlap trim. Assumes output rate equals input rate."""
    if len(patches) == 1:
        return patches[0]
    keep = sample_rate * 6
    edge = sample_rate * 2
    pieces = [patches[0][:, :keep]]
    for patch in patches[1:-1]:
        pieces.append(patch[:, edge:keep])
    pieces.append(patches[-1][:, edge:])
    return torch.cat(pieces, dim=1)


def run_long(model, audio, sample_rate):
    patches = long_patches(audio, sample_rate)
    enhanced = []
    with torch.inference_mode():
        for patch in patches:
            enhanced.append(model(patch, sr_in=sample_rate, sr_out=sample_rate, enable_plc=True))
    return stitch_long(enhanced, sample_rate)


def compare_unipase_long(device, seconds, sample_rate, seed):
    section(f'UniPASE inference_long {sample_rate} Hz {seconds}s seed {seed}')
    official, independent = unipase_pair(device)
    wave = make_wave(seconds, sample_rate, seed).to(device)
    scale = wave.abs().max()
    y_off = _timed('official_long', lambda: run_long(official, wave, sample_rate))
    y_ind = _timed('independent_long', lambda: run_long(independent, wave, sample_rate))
    tag = f'unipase_long_{sample_rate}_{seconds}s'
    report(tag + '_raw', y_off, y_ind)
    report(tag + '_peak', peak_official(y_off, scale), peak_official(y_ind, scale))
    from src.se.models.unipase.inference import UniPASEEnhancer
    api = UniPASEEnhancer(independent, device).enhance(wave.squeeze(0), sample_rate)
    report(tag + '_api', peak_official(y_off, scale), api)


def stupase_pair(device):
    original = torch.nn.Module.load_state_dict

    def wrapped(self, state, strict=True, assign=False):
        result = original(self, state, strict=strict, assign=assign)
        name = type(self).__name__
        if name in ('CFM', 'VocosVocoder', 'WavLM'):
            print(f'load {name} strict={strict} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}', flush=True)
        return result

    torch.nn.Module.load_state_dict = wrapped
    sys.path.insert(0, str(ROOT / 'src/se/official/pase/stupase'))
    from models.stupase import StuPASE as Official
    from src.se.models.stupase.stupase import StuPASE as Independent
    paths = (
        str(CKPT / 'stupase/DeWavLM-R.pt'),
        str(CKPT / 'stupase/CFM.pt'),
        str(CKPT / 'stupase/Vocoder_Mel-16k.pt'),
    )
    official = Official(*paths).to(device).eval()
    independent = Independent(*paths).to(device).eval()
    return official, independent


def _capture_trajectory(model, box):
    original = model.cfm.sample

    def sample(*args, **kwargs):
        out, trajectory = original(*args, **kwargs)
        box['trajectory'] = trajectory.detach().clone()
        return out, trajectory

    model.cfm.sample = sample


def compare_stupase(device, seconds, seed):
    section(f'StuPASE forward 16 kHz {seconds}s seed {seed}')
    official, independent = stupase_pair(device)
    captured = {'official': {}, 'independent': {}}
    _capture_trajectory(official, captured['official'])
    _capture_trajectory(independent, captured['independent'])
    wave = make_wave(seconds, 16000, seed).to(device)
    scale = wave.abs().max()
    torch.manual_seed(seed)
    with torch.inference_mode():
        y_off = _timed('official', lambda: official(wave, sr=16000, steps=8, cfg_strength=0.5, sway_sampling_coef=-1.0))
    torch.manual_seed(seed)
    with torch.inference_mode():
        y_ind = _timed('independent', lambda: independent(wave, sr=16000, steps=8, cfg_strength=0.5, sway_sampling_coef=-1.0))
    tag = f'stupase_{seconds}s_seed{seed}'
    report(tag + '_ode', captured['official']['trajectory'], captured['independent']['trajectory'])
    report(tag + '_raw', y_off, y_ind)
    report(tag + '_peak', peak_official(y_off, scale), peak_official(y_ind, scale))
    from src.se.models.stupase.inference import StuPASEEnhancer
    torch.manual_seed(seed)
    api = StuPASEEnhancer(independent, device).enhance(wave.squeeze(0), 16000)
    report(tag + '_api', peak_official(y_off, scale), api)


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
    parser.add_argument('model', choices=('unipase', 'unipase-long', 'stupase', 'mossformer'))
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    try:
        if args.model == 'unipase':
            compare_unipase(device, args.seconds, args.sr, args.seed)
        elif args.model == 'unipase-long':
            compare_unipase_long(device, args.seconds, args.sr, args.seed)
        elif args.model == 'stupase':
            compare_stupase(device, args.seconds, args.seed)
        else:
            compare_mossformer(device, args.seconds, args.seed)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
