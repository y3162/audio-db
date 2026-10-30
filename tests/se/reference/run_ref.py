"""Official-package comparisons. Run with .venvs/ref, not .venvs/lab."""
import sys
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


def rotary():
    section('StuPASE rotary vs x-transformers 2.14.2')
    from x_transformers.x_transformers import RotaryEmbedding as Official
    from x_transformers.x_transformers import apply_rotary_pos_emb as official_apply
    from src.se.models.stupase.flow.compat import RotaryEmbedding as Independent
    from src.se.models.stupase.flow.compat import apply_rotary_pos_emb as independent_apply
    official, independent = Official(64).eval(), Independent(64).eval()
    independent.load_state_dict(official.state_dict())
    freqs_o, scale_o = official.forward_from_seq_len(17)
    freqs_i, scale_i = independent.forward_from_seq_len(17)
    print('freq max', float((freqs_o - freqs_i).abs().max()), 'scale', scale_o, scale_i, flush=True)
    torch.manual_seed(0)
    values = torch.randn(2, 8, 17, 64)
    left = official_apply(values, freqs_o, scale_o)
    right = independent_apply(values, freqs_i, scale_i)
    report('rotary', left, right)


def mossformer():
    section('MossFormerGAN decode')
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
    generator = MossFormerGAN_SE_16K(args).model
    state = torch.load(CKPT / 'mossformergan/last_best_checkpoint.pt', map_location='cpu', weights_only=False)
    generator.load_state_dict(state['model'], strict=True)
    generator.eval()
    enhancer = load_enhancer('mossformergan', CKPT / 'mossformergan/last_best_checkpoint.pt', device='cpu')
    torch.manual_seed(3)
    wave = torch.randn(16000) * 0.05
    with torch.inference_mode():
        official = decode_one_audio_mossformergan_se_16k(generator, torch.device('cpu'), wave.numpy()[None], args)
    official = torch.as_tensor(official).reshape(-1).float()
    report('moss_1s', official, enhancer.enhance(wave, 16000))


def _stub_espnet():
    import torch.nn as nn
    names = ['espnet2', 'espnet2.enh', 'espnet2.enh.separator', 'espnet2.enh.separator.abs_separator',
             'espnet2.torch_utils', 'espnet2.torch_utils.get_layer_from_string']
    for name in names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['espnet2.enh.separator.abs_separator'].AbsSeparator = nn.Module

    def get_layer(l_name, library=nn):
        match = [x for x in dir(library) if l_name.lower() == x.lower()]
        return getattr(library, match[0])

    sys.modules['espnet2.torch_utils.get_layer_from_string'].get_layer = get_layer


def unipase():
    section('UniPASE forward')
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
    official = Official(**kwargs).eval()
    independent = Independent(**kwargs).eval()
    torch.manual_seed(4)
    wave16 = torch.randn(1, 8000) * 0.05
    with torch.inference_mode():
        y_off = official(wave16, sr_in=16000, sr_out=16000, enable_plc=True)
        y_ind = independent(wave16, sr_in=16000, sr_out=16000, enable_plc=True)
    report('unipase_16k_0.5s', y_off, y_ind)
    wave48 = torch.randn(1, 16000) * 0.05
    with torch.inference_mode():
        y_off = official(wave48, sr_in=48000, sr_out=48000, enable_plc=True)
        y_ind = independent(wave48, sr_in=48000, sr_out=48000, enable_plc=True)
    report('unipase_48k_input_16k_samples', y_off, y_ind)


def _trace_loads():
    original = torch.nn.Module.load_state_dict

    def wrapped(self, state, strict=True, assign=False):
        result = original(self, state, strict=strict, assign=assign)
        name = type(self).__name__
        if name in ('CFM', 'VocosVocoder', 'WavLM', 'VocosAdapter', 'TFGridNet'):
            print(f'load {name} strict={strict} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}', flush=True)
            if result.missing_keys:
                print('  missing', result.missing_keys[:40], flush=True)
            if result.unexpected_keys:
                print('  unexpected', result.unexpected_keys[:40], flush=True)
        return result

    torch.nn.Module.load_state_dict = wrapped


def stupase():
    section('StuPASE forward')
    _trace_loads()
    sys.path.insert(0, str(ROOT / 'src/se/official/pase/stupase'))
    from models.stupase import StuPASE as Official
    from src.se.models.stupase.stupase import StuPASE as Independent
    paths = (
        str(CKPT / 'stupase/DeWavLM-R.pt'),
        str(CKPT / 'stupase/CFM.pt'),
        str(CKPT / 'stupase/Vocoder_Mel-16k.pt'),
    )
    official = Official(*paths).eval()
    independent = Independent(*paths).eval()
    torch.manual_seed(5)
    wave = torch.randn(1, 8000) * 0.05
    torch.manual_seed(5)
    with torch.inference_mode():
        y_off = official(wave, sr=16000, steps=8, cfg_strength=0.5, sway_sampling_coef=-1.0)
    torch.manual_seed(5)
    with torch.inference_mode():
        y_ind = independent(wave, sr=16000, steps=8, cfg_strength=0.5, sway_sampling_coef=-1.0)
    report('stupase_0.5s', y_off, y_ind)


def main():
    torch.set_num_threads(4)
    selected = sys.argv[1:] or ['rotary', 'mossformer', 'unipase', 'stupase']
    functions = {'rotary': rotary, 'mossformer': mossformer, 'unipase': unipase, 'stupase': stupase}
    for name in selected:
        try:
            functions[name]()
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    main()
