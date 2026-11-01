"""Official-package comparisons. Run with .venvs/ref, not .venvs/lab."""
import sys
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










def main():
    torch.set_num_threads(4)
    selected = sys.argv[1:] or ['mossformer']
    functions = {'mossformer': mossformer}
    for name in selected:
        try:
            functions[name]()
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    main()
