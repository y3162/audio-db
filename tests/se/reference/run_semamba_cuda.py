"""SEMamba++ official CUDA mamba-ssm versus the independent PyTorch scan.

Run with ``.venvs/semamba`` (Python 3.10, PyTorch 2.6.0+cu124,
``causal-conv1d==1.7.0``, ``mamba-ssm==2.3.2.post1``). The library itself
does not import those packages.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/se/official/semambapp'))
from model.semambapp import SEMambapp
from model.stfts import mag_phase_istft, mag_phase_stft
from src.se import load_enhancer
from tests.se.reference.metrics import compare, format_row

CONFIG = {
    'model_cfg': {
        'hid_feature': 48, 'num_tfmamba': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'norm_epsilon': 1e-5, 'input_channel': 2, 'output_channel': 1, 'beta': 2.0,
        'dropout': 0.1, 'unet_expansion': 1,
    },
    'stft_cfg': {
        'n_fft': 400, 'hop_size': 100, 'win_size': 400, 'compress_factor': 0.3, 'sampling_rate': 16000,
    },
}


def report(name, reference, actual):
    print(format_row(name, compare(reference, actual)), flush=True)


def reference_forward(module, hidden):
    """Published selective-scan recurrence on one official ``Mamba`` module."""
    batch, seqlen, _ = hidden.shape
    projected = module.in_proj.weight @ hidden.reshape(batch * seqlen, -1).t()
    xz = projected.view(-1, batch, seqlen).transpose(0, 1)
    x, z = xz.chunk(2, dim=1)
    x = F.silu(module.conv1d(x)[..., :seqlen])
    mixed = module.x_proj(x.transpose(1, 2).reshape(batch * seqlen, -1))
    dt, b_mat, c_mat = torch.split(mixed, [module.dt_rank, module.d_state, module.d_state], dim=-1)
    dt = (module.dt_proj.weight @ dt.t()).view(-1, batch, seqlen).transpose(0, 1)
    b_mat = b_mat.view(batch, seqlen, module.d_state).transpose(1, 2)
    c_mat = c_mat.view(batch, seqlen, module.d_state).transpose(1, 2)
    delta = F.softplus(dt + module.dt_proj.bias.view(1, -1, 1))
    state_a = -torch.exp(module.A_log.float())
    delta_a = torch.exp(torch.einsum('bdl,dn->bdln', delta.float(), state_a))
    delta_b_u = torch.einsum('bdl,bnl,bdl->bdln', delta.float(), b_mat.float(), x.float())
    state = x.new_zeros(batch, x.shape[1], module.d_state)
    outputs = []
    for index in range(seqlen):
        state = delta_a[:, :, index] * state + delta_b_u[:, :, index]
        outputs.append(torch.einsum('bdn,bn->bd', state, c_mat[:, :, index].float()))
    scanned = torch.stack(outputs, dim=2) + x.float() * module.D.float().view(-1, 1)
    scanned = scanned * F.silu(z.float())
    return module.out_proj(scanned.transpose(1, 2).type_as(hidden))


def timed(label, fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f'{label}: {elapsed:.3f}s peak_vram_mib={peak:.1f}', flush=True)
    return value, elapsed, peak


def official_wave(model, audio):
    """Match infer.py: divide by (peak + 1e-9), multiply the raw peak back."""
    peak = audio.abs().max()
    mag, pha, _ = mag_phase_stft((audio / (peak + 1e-9)).view(1, -1), 400, 100, 400, 0.3)
    mag, pha, _ = model(mag, pha)
    return mag_phase_istft(mag, pha, 400, 100, 400, 0.3).squeeze(0) * peak, mag, pha


def main():
    seconds = [float(item) for item in (sys.argv[1:] or ['1', '5', '20'])]
    device = torch.device('cuda')
    official = SEMambapp(CONFIG).to(device).eval()
    state = torch.load(ROOT / 'data/checkpoints/semambapp/semambapp.pth', map_location='cpu', weights_only=False)['generator']
    missing, unexpected = official.load_state_dict(state, strict=False)
    print(f'official missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    if missing or unexpected:
        print('missing', missing[:12], flush=True)
        print('unexpected', unexpected[:12], flush=True)
    enhancer = load_enhancer('semambapp', ROOT / 'data/checkpoints/semambapp/semambapp.pth', device='cuda')
    mambas = [module for module in official.modules() if module.__class__.__name__ == 'Mamba']
    print(f'mamba_modules={len(mambas)} fast_path={mambas[0].use_fast_path}', flush=True)
    torch.manual_seed(9)
    probe = torch.randn(1, 64, mambas[0].d_model, device=device)
    with torch.inference_mode():
        fused = mambas[0](probe)
        mambas[0].use_fast_path = False
        scan = mambas[0](probe)
        mambas[0].use_fast_path = True
        ref = reference_forward(mambas[0], probe)
    report('mixer_fused_vs_cuda_scan', fused, scan)
    report('mixer_fused_vs_pytorch_ref', fused, ref)
    report('mixer_cuda_scan_vs_pytorch_ref', scan, ref)

    for duration in seconds:
        torch.manual_seed(9)
        audio = (torch.randn(int(duration * 16000), device=device) * 0.05)
        with torch.inference_mode():
            (wave_o, mag_o, pha_o), sec_o, mem_o = timed(f'official_{duration}s', lambda: official_wave(official, audio))
            (wave_i, mag_i, pha_i), sec_i, mem_i = timed(
                f'independent_model_{duration}s',
                lambda: _independent_wave(enhancer.model, audio),
            )
            wave_api, sec_api, mem_api = timed(f'api_{duration}s', lambda: enhancer.enhance(audio, 16000))
        report(f'semamba_cuda_mag_{duration}s', mag_o, mag_i)
        report(f'semamba_cuda_phase_{duration}s', pha_o, pha_i)
        report(f'semamba_cuda_wave_{duration}s', wave_o, wave_i)
        report(f'semamba_cuda_api_{duration}s', wave_o, wave_api)
        audio_seconds = duration
        print(
            f'rtf official={sec_o / audio_seconds:.3f} independent={sec_i / audio_seconds:.3f} '
            f'api={sec_api / audio_seconds:.3f} vram_mib official={mem_o:.1f} independent={mem_i:.1f} api={mem_api:.1f}',
            flush=True,
        )


def _independent_wave(model, audio):
    peak = audio.abs().max()
    window = torch.hann_window(400, device=audio.device)
    spec = torch.stft((audio / (peak + 1e-9)).view(1, -1), 400, 100, 400, window, center=True, pad_mode='reflect', return_complex=True)
    mag, pha = model(spec.abs().pow(0.3), spec.angle())
    restored = torch.complex(mag.pow(1 / 0.3) * torch.cos(pha), mag.pow(1 / 0.3) * torch.sin(pha))
    return torch.istft(restored, 400, 100, 400, window, center=True).squeeze(0) * peak, mag, pha


if __name__ == '__main__':
    main()
