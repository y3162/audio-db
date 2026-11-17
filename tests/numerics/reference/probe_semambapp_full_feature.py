"""Fingerprint the official-width FM waveform gradient after one D step.

Run each kind and warp in a fresh process with a prepared AUDIO_DB_WARP_OUT:
    python -m tests.numerics.reference.probe_semambapp_full_feature official 1
    python -m tests.numerics.reference.probe_semambapp_full_feature audiodb 1
"""
import json
import sys

import torch

from tests.numerics.reference import compare_semamba_warp as w


def main():
    kind, mode = sys.argv[1:3]
    if kind not in {'official', 'audiodb'} or mode not in {'1', '2'}:
        raise SystemExit('kind must be official/audiodb and warp must be 1/2')
    w._flags()
    w.configure(mode, False)
    payload = w._load()
    cfg, generator, cqt, mrd, mel = w._build_official() if kind == 'official' else w._build_audio('fused')
    clean, noisy = w._move(payload, generator, cqt, mrd, mel, w._device())
    _optim_g, optim_d, _scheduler_g, _scheduler_d = w._optimizers(generator, cqt, mrd)
    stft = 'official' if kind == 'official' else 'audio'
    loss_kind = 'official' if kind == 'official' else 'audio'
    parts = w._parts(generator, clean, noisy, stft)
    losses = w._losses_for(loss_kind, 'family', {**parts, 'audio': parts['audio'].detach()}, clean, cqt, mrd, mel, cfg)
    optim_d.zero_grad()
    losses['discriminator'].backward()
    optim_d.step()
    parts = w._parts(generator, clean, noisy, stft)
    losses = w._losses_for(loss_kind, 'family', parts, clean, cqt, mrd, mel, cfg)
    gradient = torch.autograd.grad(losses['matching'], parts['audio'])[0]
    print(json.dumps({
        'kind': kind, 'warp': mode, 'selected': w._selected_warps(),
        'loss': float(losses['matching'].detach()), 'gradient_sha256': w._digest(gradient),
        'finite': bool(torch.isfinite(gradient).all()), 'numel': gradient.numel(),
    }), flush=True)


if __name__ == '__main__':
    main()
