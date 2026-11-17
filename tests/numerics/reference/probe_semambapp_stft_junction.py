"""Ablate SEMamba++ STFT graph paths and inspect the joint waveform gradient.

Prepare a fresh compare_semamba_warp payload in AUDIO_DB_WARP_OUT first.
The official implementation is loaded only by this diagnostic.
"""
import json

import torch

from tests.numerics.reference import compare_semamba_warp as w


def main():
    w._flags()
    w.configure('1', False)
    payload = w._load()
    device = w._device()
    official = w._case('official', 'official', 'official', 'official', 'family', payload, device, '1')
    original_spectrum, original_waveform = w._spectrum, w._waveform
    try:
        for name, official_analysis, official_inverse in (
            ('audio', False, False), ('official_analysis', True, False),
            ('official_inverse', False, True), ('official_both', True, True),
        ):
            w._spectrum = lambda kind, x, eps: original_spectrum('official' if official_analysis else kind, x, eps)
            w._waveform = lambda kind, mag, phase, length: original_waveform(
                'official' if official_inverse else kind, mag, phase, length
            )
            candidate = w._case(name, 'audio', 'audio', 'audio', 'family', payload, device, '1')
            buckets = w._stage_report(official, candidate)['grads']['buckets']
            print(json.dumps({'case': name, 'decoder_gradient_max_abs': {
                k: buckets[k]['max_abs'] for k in ('magnitude_decoder', 'phase_decoder')
            }}), flush=True)
    finally:
        w._spectrum, w._waveform = original_spectrum, original_waveform

    gradients = []
    for kind in ('official', 'audiodb', 'official'):
        cfg, generator, cqt, mrd, mel = w._build_official() if kind == 'official' else w._build_audio('fused')
        clean, noisy = w._move(payload, generator, cqt, mrd, mel, device)
        parts = w._parts(generator, clean, noisy, 'official' if kind == 'official' else 'audio')
        losses = w._losses_for('official' if kind == 'official' else 'audio', 'family', parts, clean, cqt, mrd, mel, cfg)
        gradients.append({name: torch.autograd.grad(losses[name], parts['audio'], retain_graph=True)[0].detach().cpu()
                          for name in ('matching', 'adversarial', 'mel', 'consistency', 'generator')})
    for name in gradients[0]:
        left, right, repeat = (item[name] for item in gradients)
        difference = (left - right).abs()
        print(json.dumps({'term': name, 'cross': w._compare(left, right), 'repeat': w._compare(left, repeat),
                          'interior_mismatch': int((difference[..., 200:-200] != 0).sum())}), flush=True)


if __name__ == '__main__':
    main()
