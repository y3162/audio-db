"""Validation-only audio previews, using the training task's generator."""
from __future__ import annotations

import copy
import torch
import torch.nn.functional as F

from src.training.core.distributed import unwrap


def predict(task, batch, name):
    clean = batch['clean'].to(task.device)
    noisy = batch['noisy'].to(task.device)
    module = unwrap(task.generator)
    if name == 'mp_senet':
        # Same full-length input as validation_report. Centered iSTFT may be shorter.
        estimate = task._generate(
            clean, noisy, generator=module, allow_official_istft_length=True,
        )['audio']
    elif name == 'semambapp':
        estimate = task._generate(clean, noisy, module)['audio']
    elif name == 'tf_gridnet':
        if hasattr(task, '_enhance'):
            estimate = task._enhance(module, noisy)
        else:
            from src.training.se.waveform import _predict
            estimate = _predict(task.config['model'], module, noisy)
    elif name == 'tf_locoformer':
        estimate = module(noisy)
    elif name == 'mossformergan':
        from src.training.se.mossformergan import _istft, _uncompress
        scale = torch.sqrt(noisy.shape[-1] / (noisy.square().sum(dim=-1) + 1e-6)).unsqueeze(-1)
        _, imag, _, real, _, _ = task._forward(module, noisy * scale, clean * scale)
        estimate = _istft(_uncompress(real, imag).squeeze(1), task.n_fft, task.hop, task.win, clean.shape[-1]) / scale
    elif name == 'deepfilternet3':
        from src.training.se.deepfilternet import _vorbis_istft, vorbis_window
        old_stream = task.stream
        task.stream = copy.deepcopy(old_stream)
        try:
            task.stream.reset()
            spectrum, erb, feat, _ = task._analyze_features(noisy)
            enhanced, _, _, _ = module(spectrum, erb, feat)
            params = module.params
            window = vorbis_window(params.fft_size).to(device=noisy.device, dtype=noisy.dtype)
            estimate = _vorbis_istft(torch.view_as_complex(enhanced.squeeze(1)), params.fft_size, params.hop_size, window)
        finally:
            task.stream = old_stream
    else:
        raise ValueError(f'unsupported preview model: {name}')
    length = min(clean.shape[-1], noisy.shape[-1], estimate.shape[-1])
    return [wave[0, :length].detach().cpu() for wave in (clean, noisy, estimate)]


def write_preview(writer, task, dataset, step, config, chunks=None):
    audio_config = config['audio']
    if not (audio_config['enabled'] or config['spectrogram']):
        return
    ids = dataset.case_ids()
    indices = sorted(range(len(ids)), key=ids.__getitem__)[:audio_config['max_cases']]
    for index in indices:
        case_id = ids[index]
        try:
            if dataset.sample_rate <= 0:
                raise ValueError('invalid sample rate')
            if chunks is None:
                batch = dataset.collate([index])
            else:
                candidates = [chunk for group in chunks for chunk in group if chunk.case_id == case_id]
                if not candidates:
                    raise ValueError('case has no validation chunk')
                batch = dataset.collate_chunks([min(candidates, key=lambda c: c.start)])
            with torch.inference_mode():
                waves = predict(task, batch, task.config['model']['name'])
            waves = [wave[:int(dataset.sample_rate * audio_config['max_seconds'])] for wave in waves]
            if any(w.numel() < 2 or not torch.isfinite(w).all() for w in waves):
                raise ValueError('empty or non-finite audio')
            if audio_config['enabled']:
                for label, wave in zip(('clean', 'noisy', 'enhanced'), waves):
                    writer.add_audio(f'audio/{case_id}/{label}', wave.unsqueeze(0).clamp(-1, 1), step, sample_rate=dataset.sample_rate)
            if config['spectrogram']:
                spectra = [torch.stft(F.pad(w, (0, max(0, 512-w.numel()))), 512, 128, 512,
                                      torch.hann_window(512), return_complex=True).abs().clamp_min(1e-8).log10() for w in waves]
                low = min(float(x.min()) for x in spectra)
                high = max(float(x.max()) for x in spectra)
                image = torch.cat([(x - low) / max(high-low, 1e-8) for x in spectra], dim=1)
                writer.add_image(f'spectrogram/{case_id}', image.unsqueeze(0), step)
        except Exception as exc:
            writer.add_text(f'preview_errors/{case_id}', str(exc), step)
