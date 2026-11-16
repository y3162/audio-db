"""Common waveform validation and independent variable-length batching."""
import torch


def validate_audio(waveform, sample_rate):
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
        raise ValueError('waveform must be a one-dimensional torch.Tensor')
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise ValueError('waveform must contain finite floating-point samples')
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0 or waveform.numel() == 0:
        raise ValueError('sample_rate and waveform length must be positive')


class Enhancer:
    def enhance(self, waveform, sample_rate):
        validate_audio(waveform, sample_rate)
        return self._enhance_one(waveform, sample_rate)

    def enhance_batch(self, waveforms, sample_rate, lengths=None):
        """Process (batch, time) independently; return zero-padded outputs and lengths."""
        if not isinstance(waveforms, torch.Tensor) or waveforms.ndim != 2:
            raise ValueError('waveforms must have shape (batch, time)')
        batch, size = waveforms.shape
        if not batch or not size:
            raise ValueError('batch and time dimensions must be nonempty')
        if lengths is None:
            lens = [size] * batch
        else:
            lens = torch.as_tensor(lengths)
            if lens.ndim != 1 or lens.numel() != batch or lens.dtype == torch.bool or lens.is_floating_point() or lens.is_complex():
                raise ValueError('lengths must contain one integer per batch item')
            lens = lens.tolist()
        if any(n <= 0 or n > size for n in lens):
            raise ValueError('lengths must be within the padded time dimension')
        result = torch.zeros_like(waveforms)
        for i, n in enumerate(lens):
            result[i, :n] = self.enhance(waveforms[i, :n], sample_rate)
        return result, torch.tensor(lens, dtype=torch.long, device=waveforms.device)
