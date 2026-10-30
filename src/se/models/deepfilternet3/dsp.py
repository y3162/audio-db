"""STFT, ERB, and exponential normalization matching the published DeepFilterNet DSP.

The reference implementation lives in libDF (Rust). This module follows that
algorithm in PyTorch so inference does not require the libdf extension:
vorbis window, unnormalized real FFT with the analysis scale 2*hop/n_fft**2,
50% (or smaller) overlap-add, rectangular ERB power bands, and the exponential
mean / unit-norm states.
"""
import math
from dataclasses import dataclass

import torch


def freq_to_erb(freq_hz):
    return 9.265 * math.log1p(freq_hz / (24.7 * 9.265))


def erb_to_freq(n_erb):
    return 24.7 * 9.265 * (math.exp(n_erb / 9.265) - 1.0)


def erb_band_widths(sample_rate, n_fft, n_bands, min_bins):
    """Return the number of FFT bins assigned to each ERB band."""
    freq_width = sample_rate / n_fft
    erb_low = freq_to_erb(0.0)
    erb_high = freq_to_erb(sample_rate / 2)
    step = (erb_high - erb_low) / n_bands
    previous = 0
    overflow = 0
    widths = []
    for index in range(1, n_bands + 1):
        bin_index = int(round(erb_to_freq(erb_low + index * step) / freq_width))
        count = bin_index - previous - overflow
        if count < min_bins:
            overflow = min_bins - count
            count = min_bins
        else:
            overflow = 0
        widths.append(count)
        previous = bin_index
    widths[-1] += 1
    extra = sum(widths) - (n_fft // 2 + 1)
    if extra > 0:
        widths[-1] -= extra
    if sum(widths) != n_fft // 2 + 1:
        raise ValueError(f'ERB widths sum to {sum(widths)}, expected {n_fft // 2 + 1}')
    return widths


def rectangular_erb(widths, sample_rate, inverse=False):
    """Normalized rectangular ERB matrix used by the mask stage."""
    n_freqs = int(sum(widths))
    banks = torch.zeros(n_freqs, len(widths))
    start = 0
    for index, width in enumerate(widths):
        banks[start:start + width, index] = 1
        start += width
    if inverse:
        return banks.t().contiguous()
    banks = banks / banks.sum(dim=0, keepdim=True)
    return banks


def vorbis_window(n_fft):
    half = n_fft / 2
    bins = torch.arange(n_fft, dtype=torch.float64)
    sinusoid = torch.sin(0.5 * math.pi * (bins + 0.5) / half)
    return torch.sin(0.5 * math.pi * sinusoid * sinusoid).to(torch.float32)


def norm_alpha(sample_rate, hop, tau):
    """Round the exponential decay so it stays strictly below 1, as in df.utils."""
    raw = math.exp(-(hop / sample_rate) / tau)
    precision = 3
    alpha = 1.0
    while alpha >= 1.0:
        alpha = round(raw, precision)
        precision += 1
    return alpha


def _linspace_state(start, end, count, channels, device, dtype):
    values = torch.linspace(start, end, count, device=device, dtype=dtype)
    return values.view(1, count).repeat(channels, 1)


@dataclass
class DFStream:
    """Per-utterance STFT overlap state. Reset between independent utterances."""

    sample_rate: int
    n_fft: int
    hop: int
    n_erb: int
    n_df: int
    min_erb_bins: int
    norm_tau: float
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.widths = erb_band_widths(self.sample_rate, self.n_fft, self.n_erb, self.min_erb_bins)
        self.window = vorbis_window(self.n_fft).to(device=self.device, dtype=self.dtype)
        self.scale = 1.0 / (self.n_fft ** 2 / (2 * self.hop))
        self.alpha = norm_alpha(self.sample_rate, self.hop, self.norm_tau)
        self.reset()

    def reset(self):
        length = self.n_fft - self.hop
        self.analysis_memory = torch.zeros(length, device=self.device, dtype=self.dtype)
        self.synthesis_memory = torch.zeros(length, device=self.device, dtype=self.dtype)
        self.mean_state = None
        self.unit_state = None

    def to(self, device):
        self.device = torch.device(device)
        self.window = self.window.to(self.device)
        self.analysis_memory = self.analysis_memory.to(self.device)
        self.synthesis_memory = self.synthesis_memory.to(self.device)
        if self.mean_state is not None:
            self.mean_state = self.mean_state.to(self.device)
            self.unit_state = self.unit_state.to(self.device)
        return self

    def analyze(self, audio):
        """audio [channels, time] -> complex spectrum [channels, frames, bins]."""
        if audio.ndim != 2:
            raise ValueError('analysis expects [channels, time]')
        channels, time = audio.shape
        frames = time // self.hop
        spectrum = torch.empty(channels, frames, self.n_fft // 2 + 1, dtype=torch.complex64, device=audio.device)
        for channel in range(channels):
            self.analysis_memory.zero_()
            for frame in range(frames):
                start = frame * self.hop
                spectrum[channel, frame], self.analysis_memory = _analyze_frame(
                    audio[channel, start:start + self.hop], self.analysis_memory, self.window, self.scale)
        return spectrum

    def synthesize(self, spectrum):
        """complex spectrum [channels, frames, bins] -> waveform [channels, time]."""
        channels, frames, _ = spectrum.shape
        output = torch.empty(channels, frames * self.hop, device=spectrum.device, dtype=self.dtype)
        for channel in range(channels):
            self.synthesis_memory.zero_()
            for frame in range(frames):
                output[channel, frame * self.hop:(frame + 1) * self.hop], self.synthesis_memory = _synthesize_frame(
                    spectrum[channel, frame], self.synthesis_memory, self.window)
        return output

    def features(self, spectrum):
        """Return real spectrum, normalized ERB, and unit-normalized DF bins.

        Shapes match the official enhance path: [C, 1, T, F, 2], [C, 1, T, E], [C, 1, T, Fdf, 2].
        """
        channels, frames, _ = spectrum.shape
        device, dtype = spectrum.device, spectrum.real.dtype
        self.mean_state = _linspace_state(-60.0, -90.0, self.n_erb, channels, device, dtype)
        self.unit_state = _linspace_state(0.001, 0.0001, self.n_df, channels, device, dtype)
        erb = _erb_power_db(spectrum, self.widths)
        for frame in range(frames):
            updated = erb[:, frame] * (1.0 - self.alpha) + self.mean_state * self.alpha
            self.mean_state = updated
            erb[:, frame] = (erb[:, frame] - updated) / 40.0
        df_bins = spectrum[..., :self.n_df].clone()
        magnitude = df_bins.abs()
        for frame in range(frames):
            updated = magnitude[:, frame] * (1.0 - self.alpha) + self.unit_state * self.alpha
            self.unit_state = updated
            df_bins[:, frame] = df_bins[:, frame] / updated.sqrt().clamp_min(1e-12)
        spec = torch.view_as_real(spectrum).unsqueeze(1)
        return spec, erb.unsqueeze(1), torch.view_as_real(df_bins).unsqueeze(1)


def _analyze_frame(frame, memory, window, scale):
    hop = frame.shape[0]
    left = window.shape[0] - hop
    buffer = torch.empty(window.shape[0], device=frame.device, dtype=frame.dtype)
    buffer[:left] = memory * window[:left]
    buffer[left:] = frame * window[left:]
    split = memory.shape[0] - hop
    if split > 0:
        memory = torch.cat((memory[hop:], memory[:hop]))
    updated = memory.clone()
    updated[split:] = frame
    spectrum = torch.fft.rfft(buffer).to(torch.complex64) * scale
    return spectrum, updated


def _synthesize_frame(spectrum, memory, window):
    hop = memory.shape[0] if memory.shape[0] == window.shape[0] // 2 else (window.shape[0] - memory.shape[0])
    # frame size is the configured hop, which equals n_fft - memory for the usual setting.
    hop = window.shape[0] - memory.shape[0]
    waveform = torch.fft.irfft(spectrum, n=window.shape[0], norm='forward').to(window.dtype) * window
    output = waveform[:hop] + memory[:hop]
    split = memory.shape[0] - hop
    shifted = torch.cat((memory[hop:], memory[:hop])) if split > 0 else memory
    updated = shifted.clone()
    second = waveform[hop:]
    if split > 0:
        updated[:split] = updated[:split] + second[:split]
        updated[split:] = second[split:]
    else:
        updated = second
    return output, updated


def _erb_power_db(spectrum, widths):
    power = spectrum.real.square() + spectrum.imag.square()
    bands = []
    start = 0
    for width in widths:
        bands.append(power[..., start:start + width].mean(dim=-1))
        start += width
    erb = torch.stack(bands, dim=-1)
    return torch.log10(erb + 1e-10) * 10.0
