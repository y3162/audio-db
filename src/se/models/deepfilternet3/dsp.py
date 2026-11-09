"""STFT, ERB, and exponential normalization matching the published DeepFilterNet DSP.

The reference implementation lives in libDF (Rust). This module follows that
algorithm without importing libdf: vorbis window, unnormalized real FFT with the
analysis scale ``2 * hop / n_fft**2``, 50% (or smaller) overlap-add, rectangular
ERB power bands, and the exponential mean / unit-norm states.

``libdf_rfft.so`` (``realfft`` 3.3.0, the libdf 0.5.6 FFT) supplies the analysis
FFT, ERB ``log10``, and unit-norm square root. Those three libm operations are
where ``torch.fft.rfft``, ``torch.log10``, and ``torch.sqrt`` disagree with
libdf in float32. Without the library, analysis falls back to PyTorch.
"""
import ctypes
import hashlib
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

_ABI_VERSION = 1
_LOG = logging.getLogger(__name__)
_SELECTION_CACHE: dict | None = None
_SELECTION_KEY: tuple | None = None
_SELECTION_LOGGED = False


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
    """Match libdf's float32 ``start + i * (end - start) / (n - 1)`` state.

    ``torch.linspace`` differs in the last unit-norm bins, and forcing the last
    value to ``end`` also leaves the unit norm.
    """
    start_t = torch.tensor(start, dtype=torch.float32, device=device)
    end_t = torch.tensor(end, dtype=torch.float32, device=device)
    step = (end_t - start_t) / torch.tensor(count - 1, dtype=torch.float32, device=device)
    values = start_t + torch.arange(count, device=device, dtype=torch.float32) * step
    return values.view(1, count).repeat(channels, 1).to(dtype=dtype)


_KERNEL = None
_KERNEL_KEY = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _tracked_rfft_path() -> Path:
    return Path(__file__).with_name("libdf_rfft.so")


def _local_rfft_path() -> Path:
    return _repo_root() / "build" / "df-rfft" / "release" / "libdf_rfft.so"


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _glibc_version() -> str | None:
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")
    except (OSError, ValueError):
        return None
    return value or None


def _probe_library(path: Path) -> tuple[int | None, str]:
    """Return ``(abi, reason)``. ``reason`` is ``ok`` only for ABI 1."""

    if not path.is_file():
        return None, "missing"
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        return None, f"load failed: {error}"
    try:
        abi = library.df_rfft_abi
    except AttributeError:
        return None, "df_rfft_abi is missing"
    abi.argtypes = []
    abi.restype = ctypes.c_int
    try:
        version = int(abi())
    except OSError as error:
        return None, f"abi call failed: {error}"
    if version != _ABI_VERSION:
        return version, f"ABI {version} is not {_ABI_VERSION}"
    return version, "ok"


def _selection_key() -> tuple:
    tracked = _tracked_rfft_path()
    local = _local_rfft_path()

    def stamp(path: Path) -> tuple:
        if not path.is_file():
            return (str(path), None, None)
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)

    return (os.environ.get("DF_RFFT_LIBRARY"), stamp(tracked), stamp(local))


def rfft_selection() -> dict:
    """Choose the native library for this process without replacing tracked files.

    ``DF_RFFT_LIBRARY`` wins. Otherwise a loadable tracked file wins. If the
    tracked file exists but does not load, the local ``make df-rfft`` output is
    used when that file loads at ABI 1. An unloadable tracked file is not
    turned into the PyTorch fallback while it still exists and no local build
    loads. A missing file still falls back.
    """

    global _SELECTION_CACHE, _SELECTION_KEY
    key = _selection_key()
    if _SELECTION_CACHE is not None and key == _SELECTION_KEY:
        return dict(_SELECTION_CACHE)
    override = os.environ.get("DF_RFFT_LIBRARY")
    if override:
        path = Path(override)
        version, reason = _probe_library(path)
        source = "environment"
    else:
        tracked = _tracked_rfft_path()
        version, reason = _probe_library(tracked)
        if reason == "ok":
            path = tracked
            source = "tracked"
        else:
            local = _local_rfft_path()
            local_version, local_reason = _probe_library(local)
            if local_reason == "ok":
                path = local
                version = local_version
                reason = f"tracked library was not used: {reason}"
                source = "local_build"
            elif tracked.is_file():
                path = tracked
                source = "tracked_unreadable"
            else:
                path = tracked
                source = "missing"
                reason = "missing"
    record = {
        "abi": version,
        "glibc": _glibc_version(),
        "path": str(path),
        "reason": reason,
        "sha256": _file_sha256(path),
        "source": source,
    }
    _SELECTION_CACHE = dict(record)
    _SELECTION_KEY = key
    return record


def rfft_library_path():
    """Shared library used for libdf-matching float32 kernels.

    ``DF_RFFT_LIBRARY`` overrides discovery. Discovery never rewrites the
    tracked library and never treats an unloadable tracked file as permission
    to fall back while that file exists.
    """

    override = os.environ.get("DF_RFFT_LIBRARY")
    if override:
        return Path(override)
    return Path(rfft_selection()["path"])


def rfft_runtime_record() -> dict:
    """Selection plus the kernel that analysis actually runs.

    ``kernel`` is ``native``, ``fallback``, or ``unreadable``. ``unreadable``
    means a file is present and was refused, not that PyTorch was substituted.
    """

    global _SELECTION_LOGGED
    record = rfft_selection()
    try:
        kernel = _rfft_kernel()
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["kernel"] = "unreadable"
    else:
        record["kernel"] = "fallback" if kernel is None else "native"
        if kernel is not None:
            record["abi"] = int(kernel.df_rfft_abi())
    if not _SELECTION_LOGGED:
        _SELECTION_LOGGED = True
        _LOG.info(
            "realfft kernel=%s source=%s path=%s",
            record["kernel"],
            record["source"],
            record["path"],
        )
    return record


def _kernel_cache_key(path):
    required = os.environ.get("DF_RFFT_REQUIRED") == "1"
    if path.is_file():
        stat = path.stat()
        identity = (str(path), stat.st_mtime_ns, stat.st_size)
    else:
        identity = (str(path), None, None)
    return identity, required


def _bind_symbol(library, name, argtypes):
    try:
        function = getattr(library, name)
    except AttributeError as error:
        raise RuntimeError(
            f"{rfft_library_path()} does not export {name}. Rebuild with `make df-rfft`."
        ) from error
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def _rfft_kernel():
    """Load ``libdf_rfft.so``, or return None when the fallback is allowed.

    ``DF_RFFT_REQUIRED=1`` raises ``FileNotFoundError`` instead of falling back
    to ``torch.fft.rfft``. A library with the wrong ABI raises ``RuntimeError``.
    """
    global _KERNEL, _KERNEL_KEY
    path = rfft_library_path()
    key = _kernel_cache_key(path)
    if key == _KERNEL_KEY:
        return _KERNEL
    required = key[1]
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"DeepFilterNet3 realfft kernel is missing: {path}. "
                "Build it with `make df-rfft` from the repository root "
                "(Rust 1.76 or newer, Cargo.lock, realfft 3.3.0)."
            )
        _KERNEL_KEY = key
        _KERNEL = None
        return None
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise RuntimeError(
            f"failed to load {path}. Rebuild it on this machine with `make df-rfft`."
        ) from error
    abi = _bind_symbol(library, "df_rfft_abi", [])
    version = abi()
    if version != _ABI_VERSION:
        raise RuntimeError(
            f"{path} reports ABI {version}, expected {_ABI_VERSION}. Rebuild with `make df-rfft`."
        )
    _bind_symbol(library, "df_rfft_f32", [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ])
    _bind_symbol(library, "df_log10_db", [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
    ])
    _bind_symbol(library, "df_sqrt_f32", [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
    ])
    _bind_symbol(library, "df_unit_norm", [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ])
    _KERNEL_KEY = key
    _KERNEL = library
    return library


def _ptr(tensor):
    flat = tensor.reshape(-1)
    if flat.dtype != torch.float32 or flat.device.type != "cpu" or not flat.is_contiguous():
        raise RuntimeError("the realfft kernel expects a contiguous CPU float32 tensor")
    return ctypes.cast(flat.data_ptr(), ctypes.POINTER(ctypes.c_float))


def _rfft_unnormalized(buffer):
    """Length-N float32 waveform -> complex bins ``N // 2 + 1``, unnormalized."""
    waveform = buffer.detach().to(dtype=torch.float32, device="cpu").contiguous()
    bins = waveform.numel() // 2 + 1
    kernel = _rfft_kernel()
    if kernel is None:
        return torch.fft.rfft(waveform)
    real = torch.empty(bins, dtype=torch.float32)
    imag = torch.empty(bins, dtype=torch.float32)
    status = kernel.df_rfft_f32(_ptr(waveform), waveform.numel(), _ptr(real), _ptr(imag))
    if status != 0:
        raise RuntimeError(f"df_rfft_f32 failed with status {status}")
    return torch.complex(real, imag)


def _log10_db(power):
    """libdf ERB decibel: ``(power + 1e-10).log10() * 10`` in float32."""
    values = power.detach().to(dtype=torch.float32, device="cpu").contiguous()
    kernel = _rfft_kernel()
    if kernel is None:
        return torch.log10(values + 1e-10) * 10.0
    output = torch.empty(values.shape, dtype=torch.float32)
    status = kernel.df_log10_db(_ptr(values), values.numel(), _ptr(output))
    if status != 0:
        raise RuntimeError(f"df_log10_db failed with status {status}")
    return output


def _sqrt_f32(values):
    flat = values.detach().to(dtype=torch.float32, device="cpu").reshape(values.shape).contiguous()
    kernel = _rfft_kernel()
    if kernel is None:
        return flat.sqrt()
    output = torch.empty(flat.shape, dtype=torch.float32)
    status = kernel.df_sqrt_f32(_ptr(flat), flat.numel(), _ptr(output))
    if status != 0:
        raise RuntimeError(f"df_sqrt_f32 failed with status {status}")
    return output


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
        device = audio.device
        audio_cpu = audio.detach().to(dtype=torch.float32, device='cpu')
        window = self.window.detach().to(dtype=torch.float32, device='cpu')
        channels, time = audio_cpu.shape
        frames = time // self.hop
        spectrum = torch.empty(channels, frames, self.n_fft // 2 + 1, dtype=torch.complex64)
        memory = torch.zeros(self.n_fft - self.hop, dtype=torch.float32)
        for channel in range(channels):
            memory.zero_()
            for frame in range(frames):
                start = frame * self.hop
                spectrum[channel, frame], memory = _analyze_frame(
                    audio_cpu[channel, start:start + self.hop], memory, window, self.scale)
        self.analysis_memory = memory.to(device=self.device, dtype=self.dtype)
        return spectrum.to(device=device)

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
        device = spectrum.device
        spectrum = spectrum.detach().to(dtype=torch.complex64, device='cpu')
        channels, frames, _ = spectrum.shape
        dtype = torch.float32
        alpha = torch.tensor(self.alpha, dtype=dtype)
        one_minus = torch.tensor(1, dtype=dtype) - alpha
        self.mean_state = _linspace_state(-60.0, -90.0, self.n_erb, channels, spectrum.device, dtype)
        self.unit_state = _linspace_state(0.001, 0.0001, self.n_df, channels, spectrum.device, dtype)
        erb = _erb_power_db(spectrum, self.widths)
        for frame in range(frames):
            updated = erb[:, frame] * one_minus + self.mean_state * alpha
            self.mean_state = updated
            erb[:, frame] = (erb[:, frame] - updated) / 40.0
        df_bins = spectrum[..., :self.n_df].clone()
        kernel = _rfft_kernel()
        if kernel is not None:
            real = df_bins.real.contiguous()
            imag = df_bins.imag.contiguous()
            status = kernel.df_unit_norm(
                _ptr(real), _ptr(imag), channels, frames, self.n_df, ctypes.c_float(self.alpha),
            )
            if status != 0:
                raise RuntimeError(f"df_unit_norm failed with status {status}")
            df_bins = torch.complex(real, imag)
        else:
            magnitude = torch.hypot(df_bins.real, df_bins.imag)
            for frame in range(frames):
                updated = magnitude[:, frame] * one_minus + self.unit_state * alpha
                self.unit_state = updated
                df_bins[:, frame] = df_bins[:, frame] / _sqrt_f32(updated)
        spec = torch.view_as_real(spectrum).unsqueeze(1)
        return spec.to(device), erb.unsqueeze(1).to(device), torch.view_as_real(df_bins).unsqueeze(1).to(device)


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
    spectrum = _rfft_unnormalized(buffer).to(torch.complex64) * scale
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
    """Mean band power in decibels, summed in the same float32 order as libdf."""
    real = spectrum.real
    imag = spectrum.imag
    bands = []
    start = 0
    for width in widths:
        scale = torch.tensor(1, dtype=torch.float32) / torch.tensor(width, dtype=torch.float32)
        total = torch.zeros(real.shape[:-1], dtype=torch.float32)
        for offset in range(width):
            bin_real = real[..., start + offset]
            bin_imag = imag[..., start + offset]
            total = total + (bin_real * bin_real + bin_imag * bin_imag) * scale
        bands.append(total)
        start += width
    return _log10_db(torch.stack(bands, dim=-1))
