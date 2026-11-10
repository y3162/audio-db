"""Compare DeepFilterNet3 training synthesis with the published Istft.

Official training builds ``df.loss.Istft`` from the libdf FFT window and calls
``torch.istft``. ``libdf.DF.synthesis`` is the streaming enhancer path, not the
training loss. This script records both differences and does not change either
implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".venvs/tsubame-libdf"))
sys.path.insert(0, str(ROOT / "src/se/official/deepfilternet/DeepFilterNet"))

from libdf import DF  # noqa: E402
from src.se.models.deepfilternet3.dsp import vorbis_window  # noqa: E402
from src.experiments.calc.deepfilternet import _vorbis_istft  # noqa: E402


def official_istft(spectrum: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """``df.loss.Istft.forward``: complex STFT, one extra frame, normalized istft."""
    values = torch.view_as_complex(spectrum) if not spectrum.is_complex() else spectrum
    frames, bins = values.shape[-2:]
    leading = values.shape[:-2]
    transformed = torch.istft(
        torch.nn.functional.pad(values.reshape(-1, frames, bins).transpose(1, 2), (0, 1)),
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        normalized=True,
    )
    if values.ndim > 2:
        transformed = transformed.view(*leading, transformed.shape[-1])
    return transformed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    noisy = payload["noisy"][0]
    if noisy.dim() == 1:
        noisy = noisy.unsqueeze(0)
    sample_rate, n_fft, hop = 48000, 960, 480
    state = DF(sample_rate, n_fft, hop, 32, 32)
    official_window = torch.from_numpy(state.fft_window().copy()).float()
    local_window = vorbis_window(n_fft).float()
    # Official analysis includes the model's lookahead pad. Match that length.
    padded = torch.nn.functional.pad(noisy, (0, n_fft))
    spectrum = torch.from_numpy(state.analysis(padded.numpy()))
    official_wave = official_istft(spectrum, n_fft, hop, official_window)
    probe_wave = _vorbis_istft(spectrum.squeeze(1), n_fft, hop, local_window)
    length = min(official_wave.shape[-1], probe_wave.shape[-1])
    libdf_wave = torch.from_numpy(state.synthesis(spectrum.numpy())).float()
    lib_length = min(length, libdf_wave.shape[-1])
    result = {
        "window_max_abs": float((official_window - local_window).abs().max()),
        "official_istft_vs_probe_max_abs": float(
            (official_wave[..., :length] - probe_wave[..., :length]).abs().max()
        ),
        "libdf_synthesis_vs_official_istft_max_abs": float(
            (libdf_wave[..., :lib_length] - official_wave[..., :lib_length]).abs().max()
        ),
        "shapes": {
            "official_istft": list(official_wave.shape),
            "probe": list(probe_wave.shape),
            "libdf_synthesis": list(libdf_wave.shape),
        },
        "note": (
            "Training loss uses df.loss.Istft (torch.istft, normalized, one extra frame). "
            "libdf.DF.synthesis is not the training-loss inverse."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
