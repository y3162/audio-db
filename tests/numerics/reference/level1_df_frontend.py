"""Compare DeepFilterNet3 DFStream features with official libdf.

The numeric rules are in VALIDATION_PLAN.md. This does not train.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LIBDF = Path(os.environ.get("AUDIO_DB_LIBDF_PATH", "/tmp/audio-db-official-numerics/libdf-py"))
if str(LIBDF) not in sys.path:
    sys.path.insert(0, str(LIBDF))

OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-level1-20261111/df"))


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def main() -> None:
    import libdf
    from src.se import build_model
    from src.se.models.deepfilternet3.dsp import DFStream

    OUT.mkdir(parents=True, exist_ok=True)
    params = build_model("deepfilternet3", None).params
    stream = DFStream(
        params.sr,
        params.fft_size,
        params.hop_size,
        params.nb_erb,
        params.nb_df,
        params.min_nb_freqs,
        params.norm_tau,
        torch.device("cpu"),
    )
    reference = libdf.DF(params.sr, params.fft_size, params.hop_size, params.nb_erb, params.min_nb_freqs)
    generator = np.random.default_rng(7)
    # 1600 samples at 16 kHz is the length used by the existing official loop
    # before resample. At 48 kHz that is 4800 samples, matching the unit test.
    rows = []
    waveforms = {
        "normal": generator.standard_normal((2, 4800)).astype(np.float32),
        "quiet_second": None,
        "linspace": torch.linspace(-0.2, 0.2, 4800).view(1, -1).numpy().astype(np.float32),
    }
    waveforms["quiet_second"] = waveforms["normal"].copy()
    waveforms["quiet_second"][1] *= np.float32(1e-4)
    for name, audio in waveforms.items():
        stream.reset()
        expected = np.array(reference.analysis(audio), copy=True)
        actual = stream.analyze(torch.from_numpy(np.ascontiguousarray(audio))).numpy()
        alpha = float(stream.alpha)
        widths = reference.erb_widths()
        erb = np.array(libdf.erb_norm(libdf.erb(expected, widths), alpha), copy=True)
        unit = np.array(libdf.unit_norm(np.array(expected[..., : params.nb_df], copy=True), alpha), copy=True)
        spec, erb_hat, unit_hat = stream.features(torch.from_numpy(expected))
        unit_real = torch.view_as_real(torch.from_numpy(unit)).numpy()
        rows.append(
            {
                "name": name,
                "analysis_max_abs": _max_abs(actual, expected),
                "erb_max_abs": _max_abs(erb_hat[:, 0].numpy(), erb),
                "unit_max_abs": _max_abs(unit_hat[:, 0].numpy(), unit_real),
                "spec_max_abs": _max_abs(
                    spec[:, 0].numpy(),
                    torch.view_as_real(torch.from_numpy(expected)).numpy(),
                ),
                "shape": list(expected.shape),
            }
        )
        print(name, rows[-1], flush=True)
    report = {
        "official_commit": "d375b2d8309e0935d165700c91da9de862a99c31",
        "libdf_path": str(LIBDF),
        "params": {
            "sr": params.sr,
            "fft_size": params.fft_size,
            "hop_size": params.hop_size,
            "nb_erb": params.nb_erb,
            "nb_df": params.nb_df,
            "min_nb_freqs": params.min_nb_freqs,
            "norm_tau": params.norm_tau,
        },
        "device": "cpu",
        "rows": rows,
    }
    path = OUT / "df_frontend.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
