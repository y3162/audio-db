"""Classify MossFormerGAN buffers that the CPU comparison left unmatched.

Does not train. The CPU parameter and optimizer match is the existing log.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SE = ROOT / "src/se/official/clearervoice-studio/train/speech_enhancement"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SE))

from models.mossformer_gan.generator import SyncANet  # noqa: E402
from src.experiments.official.mossformergan.models import MossFormerGANGenerator  # noqa: E402

OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-level1-20261111/moss"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    official = SyncANet(num_channel=64, num_features=201).to(device)
    independent = MossFormerGANGenerator(seed=7).generator.to(device)
    official.load_state_dict(independent.state_dict(), strict=True)
    # Real/imag spectrogram [B, 2, T, F]. A short T only fills the RoPE cache.
    waveform = torch.zeros(1, 2, 8, 201, device=device)
    official.train()
    independent.train()
    with torch.no_grad():
        official(waveform)
        independent(waveform)
    official_buffers = {name: tuple(value.shape) for name, value in official.named_buffers()}
    independent_buffers = {name: tuple(value.shape) for name, value in independent.named_buffers()}
    missing = sorted(set(official_buffers) - set(independent_buffers))
    official_named = dict(official.named_buffers())
    report = {
        "official_commit": "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61",
        "load_state_dict_strict": True,
        "parameter_names_match": True,
        "missing_on_independent": missing,
        "missing_are_rope_cache": all(
            name.endswith("cached_freqs")
            or name.endswith("cached_freqs_seq_len")
            or name.endswith("rotary_pos_emb.dummy")
            or ".rotary_pos_emb." in name
            for name in missing
        ),
        "missing_requires_grad": any(bool(official_named[name].requires_grad) for name in missing),
        "official_buffer_count": len(official_buffers),
        "independent_buffer_count": len(independent_buffers),
    }
    path = OUT / "moss_buffers.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
