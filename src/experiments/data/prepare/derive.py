"""Pre-written 16 kHz VoiceBank wavs are no longer produced.

Training reads the 48 kHz sources and resamples them in
``src.experiments.data.prepare.resample``. This module remains so old imports fail
with a clear message instead of writing audio.
"""

from __future__ import annotations

from src.experiments.data.errors import ReproductionError


def derive_all(*_args, **_kwargs):
    raise ReproductionError(
        "the derive step no longer writes 16 kHz wavs; training resamples 48 kHz sources"
    )
