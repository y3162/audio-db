"""Clip readers shared by the online training classes and the experiment data API.

DeepFilterNet3 reads a native span with soundfile. SEMamba++ loads the whole
file through ``librosa.load``. The two readers stay separate so a change to
one path cannot change the other waveform.
"""

from __future__ import annotations

import numpy as np

from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.protocols.online_semamba import SAMPLE_RATE


class Dfn3Clip:
    """One DeepFilterNet3 asset. ``read`` returns a native-rate mono span."""

    def __init__(self, store: AssetStore, asset_id: str) -> None:
        row = store.asset(asset_id)
        self.asset_id = asset_id
        self.sample_rate = int(row["sample_rate"])
        self.frames = int(row["frames"])
        self.path = store.path(asset_id)

    def read(self, start: int, stop: int) -> np.ndarray:
        import soundfile as sf

        data, rate = sf.read(self.path, start=int(start), stop=int(stop), dtype="float32", always_2d=True)
        if int(rate) != self.sample_rate:
            raise ReproductionError(f"{self.asset_id} sample rate does not match the snapshot")
        return np.ascontiguousarray(data[:, 0], dtype=np.float32)


class SemambaClip:
    """One SEMamba++ asset. ``read`` returns the whole file at 16 kHz."""

    def __init__(self, store: AssetStore, asset_id: str) -> None:
        row = store.asset(asset_id)
        self.asset_id = asset_id
        self.sample_rate = int(row["sample_rate"])
        self.path = store.path(asset_id)

    def read(self) -> np.ndarray:
        import librosa

        audio, rate = librosa.load(self.path, sr=SAMPLE_RATE, mono=True)
        if int(rate) != SAMPLE_RATE:
            raise ReproductionError(f"{self.asset_id} did not load at 16000 Hz")
        if audio.size < 1:
            raise ReproductionError(f"{self.asset_id} is empty")
        return np.asarray(audio, dtype=np.float64)
