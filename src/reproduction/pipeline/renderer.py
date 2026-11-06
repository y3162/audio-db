"""Read one planned sample from disk.

The renderer applies the dataset protocol. It does not look at
``cases.protocol_id``. P2 stores one shared case row for both models, and that
row points at the MP-SENet protocol.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.pipeline.transform import apply_window, independent_audio_norm, noisy_rms_unit, trim_to_shorter
from src.reproduction.pipeline.types import RenderedSample, SampleRef


def render_sample(
    store: AssetStore,
    sample: SampleRef,
    *,
    normalize: str,
    sample_rate: int,
    verify_on_read: bool = False,
) -> RenderedSample:
    if not sample.case_id:
        raise ReproductionError("fixed-pair rendering requires a case id")
    case = store.case(sample.case_id)
    if case.get("teacher_kind") in (None, "none"):
        raise ReproductionError(f"case {sample.case_id} has no teacher")
    clean_id = case.get("clean_asset_id")
    noisy_id = case.get("noisy_asset_id")
    if not clean_id or not noisy_id:
        raise ReproductionError(f"case {sample.case_id} is not a recorded pair")
    clean, clean_rate = _read_mono(store.path(clean_id), verify_sha=_sha(store, clean_id, verify_on_read))
    noisy, noisy_rate = _read_mono(store.path(noisy_id), verify_sha=_sha(store, noisy_id, verify_on_read))
    if clean_rate != sample_rate or noisy_rate != sample_rate:
        raise ReproductionError(
            f"case {sample.case_id} is {noisy_rate}/{clean_rate} Hz; "
            f"refusing to resample to {sample_rate}"
        )
    _expect_frames(store, noisy_id, noisy, "noisy")
    _expect_frames(store, clean_id, clean, "clean")
    if normalize == "noisy_rms_unit_v1":
        noisy, clean = trim_to_shorter(noisy, clean)
        noisy, clean = noisy_rms_unit(noisy, clean)
    elif normalize == "independent_audio_norm_v1":
        if noisy.shape != clean.shape:
            raise ReproductionError(
                f"case {sample.case_id} clean and noisy lengths differ "
                f"({clean.shape[0]} vs {noisy.shape[0]})"
            )
        noisy = independent_audio_norm(noisy)
        clean = independent_audio_norm(clean)
    else:
        raise ReproductionError(f"normalization {normalize} is not implemented")
    noisy, clean, valid = apply_window(noisy, clean, sample.crop)
    if noisy.shape != clean.shape:
        raise ReproductionError(f"case {sample.case_id} lost clean/noisy alignment")
    if not np.isfinite(noisy).all() or not np.isfinite(clean).all():
        raise ReproductionError(f"case {sample.case_id} rendered a non-finite waveform")
    return RenderedSample(
        noisy=noisy,
        clean=clean,
        sample_rate=sample_rate,
        valid_length=valid,
        case_id=sample.case_id,
        teacher_kind=str(case["teacher_kind"]),
        extras={},
    )


def _sha(store: AssetStore, asset_id: str, verify: bool) -> str | None:
    if not verify:
        return None
    digest = store.asset(asset_id).get("sha256")
    if not isinstance(digest, str):
        raise ReproductionError(f"asset {asset_id} has no sha256 to verify")
    return digest


def _expect_frames(store: AssetStore, asset_id: str, wave: np.ndarray, label: str) -> None:
    expected = int(store.asset(asset_id)["frames"])
    if int(wave.shape[0]) != expected:
        raise ReproductionError(
            f"{label} asset {asset_id} has {wave.shape[0]} samples; the snapshot says {expected}"
        )


def _read_mono(path: Path, *, verify_sha: str | None) -> tuple[np.ndarray, int]:
    if verify_sha is not None:
        actual = _file_sha256(path)
        if actual != verify_sha:
            raise ReproductionError(f"sha256 mismatch for {path}")
    import soundfile as sf

    if not path.is_file():
        raise ReproductionError(f"audio file is missing: {path}")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size < 1 or int(audio.shape[1]) < 1:
        raise ReproductionError(f"audio file is empty: {path}")
    # Official loaders keep the first channel.
    mono = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ReproductionError(f"audio file is non-finite: {path}")
    return mono, int(rate)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
