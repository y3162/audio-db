"""Render a content-addressed VoiceBank SNR-shift snapshot case in memory."""

from __future__ import annotations

import hashlib

import numpy as np

from src.experiments.data.assets.store import AssetStore
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import digest, validate_realization
from src.experiments.data.prepare.resample import resample_float32
from src.experiments.data.render.voicebank_shift import render_voicebank_shift_files


def render_shift_case(store: AssetStore, case_id: str, *, verify_on_read: bool = False) -> tuple[np.ndarray, np.ndarray]:
    document = store.docs.get(case_id)
    if not isinstance(document, dict):
        raise ReproductionError(f"case {case_id} has no SNR-shift realization")
    validate_realization(document)
    if document["algorithm"] != "fixed_pair.voicebank_snr_shift" or digest(document) != case_id:
        raise ReproductionError(f"case {case_id} realization identity does not match")
    paths = {}
    for name, source in document["sources"].items():
        asset = store.asset(source["asset"])
        if asset.get("sha256") != source["sha256"]:
            raise ReproductionError(f"case {case_id} source hash does not match snapshot")
        path = store.path(source["asset"])
        if verify_on_read:
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != source["sha256"]:
                raise ReproductionError(f"case {case_id} source file sha256 mismatch")
        paths[name] = path
    draws = document["draws"]
    row = {"status": "ready", "noise_start_48k": draws["noise_start_48k"],
           "noise_end_exclusive_48k": draws["noise_end_exclusive_48k"],
           "clean_gain_g": draws["clean_gain"], "shifted_noise_gain": draws["shifted_noise_gain"]}
    noisy, clean = render_voicebank_shift_files(row, paths["clean"], paths["noise"])
    noisy16, clean16 = resample_float32(noisy, 48000), resample_float32(clean, 48000)
    expected = int(document["output"]["length"])
    if noisy16.shape != (expected,) or clean16.shape != (expected,):
        raise ReproductionError(f"case {case_id} resampled length does not match snapshot")
    return noisy16, clean16
