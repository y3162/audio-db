"""Official-reproduction identity checked before a training step.

The IDs are the float32 VoiceBank snapshots that the official experiment
configs open. They were read from those snapshots, not invented. A directory
name is not part of the condition. ``research`` does not consult this table.
"""

from __future__ import annotations

import json
from typing import Any

from src.experiments.errors import RuntimeFailure

SNAPSHOT_CLASS = "src.experiments.data.SnapshotDataModule"
MP_TRAINER = "src.experiments.official.mp_senet.trainer.MPSENetGANTrainer"
MOSS_TRAINER = "src.experiments.official.mossformergan.trainer.MossFormerGANTrainer"

_RESAMPLE = {
    "tool": "soxr",
    "quality": "HQ",
    "source_rate": 48000,
    "target_rate": 16000,
    "length": "source_frames_plus_one_div_3",
    "order": [
        "read_float32",
        "soxr_hq",
        "protocol_normalize",
        "protocol_window",
    ],
}

# Fields the official experiment YAML must declare, and that the opened
# snapshot identity must match. ``order_policy`` is omitted: MP-SENet resolves
# it from the world size, and both resolutions belong to the same protocol.
PUBLICATION_KEYS = (
    "algorithm",
    "claim_label",
    "dataset_id",
    "normalize",
    "profile",
    "protocol_id",
    "resample",
    "sample_rate",
    "selection_uses_test",
    "snapshot_hash",
    "split",
    "teacher",
    "validation_crop",
    "validation_scope",
    "window",
)


def _condition(**fields: Any) -> dict[str, Any]:
    body = {
        "profile": "official_faithful",
        "claim_label": "official_reproduction",
        "selection_uses_test": True,
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "split": "train",
        "window": 32000,
        "resample": {**_RESAMPLE, "order": list(_RESAMPLE["order"])},
    }
    body.update(fields)
    missing = [key for key in PUBLICATION_KEYS if key not in body]
    if missing:
        raise RuntimeError(f"official condition is missing {missing[0]}")
    return {key: body[key] for key in PUBLICATION_KEYS}


OFFICIAL_CONDITIONS = {
    MP_TRAINER: _condition(
        algorithm="fixed_pair.mp_senet",
        dataset_id="2e1ebda5109626e04198b8dd16c1948cd2bde159087a16001fa9239c2860c67f",
        protocol_id="4d3e13e34a7c6beaa8406bee8ae006951aa12a8f88791f8ca29c6ce57a9bf6ae",
        snapshot_hash="9f342b0fccf5a021971d918a14821b8c18a05ea7de05d97086f9785dbaa413dc",
        normalize="noisy_rms_unit_v1",
        validation_crop="full_length",
        validation_scope="rank0",
    ),
    MOSS_TRAINER: _condition(
        algorithm="fixed_pair.mossformergan_se_16k",
        dataset_id="18c0d64a78edc2914cc1ea71ea6558a0fa54d52f03252af843d49fc417c3c69c",
        protocol_id="6a0fcf415f437d0e0a9b2a4f5cc5670645e48dc28536babd4fef860b99f00a33",
        snapshot_hash="8d9107b91fe36e11a76c5b6e686d29dcad16064a6c3de7598c82dd195e5056f5",
        normalize="independent_audio_norm_v1",
        validation_crop="per_epoch_random",
        validation_scope="per_rank",
    ),
}


def trainer_path(trainer: object) -> str:
    kind = type(trainer)
    return f"{kind.__module__}.{kind.__qualname__}"


def official_condition(trainer: object) -> dict[str, Any] | None:
    return OFFICIAL_CONDITIONS.get(trainer_path(trainer))


def require_declared_publication(trainer: object, declared: object) -> None:
    """Refuse an ``official_faithful`` config whose publication block is not the known condition."""

    condition = official_condition(trainer)
    path = trainer_path(trainer)
    if condition is None:
        raise RuntimeFailure(f"official_faithful has no dataset condition for {path}")
    if not isinstance(declared, dict):
        raise RuntimeFailure("official_faithful publication block is missing")
    if _canon(declared) != _canon(condition):
        raise RuntimeFailure(
            "official_faithful publication block does not match the official dataset condition for " + path
        )


def require_official_identity(trainer: object, identity: dict[str, Any]) -> None:
    """Refuse a snapshot that is not this trainer's official dataset and pipeline."""

    condition = official_condition(trainer)
    path = trainer_path(trainer)
    if condition is None:
        raise RuntimeFailure(f"official_faithful has no dataset condition for {path}")
    kind = identity.get("class")
    if kind != SNAPSHOT_CLASS:
        label = kind if isinstance(kind, str) and kind else "missing class"
        raise RuntimeFailure(f"official_faithful data class {label} is not the official snapshot")
    for key in PUBLICATION_KEYS:
        if key not in identity or identity[key] is None:
            raise RuntimeFailure(f"official_faithful data is missing {key}")
        if not _equal(identity[key], condition[key]):
            if key == "selection_uses_test" and identity[key] is False:
                raise RuntimeFailure("official_faithful selection uses the dataset test split as validation")
            if key == "profile":
                raise RuntimeFailure("official_faithful data profile must be official_faithful")
            if key in {"normalize", "window", "validation_crop", "validation_scope", "resample"}:
                raise RuntimeFailure(
                    f"official_faithful pipeline {key} {identity[key]!r} does not match the official condition"
                )
            raise RuntimeFailure(
                f"official_faithful {key} {identity[key]!r} does not match the official condition"
            )


def require_leakage_identity(identity: dict[str, Any]) -> None:
    """Refuse a leakage-safe claim whose test use cannot be shown to be false.

    Snapshot modules still run their own train/valid/test overlap check when
    they open. This function does not treat an unknown flag as that check.
    """

    profile = identity.get("profile")
    if profile != "leakage_safe":
        raise RuntimeFailure("leakage_safe data profile must be leakage_safe")
    uses_test = identity.get("selection_uses_test")
    if uses_test is None:
        raise RuntimeFailure("leakage_safe selection_uses_test is unknown")
    if uses_test is not False:
        raise RuntimeFailure("leakage_safe selection keeps the test split out of best-checkpoint decisions")
    claim = identity.get("claim_label")
    if claim is not None and claim != "leakage_safe_comparison":
        raise RuntimeFailure(f"leakage_safe claim_label {claim} does not match the snapshot declaration")


def _equal(left: object, right: object) -> bool:
    return _canon(left) == _canon(right)


def _canon(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
