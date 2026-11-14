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
# snapshot identity must match. ``order_policy`` is not one of them: MP-SENet
# resolves it from the world size, so one string cannot describe the condition.
# ``require_official_sampling`` checks that resolution without extending schema v1.
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

# Configured sampler order, not the world-size resolution. A permutation of the
# MP-SENet pair is the same configuration ``resolve_order_policy`` accepts.
OFFICIAL_ORDERS: dict[str, tuple[str, ...]] = {
    MP_TRAINER: ("fixed_single_shuffle_v1", "torch_distributed_sampler_v1"),
    MOSS_TRAINER: ("rank_mod_fixed_v1",),
}
_UNSET = object()


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


def require_official_sampling(trainer: object, data: object, identity: dict[str, Any]) -> None:
    """Refuse an official run whose sampler order or sample cap is not the official one.

    The publication block stays schema v1. MP-SENet's resolved policy depends on
    the world size, so the configured order and the policy reported by
    ``sampler.policy(world_size)`` and ``identity()["order_policy"]`` are both
    required. ``max_epoch_samples`` and ``max_validation_batches`` must be unset.
    ``research`` does not call this function.
    """

    from src.experiments.data.errors import ReproductionError
    from src.experiments.data.sampling.order import resolve_order_policy
    from src.experiments.errors import DataError

    path = trainer_path(trainer)
    official = OFFICIAL_ORDERS.get(path)
    if official is None:
        raise RuntimeFailure(f"official_faithful has no sampler condition for {path}")
    sampler = getattr(data, "sampler", None)
    configured = getattr(sampler, "order", None)
    policy = getattr(sampler, "policy", None)
    if not _same_order(configured, official):
        raise RuntimeFailure(
            f"official_faithful sampler order {configured!r} does not match the official condition"
        )
    if not callable(policy):
        raise RuntimeFailure("official_faithful sampler order is missing")
    world_size = getattr(data, "world_size", None)
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise RuntimeFailure("official_faithful world_size is missing")
    try:
        expected = resolve_order_policy(_rule_order(official), world_size)
        resolved = policy(world_size)
    except (ReproductionError, DataError, TypeError, ValueError) as exc:
        raise RuntimeFailure(f"official_faithful order_policy is not valid: {exc}") from exc
    if resolved != expected:
        raise RuntimeFailure(
            f"official_faithful order_policy {resolved!r} does not match "
            f"the official condition for world_size {world_size}"
        )
    reported = identity.get("order_policy")
    if "order_policy" not in identity or reported != expected:
        raise RuntimeFailure(
            f"official_faithful order_policy {reported!r} does not match "
            f"the official condition for world_size {world_size}"
        )
    epoch_cap = getattr(sampler, "max_epoch_samples", _UNSET)
    if epoch_cap is _UNSET or epoch_cap is not None:
        shown = "missing" if epoch_cap is _UNSET else epoch_cap
        raise RuntimeFailure(f"official_faithful max_epoch_samples {shown!r} must be unset")
    valid_cap = getattr(data, "max_validation_batches", _UNSET)
    if valid_cap is _UNSET or valid_cap is not None:
        shown = "missing" if valid_cap is _UNSET else valid_cap
        raise RuntimeFailure(f"official_faithful max_validation_batches {shown!r} must be unset")


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


def _policy_names(order: object) -> tuple[str, ...] | None:
    if isinstance(order, str) and order:
        return (order,)
    if isinstance(order, list) and order and all(isinstance(item, str) and item for item in order):
        return tuple(order)
    return None


def _same_order(configured: object, official: tuple[str, ...]) -> bool:
    names = _policy_names(configured)
    if names is None:
        return False
    if len(official) == 1:
        return names == official
    return len(names) == len(official) and set(names) == set(official) and len(set(names)) == len(names)


def _rule_order(official: tuple[str, ...]) -> str | list[str]:
    if len(official) == 1:
        return official[0]
    return list(official)


def _equal(left: object, right: object) -> bool:
    return _canon(left) == _canon(right)


def _canon(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
