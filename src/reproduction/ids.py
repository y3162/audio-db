"""Content-addressed ids for the reproduction database.

Hashes reuse ``src.data.recipe`` (sorted keys, 6 decimal places, SHA-256).
Names and integer versions are not part of a protocol id or a dataset id.
"""

from __future__ import annotations

import re
from typing import Any

from src.data.recipe import RecipeError, _dump, _normalize, case_id_for, json_hash
from src.reproduction.errors import ReproductionError

SCHEMA_VERSION = "1"

CLAIM_LABELS = (
    "official_reproduction",
    "protocol_conformant_substitute_materials",
    "espnet_public_se_recipe_reproduction",
    "official_recipe_scaled_down",
    "minimal_repair_reproduction",
    "leakage_safe_comparison",
)
DEVIATION_CLASSES = (
    "official_effective",
    "minimal_launch_repair",
    "algorithm_change",
    "research_improvement",
)
PROFILES = ("official_faithful", "leakage_safe")
MATERIAL_ORIGINS = ("official", "derived_from_official", "substitute", "mixed", "unknown")
SET_ORIGINS = ("official", "substitute", "derived_from_official", "unknown")
DECLARED_STATUSES = ("ready", "blocked", "unknown")
KINDS = ("fixed_pair", "pregenerated", "online_mixing")
STORAGE_ROOTS = ("corpora", "generated", "derived")
INTEGRITIES = ("sha256", "size", "catalog")
INTEGRITY_RANK = {"sha256": 2, "size": 1, "catalog": 0}
CASE_ROLES = ("recorded_pair", "generated", "online_snapshot")
DOC_KINDS = ("recipe_v1", "realization_v1")
TEACHERS = ("dry_clean", "reverberant_clean", "official_mixed_target", "none")
SPLIT_ROLES = ("train", "valid", "test")
MATERIAL_SPLIT_ROLES = ("train", "valid", "test", "pool")
MATERIAL_ROLES = ("speech", "noise", "rir")
SOURCE_KINDS = ("resource_duckdb", "staging_parquet")
ORDER_POLICIES = (
    "global_shuffle_stride_v1",
    "torch_distributed_sampler_v1",
    "fixed_single_shuffle_v1",
    "rank_mod_fixed_v1",
    "per_rank_independent_v1",
    "espnet_chunk_v1",
    "dfn3_online_ceil_v1",
)
SELECTION_RULES = {
    "explicit": set(),
    "official_filelist": {"file"},
    "subset_exclude_speakers": {"exclude"},
    "sorted_ratio": {"key", "ratio", "part"},
    "speaker_holdout": {"speakers"},
    "fixed_subset": {"of", "k"},
}
PROVENANCE_KEYS = {"source", "repo", "commit", "path", "file", "sha256", "note"}
OFFICIAL_REF_KEYS = {"repo", "commit", "path", "section", "document"}
RULE_KEYS = {
    "sample_rate",
    "teacher",
    "normalize",
    "window",
    "order",
    "epoch_unit",
    "global_batch",
    "validation",
    "delivery",
    "resample",
    "retry_policy",
    "renderer",
    "online",
}
# Shipped DeepFilterNet3 online-mixing distributions. The protocol id hashes
# this object. audio-db does not reproduce Rust StdRng bits.
DFN3_ONLINE_V1 = {
    "sampling_factor": 1,
    "snrs": [-100, -5, 0, 5, 10, 20, 40],
    "gains": [-6, 0, 6],
    "noise_count_low": 2,
    "noise_count_high_exclusive": 6,
    "p_synthetic_noise_train": 0.05,
    "synthetic_decay": [-2.0, 2.0],
    "synthetic_gains": [-24, -12, -6, 0],
    "p_reverb": 0.1,
    "drr": 0.3,
    "rt60": 0.5,
    "offset_late_ms": 20,
    "p_decay_floor": 0.5,
    "rir_resample": [0.8, 1.2],
    "rir_max_samples": 1440000,
    "speech_aug": {
        "p_remove_dc": 0.25,
        "p_lfilt": 0.25,
        "lfilt_ab": [-0.375, 0.375],
        "p_biquad": 0.0,
        "p_resample": 0.1,
        "resample_ratio": [0.9, 1.1],
        "resample_step_hz": 500,
    },
    "noise_aug": {
        "p_lfilt": 0.25,
        "p_biquad": 0.0,
        "p_resample": 0.1,
        "p_clip_train": 0.1,
        "clip_c": [0.01, 0.5],
    },
    "disabled": {
        "p_clipping_speech": 0.0,
        "p_zeroing": 0.0,
        "p_air_absorption": 0.0,
        "p_bandwidth_ext": 0.0,
        "p_interfer_sp": 0.0,
    },
    "renderer_version": "dfn3_online_v1",
    "official_commit": "d375b2d8309e0935d165700c91da9de862a99c31",
    "seed_rule": "protocol_snapshot_seed_split_epoch_index",
    "batch_schedule": [[0, 16], [2, 24], [5, 32], [10, 64], [20, 128], [40, 9999]],
    "batch_size_limit": 64,
    "keep_remainder": True,
    "official_seed": 43,
}
# HEAD config.yaml of Eddieyjlee/SEMambapp @ cd369c89. Not config_orig.yaml
# and not the paper's 24,000-sample / −10 dB setting. Tiny constants are
# strings so the 6-decimal recipe hash does not collapse 1e-9 to 0.
SEMAMBA_GSR_V1 = {
    "config_name": "config.yaml",
    "official_commit": "cd369c89a8106072c077b468417b765b227afd74",
    "sample_rate": 16000,
    "segment_size": 48000,
    "snr": [-5, 0, 5, 10, 15, 20],
    "bandwidth_sr": [4000, 6000, 8000, 10000, 12000, 14000],
    "bandwidth_type": ["kaiser_best", "kaiser_fast", "scipy", "polyphase"],
    "lowpass_type": ["butter", "cheby1", "cheby2", "bessel", "ellip"],
    "lowpass_order": [2, 4, 6, 8],
    "lowpass_ripple": 1,
    "lowpass_stop_atten": 40,
    "clipping_min": [0.0, 0.1],
    "clipping_max": [0.9, 1.0],
    "packet_duration": [10, 20],
    "packet_loss_rate": [0.05, 0.25],
    "noise_prob": 1.0,
    "reverb_prob": 0.5,
    "bandwidth_prob": 0.5,
    "clipping_prob": 0.5,
    "packet_loss_prob": 0.2,
    "packet_loss_selected": False,
    "degrad_order": ["reverb", "noise", "clipping", "bandwidth"],
    "discard_first_draw": True,
    "vad_frame": 1024,
    "vad_shift": 512,
    "vad_threshold": 0.01,
    "vad_window": "boxcar",
    "peak_eps": "1e-9",
    "snr_power_floor": "1e-10",
    "normalize": "independent_peak_v1",
    "order": "per_rank_independent_v1",
    "teacher": "dry_clean",
    "rir_normalize": False,
    "rir_delay_compensation": False,
    "renderer_version": "semamba_gsr_v1",
    "seed_rule": "semamba-gsr protocol snapshot seed split epoch index",
    "load_resample": "librosa_load_sr16000_mono",
}
RESAMPLE_KEYS = {
    "tool",
    "quality",
    "source_rate",
    "target_rate",
    "quantization",
    "length",
    "order",
}
RESAMPLE_ORDER = (
    "read_float32",
    "soxr_hq",
    "clip_to_unit",
    "pcm16_roundtrip",
    "protocol_normalize",
    "protocol_window",
)
VALIDATION_KEYS = {"crop", "batch", "interval", "note"}
REALIZATION_KEYS = {
    "realization_version",
    "algorithm",
    "algorithm_version",
    "sample_rate",
    "sources",
    "draws",
    "steps",
    "output",
}
NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_HEX = set("0123456789abcdef")

# Closed envelope for P1. Full per-model dataclasses belong to the protocol
# package, which is a later phase. Unknown algorithms and unknown keys are rejected.
ALGORITHMS: dict[str, dict[str, Any]] = {
    "fixed_pair.mp_senet": {
        "kind": "fixed_pair",
        "version": 1,
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "normalize": "noisy_rms_unit_v1",
    },
    "fixed_pair.mossformergan_se_16k": {
        "kind": "fixed_pair",
        "version": 1,
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "normalize": "independent_audio_norm_v1",
    },
    "online.dfn3": {
        "kind": "online_mixing",
        "version": 1,
        "sample_rate": 48000,
        "teacher": "official_mixed_target",
        "normalize": None,
    },
    "online.semambapp_gsr": {
        "kind": "online_mixing",
        "version": 1,
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "normalize": "independent_peak_v1",
    },
    "pregenerated.ms_snsd_tfgridnet": {
        "kind": "pregenerated",
        "version": 1,
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "normalize": None,
    },
    "pregenerated.dns_tflocoformer": {
        "kind": "pregenerated",
        "version": 1,
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "normalize": None,
    },
}

CASE_SLOTS = ("train.cases", "valid.cases", "test.cases")
MATERIAL_SLOTS = tuple(
    f"{split}.{role}" for split in ("train", "valid", "test") for role in MATERIAL_ROLES
)
ALL_SLOTS = CASE_SLOTS + MATERIAL_SLOTS
REQUIRED_SLOTS = {
    "fixed_pair": ("train.cases", "valid.cases", "test.cases"),
    "pregenerated": ("train.cases", "valid.cases", "test.cases"),
    "online_mixing": ("train.speech", "valid.cases"),
}
ALLOWED_SLOTS = {
    "fixed_pair": CASE_SLOTS,
    "pregenerated": CASE_SLOTS + MATERIAL_SLOTS,
    "online_mixing": MATERIAL_SLOTS + ("valid.cases", "test.cases"),
}


def digest(value: Any) -> str:
    try:
        return json_hash(value)
    except RecipeError as exc:
        raise ReproductionError(str(exc)) from exc


def canonical_text(value: Any) -> str:
    try:
        return _dump(_normalize(value))
    except RecipeError as exc:
        raise ReproductionError(str(exc)) from exc


def recipe_case_id(recipe: dict[str, Any]) -> str:
    try:
        return case_id_for(recipe)
    except RecipeError as exc:
        raise ReproductionError(str(exc)) from exc


def check_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or NAME_RE.fullmatch(value) is None:
        raise ReproductionError(f"{where} must match {NAME_RE.pattern}")
    return value


def check_sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise ReproductionError(f"{where} must be 64 lowercase hex characters")
    return value


def check_optional_sha256(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return check_sha256(value, where)


def positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReproductionError(f"{where} must be a positive integer")
    return value


def non_negative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReproductionError(f"{where} must be a non-negative integer")
    return value


def check_rel_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise ReproductionError(f"{where} must be a relative path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ReproductionError(f"{where} must not contain empty, '.' , or '..' parts")
    return value


def check_segment(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or "/" in value or value in (".", ".."):
        raise ReproductionError(f"{where} must be a single path segment")
    return value


def require_object(value: Any, where: str, allowed: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReproductionError(f"{where} must be an object")
    if allowed is not None:
        extra = set(value) - allowed
        if extra:
            raise ReproductionError(f"{where} has unknown fields: {', '.join(sorted(extra))}")
    return value


def protocol_id_for(
    *,
    algorithm: str,
    algorithm_version: int,
    kind: str,
    rule: dict[str, Any],
    deviation_class: str,
    parent: str | None,
) -> str:
    return digest(
        {
            "doc_type": "protocol",
            "algorithm": algorithm,
            "algorithm_version": algorithm_version,
            "kind": kind,
            "rule": rule,
            "deviation_class": deviation_class,
            "parent": parent,
        }
    )


def dataset_id_for(
    *,
    protocol_id: str,
    bindings: list[dict[str, Any]],
    profile: str,
    material_origin: str,
    claim_label: str,
    selection_uses_test: bool,
    declared_status: str,
) -> str:
    encoded = []
    for binding in sorted(bindings, key=lambda item: item["slot"]):
        encoded.append(
            {
                "slot": binding["slot"],
                "case_set_id": binding.get("case_set_id"),
                "material_set_id": binding.get("material_set_id"),
            }
        )
    return digest(
        {
            "doc_type": "dataset",
            "protocol_id": protocol_id,
            "bindings": encoded,
            "profile": profile,
            "material_origin": material_origin,
            "claim_label": claim_label,
            "selection_uses_test": selection_uses_test,
            "declared_status": declared_status,
        }
    )


def material_members_key(asset_id: str, sha256: str | None, size_bytes: int | None) -> list[Any]:
    if sha256 is not None:
        return [asset_id, sha256]
    if size_bytes is None:
        raise ReproductionError(f"material member {asset_id} needs sha256 or size_bytes")
    return [asset_id, size_bytes]


def material_set_id_for(
    *,
    name: str,
    version: int,
    role: str,
    split_role: str,
    rule: dict[str, Any],
    members: list[list[Any]],
) -> str:
    return digest(
        {
            "doc_type": "material_set",
            "name": name,
            "version": version,
            "role": role,
            "split_role": split_role,
            "rule": rule,
            "members": members,
        }
    )


def case_set_id_for(
    *,
    name: str,
    version: int,
    split_role: str,
    rule: dict[str, Any],
    case_ids: list[str],
) -> str:
    return digest(
        {
            "doc_type": "case_set",
            "name": name,
            "version": version,
            "split_role": split_role,
            "rule": rule,
            "case_ids": case_ids,
        }
    )


def build_id_for(
    *,
    generator: str,
    generator_version: int,
    params: dict[str, Any],
    seed: int,
    inputs: dict[str, str],
) -> str:
    return digest(
        {
            "doc_type": "build",
            "generator": generator,
            "generator_version": generator_version,
            "params": params,
            "seed": seed,
            "input_material_set_ids": inputs,
        }
    )


def derivation_id_for(*, kind: str, tool: str, tool_version: str, params: dict[str, Any]) -> str:
    return digest({"kind": kind, "tool": tool, "tool_version": tool_version, "params": params})


def catalog_id_for(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: row["path"])
    return digest(ordered)


def audit_id_for(*, subject_kind: str, subject_id: str, audit_kind: str, report: dict[str, Any]) -> str:
    return digest(
        {
            "subject_kind": subject_kind,
            "subject_id": subject_id,
            "audit_kind": audit_kind,
            "report": report,
        }
    )


def asset_batch_id(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: row["asset_id"])
    return digest(ordered)


def case_content_hash(row: dict[str, Any]) -> str:
    body = {key: value for key, value in row.items() if key not in ("case_id", "content_hash")}
    return digest(body)


def generated_rel_path(
    *,
    generator: str,
    build_id: str,
    stream: str,
    gen_index: int,
    shard_size: int,
) -> str:
    check_segment(generator, "generator")
    check_sha256(build_id, "build_id")
    check_segment(stream, "stream")
    non_negative_int(gen_index, "gen_index")
    positive_int(shard_size, "shard_size")
    shard = gen_index // shard_size
    if shard > 9999 or gen_index > 9_999_999:
        raise ReproductionError("gen_index does not fit the generated path width")
    return f"{generator}/{build_id[:16]}/{stream}/s{shard:04d}/fid{gen_index:07d}.wav"


def validate_algorithm(algorithm: str, algorithm_version: int, kind: str) -> dict[str, Any]:
    spec = ALGORITHMS.get(algorithm)
    if spec is None:
        raise ReproductionError(f"unknown algorithm: {algorithm}")
    if algorithm_version != spec["version"]:
        raise ReproductionError(
            f"unsupported {algorithm} algorithm_version: {algorithm_version}"
        )
    if kind != spec["kind"]:
        raise ReproductionError(f"{algorithm} kind must be {spec['kind']}")
    return spec


def validate_rule(algorithm: str, rule: Any) -> dict[str, Any]:
    spec = ALGORITHMS[algorithm]
    body = require_object(rule, "rule_json", RULE_KEYS)
    if set(body) < {"sample_rate", "teacher"}:
        raise ReproductionError("rule_json needs sample_rate and teacher")
    if body["sample_rate"] != spec["sample_rate"]:
        raise ReproductionError(f"{algorithm} sample_rate must be {spec['sample_rate']}")
    if body["teacher"] != spec["teacher"]:
        raise ReproductionError(f"{algorithm} teacher must be {spec['teacher']}")
    if spec["normalize"] is None:
        if "normalize" in body and body["normalize"] is not None:
            raise ReproductionError(f"{algorithm} does not take normalize")
    else:
        if body.get("normalize") != spec["normalize"]:
            raise ReproductionError(f"{algorithm} normalize must be {spec['normalize']}")
    if "window" in body:
        positive_int(body["window"], "rule_json.window")
    if "order" in body:
        _check_order(body["order"])
    if "epoch_unit" in body and not isinstance(body["epoch_unit"], str):
        raise ReproductionError("rule_json.epoch_unit must be a string")
    if "global_batch" in body:
        positive_int(body["global_batch"], "rule_json.global_batch")
    if "validation" in body:
        extra = require_object(body["validation"], "rule_json.validation", VALIDATION_KEYS)
        for key, value in extra.items():
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                raise ReproductionError(f"rule_json.validation.{key} must be a string or integer")
    if "delivery" in body and body["delivery"] != "on_demand":
        raise ReproductionError("rule_json.delivery must be on_demand")
    if "retry_policy" in body and body["retry_policy"] not in {
        "length_plan_no_sample_retry",
        "deterministic_case_stream_v1",
    }:
        raise ReproductionError("rule_json.retry_policy is unknown")
    if "renderer" in body and body["renderer"] not in {"dns_pcm16_v1", "dns_pcm16_accept_v2"}:
        raise ReproductionError("rule_json.renderer is unknown")
    if body.get("renderer") == "dns_pcm16_accept_v2" and body.get("retry_policy") != "deterministic_case_stream_v1":
        raise ReproductionError("dns_pcm16_accept_v2 requires deterministic_case_stream_v1")
    if body.get("retry_policy") == "deterministic_case_stream_v1" and body.get("renderer") != "dns_pcm16_accept_v2":
        raise ReproductionError("deterministic_case_stream_v1 requires dns_pcm16_accept_v2")
    if algorithm == "online.dfn3":
        if body.get("online") != DFN3_ONLINE_V1:
            raise ReproductionError("online.dfn3 rule_json.online is not the shipped distribution")
        if body.get("order") != "dfn3_online_ceil_v1":
            raise ReproductionError("online.dfn3 order must be dfn3_online_ceil_v1")
        if body.get("window") != 144000:
            raise ReproductionError("online.dfn3 window must be 144000")
        if body.get("global_batch") != 64:
            raise ReproductionError("online.dfn3 global_batch must be the shipped cap 64")
        if body.get("delivery") != "on_demand":
            raise ReproductionError("online.dfn3 delivery must be on_demand")
    elif algorithm == "online.semambapp_gsr":
        if body.get("online") != SEMAMBA_GSR_V1:
            raise ReproductionError(
                "online.semambapp_gsr rule_json.online is not the HEAD config.yaml distribution"
            )
        if body.get("order") != "per_rank_independent_v1":
            raise ReproductionError("online.semambapp_gsr order must be per_rank_independent_v1")
        if body.get("window") != 48000:
            raise ReproductionError("online.semambapp_gsr window must be 48000")
        if body.get("global_batch") != 8:
            raise ReproductionError("online.semambapp_gsr global_batch must be the shipped 8")
        if body.get("delivery") != "on_demand":
            raise ReproductionError("online.semambapp_gsr delivery must be on_demand")
    elif "online" in body:
        raise ReproductionError(f"{algorithm} does not take online")
    if "resample" in body:
        if not algorithm.startswith("fixed_pair."):
            raise ReproductionError(f"{algorithm} does not take resample")
        if body.get("delivery") != "on_demand":
            raise ReproductionError("resample requires delivery on_demand")
        _check_resample(body["resample"])
    elif body.get("delivery") == "on_demand" and algorithm.startswith("fixed_pair."):
        raise ReproductionError("on-demand fixed-pair rule needs resample")
    return body


def _check_resample(value: Any) -> None:
    body = require_object(value, "rule_json.resample", RESAMPLE_KEYS)
    if body.get("tool") != "soxr" or body.get("quality") != "HQ":
        raise ReproductionError("resample tool must be soxr HQ")
    if body.get("source_rate") != 48000 or body.get("target_rate") != 16000:
        raise ReproductionError("resample rates must be 48000 to 16000")
    if body.get("quantization") != "pcm16":
        raise ReproductionError("resample quantization must be pcm16")
    if body.get("length") != "source_frames_plus_one_div_3":
        raise ReproductionError("resample length formula is not the recorded soxr HQ count")
    if tuple(body.get("order") or ()) != RESAMPLE_ORDER:
        raise ReproductionError("resample order does not match the recorded VoiceBank order")


def _check_order(value: Any) -> None:
    if isinstance(value, str):
        if value not in ORDER_POLICIES:
            raise ReproductionError(f"unknown order policy: {value}")
        return
    if isinstance(value, list) and value and all(isinstance(item, str) and item in ORDER_POLICIES for item in value):
        return
    raise ReproductionError("rule_json.order must be a known order policy")


def validate_selection_rule(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or "rule" not in value:
        raise ReproductionError("selection rule needs a rule name")
    name = value["rule"]
    if name not in SELECTION_RULES:
        raise ReproductionError(f"unknown selection rule: {name}")
    extra = set(value) - {"rule"} - SELECTION_RULES[name]
    if extra:
        raise ReproductionError(f"selection rule {name} has unknown fields: {', '.join(sorted(extra))}")
    missing = SELECTION_RULES[name] - set(value)
    if missing:
        raise ReproductionError(f"selection rule {name} is missing {', '.join(sorted(missing))}")
    return value


def validate_provenance(value: Any, *, origin: str) -> dict[str, Any]:
    body = require_object(value, "provenance_json", PROVENANCE_KEYS)
    if not isinstance(body.get("source"), str) or not body["source"]:
        raise ReproductionError("provenance_json.source must be a string")
    if origin == "official":
        check_sha256(body.get("sha256"), "provenance_json.sha256")
        if not isinstance(body.get("path"), str) and not isinstance(body.get("file"), str):
            raise ReproductionError("official provenance needs path or file")
    if "sha256" in body and body["sha256"] is not None:
        check_sha256(body["sha256"], "provenance_json.sha256")
    return body


def validate_official_ref(value: Any) -> dict[str, Any]:
    body = require_object(value, "official_ref_json", OFFICIAL_REF_KEYS)
    if not body:
        raise ReproductionError("official_ref_json must not be empty")
    for key, item in body.items():
        if not isinstance(item, str) or not item:
            raise ReproductionError(f"official_ref_json.{key} must be a string")
    return body


def validate_deviations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ReproductionError("deviations_json must be a list")
    items = []
    for index, item in enumerate(value):
        body = require_object(item, f"deviations_json[{index}]", {"id", "class", "text", "ref"})
        if not isinstance(body.get("id"), str) or not isinstance(body.get("text"), str):
            raise ReproductionError(f"deviations_json[{index}] needs id and text")
        if body.get("class") not in DEVIATION_CLASSES:
            raise ReproductionError(f"deviations_json[{index}].class is unknown")
        if "ref" in body and not isinstance(body["ref"], str):
            raise ReproductionError(f"deviations_json[{index}].ref must be a string")
        items.append(body)
    return items


def validate_realization(document: Any) -> dict[str, Any]:
    body = require_object(document, "realization", REALIZATION_KEYS)
    if body.get("realization_version") != 1:
        raise ReproductionError("unsupported realization_version")
    algorithm = body.get("algorithm")
    if algorithm not in ALGORITHMS:
        raise ReproductionError(f"unknown realization algorithm: {algorithm}")
    if body.get("algorithm_version") != ALGORITHMS[algorithm]["version"]:
        raise ReproductionError("realization algorithm_version does not match the registry")
    positive_int(body.get("sample_rate"), "realization.sample_rate")
    if not isinstance(body.get("sources"), dict):
        raise ReproductionError("realization.sources must be an object")
    if not isinstance(body.get("draws"), dict):
        raise ReproductionError("realization.draws must be an object")
    if "steps" in body:
        steps = body["steps"]
        if not isinstance(steps, list) or not all(isinstance(item, str) for item in steps):
            raise ReproductionError("realization.steps must be a list of strings")
    output = body.get("output")
    if not isinstance(output, dict):
        raise ReproductionError("realization.output must be an object")
    extra = set(output) - {"length", "teacher"}
    if extra:
        raise ReproductionError(f"realization.output has unknown fields: {', '.join(sorted(extra))}")
    if output.get("teacher") not in TEACHERS:
        raise ReproductionError("realization.output.teacher is unknown")
    if "length" in output:
        positive_int(output["length"], "realization.output.length")
    return body


def asset_ids_in(value: Any) -> set[str]:
    found: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            asset = item.get("asset")
            if isinstance(asset, str):
                found.add(asset)
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return found


def rule_mentions(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        if value.get("of") == target:
            return True
        return any(rule_mentions(child, target) for child in value.values())
    if isinstance(value, list):
        return any(rule_mentions(child, target) for child in value)
    return value == target


def weakest_integrity(values: list[str]) -> str:
    if not values:
        raise ReproductionError("integrity_min needs at least one member")
    return min(values, key=lambda item: INTEGRITY_RANK[item])
