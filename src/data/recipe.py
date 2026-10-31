"""Canonical speech-generation recipes and the case ids derived from them.

A recipe is the only description of how a case is produced. The case id is the
SHA-256 of the canonical recipe. It is not part of the hashed document.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

RECIPE_VERSION = 1
OPERATION_VERSIONS = {
    "additive_noise": {1},
    "rir": {1},
    "peak_normalize": {1},
}
REFERENCES = {"dry_clean", "reverberant_clean", "none"}
RIR_LENGTH = "trim_to_input"
NOISE_START_UNIT = "sample"
SOURCE_NAMES = {"clean", "noise", "noisy", "rir"}
_TOP_LEVEL = {"recipe_version", "sources", "operations", "output"}
_SOURCE_KEYS = {"path", "sha256"}
_OUTPUT_KEYS = {"sample_rate", "reference", "max_seconds"}
_OP_KEYS = {
    "additive_noise": {"type", "version", "source", "snr_db", "noise_start", "noise_start_unit", "seed"},
    "rir": {"type", "version", "source", "channel", "length"},
    "peak_normalize": {"type", "version", "limit"},
}


class RecipeError(ValueError):
    """The recipe is missing a required field or names an unknown operation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate a recipe and return the normalized document that is hashed."""
    if not isinstance(recipe, dict):
        raise RecipeError("recipe must be an object")
    extra = set(recipe) - _TOP_LEVEL
    if extra:
        raise RecipeError(f"unknown recipe fields: {', '.join(sorted(extra))}")
    version = recipe.get("recipe_version")
    if version != RECIPE_VERSION:
        raise RecipeError(f"unsupported recipe_version: {version!r}")
    sources = _sources(recipe.get("sources"))
    operations = _operations(recipe.get("operations"), sources)
    output = _output(recipe.get("output"), sources, operations)
    return {
        "recipe_version": RECIPE_VERSION,
        "sources": sources,
        "operations": operations,
        "output": output,
    }


def canonical_json(recipe: dict[str, Any]) -> str:
    return _dump(canonical_recipe(recipe))


def case_id_for(recipe: dict[str, Any]) -> str:
    """SHA-256 of the canonical recipe. The id itself is not an input."""
    return hashlib.sha256(canonical_json(recipe).encode("utf-8")).hexdigest()


def case_content_hash(record: dict[str, Any]) -> str:
    """Hash of the registered case body. ``case_id`` is not included."""
    if "case_id" in record:
        raise RecipeError("case content hash must not include case_id")
    return hashlib.sha256(_dump(_normalize(record)).encode("utf-8")).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(_dump(_normalize(value)).encode("utf-8")).hexdigest()


def _sources(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or not value:
        raise RecipeError("recipe sources must be a non-empty object")
    unknown = set(value) - SOURCE_NAMES
    if unknown:
        raise RecipeError(f"unknown source names: {', '.join(sorted(unknown))}")
    sources: dict[str, dict[str, str]] = {}
    for name in sorted(value):
        item = value[name]
        if not isinstance(item, dict):
            raise RecipeError(f"source {name} must be an object")
        extra = set(item) - _SOURCE_KEYS
        if extra:
            raise RecipeError(f"source {name} has unknown fields: {', '.join(sorted(extra))}")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise RecipeError(f"source {name} path must be a relative path")
        if any(part in ("", ".") for part in Path(path).parts):
            raise RecipeError(f"source {name} path must not contain empty or '.' parts")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RecipeError(f"source {name} sha256 must be 64 lowercase hex characters")
        sources[name] = {"path": Path(path).as_posix(), "sha256": digest}
    return sources


def _operations(value: Any, sources: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RecipeError("recipe operations must be a list")
    operations = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise RecipeError(f"operation {index} must be an object")
        kind = item.get("type")
        if kind not in OPERATION_VERSIONS:
            raise RecipeError(f"unknown operation type: {kind!r}")
        version = item.get("version")
        if version not in OPERATION_VERSIONS[kind]:
            raise RecipeError(f"unsupported {kind} version: {version!r}")
        extra = set(item) - _OP_KEYS[kind]
        if extra:
            raise RecipeError(f"operation {index} has unknown fields: {', '.join(sorted(extra))}")
        if kind == "additive_noise":
            operations.append(_additive(item, index, sources))
        elif kind == "rir":
            operations.append(_rir(item, index, sources))
        else:
            operations.append(_peak(item, index))
    return operations


def _additive(item: dict[str, Any], index: int, sources: dict[str, dict[str, str]]) -> dict[str, Any]:
    source = item.get("source", "noise")
    _require_source(source, "noise", sources, index)
    if "clean" not in sources:
        raise RecipeError("additive_noise requires a clean source")
    start = item.get("noise_start")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise RecipeError(f"operation {index} noise_start must be a non-negative integer sample index")
    unit = item.get("noise_start_unit")
    if unit != NOISE_START_UNIT:
        raise RecipeError(
            f"operation {index} noise_start_unit must be {NOISE_START_UNIT!r} "
            "(index into the noise after resampling to output.sample_rate, then tiling)"
        )
    seed = item.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RecipeError(f"operation {index} seed must be an integer")
    return {
        "type": "additive_noise",
        "version": 1,
        "source": source,
        "snr_db": _finite_number(item.get("snr_db"), f"operation {index} snr_db"),
        "noise_start": start,
        "noise_start_unit": NOISE_START_UNIT,
        "seed": seed,
    }


def _rir(item: dict[str, Any], index: int, sources: dict[str, dict[str, str]]) -> dict[str, Any]:
    source = item.get("source", "rir")
    _require_source(source, "rir", sources, index)
    if "clean" not in sources:
        raise RecipeError("rir requires a clean source")
    channel = item.get("channel", 0)
    if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
        raise RecipeError(f"operation {index} channel must be a non-negative integer")
    length = item.get("length", RIR_LENGTH)
    if length != RIR_LENGTH:
        raise RecipeError(
            f"operation {index} length must be {RIR_LENGTH!r}: "
            "convolve causally and keep the input length, discarding the reverberant tail"
        )
    return {
        "type": "rir",
        "version": 1,
        "source": source,
        "channel": channel,
        "length": RIR_LENGTH,
    }


def _peak(item: dict[str, Any], index: int) -> dict[str, Any]:
    limit = _finite_number(item.get("limit"), f"operation {index} limit")
    if limit <= 0.0 or limit > 1.0:
        raise RecipeError(f"operation {index} limit must be in (0, 1]")
    return {"type": "peak_normalize", "version": 1, "limit": limit}


def _output(value: Any, sources: dict[str, dict[str, str]], operations: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeError("recipe output must be an object")
    extra = set(value) - _OUTPUT_KEYS
    if extra:
        raise RecipeError(f"unknown output fields: {', '.join(sorted(extra))}")
    rate = value.get("sample_rate")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise RecipeError("output.sample_rate must be a positive integer")
    reference = value.get("reference")
    if reference not in REFERENCES:
        raise RecipeError(f"output.reference must be one of: {', '.join(sorted(REFERENCES))}")
    has_rir = any(item["type"] == "rir" for item in operations)
    if reference == "reverberant_clean" and not has_rir:
        raise RecipeError("reverberant_clean requires an rir operation")
    if reference == "dry_clean" and "clean" not in sources:
        raise RecipeError("dry_clean requires a clean source")
    if not operations:
        _check_recorded(sources, reference)
    output: dict[str, Any] = {"sample_rate": rate, "reference": reference}
    if "max_seconds" in value and value["max_seconds"] is not None:
        limit = _finite_number(value["max_seconds"], "output.max_seconds")
        if limit <= 0.0:
            raise RecipeError("output.max_seconds must be positive")
        output["max_seconds"] = limit
    return output


def _check_recorded(sources: dict[str, dict[str, str]], reference: str) -> None:
    names = set(sources)
    if names == {"clean"} and reference != "dry_clean":
        raise RecipeError("a clean recording must use reference dry_clean")
    if names == {"noisy"} and reference != "none":
        raise RecipeError("a noisy recording has no clean reference")
    if names == {"clean", "noisy"} and reference != "dry_clean":
        raise RecipeError("a paired recording must use the clean file as dry_clean")
    if names - {"clean", "noisy"}:
        raise RecipeError("recorded recipes only use clean and noisy sources")
    if names not in ({"clean"}, {"noisy"}, {"clean", "noisy"}):
        raise RecipeError("a recorded recipe needs clean, noisy, or both")


def _require_source(name: Any, expected: str, sources: dict[str, dict[str, str]], index: int) -> None:
    if name != expected or name not in sources:
        raise RecipeError(f"operation {index} source must be {expected!r}")


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise RecipeError(f"{where} must be finite")
    return float(f"{number:.6f}")


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecipeError("non-finite numbers are not allowed")
        return float(f"{value:.6f}")
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    raise RecipeError(f"unsupported value type: {type(value).__name__}")


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
