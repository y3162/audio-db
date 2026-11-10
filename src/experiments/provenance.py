"""Bind ``audio-db-data-provenance-v1`` to an experiment checkpoint.

The document is the training provenance contract. This module does not invent
a second schema and does not treat a missing material hash as independence.
Rank 0 is the only writer. A legacy checkpoint whose ``provenance_sha256`` is
``None`` stays unproven: a provenance file written by a later step is not
applied to it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.evaluation.provenance import (
    PROVENANCE_NAME,
    judge_checkpoint,
    judge_independence,
    parse_provenance_bytes,
    sidecar_path,
)
from src.experiments.config import LEGACY_CLASS_PATHS, ExperimentConfig, config_sha256
from src.experiments.errors import CheckpointError
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.provenance import build_provenance, provenance_sha256, publish_provenance


class _Process:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size


class _DatasetView:
    def __init__(self, identity: dict[str, Any], parts: dict[str, Any]) -> None:
        self._identity = identity
        self._parts = parts

    def identity(self) -> dict[str, Any]:
        return self._identity

    def provenance_parts(self) -> dict[str, Any]:
        return self._parts


def dataset_identity_map(config: ExperimentConfig, components: Any) -> dict[str, Any] | None:
    """Snapshot identity for provenance.

    A sampler that can ``align_scheduled_batch`` changes ``micro_batch`` during
    a run. That width stays in the checkpoint payload and is not part of the
    run's data provenance, so a later epoch can keep the same provenance file.
    """

    found: dict[str, Any] = {}
    for component_id, spec in config.components.items():
        if spec.ref is not None or spec.role != "data":
            continue
        component = components.get(component_id)
        identity = getattr(component, "identity", None)
        found[component_id] = _stable_identity(component, identity() if callable(identity) else None)
    if not found:
        return None
    return found


def _stable_identity(component: Any, identity: Any) -> Any:
    if not isinstance(identity, dict):
        return identity
    if not callable(getattr(component, "align_scheduled_batch", None)):
        return identity
    stable = dict(identity)
    stable.pop("micro_batch", None)
    return stable


def bind_provenance(
    directory: Path,
    config: ExperimentConfig,
    components: Any,
    runtime: Any,
) -> tuple[bytes, str]:
    """Publish the run provenance once and return its canonical bytes and SHA-256."""

    generations = _config_generations(config)
    provenance_path = directory / PROVENANCE_NAME
    if not provenance_path.is_file():
        generations = generations[:1]
    last_error: TrainingError | None = None
    for class_paths, digest in generations:
        try:
            document = _document(
                config,
                components,
                runtime,
                config_digest=digest,
                class_paths=class_paths,
            )
            raw = publish_provenance(directory, document, rank=0)
        except TrainingError as exc:
            if provenance_path.is_file() and "does not match this run" in str(exc):
                last_error = exc
                continue
            raise CheckpointError(str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise CheckpointError(f"data provenance could not be written: {exc}") from exc
        return raw, provenance_sha256(raw)
    raise CheckpointError(str(last_error) if last_error is not None else "data provenance does not match this run")


def attach_provenance_sidecar(checkpoint: Path, raw: bytes) -> None:
    """Copy the run provenance next to one checkpoint. The previous bytes must match."""

    from src.experiments.calc.provenance import write_provenance_sidecar

    try:
        write_provenance_sidecar(checkpoint, raw)
    except TrainingError as exc:
        raise CheckpointError(str(exc)) from exc


def require_provenance(
    payload: dict[str, Any],
    source: Path,
    config: ExperimentConfig,
    components: Any,
    *,
    mode: str,
) -> None:
    """Check a new checkpoint's sidecar. ``None`` is a legacy file and is not upgraded."""

    expected = payload.get("provenance_sha256")
    if expected is None:
        return
    if not isinstance(expected, str) or len(expected) != 64:
        raise CheckpointError("checkpoint provenance_sha256 is invalid")
    directory_file = source.parent / PROVENANCE_NAME
    side = sidecar_path(source)
    paths = (side, directory_file)
    for path in paths:
        if not path.is_file():
            raise CheckpointError(f"checkpoint provenance is missing: {path.name}")
    document = None
    digests: list[str] = []
    for path in paths:
        try:
            raw = path.read_bytes()
            parsed = parse_provenance_bytes(raw)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"data provenance is unreadable: {path.name}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise CheckpointError("checkpoint provenance does not match this experiment")
        digests.append(digest)
        document = parsed
    if len(set(digests)) != 1 or document is None:
        raise CheckpointError("checkpoint provenance files disagree")
    if mode == "initialize":
        return
    live = dataset_identity_map(config, components)
    if not _identities_match(document.get("dataset_identity"), live):
        raise CheckpointError("checkpoint provenance does not match this experiment")


def judge_experiment_checkpoint(
    checkpoint: str | Path,
    eval_materials: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Judge one experiment checkpoint without promoting a legacy file.

    A later ``data_provenance.json`` in the same directory does not become
    evidence for a checkpoint whose own ``provenance_sha256`` is ``None``.
    """

    path = Path(checkpoint)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as exc:
        result = judge_independence(None, eval_materials, load_problem=f"checkpoint is unreadable: {exc}")
        result["checkpoint"] = str(path)
        return result
    if not isinstance(payload, dict):
        result = judge_independence(None, eval_materials, load_problem="checkpoint is not a mapping")
        result["checkpoint"] = str(path)
        return result
    if payload.get("provenance_sha256") is None:
        result = judge_independence(
            None,
            eval_materials,
            load_problem="legacy checkpoint has no provenance_sha256",
        )
        result["checkpoint"] = str(path)
        result["legacy_checkpoint"] = True
        return result
    result = judge_checkpoint(path, eval_materials)
    result["legacy_checkpoint"] = False
    return result


def _config_generations(config: ExperimentConfig) -> list[tuple[dict[str, str] | None, str]]:
    """Current class paths, then each explicit historical generation.

    A generation is one complete rewrite. Mixing paths from different
    generations is not a separate accepted identity.
    """

    generations: list[tuple[dict[str, str] | None, str]] = [(None, config_sha256(config))]
    seen = {generations[0][1]}
    for mapping in LEGACY_CLASS_PATHS:
        digest = config_sha256(config, class_paths=mapping)
        if digest in seen:
            continue
        seen.add(digest)
        generations.append((mapping, digest))
    return generations


def _document(
    config: ExperimentConfig,
    components: Any,
    runtime: Any,
    *,
    config_digest: str | None = None,
    class_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    identity_map = dataset_identity_map(config, components)
    view = _DatasetView(_view_identity(identity_map), _parts(components, identity_map))
    process = _Process(int(getattr(runtime, "world_size", 1)))
    digest = config_sha256(config) if config_digest is None else config_digest
    training_config = {
        "model": {"name": _model_name(config, class_paths)},
        "seed": config.seed,
        "loss": {"experiment_config_sha256": digest},
        "data": {},
    }
    document = build_provenance(view, training_config, process)
    document["dataset_identity"] = identity_map
    document["experiment_config_sha256"] = digest
    limitations = list(document.get("limitations") or [])
    if not _parts_have_content(document):
        note = "a provenance file without content hashes is not evidence of independence"
        if note not in limitations:
            limitations.append(note)
    document["limitations"] = limitations
    return document


def _view_identity(identity_map: dict[str, Any] | None) -> dict[str, Any]:
    if not identity_map:
        return {"class": "experiment", "dataset_id": None}
    if len(identity_map) == 1:
        only = next(iter(identity_map.values()))
        if isinstance(only, dict):
            return only
    return {"class": "experiment", "components": identity_map}


def _parts(components: Any, identity_map: dict[str, Any] | None) -> dict[str, Any]:
    materials: list[dict[str, Any]] = []
    sets: list[dict[str, Any]] = []
    limitations: list[str] = []
    lineage_counts = {"source": 0, "derived": 0, "unknown": 0}
    header: dict[str, Any] = {
        "pipeline": "experiment",
        "dataset_id": None,
        "dataset_name": None,
        "dataset_version": None,
        "snapshot_hash": None,
        "protocol_id": None,
        "profile": None,
        "claim_label": None,
        "selection_uses_test": None,
    }
    if not identity_map:
        limitations.append("this experiment has no data module")
        header.update({"materials": materials, "sets": sets, "lineage_counts": lineage_counts, "limitations": limitations})
        return header
    used_lineage = False
    for component_id, identity in identity_map.items():
        component = _component_for(components, component_id)
        lineage = getattr(component, "lineage", None)
        if not callable(lineage):
            limitations.append(f"{component_id} has no snapshot material hashes")
            continue
        try:
            parts = lineage()
        except Exception as exc:
            raise CheckpointError(f"data lineage for {component_id} failed: {exc}") from exc
        if not isinstance(parts, dict):
            raise CheckpointError(f"data lineage for {component_id} is not an object")
        used_lineage = True
        if header["dataset_id"] is None:
            for key in (
                "dataset_id",
                "dataset_name",
                "dataset_version",
                "snapshot_hash",
                "protocol_id",
                "profile",
                "claim_label",
                "selection_uses_test",
            ):
                header[key] = parts.get(key)
        materials.extend(item for item in parts.get("materials") or [] if isinstance(item, dict))
        sets.extend(item for item in parts.get("sets") or [] if isinstance(item, dict))
        counts = parts.get("lineage_counts") or {}
        if isinstance(counts, dict):
            for key in lineage_counts:
                lineage_counts[key] += int(counts.get(key) or 0)
        limitations.extend(str(item) for item in parts.get("limitations") or [])
    if len(identity_map) > 1:
        limitations.append("multiple data modules; the material list is their concatenation")
    if not used_lineage:
        limitations.append("this experiment has no snapshot material hashes")
    header["pipeline"] = "reproduction" if used_lineage else "experiment"
    header["materials"] = materials
    header["sets"] = sets
    header["lineage_counts"] = lineage_counts
    header["limitations"] = limitations
    return header


def _component_for(components: Any, component_id: str) -> Any:
    getter = getattr(components, "get", None)
    if callable(getter):
        return getter(component_id)
    return components[component_id]


def _parts_have_content(document: dict[str, Any]) -> bool:
    if document.get("pipeline") != "reproduction":
        return False
    materials = document.get("materials")
    return isinstance(materials, list) and bool(materials)


def _identities_match(saved: object, live: object) -> bool:
    return _comparable_map(saved) == _comparable_map(live)


def _comparable_map(value: object) -> object:
    from src.experiments.data.identity import comparable_identity

    if not isinstance(value, dict):
        return value
    converted: dict[str, Any] = {}
    for key, item in value.items():
        converted[str(key)] = comparable_identity(item) if isinstance(item, dict) else item
    return converted


def _model_name(config: ExperimentConfig, class_paths: dict[str, str] | None = None) -> str:
    names = []
    for _, spec in sorted(config.components.items()):
        if spec.ref is not None or spec.role != "module" or not spec.class_path:
            continue
        names.append((class_paths or {}).get(spec.class_path, spec.class_path))
    if not names:
        return "experiment"
    return ",".join(names)
