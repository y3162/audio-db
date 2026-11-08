"""Atomic ``audio-db-experiment-v1`` checkpoints.

CPU / FP32 restore happens after ``build_experiment`` has prepared the module
and constructed the optimizer. CPU ``.to(cpu)`` keeps parameter identity, so
``load_state_dict`` updates the tensors the optimizer already holds.

Later strategies must not reuse that order:

- DDP loads module weights onto the unwrapped module, then ``prepare_model``
  wraps it, then the optimizer is built, then optimizer and the other owners
  load. Every rank must hold the same weights before wrap.
- FSDP may replace parameters inside ``prepare_model``. Weights load through
  the FSDP state API after prepare, and the optimizer is built on those
  replaced parameters. A version 1 file is not a sharded checkpoint.

``audio-db-training-v1`` is a different format and is refused.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from src.experiments.config import ExperimentConfig, config_sha256
from src.experiments.errors import CheckpointError, ExperimentError, StateError
from src.experiments.registry import StateRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAT = "audio-db-experiment-v1"
SCHEMA_VERSION = 1
_PROTECTED = (
    "data/checkpoints",
    "data/corpora",
    "data/db",
    "data/staging",
)
_LABELS = frozenset({"step", "init", "validation_pending"})
_PAYLOAD_KEYS = frozenset(
    {
        "aliases",
        "code_identity",
        "component_ids",
        "config_sha256",
        "dataset_identity",
        "format",
        "initialize",
        "label",
        "logical_step",
        "optimizer_updates",
        "output_directory",
        "parameter_ranks_agree",
        "provenance_sha256",
        "rank_count_in_rng",
        "rng_by_rank",
        "schema_version",
        "state_ids",
        "states",
        "validation_pending",
        "world_size",
    }
)
_CODE_KEYS = frozenset({"dirty", "git_commit"})
_LOAD_ORDER = ("module", "optimizer", "scheduler", "data", "trainer", "runtime")


@dataclass(frozen=True)
class RestoreReport:
    """What one restore call applied."""

    mode: Literal["resume", "initialize"]
    logical_step: int
    state_ids: tuple[str, ...]
    initialized_states: tuple[str, ...]


def assert_output_allowed(path: Path) -> None:
    """Refuse experiment output under published data directories."""

    resolved = path.resolve()
    for relative in _PROTECTED:
        protected = (REPO_ROOT / relative).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise CheckpointError(f"refusing to write experiment output under {relative}: {path}")


class DirectoryLock:
    """Non-blocking exclusive lock on ``directory/.lock``."""

    def __init__(self, directory: Path) -> None:
        self._handle = (directory / ".lock").open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise ExperimentError(f"experiment is already running: {directory}") from exc

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()

    def __del__(self) -> None:
        self.release()


class CheckpointManager:
    """Save and restore one experiment directory.

    The trainer decides when a step is safe to publish. This class writes the
    file and checks its identity. It does not branch on a model or a loss.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._config: ExperimentConfig | None = None
        self._states: StateRegistry | None = None
        self._components: Any = None
        self._runtime: Any = None

    def attach(self, config: ExperimentConfig, graph: Any) -> None:
        """Bind the live graph. Save and restore require this."""

        self._config = config
        self._states = graph.states
        self._components = graph.components
        self._runtime = graph.runtime

    def save(
        self,
        directory: Path,
        *,
        logical_step: int,
        label: str,
        initialize_allow: list[str] | None = None,
    ) -> Path:
        """Write one checkpoint atomically. The finished file is replaced last."""

        config, states, runtime = self._bound()
        if Path(directory).resolve() != self.directory.resolve():
            raise CheckpointError("checkpoint directory does not match the experiment directory")
        if label not in _LABELS:
            raise CheckpointError(f"unknown checkpoint label {label!r}")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or logical_step < 0:
            raise CheckpointError("logical_step must be an integer >= 0")
        if label == "init":
            if not initialize_allow:
                raise CheckpointError("an init checkpoint requires initialize.allow")
            allow = _allow_list(initialize_allow)
        elif initialize_allow is not None:
            raise CheckpointError("initialize.allow is only valid when label is init")
        else:
            allow = None
        if runtime.world_size != 1:
            raise CheckpointError("Phase 2-A saves world size 1 only")
        payload = _payload(
            config=config,
            states=states,
            components=self._components,
            runtime=runtime,
            directory=self.directory,
            logical_step=logical_step,
            label=label,
            initialize_allow=allow,
        )
        filename = _filename(label, logical_step)
        target = self.directory / filename
        temporary = self.directory / f".{filename}.tmp"
        try:
            with temporary.open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            _verify_written(temporary, payload)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        try:
            os.replace(temporary, target)
        except Exception as exc:
            raise CheckpointError(f"checkpoint replace failed: {exc}") from exc
        _fsync_directory(self.directory)
        return target

    def restore(self, path: Path, *, mode: Literal["resume", "initialize"]) -> RestoreReport:
        """Restore a finished checkpoint. Dotfiles and foreign formats are refused."""

        config, states, runtime = self._bound()
        if mode not in {"resume", "initialize"}:
            raise CheckpointError(f"unknown restore mode {mode!r}")
        source = Path(path)
        if source.name.startswith("."):
            raise CheckpointError(f"temporary checkpoint files cannot be restored: {source.name}")
        if not source.is_file():
            raise CheckpointError(f"checkpoint not found: {source}")
        payload = _read_payload(source)
        _validate_payload(
            payload,
            states,
            runtime,
            config,
            self._components,
            mode,
            source,
            self.directory,
        )
        logical_step = int(payload["logical_step"])
        if mode == "initialize":
            applied = _apply_initialize(payload, states)
            return RestoreReport("initialize", logical_step, states.ids(), tuple(applied))
        _apply_resume(payload, states, runtime)
        return RestoreReport("resume", logical_step, states.ids(), ())

    def _bound(self) -> tuple[ExperimentConfig, StateRegistry, Any]:
        if self._config is None or self._states is None or self._runtime is None:
            raise CheckpointError("checkpoint manager is not attached to an experiment")
        return self._config, self._states, self._runtime


def write_status(directory: Path, status: str) -> None:
    """Replace ``status.json`` in ``directory``."""

    _replace_text(directory / "status.json", json.dumps({"status": status}) + "\n")


def write_complete(directory: Path, logical_step: int) -> None:
    """Publish ``complete.json``. Its presence refuses a later resume."""

    body = json.dumps({"logical_step": logical_step}, sort_keys=True) + "\n"
    _replace_text(directory / "complete.json", body)


def _replace_text(target: Path, body: str) -> None:
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, target)


def _payload(
    *,
    config: ExperimentConfig,
    states: StateRegistry,
    components: Any,
    runtime: Any,
    directory: Path,
    logical_step: int,
    label: str,
    initialize_allow: list[str] | None,
) -> dict[str, Any]:
    blobs: dict[str, Any] = {}
    for state_id in states.ids():
        blobs[state_id] = states.owner(state_id).state_dict()
    trainer_blob = _trainer_blob(states, blobs)
    if trainer_blob is not None and trainer_blob.get("logical_step") != logical_step:
        raise CheckpointError("checkpoint logical_step does not match the trainer")
    updates = trainer_blob.get("optimizer_updates") if trainer_blob is not None else {}
    if not isinstance(updates, dict):
        raise CheckpointError("trainer optimizer_updates must be a mapping")
    rng = runtime.capture_rng()
    return {
        "aliases": dict(states.aliases()),
        "code_identity": _code_identity(),
        "component_ids": sorted(config.components),
        "config_sha256": config_sha256(config),
        "dataset_identity": _dataset_identity(config, components),
        "format": FORMAT,
        "initialize": None if initialize_allow is None else {"allow": list(initialize_allow)},
        "label": label,
        "logical_step": logical_step,
        "optimizer_updates": {str(key): int(value) for key, value in updates.items()},
        "output_directory": str(directory.resolve()),
        "parameter_ranks_agree": True,
        "provenance_sha256": None,
        "rank_count_in_rng": 1,
        "rng_by_rank": [rng],
        "schema_version": SCHEMA_VERSION,
        "state_ids": list(states.ids()),
        "states": blobs,
        "validation_pending": {"logical_step": logical_step} if label == "validation_pending" else None,
        "world_size": 1,
    }


def _validate_payload(
    payload: dict[str, Any],
    states: StateRegistry,
    runtime: Any,
    config: ExperimentConfig,
    components: Any,
    mode: str,
    source: Path,
    directory: Path,
) -> None:
    if payload.get("world_size") != runtime.world_size:
        raise CheckpointError(
            f"checkpoint world size {payload.get('world_size')} does not match {runtime.world_size}"
        )
    rng = payload.get("rng_by_rank")
    if not isinstance(rng, list) or len(rng) != payload["world_size"]:
        raise CheckpointError("rng_by_rank length does not match world size")
    if payload.get("rank_count_in_rng") != payload["world_size"]:
        raise CheckpointError("rank_count_in_rng does not match world size")
    saved_ids = payload.get("state_ids")
    saved_states = payload.get("states")
    if not isinstance(saved_ids, list) or not isinstance(saved_states, dict):
        raise CheckpointError("checkpoint state manifest is invalid")
    if set(saved_ids) != set(saved_states):
        raise CheckpointError("checkpoint state_ids and states do not match")
    current = set(states.ids())
    missing = sorted(current - set(saved_ids))
    extra = sorted(set(saved_ids) - current)
    if missing or extra:
        raise StateError(f"checkpoint state ids missing {missing} extra {extra}")
    aliases = payload.get("aliases")
    if not isinstance(aliases, dict) or aliases != dict(states.aliases()):
        raise StateError("checkpoint aliases do not match this experiment")
    component_ids = payload.get("component_ids")
    if component_ids != sorted(config.components):
        raise CheckpointError("checkpoint component ids do not match this experiment")
    if payload.get("parameter_ranks_agree") is not True:
        raise CheckpointError("checkpoint parameters did not agree across ranks")
    if mode == "resume":
        if source.resolve().parent != directory.resolve():
            raise CheckpointError("resume from a different directory is not supported")
        if payload.get("output_directory") != str(directory.resolve()):
            raise CheckpointError("resume from a different directory is not supported")
        if payload.get("config_sha256") != config_sha256(config):
            raise CheckpointError("checkpoint config identity does not match this experiment")
        if payload.get("dataset_identity") != _dataset_identity(config, components):
            raise CheckpointError("checkpoint dataset identity does not match this experiment")
        if payload.get("provenance_sha256") is not None:
            raise CheckpointError("checkpoint provenance does not match this experiment")
        _require_plan_hashes(payload, states)
        return
    initialize = payload.get("initialize")
    if not isinstance(initialize, dict) or "allow" not in initialize:
        raise CheckpointError("initialize_from requires initialize.allow in the checkpoint")
    allow = initialize["allow"]
    if not isinstance(allow, list) or not allow or not all(isinstance(item, str) for item in allow):
        raise CheckpointError("initialize.allow must be a list of state ids")
    for state_id in allow:
        if state_id not in current:
            raise StateError(f"unknown state id {state_id}")
        owner = states.owner(state_id)
        if getattr(owner, "kind", None) != "module":
            raise CheckpointError(f"initialize.allow cannot include {state_id}")


def _apply_resume(payload: dict[str, Any], states: StateRegistry, runtime: Any) -> None:
    blobs: dict[str, Any] = payload["states"]
    for kind in _LOAD_ORDER:
        for state_id in states.ids():
            owner = states.owner(state_id)
            if getattr(owner, "kind", None) != kind:
                continue
            blob = blobs[state_id]
            if not isinstance(blob, dict):
                raise CheckpointError(f"state {state_id} is not a mapping")
            if kind == "runtime" and blob.get("grad_scaler") is not None:
                raise CheckpointError("grad scaler restore is not implemented")
            owner.load_state_dict(blob)
    rank = payload["rng_by_rank"][0]
    if not isinstance(rank, dict):
        raise CheckpointError("rng_by_rank[0] is not a mapping")
    try:
        runtime.restore_rng(rank)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError(f"failed to restore RNG: {exc}") from exc


def _apply_initialize(payload: dict[str, Any], states: StateRegistry) -> list[str]:
    allow: list[str] = list(payload["initialize"]["allow"])
    blobs: dict[str, Any] = payload["states"]
    for state_id in allow:
        blob = blobs.get(state_id)
        if not isinstance(blob, dict):
            raise CheckpointError(f"state {state_id} is not a mapping")
        states.owner(state_id).load_state_dict(blob)
    return allow


def _require_plan_hashes(payload: dict[str, Any], states: StateRegistry) -> None:
    blobs: dict[str, Any] = payload["states"]
    for state_id in states.ids():
        owner = states.owner(state_id)
        if getattr(owner, "kind", None) != "data":
            continue
        blob = blobs[state_id]
        saved = blob.get("plan_hash") if isinstance(blob, dict) else None
        live = getattr(owner.target, "plan_hash", None)
        if not isinstance(saved, str) or saved != live:
            raise CheckpointError(f"data plan_hash does not match state {state_id}")


def _dataset_identity(config: ExperimentConfig, components: Any) -> dict[str, Any] | None:
    found: dict[str, Any] = {}
    for component_id, spec in config.components.items():
        if spec.ref is not None or spec.role != "data":
            continue
        component = components.get(component_id)
        identity = getattr(component, "identity", None)
        found[component_id] = identity() if callable(identity) else None
    if not found:
        return None
    return found


def _trainer_blob(states: StateRegistry, blobs: dict[str, Any]) -> dict[str, Any] | None:
    trainers = [state_id for state_id in states.ids() if states.owner(state_id).kind == "trainer"]
    if not trainers:
        return None
    blob = blobs[trainers[0]]
    if not isinstance(blob, dict):
        raise CheckpointError("trainer state is not a mapping")
    return blob


def _verify_written(path: Path, payload: dict[str, Any]) -> None:
    restored = _read_payload(path)
    if restored.get("format") != payload["format"]:
        raise CheckpointError("checkpoint verification failed: format")
    if restored.get("logical_step") != payload["logical_step"]:
        raise CheckpointError("checkpoint verification failed: logical_step")
    if restored.get("state_ids") != payload["state_ids"]:
        raise CheckpointError("checkpoint verification failed: state_ids")
    if restored.get("config_sha256") != payload["config_sha256"]:
        raise CheckpointError("checkpoint verification failed: config_sha256")


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError, pickle_error()) as exc:
        raise CheckpointError(f"checkpoint {path.name} is unreadable") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint {path.name} is not a mapping")
    found = payload.get("format")
    if found != FORMAT:
        raise CheckpointError(f"checkpoint format {found!r} is not {FORMAT}")
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema version {version!r}")
    unknown = sorted(set(payload) - _PAYLOAD_KEYS)
    if unknown:
        raise CheckpointError(f"unknown checkpoint field {unknown[0]}")
    missing = sorted(_PAYLOAD_KEYS - set(payload))
    if missing:
        raise CheckpointError(f"checkpoint is missing {missing[0]}")
    code = payload.get("code_identity")
    if not isinstance(code, dict) or set(code) != _CODE_KEYS:
        raise CheckpointError("checkpoint code_identity is invalid")
    label = payload.get("label")
    if label not in _LABELS:
        raise CheckpointError(f"unknown checkpoint label {label!r}")
    logical_step = payload.get("logical_step")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or logical_step < 0:
        raise CheckpointError("checkpoint logical_step is invalid")
    return payload


def pickle_error() -> type[BaseException]:
    import pickle

    return pickle.UnpicklingError


def _filename(label: str, logical_step: int) -> str:
    prefix = "init" if label == "init" else "step"
    return f"{prefix}_{logical_step:08d}.pt"


def _allow_list(values: list[str]) -> list[str]:
    if not isinstance(values, list) or not values:
        raise CheckpointError("initialize.allow must be a list of state ids")
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise CheckpointError("initialize.allow must be a list of state ids")
        if value in cleaned:
            raise CheckpointError(f"initialize.allow repeats {value}")
        cleaned.append(value)
    return cleaned


def _code_identity() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=REPO_ROOT,
            timeout=10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return {"dirty": None, "git_commit": None}
    return {"dirty": dirty != 0, "git_commit": commit or None}


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        return
    finally:
        os.close(fd)

