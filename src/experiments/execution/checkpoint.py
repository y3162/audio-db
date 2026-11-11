"""Atomic ``audio-db-experiment-v1`` checkpoints.

Restore happens after ``build_experiment`` has prepared the module and
constructed the optimizer. CPU ``.to(cpu)`` and single-device CUDA DDP keep
parameter identity: the wrapper does not replace those parameter objects.
``load_state_dict`` therefore writes into the unwrapped ``state_module`` in
place, which is the same storage the optimizer holds.
Every rank loads the same file. Rank 0 writes it.

FSDP may replace parameters inside ``prepare_model``. That path is not
implemented. A version 1 file is not a sharded checkpoint.

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

from src.experiments.config import LEGACY_CLASS_PATHS, ExperimentConfig, config_sha256
from src.experiments.data.identity import comparable_identity
from src.experiments.errors import CheckpointError, ExperimentError, StateError
from src.experiments.calc.errors import TrainingError
from src.experiments.core.registry import StateRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]
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


def acquire_directory_lock(directory: Path, runtime: object) -> DirectoryLock | None:
    """Take the exclusive directory lock on rank 0 and tell the other ranks.

    Other ranks do not take the same exclusive lock. They wait on one
    collective for rank 0's result, then continue as members of this run. A
    second experiment's rank 0 fails the non-blocking lock and every rank of
    that launch raises. This function does not write ``status.json``.
    """

    world = int(getattr(runtime, "world_size", 1))
    rank = int(getattr(runtime, "rank", 0))
    if world == 1:
        return DirectoryLock(directory)
    status = ""
    lock: DirectoryLock | None = None
    if rank == 0:
        try:
            lock = DirectoryLock(directory)
        except ExperimentError as exc:
            status = str(exc)
    statuses = runtime.gather_objects(status)  # type: ignore[attr-defined]
    failure = statuses[0]
    if not isinstance(failure, str):
        failure = "rank 0 did not report a lock result"
    if failure:
        if lock is not None:
            lock.release()
        if rank == 0:
            raise ExperimentError(failure)
        raise ExperimentError(f"rank 0 failed to lock the experiment directory: {failure}")
    runtime.barrier()  # type: ignore[attr-defined]
    return lock


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
        filename = _filename(label, logical_step)
        target = self.directory / filename
        error: BaseException | None = None
        try:
            from src.experiments.execution.profile import Span

            with Span("replicated_state_compare"):
                _assert_replicated_states(states, runtime)
            local_rng = dict(runtime.capture_rng())
            local_rng["rank"] = int(runtime.rank)
            rngs = runtime.gather_objects(local_rng)
            if len(rngs) != runtime.world_size:
                raise CheckpointError("rng_by_rank length does not match world size")
            if int(runtime.rank) == 0:
                from src.experiments.execution.profile import Span
                from src.experiments.execution.provenance import attach_provenance_sidecar, bind_provenance

                with Span("checkpoint_write"):
                    raw, provenance = bind_provenance(self.directory, config, self._components, runtime)
                    payload = _payload(
                        config=config,
                        states=states,
                        components=self._components,
                        runtime=runtime,
                        directory=self.directory,
                        logical_step=logical_step,
                        label=label,
                        initialize_allow=allow,
                        rngs=rngs,
                        provenance_sha256=provenance,
                    )
                    temporary = self.directory / f".{filename}.tmp"
                    try:
                        with temporary.open("wb") as stream:
                            torch.save(payload, stream)
                            stream.flush()
                            os.fsync(stream.fileno())
                        _verify_written(temporary, payload)
                        attach_provenance_sidecar(target, raw)
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        raise
                    try:
                        os.replace(temporary, target)
                    except Exception as exc:
                        raise CheckpointError(f"checkpoint replace failed: {exc}") from exc
                    _fsync_directory(self.directory)
        except Exception as exc:
            error = exc
        if runtime.world_size > 1:
            errors = runtime.gather_objects(None if error is None else f"{type(error).__name__}: {error}")
            if error is not None:
                raise error
            if any(errors):
                message = next(item for item in errors if item)
                raise CheckpointError(str(message))
            runtime.barrier()
        elif error is not None:
            raise error
        return target

    def restore(self, path: Path, *, mode: Literal["resume", "initialize"]) -> RestoreReport:
        """Restore a finished checkpoint. Dotfiles and foreign formats are refused."""

        config, states, runtime = self._bound()
        if mode not in {"resume", "initialize"}:
            raise CheckpointError(f"unknown restore mode {mode!r}")
        failure: BaseException | None = None
        report: RestoreReport | None = None
        try:
            report = self._restore_local(path, mode=mode, config=config, states=states, runtime=runtime)
        except Exception as exc:
            failure = exc
        if runtime.world_size > 1:
            errors = runtime.gather_objects(None if failure is None else f"{type(failure).__name__}: {failure}")
            if failure is not None:
                raise failure
            if any(errors):
                message = next(item for item in errors if item)
                raise CheckpointError(str(message))
            runtime.barrier()
        elif failure is not None:
            raise failure
        if report is None:
            raise CheckpointError("restore did not produce a report")
        return report

    def _restore_local(
        self,
        path: Path,
        *,
        mode: Literal["resume", "initialize"],
        config: ExperimentConfig,
        states: StateRegistry,
        runtime: Any,
    ) -> RestoreReport:
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
    rngs: list[Any],
    provenance_sha256: str,
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
    if len(rngs) != runtime.world_size:
        raise CheckpointError("rng_by_rank length does not match world size")
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
        "provenance_sha256": provenance_sha256,
        "rank_count_in_rng": runtime.world_size,
        "rng_by_rank": list(rngs),
        "schema_version": SCHEMA_VERSION,
        "state_ids": list(states.ids()),
        "states": blobs,
        "validation_pending": {"logical_step": logical_step} if label == "validation_pending" else None,
        "world_size": runtime.world_size,
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
        acceptable = {config_sha256(config)}
        for mapping in LEGACY_CLASS_PATHS:
            acceptable.add(config_sha256(config, class_paths=mapping))
        if payload.get("config_sha256") not in acceptable:
            raise CheckpointError("checkpoint config identity does not match this experiment")
        _align_scheduled_batches(payload, components)
        if not _dataset_identities_match(payload.get("dataset_identity"), _dataset_identity(config, components)):
            raise CheckpointError("checkpoint dataset identity does not match this experiment")
        _require_plan_hashes(payload, states)
        from src.experiments.execution.provenance import require_provenance

        require_provenance(payload, source, config, components, mode="resume")
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
    from src.experiments.execution.provenance import require_provenance

    require_provenance(payload, source, config, components, mode="initialize")


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
            owner.load_state_dict(blob)
    index = int(getattr(runtime, "rank", 0))
    ranks = payload["rng_by_rank"]
    if index >= len(ranks):
        raise CheckpointError("rng_by_rank has no blob for this rank")
    rank = ranks[index]
    if not isinstance(rank, dict):
        raise CheckpointError(f"rng_by_rank[{index}] is not a mapping")
    saved_rank = rank.get("rank")
    if saved_rank is not None and saved_rank != index:
        raise CheckpointError(f"rng_by_rank[{index}] belongs to rank {saved_rank}")
    try:
        runtime.restore_rng(rank)
    except (KeyError, RuntimeError, TypeError, ValueError, TrainingError, ExperimentError) as exc:
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
        recompute = getattr(owner.target, "plan_hash_for_state", None)
        if callable(recompute) and isinstance(blob, dict):
            try:
                live = recompute(blob)
            except StateError:
                raise
            except ExperimentError as exc:
                raise CheckpointError(f"data plan_hash does not match state {state_id}: {exc}") from exc
        else:
            live = getattr(owner.target, "plan_hash", None)
        if not isinstance(saved, str) or saved != live:
            raise CheckpointError(f"data plan_hash does not match state {state_id}")


def _align_scheduled_batches(payload: dict[str, Any], components: Any) -> None:
    """Rebind DeepFilterNet3 to the micro-batch stored in the checkpoint.

    The YAML width is the width at construction. A later epoch may have been
    saved at another width. Aligning before the identity compare keeps that
    width from being treated as a different dataset. Other samplers have no
    ``align_scheduled_batch`` and are not rebound.
    """

    saved = payload.get("dataset_identity")
    if not isinstance(saved, dict) or components is None:
        return
    for component_id, identity in saved.items():
        if not isinstance(identity, dict):
            continue
        try:
            component = components.get(component_id)
        except ExperimentError:
            continue
        align = getattr(component, "align_scheduled_batch", None)
        if callable(align):
            align(identity)


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


def _dataset_identities_match(saved: object, live: object) -> bool:
    """Compare dataset identity without display-only snapshot fields.

    ``TensorPairData`` identities have no ``dataset_id``, so they stay on
    full equality. Snapshot identities ignore ``dataset_name`` and
    ``dataset_version`` and still require the dataset id and snapshot hash.
    """

    return _comparable_dataset_identity(saved) == _comparable_dataset_identity(live)


def _comparable_dataset_identity(value: object) -> object:
    if not isinstance(value, dict):
        return value
    converted: dict[str, Any] = {}
    for key, item in value.items():
        converted[str(key)] = comparable_identity(item) if isinstance(item, dict) else item
    return converted


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


def _assert_replicated_states(states: StateRegistry, runtime: Any) -> None:
    """Refuse a checkpoint when replicated state differs across ranks.

    RNG is not compared here. It is gathered separately because each rank keeps
    its own generators. Runtime state is compared, including the GradScaler
    scale and growth tracker. Comparison uses dtype, shape, and values.
    """

    if runtime.world_size == 1:
        return
    # Rank agreement pickles every replicated tensor. Moving to CPU first keeps
    # the NCCL object-collective payload off the GPU. This still sends the full
    # optimizer state to every rank and is not a large-model checkpoint.
    local: dict[str, Any] = {}
    for state_id in states.ids():
        owner = states.owner(state_id)
        local[state_id] = _cpu_tree(owner.state_dict())
    gathered = runtime.gather_objects(local)
    for rank, other in enumerate(gathered):
        mismatch = _first_difference(gathered[0], other, "state")
        if mismatch is not None:
            raise CheckpointError(f"ranks disagree at {mismatch} on rank {rank}")


def _cpu_tree(value: Any) -> Any:
    """Copy tensors to CPU so rank comparison does not gather GPU storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _first_difference(left: object, right: object, path: str) -> str | None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            return path
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return path
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            return path
        return None
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return path
        for key in left:
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            return path
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            found = _first_difference(lhs, rhs, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if isinstance(left, float) or isinstance(right, float):
        if left != right:
            return path
        return None
    if left != right:
        return path
    return None


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

