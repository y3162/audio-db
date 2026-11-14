"""Atomic ``audio-db-experiment-v1`` checkpoints: one Latest and named Archives.

The trainer offers each point where its state is consistent. The selector
named in the experiment YAML decides what to keep. This module collects the
state, checks that ranks agree, writes one verified temporary file, and
publishes it:

- ``latest.pt`` is the single Latest. ``os.replace`` swaps it only after the
  new file is complete and verified, so a failed save keeps the previous one.
- ``archive/<name>.pt`` are Archives. The selector chooses their names and
  when they are deleted. Deletion can only address ``archive/``.

Latest and an Archive written at the same point hold the same bytes. Both are
full checkpoints and both can be resumed.

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
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import torch

from src.experiments.config import ExperimentConfig, config_sha256
from src.experiments.data.identity import comparable_identity
from src.experiments.errors import BuildError, CheckpointError, ExperimentError, StateError
from src.experiments.calc.errors import TrainingError
from src.experiments.core.registry import StateRegistry
from src.experiments.execution.selector import (
    EVENTS,
    Decision,
    SavePoint,
    Selector,
    build_selector,
    selector_identity,
)
from src.experiments.execution.status import replace_text as _replace_text
from src.experiments.execution.status import read_status, write_complete, write_status  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "audio-db-experiment-v1"
SCHEMA_VERSION = 1
LATEST_NAME = "latest.pt"
ARCHIVE_DIR = "archive"
INITIALIZATION_NAME = "initialization.json"
INITIALIZATION_PROVENANCE_NAME = "initialization.data_provenance.json"
INITIALIZATION_FORMAT = "audio-db-initialization-v1"
_PROTECTED = (
    "data/checkpoints",
    "data/corpora",
    "data/db",
    "data/staging",
)
_LABELS = frozenset({"step", "validation_pending"})
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
        "progress",
        "output_directory",
        "parameter_ranks_agree",
        "provenance_sha256",
        "rank_count_in_rng",
        "rng_by_rank",
        "schema_version",
        "selector",
        "state_ids",
        "states",
        "validation_pending",
        "world_size",
    }
)
_CODE_KEYS = frozenset({"dirty", "git_commit"})
_LOAD_ORDER = ("module", "optimizer", "scheduler", "data", "trainer", "stopping", "evaluator", "other", "runtime")
# Data position, trainer history, RNG, and runtime state belong to the source
# experiment. Loading them continues that experiment, which is resume.
_INITIALIZE_KINDS = ("module", "optimizer", "scheduler")


@dataclass(frozen=True)
class RestoreReport:
    """What one restore call applied."""

    mode: Literal["resume", "initialize"]
    logical_step: int | None
    state_ids: tuple[str, ...]
    initialized_states: tuple[str, ...]


@dataclass(frozen=True)
class SaveReport:
    """What one offer wrote and deleted."""

    latest: Path | None = None
    archives: tuple[Path, ...] = ()
    deleted: tuple[str, ...] = ()

    @property
    def saved(self) -> bool:
        return self.latest is not None or bool(self.archives)


def assert_output_allowed(path: Path) -> None:
    """Refuse experiment output under published data directories."""

    resolved = path.resolve()
    for relative in _PROTECTED:
        protected = (REPO_ROOT / relative).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise CheckpointError(f"refusing to write experiment output under {relative}: {path}")


class DirectoryLock:
    """Non-blocking exclusive lock on a file in ``directory``, ``.lock`` unless named."""

    def __init__(self, directory: Path, name: str = ".lock") -> None:
        self._handle = (directory / name).open("a+")
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


def used_directory_marker(directory: Path) -> str | None:
    """Name one file that shows ``directory`` already holds a run, or ``None``."""

    if (directory / "complete.json").is_file():
        return "complete.json"
    if (directory / LATEST_NAME).is_file():
        return LATEST_NAME
    for pattern in (f"{ARCHIVE_DIR}/*.pt", "step_*.pt"):
        found = next((path for path in sorted(directory.glob(pattern)) if path.is_file()), None)
        if found is not None:
            return str(found.relative_to(directory))
    return None


class CheckpointManager:
    """Save and restore one experiment directory.

    The trainer decides which points are safe and offers them. The selector
    decides which offers are saved. This class writes the files and checks
    their identity. It does not branch on a model or a loss.
    """

    def __init__(self, directory: Path, selector: Selector | None = None) -> None:
        self.directory = directory
        self._selector = selector
        self._config: ExperimentConfig | None = None
        self._states: StateRegistry | None = None
        self._components: Any = None
        self._runtime: Any = None
        # True while Latest holds exactly the state of the previous offer.
        self._latest_current = False

    @property
    def latest_path(self) -> Path:
        return self.directory / LATEST_NAME

    @property
    def latest_is_current(self) -> bool:
        """Whether Latest contains the state offered most recently."""
        return self._latest_current

    @property
    def archive_directory(self) -> Path:
        return self.directory / ARCHIVE_DIR

    @property
    def selector(self) -> Selector:
        if self._selector is None:
            raise CheckpointError("checkpoint manager is not attached to an experiment")
        return self._selector

    def attach(self, config: ExperimentConfig, graph: Any) -> None:
        """Bind the live graph and build the selector. Save and restore require this."""

        if self._selector is None:
            self._selector = build_selector(config.selector_class, config.selector_kwargs)
        from src.experiments.core.factory import ComponentFactory

        if type(self._selector) is not ComponentFactory.resolve(config.selector_class):
            raise CheckpointError("selector class does not match the configured class")
        if selector_identity(self._selector) != config.selector_class:
            raise CheckpointError("selector checkpoint_class_id must equal its public class path")
        _require_declared_progress(graph.trainer, self._selector)
        self._config = config
        self._states = graph.states
        self._components = graph.components
        self._runtime = graph.runtime

    def offer(
        self,
        event: str,
        *,
        logical_step: int | None = None,
        metrics: Mapping[str, float] | None = None,
        prepare: Callable[[Any], None] | None = None,
        progress: Mapping[str, int] | None = None,
    ) -> SaveReport:
        """Offer one safe point to the selector and carry out its decision.

        Every rank calls this at the same point. ``prepare`` runs on every
        rank only when a file will be written; trainers use it to make
        replicated state equal across ranks, for example by broadcasting
        buffers. ``stop`` and ``final`` describe the state of the previous
        offer, so Latest is reported as current when that offer replaced it.
        ``progress`` is the trainer's named counters. ``logical_step`` is that
        map's ``logical_step`` when the trainer has one.
        """

        config, states, runtime = self._bound()
        if event not in EVENTS:
            raise CheckpointError(f"unknown save point event {event!r}")
        progress_map = _offer_progress(logical_step, progress)
        step = int(progress_map["logical_step"]) if "logical_step" in progress_map else 0
        current = self._latest_current and event in ("stop", "final")
        point = SavePoint(
            event=event,
            logical_step=step,
            metrics=_metrics(metrics),
            latest_is_current=current,
            progress=progress_map,
            progress_explicit=True,
        )
        decision = self._decide(point, runtime)
        report = SaveReport()
        if decision.latest or decision.archive or decision.delete:
            report = self._commit(point, decision, prepare, config, states, runtime)
        self._latest_current = decision.latest or current
        return report

    def restore(
        self,
        path: Path,
        *,
        mode: Literal["resume", "initialize"] = "resume",
        states: list[str] | tuple[str, ...] | None = None,
    ) -> RestoreReport:
        """Restore a finished checkpoint. Dotfiles and foreign formats are refused.

        ``resume`` loads every state, the RNG of each rank, and the selector
        of one of this experiment's own checkpoints. ``path`` may be the
        experiment directory, which means its Latest. ``initialize`` loads
        only ``states`` from a checkpoint of any experiment and leaves that
        source untouched. Neither changes a checkpoint file.
        """

        config, registry, runtime = self._bound()
        if mode not in {"resume", "initialize"}:
            raise CheckpointError(f"unknown restore mode {mode!r}")
        failure: BaseException | None = None
        report: RestoreReport | None = None
        source = Path(path)
        try:
            if mode == "resume":
                if states is not None:
                    raise CheckpointError("states are only valid when mode is initialize")
                if source.is_dir():
                    source = source / LATEST_NAME
                report = self._resume_local(source, config, registry, runtime)
            else:
                report = self._initialize_local(source, _state_list(states), config, registry, runtime)
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
        if mode == "resume":
            self._latest_current = source.resolve() == self.latest_path.resolve()
        return report

    def refuse_rewind(self, logical_step: int | None) -> None:
        """Refuse to continue from a restored state that is behind Latest.

        A continued run drops metric rows past the restored step and its
        saves replace Latest. From an older Archive that would discard
        finished work without a request to do so. An older state starts a
        new experiment instead. Every rank reads the same files, so every
        rank reaches the same answer.
        """

        latest = self.latest_path
        if self._latest_current or not latest.is_file():
            return
        try:
            newest = _read_payload(latest)["logical_step"]
        except CheckpointError as exc:
            raise CheckpointError(
                f"{LATEST_NAME} is unreadable ({exc}); move it aside before resuming from another checkpoint"
            ) from exc
        if newest is None or logical_step is None:
            return
        if int(logical_step) < int(newest):
            raise CheckpointError(
                f"the checkpoint is at logical step {logical_step}, behind {LATEST_NAME} at {newest}. "
                "Resume continues the newest state. To start from an older checkpoint, "
                "create a new experiment that names it under initialize."
            )

    def _decide(self, point: SavePoint, runtime: Any) -> Decision:
        failure: BaseException | None = None
        decision = Decision()
        try:
            decision = _checked_decision(self.selector.decide(point))
        except Exception as exc:
            failure = exc
        if runtime.world_size > 1:
            if failure is None:
                local: tuple[Any, ...] = ("ok", decision.latest, decision.archive, decision.delete)
            else:
                local = ("error", f"{type(failure).__name__}: {failure}")
            gathered = runtime.gather_objects(local)
            if failure is not None:
                raise failure
            for item in gathered:
                if isinstance(item, tuple) and item and item[0] == "error":
                    raise CheckpointError(str(item[1]))
            if any(item != local for item in gathered):
                raise CheckpointError("ranks disagree on the checkpoint decision")
        elif failure is not None:
            raise failure
        return decision

    def _commit(
        self,
        point: SavePoint,
        decision: Decision,
        prepare: Callable[[Any], None] | None,
        config: ExperimentConfig,
        states: StateRegistry,
        runtime: Any,
    ) -> SaveReport:
        writes = decision.latest or bool(decision.archive)
        error: BaseException | None = None
        try:
            from src.experiments.execution.profile import Span

            rngs: list[Any] = []
            if writes:
                if prepare is not None:
                    prepare(runtime)
                with Span("replicated_state_compare"):
                    _assert_replicated_states(states, self.selector, runtime, config.selector_kwargs)
                local_rng = dict(runtime.capture_rng())
                local_rng["rank"] = int(runtime.rank)
                rngs = runtime.gather_objects(local_rng)
                if len(rngs) != runtime.world_size:
                    raise CheckpointError("rng_by_rank length does not match world size")
            if int(runtime.rank) == 0:
                if writes:
                    from src.experiments.execution.provenance import bind_provenance

                    with Span("checkpoint_write"):
                        raw, provenance = bind_provenance(self.directory, config, self._components, runtime)
                        payload = _payload(
                            config=config,
                            states=states,
                            components=self._components,
                            runtime=runtime,
                            selector=self.selector,
                            directory=self.directory,
                            logical_step=point.progress.get("logical_step"),
                            progress=point.progress,
                            label="validation_pending" if point.event == "boundary_pending" else "step",
                            rngs=rngs,
                            provenance_sha256=provenance,
                        )
                        self._write(payload, raw, decision)
                # Deletion comes last: a state that names an Archive is on disk before the file can go.
                self._delete(decision.delete)
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
        return SaveReport(
            latest=self.latest_path if decision.latest else None,
            archives=tuple(self.archive_directory / name for name in decision.archive),
            deleted=decision.delete,
        )

    def _write(self, payload: dict[str, Any], provenance: bytes, decision: Decision) -> None:
        """Write one verified file, publish the Archives, then replace Latest.

        Archives come first. If the process dies in between, an Archive may
        exist that the surviving Latest does not name, and the next run writes
        it again. The reverse order could leave a Latest that names a missing
        Archive.
        """

        from src.experiments.execution.provenance import attach_provenance_sidecar

        temporary = self.directory / f".{LATEST_NAME}.tmp"
        staged: list[Path] = []
        try:
            try:
                with temporary.open("wb") as stream:
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise CheckpointError(f"checkpoint write failed: {exc}") from exc
            _verify_written(temporary, payload)
            if decision.archive:
                try:
                    self.archive_directory.mkdir(exist_ok=True)
                except OSError as exc:
                    raise CheckpointError(f"cannot create {ARCHIVE_DIR}/: {exc}") from exc
            for name in decision.archive:
                target = self.archive_directory / name
                attach_provenance_sidecar(target, provenance)
                stage = self.archive_directory / f".{name}.tmp"
                staged.append(stage)
                _link_or_copy(temporary, stage)
                _replace(stage, target)
            if decision.archive:
                _fsync_directory(self.archive_directory)
            if decision.latest:
                attach_provenance_sidecar(self.latest_path, provenance)
                _replace(temporary, self.latest_path)
        finally:
            for leftover in (temporary, *staged):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
        _fsync_directory(self.directory)

    def _delete(self, names: tuple[str, ...]) -> None:
        """Delete Archives by name. A missing file is not an error."""

        from src.evaluation.provenance import sidecar_path

        for name in names:
            target = self.archive_directory / name
            try:
                target.unlink(missing_ok=True)
                sidecar_path(target).unlink(missing_ok=True)
            except OSError as exc:
                raise CheckpointError(f"cannot delete {ARCHIVE_DIR}/{name}: {exc}") from exc

    def _resume_local(
        self,
        source: Path,
        config: ExperimentConfig,
        states: StateRegistry,
        runtime: Any,
    ) -> RestoreReport:
        payload = _read_finished(source)
        _validate_resume(payload, states, runtime, config, self._components, source, self.directory, self.selector)
        _apply_resume(payload, states, runtime, self.selector)
        return RestoreReport("resume", _stored_step(payload), states.ids(), ())

    def _initialize_local(
        self,
        source: Path,
        requested: tuple[str, ...],
        config: ExperimentConfig,
        states: StateRegistry,
        runtime: Any,
    ) -> RestoreReport:
        payload = _read_finished(source)
        _validate_initialize(payload, states, config, self._components, source, requested)
        applied = _apply_initialize(payload, states, requested)
        if int(runtime.rank) == 0:
            _record_initialization(self.directory, source, payload, requested)
        return RestoreReport("initialize", _stored_step(payload), states.ids(), tuple(applied))

    def _bound(self) -> tuple[ExperimentConfig, StateRegistry, Any]:
        if self._config is None or self._states is None or self._runtime is None:
            raise CheckpointError("checkpoint manager is not attached to an experiment")
        return self._config, self._states, self._runtime


def _payload(
    *,
    config: ExperimentConfig,
    states: StateRegistry,
    components: Any,
    runtime: Any,
    selector: Selector,
    directory: Path,
    logical_step: int | None,
    progress: Mapping[str, int],
    label: str,
    rngs: list[Any],
    provenance_sha256: str,
) -> dict[str, Any]:
    blobs: dict[str, Any] = {}
    for state_id in states.ids():
        blobs[state_id] = states.owner(state_id).state_dict()
    trainer_blob = _trainer_blob(states, blobs)
    if trainer_blob is not None and "logical_step" in trainer_blob and trainer_blob.get("logical_step") != logical_step:
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
        "initialize": None,
        "label": label,
        "logical_step": logical_step,
        "optimizer_updates": {str(key): int(value) for key, value in updates.items()},
        "progress": {str(key): int(value) for key, value in progress.items()},
        "output_directory": str(directory.resolve()),
        "parameter_ranks_agree": True,
        "provenance_sha256": provenance_sha256,
        "rank_count_in_rng": runtime.world_size,
        "rng_by_rank": list(rngs),
        "schema_version": SCHEMA_VERSION,
        "selector": {"class": selector_identity(selector), "kwargs": config.selector_kwargs, "state": _selector_state(selector)},
        "state_ids": list(states.ids()),
        "states": blobs,
        "validation_pending": {"logical_step": logical_step} if label == "validation_pending" else None,
        "world_size": runtime.world_size,
    }


def _validate_resume(
    payload: dict[str, Any],
    states: StateRegistry,
    runtime: Any,
    config: ExperimentConfig,
    components: Any,
    source: Path,
    directory: Path,
    selector: Selector,
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
    saved_states = _saved_states(payload)
    current = set(states.ids())
    missing = sorted(current - set(saved_states))
    extra = sorted(set(saved_states) - current)
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
    root = directory.resolve()
    resolved = source.resolve()
    if resolved.parent not in (root, root / ARCHIVE_DIR):
        raise CheckpointError("resume from a different directory is not supported")
    if payload.get("output_directory") != str(root):
        raise CheckpointError("resume from a different directory is not supported")
    if payload.get("config_sha256") != config_sha256(config):
        raise CheckpointError("checkpoint config identity does not match this experiment")
    saved_selector = payload.get("selector")
    if not isinstance(saved_selector, dict) or set(saved_selector) != {"class", "kwargs", "state"} or not isinstance(saved_selector.get("state"), dict):
        raise CheckpointError("checkpoint selector state is invalid")
    if saved_selector.get("kwargs") != config.selector_kwargs:
        raise CheckpointError("checkpoint selector settings do not match this experiment")
    if saved_selector.get("class") != selector_identity(selector):
        raise CheckpointError(
            f"checkpoint selector {saved_selector.get('class')} does not match "
            f"{selector_identity(selector)}; its state cannot be restored"
        )
    _align_scheduled_batches(payload, components)
    if not _dataset_identities_match(payload.get("dataset_identity"), _dataset_identity(config, components)):
        raise CheckpointError("checkpoint dataset identity does not match this experiment")
    _require_plan_hashes(payload, states)
    from src.experiments.execution.provenance import require_provenance

    require_provenance(payload, source, config, components, mode="resume")


def _validate_initialize(
    payload: dict[str, Any],
    states: StateRegistry,
    config: ExperimentConfig,
    components: Any,
    source: Path,
    requested: tuple[str, ...],
) -> None:
    saved_states = _saved_states(payload)
    current = set(states.ids())
    if payload.get("initialize") is not None:
        raise CheckpointError("checkpoint initialize field must be null")
    for state_id in requested:
        if state_id not in current:
            raise StateError(f"unknown state id {state_id}")
        kind = getattr(states.owner(state_id), "kind", None)
        if kind not in _INITIALIZE_KINDS:
            raise CheckpointError(
                f"state {state_id} is {kind} state and cannot initialize a new experiment; "
                "data position, trainer, and runtime state continue the same experiment through resume"
            )
        if not isinstance(saved_states.get(state_id), dict):
            raise CheckpointError(f"checkpoint has no state {state_id}")
    chosen = set(requested)
    for state_id in requested:
        for needed in _initialize_dependencies(config, state_id):
            if needed not in chosen:
                raise CheckpointError(
                    f"state {state_id} cannot be initialized without {needed}: "
                    "an optimizer needs the modules it updates and a scheduler needs its optimizer"
                )
    from src.experiments.execution.provenance import require_provenance

    require_provenance(payload, source, config, components, mode="initialize")


def _initialize_dependencies(config: ExperimentConfig, state_id: str) -> list[str]:
    """State ids that must be loaded together with ``state_id``."""

    spec = config.components.get(state_id)
    if spec is None or spec.ref is not None:
        return []
    needed: list[str] = []
    if spec.role == "optimizer":
        for binding in spec.parameters_of or ():
            needed.append(_resolve_alias(config, binding.component))
    elif spec.role == "scheduler":
        for value in spec.inject.values():
            for item in value if isinstance(value, tuple) else (value,):
                target = _resolve_alias(config, item.ref)
                target_spec = config.components.get(target)
                if target_spec is not None and target_spec.role == "optimizer":
                    needed.append(target)
    return needed


def _resolve_alias(config: ExperimentConfig, component_id: str) -> str:
    seen: set[str] = set()
    while component_id not in seen:
        seen.add(component_id)
        spec = config.components.get(component_id)
        if spec is None or spec.ref is None:
            break
        component_id = spec.ref
    return component_id


def _saved_states(payload: dict[str, Any]) -> dict[str, Any]:
    saved_ids = payload.get("state_ids")
    saved_states = payload.get("states")
    if not isinstance(saved_ids, list) or not isinstance(saved_states, dict):
        raise CheckpointError("checkpoint state manifest is invalid")
    if set(saved_ids) != set(saved_states):
        raise CheckpointError("checkpoint state_ids and states do not match")
    return saved_states


def _apply_resume(payload: dict[str, Any], states: StateRegistry, runtime: Any, selector: Selector) -> None:
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
    saved_selector = payload.get("selector")
    if saved_selector is not None:
        try:
            selector.load_state_dict(dict(saved_selector["state"]))
        except ExperimentError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"selector state failed to load: {exc}") from exc
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


def _apply_initialize(payload: dict[str, Any], states: StateRegistry, requested: tuple[str, ...]) -> list[str]:
    blobs: dict[str, Any] = payload["states"]
    applied: list[str] = []
    for kind in _INITIALIZE_KINDS:
        for state_id in requested:
            owner = states.owner(state_id)
            if getattr(owner, "kind", None) != kind:
                continue
            owner.load_state_dict(blobs[state_id])
            applied.append(state_id)
    return applied


def _record_initialization(
    directory: Path,
    source: Path,
    payload: dict[str, Any],
    requested: tuple[str, ...],
) -> None:
    """Record where the initial state came from, with the source's data provenance.

    The new experiment's weights depend on the data the source was trained
    and selected on. Its own provenance has to carry that lineage, and the
    source may be moved later, so the bytes are copied here once.
    """

    from src.evaluation.provenance import sidecar_path

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    record = {
        "format": INITIALIZATION_FORMAT,
        "checkpoint": str(source.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "config_sha256": payload.get("config_sha256"),
        "logical_step": None if payload["logical_step"] is None else int(payload["logical_step"]),
        "provenance_sha256": payload.get("provenance_sha256"),
        "states": list(requested),
    }
    copied = directory / INITIALIZATION_PROVENANCE_NAME
    try:
        if payload.get("provenance_sha256") is None:
            copied.unlink(missing_ok=True)
        else:
            temporary = copied.with_name(f".{copied.name}.tmp")
            shutil.copyfile(sidecar_path(source), temporary)
            os.replace(temporary, copied)
        _replace_text(directory / INITIALIZATION_NAME, json.dumps(record, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        raise CheckpointError(f"cannot record the initialization source: {exc}") from exc


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
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        raise CheckpointError("checkpoint progress is missing")
    for key, value in progress.items():
        if not isinstance(key, str) or not key or isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckpointError("checkpoint progress is invalid")
    logical_step = payload.get("logical_step")
    if "logical_step" in progress:
        if progress["logical_step"] != logical_step:
            raise CheckpointError("checkpoint logical_step does not match progress")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or logical_step < 0:
            raise CheckpointError("checkpoint logical_step is invalid")
    elif logical_step is not None:
        raise CheckpointError("checkpoint logical_step is set without that progress key")
    return payload


def pickle_error() -> type[BaseException]:
    import pickle

    return pickle.UnpicklingError


def _read_finished(source: Path) -> dict[str, Any]:
    if source.name.startswith("."):
        raise CheckpointError(f"temporary checkpoint files cannot be restored: {source.name}")
    if not source.is_file():
        raise CheckpointError(f"checkpoint not found: {source}")
    return _read_payload(source)


def _offer_progress(logical_step: int | None, progress: Mapping[str, int] | None) -> dict[str, int]:
    if progress is None:
        progress = {}
    elif not isinstance(progress, Mapping):
        raise CheckpointError("progress must be a mapping")
    merged: dict[str, int] = {}
    for key, value in progress.items():
        if not isinstance(key, str) or not key:
            raise CheckpointError("progress keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckpointError(f"progress {key} must be an integer >= 0")
        merged[key] = int(value)
    if logical_step is not None:
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or logical_step < 0:
            raise CheckpointError("logical_step must be an integer >= 0")
        if "logical_step" in merged and merged["logical_step"] != int(logical_step):
            raise CheckpointError("logical_step does not match progress")
        merged["logical_step"] = int(logical_step)
    return merged


def _stored_step(payload: dict[str, Any]) -> int | None:
    value = payload["logical_step"]
    return None if value is None else int(value)


def _require_declared_progress(trainer: object, selector: Selector) -> None:
    """Reject a periodic selector or evaluator whose counter the trainer does not publish."""

    declared = getattr(trainer, "progress_keys", None)
    if declared is None:
        return
    if not isinstance(declared, tuple) or not all(isinstance(item, str) and item for item in declared):
        raise BuildError("progress_keys must be a tuple of names")
    key = getattr(selector, "progress_key", None)
    if isinstance(key, str) and key not in declared:
        published = ", ".join(declared) if declared else "no progress"
        raise BuildError(f"selector requires progress {key}, but the trainer publishes {published}")
    for evaluator in getattr(trainer, "evaluators", ()) or ():
        unit = getattr(getattr(evaluator, "schedule", None), "unit", None)
        if unit in {"logical_step", "epoch"} and unit not in declared:
            raise BuildError(f"evaluator requires progress {unit}")


def _metrics(metrics: Mapping[str, float] | None) -> dict[str, float]:
    if metrics is None:
        return {}
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CheckpointError("save point metrics must map names to numbers")
        cleaned[key] = float(value)
    return cleaned


def _checked_decision(decision: object) -> Decision:
    if not isinstance(decision, Decision) or not isinstance(decision.latest, bool):
        raise CheckpointError("a selector must return a Decision")
    archive = tuple(_archive_name(name) for name in decision.archive)
    delete = tuple(_archive_name(name) for name in decision.delete)
    if len(set(archive)) != len(archive) or len(set(delete)) != len(delete):
        raise CheckpointError("a checkpoint decision repeats an Archive name")
    if set(archive) & set(delete):
        raise CheckpointError("a checkpoint decision writes and deletes the same Archive")
    return Decision(latest=decision.latest, archive=archive, delete=delete)


def _archive_name(name: object) -> str:
    """An Archive is one ``*.pt`` file directly inside ``archive/``."""

    if (
        not isinstance(name, str)
        or not name.endswith(".pt")
        or name.startswith(".")
        or name != Path(name).name
        or "\\" in name
    ):
        raise CheckpointError(f"invalid Archive name {name!r}")
    return name


def _state_list(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise CheckpointError("initialize requires a non-empty list of state ids")
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise CheckpointError("initialize requires a non-empty list of state ids")
        if value in cleaned:
            raise CheckpointError(f"initialize repeats state {value}")
        cleaned.append(value)
    return tuple(cleaned)


def _selector_state(selector: Selector) -> dict[str, Any]:
    state = selector.state_dict()
    if not isinstance(state, dict):
        raise CheckpointError("selector state_dict must return a mapping")
    return state


def _link_or_copy(source: Path, target: Path) -> None:
    """Give ``target`` the bytes of ``source``. A hard link is used where the file system has one."""

    try:
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
            return
        except OSError:
            pass
        shutil.copyfile(source, target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError as exc:
        raise CheckpointError(f"checkpoint write failed: {exc}") from exc


def _replace(temporary: Path, target: Path) -> None:
    try:
        os.replace(temporary, target)
    except OSError as exc:
        raise CheckpointError(f"checkpoint replace failed: {exc}") from exc


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


def _assert_replicated_states(
    states: StateRegistry, selector: Selector, runtime: Any, selector_kwargs: Mapping[str, Any] | None = None
) -> None:
    """Refuse a checkpoint when replicated state differs across ranks.

    RNG is not compared here. It is gathered separately because each rank keeps
    its own generators. Runtime state is compared, including the GradScaler
    scale and growth tracker, and so is the selector state. Comparison uses
    dtype, shape, and values.
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
    local["checkpoint.selector"] = {"class": selector_identity(selector), "kwargs": selector_kwargs, "state": _selector_state(selector)}
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

