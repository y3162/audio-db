"""Start, offer, and finish one experiment without owning the training order.

The trainer calls :meth:`Lifecycle.save_if_needed` only at a point it already
considers consistent. This class does not inspect the model graph to decide
whether that point is safe. The selector still decides whether a file is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.experiments.config import ExperimentConfig
from src.experiments.errors import BuildError, CheckpointError
from src.experiments.execution.checkpoint import CheckpointManager, SaveReport
from src.experiments.execution.results import RunResult

_REASONS = frozenset({"completed", "interrupted", "early_stopping"})
_UNSET = object()


class Lifecycle:
    """One start and one finish around a trainer run.

    ``start`` restores a resume or an initialization. When Experiment constructed
    this object with ``session=True``, ``start`` also seals state, installs the
    stop signal, and publishes ``RUNNING``. ``finish`` publishes the process
    outcome once. The trainer calls both. Training calls ``save_if_needed`` in
    between. This class does not choose the training order.
    """

    def __init__(
        self,
        checkpoint: CheckpointManager,
        runtime: Any,
        config: ExperimentConfig,
        *,
        resume_request: str | Path | None = None,
        states: Any = None,
        control: Any = None,
        session: bool = False,
    ) -> None:
        if not isinstance(checkpoint, CheckpointManager):
            raise CheckpointError("lifecycle requires a checkpoint manager")
        if not isinstance(config, ExperimentConfig):
            raise CheckpointError("lifecycle requires an experiment config")
        self.checkpoint = checkpoint
        self.runtime = runtime
        self.config = config
        self.resume_request = resume_request
        self.states = states
        self.control = control
        self.session = bool(session)
        self.restore_report: Any = None
        self.reason: str | None = None
        self.engaged = False
        self._started = False
        self._finished = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def finished(self) -> bool:
        return self._finished

    def start(self, *, resume: str | Path | None | object = _UNSET) -> Any:
        """Restore once, then open the session when Experiment asked for one.

        A second call is an error, including after a failure. Omit ``resume``
        to use the request Experiment injected. Pass ``None`` to start without
        a resume even when a request was injected.
        """

        if self._started:
            raise CheckpointError("lifecycle has already started")
        if self._finished:
            raise CheckpointError("lifecycle has finished")
        chosen = self.resume_request if resume is _UNSET else resume
        directory = self.checkpoint.directory
        if chosen is not None:
            if (directory / "complete.json").is_file():
                raise CheckpointError("resume refused because complete.json exists")
            restored = self.checkpoint.restore(Path(chosen), mode="resume")
            self.checkpoint.refuse_rewind(restored.logical_step)
            self.restore_report = restored
        elif self.config.initialize is not None:
            self.restore_report = self.checkpoint.restore(
                self.config.initialize.checkpoint,
                mode="initialize",
                states=self.config.initialize.states,
            )
        self._started = True
        if self.control is not None:
            synchronize = getattr(self.control, "synchronize", None)
            if callable(synchronize):
                synchronize()
        if self.states is not None:
            self.states.seal()
        if self.session:
            # Matches the old Experiment flag: a failure while opening the
            # session replaces status, a failure during restore does not.
            self.engaged = True
            self._open_session()
        return self.restore_report

    def save_if_needed(
        self,
        event: str,
        *,
        logical_step: int | None = None,
        metrics: Mapping[str, float] | None = None,
        progress: Mapping[str, int] | None = None,
        prepare: Any = None,
    ) -> SaveReport:
        """Offer one safe point. Write nothing when the selector declines it.

        ``logical_step`` is stored when the trainer has that counter.
        ``progress`` may carry it under the same name. An empty progress map
        is an event-only point: ``step`` still fails in the selector when the
        configured key is missing, and the file stores null rather than 0.
        """

        if not self._started:
            raise CheckpointError("lifecycle has not started")
        if self._finished:
            raise CheckpointError("lifecycle has finished")
        stored = _merge_progress(logical_step, progress)
        return self.checkpoint.offer(
            event,
            logical_step=stored.get("logical_step"),
            metrics=metrics,
            progress=stored,
            prepare=prepare,
        )

    def finish(self, result: RunResult, *, primary: bool | None = None, control: Any = None) -> RunResult:
        """Record the outcome once and return it. The launcher still writes the final verdict.

        A direct run writes ``complete.json`` for a normal completion. Under
        the launcher this only reports the outcome on the still-running status.
        ``primary`` defaults to rank 0. ``control`` defaults to the stopping
        control Experiment injected.
        """

        if not self._started:
            raise CheckpointError("lifecycle has not started")
        if self._finished:
            raise CheckpointError("lifecycle has already finished")
        if not isinstance(result, RunResult):
            raise BuildError("trainer did not return RunResult")
        if result.stop_reason not in _REASONS:
            raise BuildError(f"unknown stop reason {result.stop_reason}")
        if control is None:
            control = self.control
        if primary is None:
            primary = self._rank() == 0
        details: dict[str, Any] = {}
        if result.stop_reason == "early_stopping":
            if control is None or not getattr(control, "stopped", False) or not self.checkpoint.latest_is_current:
                raise BuildError("early stopping requires a saved current Latest")
            details = {"reason": "early_stopping", "stopping": control.state_dict()}
        self._finished = True
        self.reason = result.stop_reason
        if not primary:
            return result
        from src.experiments.api import launch_id
        from src.experiments.execution.status import write_complete, write_status

        directory = self.checkpoint.directory
        if launch_id() is not None:
            write_status(
                directory,
                "RUNNING",
                outcome=result.stop_reason,
                logical_step=result.logical_step,
                launch_id=launch_id(),
                **details,
            )
            return result
        if result.stop_reason in {"completed", "early_stopping"}:
            write_complete(directory, result.logical_step, **details)
            write_status(directory, "COMPLETED", logical_step=result.logical_step, **details)
            return result
        write_status(directory, "INTERRUPTED", logical_step=result.logical_step)
        return result

    def _rank(self) -> int:
        return int(getattr(self.runtime, "rank", 0))

    def _barrier(self) -> None:
        barrier = getattr(self.runtime, "barrier", None)
        if callable(barrier):
            barrier()

    def _open_session(self) -> None:
        """Publish RUNNING after every rank can see a stop request."""

        from src.experiments.api import launch_id
        from src.experiments.execution.checkpoint import write_status
        from src.experiments.execution.signals import clear_interrupt, install_interrupt_handler

        clear_interrupt()
        install_interrupt_handler()
        self._barrier()
        if self._rank() == 0:
            details: dict[str, Any] = {}
            supervisor = launch_id()
            if supervisor is not None:
                details["launch_id"] = supervisor
            write_status(self.checkpoint.directory, "RUNNING", **details)
        self._barrier()
        from src.experiments.execution.profile import enabled as profile_enabled
        from src.experiments.execution.profile import reset as reset_profile

        if profile_enabled():
            reset_profile()
            device = getattr(self.runtime, "device", None)
            if device is not None and getattr(device, "type", None) == "cuda":
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(int(device.index or 0))


def _merge_progress(logical_step: int | None, progress: Mapping[str, int] | None) -> dict[str, int]:
    if progress is not None and not isinstance(progress, Mapping):
        raise CheckpointError("progress must be a mapping")
    merged: dict[str, int] = {} if progress is None else {}
    if progress is not None:
        for key, value in progress.items():
            if not isinstance(key, str) or not key:
                raise CheckpointError("progress keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CheckpointError(f"progress {key} must be an integer >= 0")
            merged[key] = int(value)
    if logical_step is not None:
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or logical_step < 0:
            raise CheckpointError("logical_step must be an integer >= 0")
        if "logical_step" in merged and merged["logical_step"] != logical_step:
            raise CheckpointError("logical_step does not match progress")
        merged["logical_step"] = int(logical_step)
    return merged
