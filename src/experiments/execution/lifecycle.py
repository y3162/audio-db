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


class Lifecycle:
    """One start and one finish around a trainer run.

    ``start`` restores a resume or an initialization. ``finish`` publishes the
    process outcome once. Training calls ``save_if_needed`` in between.
    """

    def __init__(self, checkpoint: CheckpointManager, runtime: Any, config: ExperimentConfig) -> None:
        if not isinstance(checkpoint, CheckpointManager):
            raise CheckpointError("lifecycle requires a checkpoint manager")
        if not isinstance(config, ExperimentConfig):
            raise CheckpointError("lifecycle requires an experiment config")
        self.checkpoint = checkpoint
        self.runtime = runtime
        self.config = config
        self.restore_report: Any = None
        self.reason: str | None = None
        self._started = False
        self._finished = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def finished(self) -> bool:
        return self._finished

    def start(self, *, resume: str | Path | None = None) -> Any:
        """Restore once. A second call is an error, including after a failure."""

        if self._started:
            raise CheckpointError("lifecycle has already started")
        if self._finished:
            raise CheckpointError("lifecycle has finished")
        directory = self.checkpoint.directory
        if resume is not None:
            if (directory / "complete.json").is_file():
                raise CheckpointError("resume refused because complete.json exists")
            restored = self.checkpoint.restore(Path(resume), mode="resume")
            self.checkpoint.refuse_rewind(restored.logical_step)
            self.restore_report = restored
        elif self.config.initialize is not None:
            self.restore_report = self.checkpoint.restore(
                self.config.initialize.checkpoint,
                mode="initialize",
                states=self.config.initialize.states,
            )
        self._started = True
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

        ``logical_step`` remains the stored counter when the trainer has one.
        ``progress`` may carry that counter under the same name. A missing
        counter is refused here until a save point can be stored without it.
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

    def finish(self, result: RunResult, *, primary: bool, control: Any = None) -> None:
        """Record the outcome once. The launcher still writes the final verdict.

        A direct run writes ``complete.json`` for a normal completion. Under
        the launcher this only reports the outcome on the still-running status.
        """

        if not self._started:
            raise CheckpointError("lifecycle has not started")
        if self._finished:
            raise CheckpointError("lifecycle has already finished")
        if not isinstance(result, RunResult):
            raise BuildError("trainer did not return RunResult")
        if result.stop_reason not in _REASONS:
            raise BuildError(f"unknown stop reason {result.stop_reason}")
        details: dict[str, Any] = {}
        if result.stop_reason == "early_stopping":
            if control is None or not getattr(control, "stopped", False) or not self.checkpoint.latest_is_current:
                raise BuildError("early stopping requires a saved current Latest")
            details = {"reason": "early_stopping", "stopping": control.state_dict()}
        self._finished = True
        self.reason = result.stop_reason
        if not primary:
            return
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
            return
        if result.stop_reason in {"completed", "early_stopping"}:
            write_complete(directory, result.logical_step, **details)
            write_status(directory, "COMPLETED", logical_step=result.logical_step, **details)
            return
        write_status(directory, "INTERRUPTED", logical_step=result.logical_step)


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
    elif progress is None:
        raise CheckpointError("logical_step must be an integer >= 0")
    return merged
