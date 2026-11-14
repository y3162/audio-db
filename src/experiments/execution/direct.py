"""Drive one trainer without an Experiment.

Comparison scripts pass a runtime and a checkpoint-like object. Experiment
injects :class:`~src.experiments.execution.lifecycle.Lifecycle` instead and
does not use this session. The session does not restore, write ``status.json``,
or decide a launcher outcome.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.errors import BuildError, CheckpointError
from src.experiments.execution.results import RunResult


class DirectSession:
    """Save adapter for one direct ``fit`` call.

    Comparison scripts still pass a checkpoint-like object here. Trainers use
    the same attributes as :class:`~src.experiments.execution.lifecycle.Lifecycle`:
    ``output_directory``, ``config_identity``, ``restore_report``, and
    ``save_if_needed``. This is the only adapter that holds that object.
    """

    def __init__(self, checkpoint: Any) -> None:
        self.checkpoint = checkpoint
        self.restore_report = None
        self.reason: str | None = None
        self.control = None
        self._started = False
        self._finished = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def output_directory(self) -> Path:
        """Directory the comparison script gave the checkpoint-like object."""

        directory = getattr(self.checkpoint, "directory", None)
        if directory is None:
            raise CheckpointError("direct session has no output directory")
        return Path(directory)

    @property
    def config_identity(self) -> str | None:
        """Experiment identity when the checkpoint-like object carries a config.

        A comparison stub has no experiment document. Research rows then store
        a null identity, the same value as before this session existed.
        """

        config = getattr(self.checkpoint, "_config", None)
        if config is None:
            return None
        from src.experiments.config import ExperimentConfig, config_sha256

        if not isinstance(config, ExperimentConfig):
            return None
        return config_sha256(config)

    @property
    def restored(self) -> bool:
        """True only when this session itself recorded a resume. Direct callers do not."""

        return getattr(self.restore_report, "mode", None) == "resume"

    def start(self, *, resume: Any = None) -> None:
        """Mark the session open. Direct callers have already built their state."""

        del resume
        if self._started:
            raise CheckpointError("lifecycle has already started")
        if self._finished:
            raise CheckpointError("lifecycle has finished")
        self._started = True
        return None

    def save_if_needed(
        self,
        event: str,
        *,
        logical_step: int | None = None,
        metrics: Any = None,
        progress: Any = None,
        prepare: Any = None,
    ) -> Any:
        """Forward one safe point. A stub that has no ``progress`` argument still works."""

        if not self._started:
            raise CheckpointError("lifecycle has not started")
        if self._finished:
            raise CheckpointError("lifecycle has finished")
        kwargs: dict[str, Any] = {}
        if logical_step is not None:
            kwargs["logical_step"] = logical_step
        if metrics is not None:
            kwargs["metrics"] = metrics
        if prepare is not None:
            kwargs["prepare"] = prepare
        if progress is not None:
            kwargs["progress"] = progress
        try:
            return self.checkpoint.offer(event, **kwargs)
        except TypeError:
            if "progress" not in kwargs:
                raise
            kwargs.pop("progress")
            return self.checkpoint.offer(event, **kwargs)

    def finish(self, result: RunResult, *, primary: bool | None = None, control: Any = None) -> RunResult:
        """Accept the trainer's result without publishing a launcher status."""

        del primary, control
        if not self._started:
            raise CheckpointError("lifecycle has not started")
        if self._finished:
            raise CheckpointError("lifecycle has already finished")
        if not isinstance(result, RunResult):
            raise BuildError("trainer did not return RunResult")
        if result.stop_reason not in {"completed", "interrupted", "early_stopping"}:
            raise BuildError(f"unknown stop reason {result.stop_reason}")
        self._finished = True
        self.reason = result.stop_reason
        return result


def run_direct(trainer: Any, runtime: Any, checkpoint: Any) -> RunResult:
    """Bind ``runtime`` and a direct session, then call ``fit`` once."""

    trainer.runtime = runtime
    current = getattr(trainer, "lifecycle", None)
    if current is None or getattr(current, "checkpoint", None) is not checkpoint:
        trainer.lifecycle = DirectSession(checkpoint)
    result = trainer.fit()
    if not isinstance(result, RunResult):
        raise BuildError("trainer did not return RunResult")
    return result
