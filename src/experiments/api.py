"""Public experiment entry points.

``load_experiment`` validates YAML. ``build_experiment`` constructs the graph
through state registration. ``Experiment.run`` executes the trainer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.experiments.builder import BuiltGraph, ExperimentBuilder
from src.experiments.checkpoint import (
    CheckpointManager,
    DirectoryLock,
    assert_output_allowed,
    write_complete,
    write_status,
)
from src.experiments.config import ExperimentConfig, load_experiment
from src.experiments.errors import (
    BuildError,
    CheckpointError,
    ConfigError,
    ExperimentError,
    PrepareError,
    StateError,
)
from src.experiments.factory import ComponentFactory, ComponentSpec
from src.experiments.results import RunResult

__all__ = [
    "Experiment",
    "RunResult",
    "build_experiment",
    "load_experiment",
    "preflight_experiment",
]


class Experiment:
    """One experiment configuration and, after ``build``, its live graph."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._graph: BuiltGraph | None = None
        self._built = False
        self._closed = False
        self._ran = False
        self._directory: Path | None = None
        self._lock: DirectoryLock | None = None
        self._checkpoint: CheckpointManager | None = None
        self._running = False

    def build(self) -> Experiment:
        """Construct, prepare, and register state. A second call is an error."""

        if self._built:
            raise StateError("experiment is already built")
        if self._closed:
            raise StateError("experiment is closed")
        self._open_directory()
        try:
            runtime = _create_runtime(self.config)
            self._graph = ExperimentBuilder().build(self.config, runtime)
            if self._checkpoint is not None:
                self._checkpoint.attach(self.config, self._graph)
        except Exception:
            self._mark_failed()
            raise
        self._built = True
        return self

    def run(
        self,
        *,
        resume: str | Path | None = None,
        initialize_from: str | Path | None = None,
    ) -> RunResult:
        """Build if needed, restore when asked, then run the entry trainer.

        ``resume`` and ``initialize_from`` cannot be combined. A failed
        restore does not start the trainer. CPU restore writes into the
        prepared module in place, so the optimizer keeps those parameters.
        """

        if resume is not None and initialize_from is not None:
            raise ConfigError("resume and initialize_from cannot both be set")
        if self._closed:
            raise StateError("experiment is closed")
        if self._ran:
            raise StateError("experiment has already been run")
        try:
            if not self._built:
                self.build()
            if self._graph is None or self._checkpoint is None or self._directory is None:
                raise BuildError("experiment is not built")
            if resume is not None and (self._directory / "complete.json").is_file():
                raise CheckpointError("resume refused because complete.json exists")
            if resume is not None:
                self._checkpoint.restore(Path(resume), mode="resume")
            elif initialize_from is not None:
                self._checkpoint.restore(Path(initialize_from), mode="initialize")
            self._graph.states.seal()
            self._running = True
            self._write_status("RUNNING")
            result = self._graph.trainer.run(self._graph.runtime, self._checkpoint)  # type: ignore[attr-defined]
            if not isinstance(result, RunResult):
                raise BuildError(f"component {self.config.entry}: trainer did not return RunResult")
            if result.stop_reason == "completed":
                write_complete(self._directory, result.logical_step)
                self._write_status("COMPLETED")
            elif result.stop_reason == "interrupted":
                self._write_status("INTERRUPTED")
            else:
                raise BuildError(f"component {self.config.entry}: unknown stop reason {result.stop_reason}")
            self._ran = True
            return result
        except Exception:
            self._mark_failed()
            raise
        finally:
            self.close()

    def get(self, component_id: str) -> object:
        """Return the live component. Module ids return their prepared handle."""

        if not self._built or self._graph is None:
            raise BuildError("experiment is not built")
        return self._graph.components.get(component_id)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return

    def close(self) -> None:
        """Close data modules and release the directory lock."""

        if self._closed:
            return
        self._closed = True
        graph = self._graph
        if graph is not None:
            for component_id, spec in self.config.components.items():
                if spec.ref is not None or spec.role != "data":
                    continue
                component = graph.components.get(component_id)
                close = getattr(component, "close", None)
                if callable(close):
                    close()
            _destroy_process_group()
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def _open_directory(self) -> None:
        directory = self.config.output_directory
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        directory = directory.resolve()
        assert_output_allowed(directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CheckpointError(f"cannot create output directory {directory}: {exc}") from exc
        self._directory = directory
        self._lock = DirectoryLock(directory)
        self._checkpoint = CheckpointManager(directory)

    def _write_status(self, status: str) -> None:
        if self._directory is None:
            return
        write_status(self._directory, status)

    def _mark_failed(self) -> None:
        if self._directory is None or self._closed:
            return
        status_path = self._directory / "status.json"
        if self._running or not status_path.is_file():
            try:
                self._write_status("FAILED")
            except OSError:
                pass
        self.close()


def build_experiment(config: ExperimentConfig) -> Experiment:
    """Create an experiment and build it. This does not call ``run``."""

    return Experiment(config).build()


def preflight_experiment(experiment: Experiment, directory: str | Path | None = None) -> Path:
    """Write ``preflight.json`` for a built experiment and do not train."""

    if not experiment._built or experiment._graph is None:
        raise BuildError("experiment is not built")
    target = Path(directory) if directory is not None else experiment._directory
    if target is None:
        raise BuildError("experiment has no output directory")
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    else:
        target = target.resolve()
    assert_output_allowed(target)
    target.mkdir(parents=True, exist_ok=True)
    graph = experiment._graph
    payload = {
        "status": "PASS",
        "training_executed": False,
        "entry": experiment.config.entry,
        "component_ids": sorted(experiment.config.components),
        "state_ids": list(graph.states.ids()),
        "logical_step": 0,
        "optimizer_updates": {},
    }
    path = target / "preflight.json"
    temporary = target / ".preflight.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _create_runtime(config: ExperimentConfig) -> object:
    factory = ComponentFactory()
    try:
        runtime = factory.create(ComponentSpec(config.runtime_class, config.runtime_kwargs))
    except PrepareError:
        raise
    except BuildError as exc:
        raise BuildError(f"runtime: {exc}") from exc
    if not hasattr(runtime, "prepare_model") or not hasattr(runtime, "step"):
        raise PrepareError(f"runtime {config.runtime_class} does not implement the runtime contract")
    return runtime


def _destroy_process_group() -> None:
    """Destroy a process group only when this process already created one.

    Importing ``torch.distributed`` constructs a process-lifetime temporary
    directory inside PyTorch. Phase 1 does not initialize a group, so the
    import is skipped unless some other code already loaded the module.
    """

    import sys

    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
