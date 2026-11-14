"""Slow trainer used to observe SIGTERM without a real SE step."""

from __future__ import annotations

import time
from pathlib import Path

from src.experiments.execution.direct import run_direct
from src.experiments.results import RunResult
from src.experiments.signals import interrupt_requested, ranks_should_stop
from src.experiments.trainers import _append_jsonl, _rewind_metrics


class InterruptProbe:
    """Sleeps between logical steps and offers each one to the selector.

    ``SIGTERM`` is observed at the next loop check, after the step that was
    already in progress has been saved. It does not save from inside a step.
    """

    def __init__(self, max_logical_steps: int = 20) -> None:
        self.max_logical_steps = max_logical_steps
        self.logical_step = 0
        self.optimizer_updates = {"probe": 0}
        self.stop_reason = "initialized"
        self._restored = False

    def state_dict(self) -> dict[str, object]:
        return {
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.logical_step = int(state["logical_step"])
        updates = state.get("optimizer_updates")
        self.optimizer_updates = dict(updates) if isinstance(updates, dict) else {"probe": 0}
        stop_reason = state.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason
        self._restored = True

    def run(self, runtime: object, checkpoint: object):  # type: ignore[no-untyped-def]
        return run_direct(self, runtime, checkpoint)

    def fit(self):  # type: ignore[no-untyped-def]
        self.lifecycle.start()
        checkpoint = self.lifecycle.checkpoint
        runtime = self.runtime
        path = Path(checkpoint.directory) / "metrics.jsonl"
        rank = int(getattr(runtime, "rank", 0))
        if self._restored:
            if rank == 0:
                _rewind_metrics(path, self.logical_step)
        else:
            if rank == 0:
                path.write_text("", encoding="utf-8")
            self.logical_step = 0
            self.optimizer_updates = {"probe": 0}
        while self.logical_step < self.max_logical_steps:
            if ranks_should_stop(runtime):
                self.stop_reason = "interrupted"
                if self.logical_step > 0:
                    self.lifecycle.save_if_needed("stop", logical_step=self.logical_step)
                return self.lifecycle.finish(
                    RunResult(
                        stop_reason="interrupted",
                        logical_step=self.logical_step,
                        optimizer_updates=dict(self.optimizer_updates),
                        output_dir=checkpoint.directory,
                    )
                )
            _pause(0.2)
            self.logical_step += 1
            self.optimizer_updates["probe"] += 1
            if rank == 0:
                _append_jsonl(
                    path,
                    {
                        "applied": True,
                        "logical_step": self.logical_step,
                        "loss": 0.0,
                        "sample_id": f"probe-{self.logical_step}",
                    },
                )
            self.lifecycle.save_if_needed("step", logical_step=self.logical_step)
        self.stop_reason = "completed"
        self.lifecycle.save_if_needed("final", logical_step=self.logical_step)
        return self.lifecycle.finish(
            RunResult(
                stop_reason="completed",
                logical_step=self.logical_step,
                optimizer_updates=dict(self.optimizer_updates),
                output_dir=checkpoint.directory,
            )
        )


def _pause(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if interrupt_requested():
            return
        try:
            time.sleep(0.05)
        except InterruptedError:
            return
