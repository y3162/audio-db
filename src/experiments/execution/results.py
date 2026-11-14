"""Run results returned to the caller. Checkpoint files are not part of this object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    """Outcome of one finished trainer run."""

    stop_reason: str
    logical_step: int | None
    optimizer_updates: dict[str, int]
    output_dir: Path

    def __post_init__(self) -> None:
        step = self.logical_step
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("logical_step must be a non-negative integer or omitted")
        object.__setattr__(self, "optimizer_updates", dict(self.optimizer_updates))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
