"""Run results returned to the caller. Checkpoint files are not part of this object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    """Outcome of one finished trainer run."""

    stop_reason: str
    logical_step: int
    optimizer_updates: dict[str, int]
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "optimizer_updates", dict(self.optimizer_updates))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
