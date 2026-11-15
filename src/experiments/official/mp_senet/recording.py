"""Step log rows for MP-SENet.

The progress object owns ``step_log``. This module builds one JSON row,
writes it through ``StepJournal``, and can hold that write until a later
flush. The held row is not part of the checkpoint. A flush does not score
research and does not save.

CUDA overlap may flush the previous row while the next signal backward is
still running. That row already describes the completed step. The trainer
flushes again, via the checkpoint ``prepare`` hook, before any file is published.
"""

from __future__ import annotations

from typing import Any

from src.experiments.errors import RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.runtime import ProcessRuntime


class StepRecorder:
    """One unflushed metrics row. Empty at every save that actually writes."""

    def __init__(self) -> None:
        self._pending: tuple[dict[str, Any], dict[str, Any]] | None = None

    @property
    def pending_logical_step(self) -> int | None:
        if self._pending is None:
            return None
        return int(self._pending[0]["logical_step"])

    def share(self, runtime: ProcessRuntime, local: dict[str, Any]) -> dict[str, Any]:
        """Rank 0's scalars, copied onto every rank so the checkpoint matches."""

        gathered = runtime.gather_objects(dict(local))
        shared = gathered[0]
        if not isinstance(shared, dict):
            raise RuntimeFailure("rank 0 did not return GAN metrics")
        return shared

    def entry(self, shared: dict[str, Any], *, epoch: int, logical_step: int) -> dict[str, Any]:
        return {
            "applied_d": True,
            "applied_g": True,
            "complex": shared["complex"],
            "consistency": shared["consistency"],
            "discriminator": shared["discriminator"],
            "epoch": epoch,
            "generator": shared["generator"],
            "logical_step": logical_step,
            "loss": shared["generator"],
            "lr_d": shared["lr_d"],
            "lr_g": shared["lr_g"],
            "magnitude": shared["magnitude"],
            "metric": shared["metric"],
            "phase": shared["phase"],
            "sample_id": shared["sample_id"],
            "time": shared["time"],
        }

    def stage(self, journal: StepJournal, entry: dict[str, Any], local: dict[str, Any]) -> None:
        """Keep this row. Write an older row first so it is not replaced."""

        if self._pending is not None:
            self.flush(journal)
        self._pending = (entry, local)

    def write_now(self, journal: StepJournal, entry: dict[str, Any], local: dict[str, Any]) -> None:
        """Write any older row, then this one. Nothing stays pending."""

        self.flush(journal)
        _write(journal, entry, local)

    def flush(self, journal: StepJournal) -> None:
        """Write the held row, if there is one. This does not offer a checkpoint."""

        if self._pending is None:
            return
        entry, local = self._pending
        self._pending = None
        _write(journal, entry, local)

    def publish(self, journal: StepJournal, entry: dict[str, Any], local: dict[str, Any], *, defer: bool) -> None:
        """Stage the row when the next backward may overlap its write. Otherwise write it now."""

        if defer:
            self.stage(journal, entry, local)
        else:
            self.write_now(journal, entry, local)


def _write(journal: StepJournal, entry: dict[str, Any], local: dict[str, Any]) -> None:
    journal.write_metrics(entry)
    journal.write_sample({**local, "logical_step": entry["logical_step"], "loss": local["generator"]})
