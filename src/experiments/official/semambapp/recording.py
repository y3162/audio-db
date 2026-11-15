"""Write one shared SEMamba++ step row. The trainer offers the step save afterwards."""

from __future__ import annotations

from typing import Any

from src.experiments.errors import RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.semambapp.state import SEMambaProgress
from src.experiments.runtime import ProcessRuntime


class StepLog:
    def write(
        self,
        progress: SEMambaProgress,
        runtime: ProcessRuntime,
        journal: StepJournal,
        local: dict[str, Any],
        *,
        epoch: int,
    ) -> None:
        shared = _shared_log(runtime, local)
        entry = {
            "adversarial": shared["adversarial"],
            "applied_d": True,
            "applied_g": True,
            "complex": shared["complex"],
            "consistency": shared["consistency"],
            "discriminator": shared["discriminator"],
            "epoch": epoch,
            "generator": shared["generator"],
            "generator_forwards": shared["generator_forwards"],
            "generator_previews": shared["generator_previews"],
            "generator_reforwards": shared["generator_reforwards"],
            "logical_step": progress.logical_step,
            "loss": shared["generator"],
            "lr_d": shared["lr_d"],
            "lr_g": shared["lr_g"],
            "magnitude": shared["magnitude"],
            "matching": shared["matching"],
            "mel": shared["mel"],
            "phase": shared["phase"],
            "sample_id": shared["sample_id"],
            "scan_configured": shared["scan_configured"],
            "scan_executed": shared["scan_executed"],
        }
        progress.step_log.append(entry)
        journal.write_metrics(entry)
        journal.write_sample(
            {
                "logical_step": progress.logical_step,
                "loss": local["generator"],
                "realization_id": local["realization_id"],
                "sample_id": local["sample_id"],
            }
        )


def _shared_log(runtime: ProcessRuntime, local: dict[str, float | str]) -> dict[str, float | str]:
    gathered = runtime.gather_objects(dict(local))
    shared = gathered[0]
    if not isinstance(shared, dict):
        raise RuntimeFailure("rank 0 did not return GAN metrics")
    return shared
