"""SEMamba++ restore assigns the progress only after the payload checks succeed."""

from __future__ import annotations

import unittest

from src.experiments.errors import RuntimeFailure
from src.experiments.official.semambapp.state import SEMambaProgress


def _progress() -> SEMambaProgress:
    progress = SEMambaProgress.initial("optim_g", "optim_d", "protocol_conformant_substitute_materials")
    progress.logical_step = 4
    progress.optimizer_updates = {"optim_g": 4, "optim_d": 4, "spare": 2}
    progress.history.append({"epoch": 1})
    progress.step_log.append({"logical_step": 4})
    progress.boundary = {"trainer_epoch": 1, "closed": False}
    return progress


class SEMambaRestoreTests(unittest.TestCase):
    def test_failed_restore_keeps_the_live_progress(self) -> None:
        progress = _progress()
        before = progress.export()
        history = progress.history
        boundary = progress.boundary
        with self.assertRaises(RuntimeFailure):
            progress.restore(
                {
                    "logical_step": True,
                    "optimizer_updates": {"optim_g": 1, "optim_d": 1},
                    "phase": "in_epoch",
                    "claim_label": progress.claim_label,
                    "history": [],
                    "completed_boundaries": [],
                    "step_log": [],
                }
            )
        self.assertEqual(progress.export(), before)
        self.assertIs(progress.history, history)
        self.assertIs(progress.boundary, boundary)

    def test_missing_registered_optimizer_is_rejected(self) -> None:
        progress = _progress()
        before = progress.export()
        payload = progress.export()
        payload["optimizer_updates"] = {"optim_g": 3, "spare": 9}
        with self.assertRaises(RuntimeFailure):
            progress.restore(payload)
        self.assertEqual(progress.export(), before)

    def test_complete_update_counts_only_the_registered_pair(self) -> None:
        progress = _progress()
        payload = progress.export()
        payload["optimizer_updates"] = {"optim_d": 3, "spare": 9, "optim_g": 5}
        progress.restore(payload)
        progress.complete_update()
        self.assertEqual(
            list(progress.optimizer_updates.items()),
            [("optim_d", 4), ("spare", 9), ("optim_g", 6)],
        )
        self.assertEqual(progress.logical_step, 5)


if __name__ == "__main__":
    unittest.main()
