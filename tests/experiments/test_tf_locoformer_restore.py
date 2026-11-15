"""Failure-atomic restore for TF-Locoformer progress."""

from __future__ import annotations

import unittest

from src.experiments.errors import RuntimeFailure
from src.experiments.official.tf_locoformer.trainer import LocoformerProgress

PROFILE = "tf_locoformer.dns.accept_recipe.research_valid_fixed@1"


def _progress() -> LocoformerProgress:
    progress = LocoformerProgress.initial("optimizer", PROFILE)
    progress.logical_step = 2
    progress.optimizer_updates = {"optimizer": 1}
    progress.learning_rates = [0.001, 0.0005]
    progress.scheduled_rates = [0.0005, 0.0005]
    progress.history = [{"epoch": 1, "loss": 0.4, "si_snr": 1.0}]
    progress.phase = "in_epoch"
    progress.stop_reason = "kept"
    progress.losses = [0.4]
    return progress


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "boundary": None,
        "completed_boundaries": [],
        "early_stopped": False,
        "finished_epoch": None,
        "history": [],
        "learning_rates": [0.001],
        "logical_step": 1,
        "optimizer_updates": {"optimizer": 1},
        "phase": "in_epoch",
        "scheduled_rates": [0.0005],
        "stop_reason": "resumed",
        "validation_profile": PROFILE,
    }
    payload.update(overrides)
    return payload


class LocoformerRestoreTests(unittest.TestCase):
    def test_failed_restore_keeps_both_rate_histories(self) -> None:
        progress = _progress()
        before = progress.export()
        rates = progress.learning_rates
        scheduled = progress.scheduled_rates
        losses = progress.losses
        with self.assertRaisesRegex(RuntimeFailure, "validation profile does not match"):
            progress.restore(_payload(validation_profile="other"))
        with self.assertRaisesRegex(ValueError, "could not convert string to float"):
            progress.restore(_payload(scheduled_rates=["nope"]))
        with self.assertRaisesRegex(RuntimeFailure, "registered optimizer id"):
            progress.restore(_payload(optimizer_updates={"other": 1}))
        self.assertEqual(progress.export(), before)
        self.assertIs(progress.learning_rates, rates)
        self.assertIs(progress.scheduled_rates, scheduled)
        self.assertIs(progress.losses, losses)
        self.assertEqual(losses, [0.4])

    def test_applied_update_counts_only_the_registered_optimizer(self) -> None:
        progress = _progress()
        progress.restore(_payload(optimizer_updates={"spare": 9, "optimizer": 2}))
        progress.count_applied_optimizer()
        progress.record_step(loss=0.2, used_lr=0.001, scheduled_lr=0.0008)
        self.assertEqual(progress.optimizer_updates, {"spare": 9, "optimizer": 3})
        self.assertEqual(progress.learning_rates[-1], 0.001)
        self.assertEqual(progress.scheduled_rates[-1], 0.0008)
        self.assertNotIn("losses", progress.export())


if __name__ == "__main__":
    unittest.main()
