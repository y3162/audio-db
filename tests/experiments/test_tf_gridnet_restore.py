"""Failure-atomic restore for TF-GridNet progress.

A valid payload keeps the registered optimizer id and any extra id. The extra
id is not incremented. A missing registered id is rejected. ``losses`` is not
part of the payload.
"""

from __future__ import annotations

import unittest

from src.experiments.errors import RuntimeFailure
from src.experiments.official.tf_gridnet.trainer import GridNetProgress


def _progress() -> GridNetProgress:
    progress = GridNetProgress.initial("optimizer")
    progress.logical_step = 2
    progress.optimizer_updates = {"optimizer": 2}
    progress.learning_rates = [0.001, 0.001]
    progress.history = [{"epoch": 1, "loss": 0.5, "inner": {"k": 1}}]
    progress.completed_boundaries = [{"closed": True}]
    progress.boundary = {"nested": {"n": 1}}
    progress.phase = "pending_validation"
    progress.finished_epoch = 1
    progress.early_stopped = False
    progress.stop_reason = "kept"
    progress.losses = [0.2, 0.3]
    return progress


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "boundary": {"nested": {"n": 9}},
        "completed_boundaries": [],
        "early_stopped": True,
        "finished_epoch": 3,
        "history": [{"epoch": 4, "loss": 0.1}],
        "learning_rates": [0.001, 0.001, 0.0007],
        "logical_step": 3,
        "optimizer_updates": {"optimizer": 3},
        "phase": "in_epoch",
        "stop_reason": "resumed",
    }
    payload.update(overrides)
    return payload


class GridNetRestoreTests(unittest.TestCase):
    def test_failed_restore_keeps_live_containers_and_losses(self) -> None:
        cases = (
            ("logical", _payload(logical_step=True), RuntimeFailure, "trainer state is invalid"),
            ("phase", _payload(phase="pending_scheduler"), RuntimeFailure, "trainer phase 'pending_scheduler' is invalid"),
            ("history", _payload(history=None), RuntimeFailure, "trainer history is invalid"),
            ("rates", _payload(learning_rates="nope"), RuntimeFailure, "trainer history is invalid"),
            ("rate_value", _payload(learning_rates=[0.1, "nope"]), ValueError, "could not convert string to float"),
            ("finished", _payload(finished_epoch=True), RuntimeFailure, "finished_epoch is invalid"),
            ("boundary", _payload(boundary=[1, 2]), TypeError, "dictionary update sequence element"),
            ("missing_id", _payload(optimizer_updates={"other": 3}), RuntimeFailure, "registered optimizer id"),
            ("bad_count", _payload(optimizer_updates={"optimizer": "nope"}), ValueError, "invalid literal for int"),
        )
        for name, payload, error, message in cases:
            with self.subTest(name=name):
                progress = _progress()
                before = progress.export()
                losses = progress.losses
                history = progress.history
                boundary = progress.boundary
                nested = boundary["nested"]
                with self.assertRaisesRegex(error, message):
                    progress.restore(payload)
                self.assertEqual(progress.export(), before)
                self.assertIs(progress.losses, losses)
                self.assertEqual(losses, [0.2, 0.3])
                self.assertIs(progress.history, history)
                self.assertIs(progress.boundary, boundary)
                self.assertIs(nested, boundary["nested"])
                self.assertEqual(nested["n"], 1)
                self.assertEqual(progress.stop_reason, "kept")

    def test_extra_optimizer_id_is_kept_and_not_incremented(self) -> None:
        progress = _progress()
        progress.restore(_payload(optimizer_updates={"spare": 4, "optimizer": 1}))
        self.assertEqual(list(progress.optimizer_updates), ["spare", "optimizer"])
        progress.record_step(loss=0.4, applied=True, learning_rate=0.0005)
        self.assertEqual(progress.optimizer_updates, {"spare": 4, "optimizer": 2})
        self.assertEqual(progress.logical_step, 4)
        self.assertEqual(progress.losses, [0.2, 0.3, 0.4])
        exported = progress.export()
        self.assertNotIn("losses", exported)

    def test_skipped_optimizer_step_still_advances_the_logical_step(self) -> None:
        progress = _progress()
        progress.restore(_payload(optimizer_updates={"optimizer": 1}, logical_step=1, learning_rates=[0.001]))
        progress.losses = []
        progress.record_step(loss=0.4, applied=False, learning_rate=0.001)
        self.assertEqual(progress.logical_step, 2)
        self.assertEqual(progress.optimizer_updates, {"optimizer": 1})


if __name__ == "__main__":
    unittest.main()
