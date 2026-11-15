"""DeepFilterNet3 restore checks the payload before it changes the live progress."""

from __future__ import annotations

import unittest

from src.experiments.errors import RuntimeFailure
from src.experiments.official.deepfilternet3.state import DFProgress


def _tables(steps: int) -> tuple[list[float], list[float]]:
    return [0.001] * steps, [1e-12] * steps


class DFRestoreTests(unittest.TestCase):
    def test_failed_restore_keeps_the_rate_history(self) -> None:
        progress = DFProgress.initial("optimizer", 4)
        progress.logical_step = 3
        progress.learning_rates.append(0.001)
        progress.losses.append(1.0)
        before = progress.export()
        rates = progress.learning_rates
        with self.assertRaises(RuntimeFailure):
            progress.decode(
                {
                    "logical_step": 1,
                    "optimizer_updates": {"other": 1},
                    "phase": "in_epoch",
                    "schedule_steps": 4,
                },
                _tables,
            )
        self.assertEqual(progress.export(), before)
        self.assertIs(progress.learning_rates, rates)
        self.assertEqual(progress.losses, [1.0])

    def test_applied_update_counts_only_the_registered_optimizer(self) -> None:
        progress = DFProgress.initial("optimizer", 2)
        decoded = progress.decode(
            {
                "logical_step": 2,
                "optimizer_updates": {"optimizer": 2, "spare": 7},
                "phase": "in_epoch",
                "schedule_steps": 2,
                "learning_rates": [0.001, 0.0009],
                "weight_decays": [1e-12, 1e-12],
            },
            _tables,
        )
        progress.apply(decoded)
        progress.record_step(applied=True, loss=0.5, learning_rate=0.0008, decay=1e-12)
        self.assertEqual(progress.optimizer_updates["optimizer"], 3)
        self.assertEqual(progress.optimizer_updates["spare"], 7)
        self.assertEqual(progress.logical_step, 3)
        self.assertNotIn("losses", progress.export())


if __name__ == "__main__":
    unittest.main()
