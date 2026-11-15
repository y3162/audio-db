"""MossFormerGAN restore checks the window before it replaces the live progress."""

from __future__ import annotations

import unittest

from src.experiments.errors import RuntimeFailure
from src.experiments.official.mossformergan.trainer import MossProgress


class MossRestoreTests(unittest.TestCase):
    def test_failed_restore_keeps_the_window(self) -> None:
        progress = MossProgress.initial("optim_g", "optim_d", "research")
        progress.logical_step = 2
        progress.accu_count = 1
        progress.window_generator.append(0.5)
        progress.counters["official_step"] = 3
        before = progress.export({}, {})
        window = progress.window_generator
        payload = progress.export({}, {})
        payload["accu_count"] = 0
        payload["generator_grads"] = {"weight": object()}
        with self.assertRaises(RuntimeFailure):
            progress.decode(payload, accumulation=2)
        self.assertEqual(progress.export({}, {}), before)
        self.assertIs(progress.window_generator, window)

    def test_closed_pair_counts_only_the_registered_optimizers(self) -> None:
        progress = MossProgress.initial("optim_g", "optim_d", "research")
        decoded = progress.decode(
            {
                **progress.export({}, {}),
                "logical_step": 1,
                "optimizer_updates": {"optim_d": 1, "spare": 4, "optim_g": 1},
                "accu_count": 0,
            },
            accumulation=2,
        )
        progress.apply(decoded)
        for name in progress._optimizer_ids:
            progress.optimizer_updates[name] += 1
        self.assertEqual(progress.optimizer_updates["optim_g"], 2)
        self.assertEqual(progress.optimizer_updates["optim_d"], 2)
        self.assertEqual(progress.optimizer_updates["spare"], 4)


if __name__ == "__main__":
    unittest.main()
