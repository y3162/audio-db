"""MossFormerGAN optimizer defaults and the released validation plateau."""

import unittest

import torch
from torch import nn

from src.experiments.calc.mossformergan import (
    OFFICIAL_ADAMW_WEIGHT_DECAY,
    MossFormerGANTask,
    mossformer_validation_decision,
)
from tests.numerics.four_process.real_models import MossFormerGAN


def _validation(value: float) -> dict:
    return {"losses": {"generator": {"value": value, "status": "ok"}}}


class _PlateauStandIn:
    def __init__(self) -> None:
        torch.manual_seed(0)
        self.generator = nn.Linear(2, 2)
        self.discriminator = nn.Linear(2, 2)
        self.optim_g = torch.optim.AdamW(self.generator.parameters(), lr=0.5, weight_decay=0.01)
        self.optim_d = torch.optim.AdamW(self.discriminator.parameters(), lr=1.0, weight_decay=0.01)
        self.sched_g = None
        self.sched_d = None
        self._best_val = float("inf")
        self._stalls = 0
        self._best_snapshot = None
        self.stop_training = False
        self.generator_state_dict = lambda: MossFormerGANTask.generator_state_dict(self)
        self._capture_training_state = lambda: MossFormerGANTask._capture_training_state(self)
        self._restore_best_and_halve_generator = lambda: MossFormerGANTask._restore_best_and_halve_generator(self)

    def step(self, value: float) -> None:
        MossFormerGANTask.step_schedulers(self, _validation(value))


class MossFormerPlateauTests(unittest.TestCase):
    def test_adamw_default_matches_the_probe(self):
        parameter = nn.Parameter(torch.zeros(1))
        published = torch.optim.AdamW([parameter], lr=5e-4)
        group = published.param_groups[0]
        self.assertEqual(group["weight_decay"], 0.01)
        self.assertEqual(group["betas"], (0.9, 0.999))
        self.assertEqual(group["eps"], 1e-8)
        self.assertFalse(group["amsgrad"])
        self.assertEqual(OFFICIAL_ADAMW_WEIGHT_DECAY, group["weight_decay"])
        self.assertEqual(MossFormerGAN.weight_decay, group["weight_decay"])
        self.assertEqual(MossFormerGAN.lr, 5e-4)
        self.assertEqual(MossFormerGAN.betas, (0.9, 0.999))

    def test_equal_loss_stalls_and_keeps_the_recorded_value(self):
        best, stalls, action = mossformer_validation_decision(1.25, 2, 1.25)
        self.assertEqual(best, 1.25)
        self.assertEqual(stalls, 3)
        self.assertEqual(action, "hold")

    def test_decision_replaces_best_on_a_worse_loss(self):
        best, stalls, action = mossformer_validation_decision(float("inf"), 0, 1.0)
        self.assertEqual((best, stalls, action), (1.0, 0, "improve"))
        best, stalls, action = mossformer_validation_decision(best, stalls, 3.0)
        self.assertEqual((best, stalls, action), (3.0, 1, "hold"))
        best, stalls, action = mossformer_validation_decision(best, stalls, 2.0)
        self.assertEqual((best, stalls, action), (2.0, 0, "improve"))

    def test_fifth_stall_restores_best_and_halves_generator_only(self):
        standin = _PlateauStandIn()
        standin.step(1.0)
        saved = standin.generator.weight.detach().clone()
        saved_disc_lr = standin.optim_d.param_groups[0]["lr"]
        with torch.no_grad():
            standin.generator.weight.add_(1.0)
        for value in (1.1, 1.2, 1.3, 1.4):
            standin.step(value)
            self.assertFalse(torch.equal(standin.generator.weight, saved))
            self.assertEqual(standin.optim_g.param_groups[0]["lr"], 0.5)
        standin.step(1.5)
        self.assertTrue(torch.equal(standin.generator.weight, saved))
        self.assertEqual(standin.optim_g.param_groups[0]["lr"], 0.25)
        self.assertEqual(standin.optim_d.param_groups[0]["lr"], saved_disc_lr)
        self.assertEqual(standin._stalls, 5)
        self.assertFalse(standin.stop_training)

    def test_tenth_stall_stops_without_another_halving(self):
        standin = _PlateauStandIn()
        standin.step(1.0)
        for value in (1.1, 1.2, 1.3, 1.4, 1.5):
            standin.step(value)
        halved = standin.optim_g.param_groups[0]["lr"]
        self.assertEqual(halved, 0.25)
        for value in (1.6, 1.7, 1.8, 1.9):
            standin.step(value)
            self.assertFalse(standin.stop_training)
            self.assertEqual(standin.optim_g.param_groups[0]["lr"], halved)
        standin.step(2.0)
        self.assertTrue(standin.stop_training)
        self.assertEqual(standin._stalls, 10)
        self.assertEqual(standin.optim_g.param_groups[0]["lr"], halved)

    def test_checkpoint_roundtrip_keeps_the_stall_count(self):
        standin = _PlateauStandIn()
        standin.step(1.0)
        standin.step(1.4)
        restored = _PlateauStandIn()
        MossFormerGANTask.load_state_dict(restored, MossFormerGANTask.state_dict(standin))
        self.assertEqual(restored._stalls, 1)
        self.assertEqual(restored._best_val, standin._best_val)
        for value in (1.5, 1.6, 1.7, 1.8):
            restored.step(value)
        self.assertEqual(restored._stalls, 5)
        self.assertEqual(restored.optim_g.param_groups[0]["lr"], 0.25)


if __name__ == "__main__":
    unittest.main()
