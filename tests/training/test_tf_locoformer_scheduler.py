"""Warmup plateau state against the pinned ESPnet scheduler.

The reference class is ``espnet2.schedulers.warmup_reducelronplateau`` from
the checkout, commit ``90eed8e53498e7af682bc6ff39d9067ae440d6a4``. It is not
``src.training.se.optim.WarmupReduceLROnPlateau``. A checkout that lacks that
file skips the comparison. The literal-state tests stay and do not import
ESPnet.
"""

from __future__ import annotations

import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch

from src.training.se.optim import WarmupReduceLROnPlateau


ROOT = Path(__file__).resolve().parents[2]
ESP = ROOT / "src/se/official/espnet"
PINNED = "90eed8e53498e7af682bc6ff39d9067ae440d6a4"
SCHEDULER_PATH = "espnet2/schedulers/warmup_reducelronplateau.py"
# Learning rates in these tests must match the pinned class exactly.
LR_ABS_TOL = 0.0


def _optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _port(**kwargs) -> tuple[WarmupReduceLROnPlateau, torch.optim.Optimizer]:
    optimizer = _optimizer(kwargs.pop("lr", 1e-3))
    schedule = WarmupReduceLROnPlateau(optimizer, **kwargs)
    return schedule, optimizer


def _official_class():
    source = ESP / SCHEDULER_PATH
    if not source.is_file():
        raise unittest.SkipTest("official ESPnet checkout missing")
    pinned = subprocess.check_output(
        ["git", "-C", str(ESP), "rev-parse", f"{PINNED}:{SCHEDULER_PATH}"],
        text=True,
    ).strip()
    current = subprocess.check_output(
        ["git", "-C", str(ESP), "hash-object", str(source)],
        text=True,
    ).strip()
    if pinned != current:
        raise AssertionError(f"{SCHEDULER_PATH} differs from {PINNED}")
    if "typeguard" not in sys.modules:
        stub = types.ModuleType("typeguard")
        stub.typechecked = lambda fn: fn
        sys.modules["typeguard"] = stub
    esp = str(ESP)
    if esp not in sys.path:
        sys.path.insert(0, esp)
    from espnet2.schedulers.warmup_reducelronplateau import (
        WarmupReduceLROnPlateau as Official,
    )

    return Official


def _official(official_cls, **kwargs):
    optimizer = _optimizer(kwargs.pop("lr", 1e-3))
    schedule = official_cls(optimizer, **kwargs)
    return schedule, optimizer


def _batch(port, official) -> None:
    port.step_batch()
    official.step()


def _valid(port, official, metric: float) -> None:
    port.step_validation(metric)
    official.step(metric)


def _assert_same(test: unittest.TestCase, port, official, message: str) -> None:
    for name in ("step_num", "best", "num_bad_epochs", "last_epoch", "cooldown_counter"):
        test.assertEqual(getattr(port, name), getattr(official, name), f"{message} {name}")
    left = float(port.optimizer.param_groups[0]["lr"])
    right = float(official.optimizer.param_groups[0]["lr"])
    test.assertLessEqual(abs(left - right), LR_ABS_TOL, f"{message} lr {left} vs {right}")


class WarmupPlateauLiteralTests(unittest.TestCase):
    """Known states for the DNS warmup length. No ESPnet import."""

    def _dns(self):
        return _port(
            warmup_steps=4000,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0.0,
            eps=1e-8,
        )

    def test_epoch1_validation_at_3834_updates_plateau_state(self):
        schedule, optimizer = self._dns()
        for _ in range(3834):
            schedule.step_batch()
        schedule.step_validation(2.0)
        self.assertEqual(schedule.step_num, 3834)
        self.assertEqual(schedule.last_epoch, 1)
        self.assertEqual(schedule.best, 2.0)
        self.assertEqual(schedule.num_bad_epochs, 0)
        self.assertEqual(schedule.cooldown_counter, 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3 * (4000 ** -1) * 3834)

    def test_later_epochs_count_improvements_and_non_improvements(self):
        schedule, _optimizer = self._dns()
        for _ in range(3834):
            schedule.step_batch()
        schedule.step_validation(2.0)
        schedule.step_validation(1.0)
        self.assertEqual(schedule.best, 1.0)
        self.assertEqual(schedule.num_bad_epochs, 0)
        self.assertEqual(schedule.last_epoch, 2)
        schedule.step_validation(1.0)
        self.assertEqual(schedule.best, 1.0)
        self.assertEqual(schedule.num_bad_epochs, 1)
        self.assertEqual(schedule.last_epoch, 3)
        schedule.step_validation(0.5)
        self.assertEqual(schedule.best, 0.5)
        self.assertEqual(schedule.num_bad_epochs, 0)
        self.assertEqual(schedule.last_epoch, 4)

    def test_warmup_boundary_learning_rates(self):
        schedule, optimizer = self._dns()
        for step in range(1, 4002):
            schedule.step_batch()
            if step in (3999, 4000):
                self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3 * (4000 ** -1) * step)
            if step == 4001:
                self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(schedule.step_num, 4001)
        self.assertEqual(schedule.last_epoch, 0)
        self.assertEqual(schedule.best, float("inf"))

    def test_patience_reduces_learning_rate_on_the_exceeding_epoch(self):
        schedule, optimizer = _port(
            warmup_steps=1, mode="min", factor=0.5, patience=1, threshold=0.0, threshold_mode="abs"
        )
        schedule.step_batch()
        schedule.step_validation(1.0)
        schedule.step_validation(1.0)
        self.assertEqual(schedule.num_bad_epochs, 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        schedule.step_validation(1.0)
        self.assertEqual(schedule.last_epoch, 3)
        self.assertEqual(schedule.num_bad_epochs, 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-4)
        schedule.step_validation(1.0)
        self.assertEqual(schedule.num_bad_epochs, 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-4)

    def test_resume_matches_an_uninterrupted_run(self):
        def fresh():
            return _port(
                warmup_steps=4000,
                mode="min",
                factor=0.5,
                patience=3,
                threshold=1e-4,
                threshold_mode="rel",
            )

        continuous, continuous_opt = fresh()
        for _ in range(3834):
            continuous.step_batch()
        continuous.step_validation(2.0)
        saved_sched = continuous.state_dict()
        saved_opt = continuous_opt.state_dict()
        for _ in range(200):
            continuous.step_batch()
        continuous.step_validation(2.5)
        continuous.step_validation(1.0)

        resumed, resumed_opt = fresh()
        resumed_opt.load_state_dict(saved_opt)
        resumed.load_state_dict(saved_sched)
        for _ in range(200):
            resumed.step_batch()
        resumed.step_validation(2.5)
        resumed.step_validation(1.0)
        self.assertEqual(resumed.step_num, continuous.step_num)
        self.assertEqual(resumed.best, continuous.best)
        self.assertEqual(resumed.num_bad_epochs, continuous.num_bad_epochs)
        self.assertEqual(resumed.last_epoch, continuous.last_epoch)
        self.assertEqual(resumed_opt.param_groups[0]["lr"], continuous_opt.param_groups[0]["lr"])
        self.assertEqual(resumed.step_num, 4034)
        self.assertEqual(resumed.best, 1.0)
        self.assertEqual(resumed.num_bad_epochs, 0)
        self.assertEqual(resumed.last_epoch, 3)


class OfficialWarmupPlateauTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official_cls = _official_class()

    def _pair(self, **kwargs):
        kwargs.setdefault("warmup_steps", 4000)
        kwargs.setdefault("mode", "min")
        kwargs.setdefault("factor", 0.5)
        kwargs.setdefault("patience", 3)
        kwargs.setdefault("threshold", 1e-4)
        kwargs.setdefault("threshold_mode", "rel")
        kwargs.setdefault("cooldown", 0)
        kwargs.setdefault("min_lr", 0.0)
        kwargs.setdefault("eps", 1e-8)
        port, port_opt = _port(**kwargs)
        official, official_opt = _official(self.official_cls, **kwargs)
        return port, official, port_opt, official_opt

    def test_epoch1_and_following_validations_match(self):
        port, official, _port_opt, _official_opt = self._pair()
        for _ in range(3834):
            _batch(port, official)
        _valid(port, official, 2.0)
        _assert_same(self, port, official, "epoch 1")
        self.assertEqual(port.best, 2.0)
        self.assertEqual(port.last_epoch, 1)
        _valid(port, official, 1.0)
        _assert_same(self, port, official, "improved")
        self.assertEqual(port.num_bad_epochs, 0)
        _valid(port, official, 1.0)
        _assert_same(self, port, official, "not improved")
        self.assertEqual(port.num_bad_epochs, 1)
        _valid(port, official, 0.5)
        _assert_same(self, port, official, "improved again")
        self.assertEqual(port.num_bad_epochs, 0)

    def test_warmup_boundary_matches(self):
        port, official, _port_opt, _official_opt = self._pair()
        for step in range(1, 4002):
            _batch(port, official)
            if step in (3999, 4000, 4001):
                _assert_same(self, port, official, f"step {step}")
        self.assertEqual(port.optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(official.optimizer.param_groups[0]["lr"], 1e-3)

    def test_patience_decay_matches_after_warmup(self):
        port, official, port_opt, official_opt = self._pair(warmup_steps=2, patience=1, threshold=0.0, threshold_mode="abs")
        _batch(port, official)
        _batch(port, official)
        _assert_same(self, port, official, "warmed")
        _valid(port, official, 1.0)
        _valid(port, official, 1.0)
        self.assertEqual(port.last_epoch, 2)
        self.assertEqual(port_opt.param_groups[0]["lr"], 1e-3)
        _assert_same(self, port, official, "before decay")
        _valid(port, official, 1.0)
        _assert_same(self, port, official, "decay epoch")
        self.assertEqual(port.last_epoch, 3)
        self.assertEqual(port.num_bad_epochs, 0)
        self.assertEqual(port_opt.param_groups[0]["lr"], 5e-4)
        self.assertEqual(official_opt.param_groups[0]["lr"], 5e-4)
        _valid(port, official, 1.0)
        _assert_same(self, port, official, "after decay")
        self.assertEqual(port_opt.param_groups[0]["lr"], 5e-4)

    def test_cooldown_matches(self):
        port, official, port_opt, official_opt = self._pair(
            warmup_steps=1, patience=0, cooldown=1, threshold=0.0, threshold_mode="abs"
        )
        _batch(port, official)
        _valid(port, official, 1.0)
        _valid(port, official, 2.0)
        _assert_same(self, port, official, "reduce into cooldown")
        self.assertEqual(port.cooldown_counter, 1)
        self.assertEqual(port.num_bad_epochs, 0)
        self.assertEqual(port_opt.param_groups[0]["lr"], 5e-4)
        _valid(port, official, 2.0)
        _assert_same(self, port, official, "inside cooldown")
        self.assertEqual(port.cooldown_counter, 0)
        self.assertEqual(port_opt.param_groups[0]["lr"], 5e-4)
        self.assertEqual(official_opt.param_groups[0]["lr"], 5e-4)
        _valid(port, official, 2.0)
        _assert_same(self, port, official, "after cooldown")
        self.assertEqual(port_opt.param_groups[0]["lr"], 2.5e-4)

    def test_min_lr_and_eps_match(self):
        port, official, _port_opt, _official_opt = self._pair(
            warmup_steps=1, patience=0, threshold=0.0, threshold_mode="abs", min_lr=6e-4
        )
        _batch(port, official)
        _valid(port, official, 1.0)
        _valid(port, official, 2.0)
        _assert_same(self, port, official, "min_lr")
        self.assertEqual(port.optimizer.param_groups[0]["lr"], 6e-4)

        blocked, blocked_official, _bopt, _oopt = self._pair(
            lr=1e-8, warmup_steps=1, patience=0, threshold=0.0, threshold_mode="abs", eps=1e-8
        )
        _batch(blocked, blocked_official)
        _valid(blocked, blocked_official, 1.0)
        _valid(blocked, blocked_official, 2.0)
        _assert_same(self, blocked, blocked_official, "eps")
        self.assertEqual(blocked.optimizer.param_groups[0]["lr"], 1e-8)

    def test_resume_matches_official_and_uninterrupted(self):
        def apply(port, official, events):
            for kind, value in events:
                if kind == "batch":
                    for _ in range(value):
                        _batch(port, official)
                else:
                    _valid(port, official, value)
                _assert_same(self, port, official, kind)

        prefix = (("batch", 3834), ("valid", 2.0))
        tail = (("batch", 200), ("valid", 2.5), ("valid", 1.0))
        port, official, port_opt, official_opt = self._pair()
        apply(port, official, prefix)
        split = {
            "port": port.state_dict(),
            "port_opt": port_opt.state_dict(),
            "official": official.state_dict(),
            "official_opt": official_opt.state_dict(),
        }
        apply(port, official, tail)

        resumed, resumed_official, resumed_opt, resumed_official_opt = self._pair()
        resumed_opt.load_state_dict(split["port_opt"])
        resumed.load_state_dict(split["port"])
        resumed_official_opt.load_state_dict(split["official_opt"])
        resumed_official.load_state_dict(split["official"])
        apply(resumed, resumed_official, tail)
        _assert_same(self, resumed, port, "port resume")
        _assert_same(self, resumed_official, official, "official resume")
        self.assertEqual(resumed.step_num, 4034)
        self.assertEqual(resumed.best, 1.0)
        self.assertEqual(resumed.num_bad_epochs, 0)


if __name__ == "__main__":
    unittest.main()
