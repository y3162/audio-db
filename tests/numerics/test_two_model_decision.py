"""CPU checks for the declared SEMamba++ / MP-SENet numerics decision rule."""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.numerics.reference.two_model_decision import (
    FORMULA_ATOL,
    GRAD_ATOL,
    LOSS_ATOL,
    MP_ORDER,
    MP_TRAIN,
    SEMAMBA_ORDER,
    SEMAMBA_TRAIN,
    WEIGHT_ATOL,
    combine_gates,
    evaluate_campaign,
    source_inventory,
    strict_reasons,
    systematic_oa,
    trainer_status,
)


def _row(**overrides):
    base = {
        "step": 0,
        "finite": True,
        "loss_gap": {"generator": 0.0, "discriminator": 0.0},
        "generator_grad_max_abs": 0.0,
        "generator_grad_real_max_abs": 0.0,
        "large_cosine_min": 1.0,
        "weight_real_max_abs": 0.0,
        "weight_tiny_max_abs": 0.0,
        "weight_max_abs": 0.0,
        "wave_max_abs": 0.0,
        "optimizer_max_abs": 0.0,
        "discriminator_grad_max_abs": 0.0,
        "loss_generator_gap": 0.0,
        "formula_residual_max_abs": 0.0,
    }
    base.update(overrides)
    return base


def _payload(oa_grad: float, self_grad: float, *, oa_weight: float, sgd_weight: float, sgd_grad: float) -> dict:
    def block(grad: float, weight: float, formula: float = 0.0) -> list[dict]:
        return [_row(step=index, generator_grad_max_abs=grad, generator_grad_real_max_abs=grad, weight_real_max_abs=weight, weight_max_abs=weight, loss_generator_gap=0.0, formula_residual_max_abs=formula) for index in range(3)]

    payload = {}
    for seed in ("101", "202", "303"):
        payload[seed] = {
            "init_ok": True,
            "adamw": {
                "oo": block(self_grad, oa_weight / 10),
                "aa": block(self_grad, oa_weight / 10),
                "oa": block(oa_grad, oa_weight, formula=0.0),
                "oa_resync": block(1e-8, 1e-8),
            },
            "sgd_matched": {
                "oo": block(sgd_grad, sgd_weight),
                "aa": block(sgd_grad, sgd_weight),
                "oa": block(sgd_grad, sgd_weight),
            },
        }
    return payload


class DecisionTests(unittest.TestCase):
    def test_thresholds_are_the_existing_allowances(self) -> None:
        self.assertEqual(LOSS_ATOL, 1e-5)
        self.assertEqual(GRAD_ATOL, 1e-4)
        self.assertEqual(WEIGHT_ATOL, 1e-4)
        self.assertEqual(FORMULA_ATOL, 1e-4)

    def test_same_magnitude_alone_is_not_a_porting_gap(self) -> None:
        report = systematic_oa([1e-2, 2e-2, 4e-2], [1.5e-2, 2.5e-2, 3e-2])
        self.assertEqual(report["status"], "WITHIN_SELF_ENVELOPE")

    def test_oa_above_the_self_envelope_is_systematic(self) -> None:
        report = systematic_oa([1e-4, 2e-4, 3e-4], [1e-2, 1.2e-2, 9e-3])
        self.assertEqual(report["status"], "SYSTEMATIC")

    def test_strict_failure_is_not_relabeled(self) -> None:
        reasons = strict_reasons("semamba", _row(generator_grad_max_abs=4.53e-2, loss_gap={"generator": 1e-6}))
        self.assertTrue(any(item.startswith("grad") for item in reasons))
        campaign = evaluate_campaign("semamba", _payload(4.53e-2, 4.02e-2, oa_weight=1e-3, sgd_weight=1e-6, sgd_grad=1e-7))
        self.assertEqual(campaign["STRICT_NATIVE_3STEP_ABSOLUTE_THRESHOLD"], "FAIL")
        self.assertEqual(campaign["distribution"]["status"], "GO")

    def test_sgd_shrink_supports_amplification_and_is_not_the_whole_gate(self) -> None:
        campaign = evaluate_campaign("mp", _payload(2e-2, 2e-2, oa_weight=5e-3, sgd_weight=1e-6, sgd_grad=1e-7))
        self.assertEqual(campaign["sgd"]["status"], "GO")
        gates = {
            "init_and_same_state": "GO",
            "distribution": "GO",
            "sgd": "GO",
            "trainer": "UNKNOWN",
            "regression": "GO",
        }
        self.assertEqual(combine_gates(gates), "CONDITIONAL GO")
        gates["trainer"] = "FAIL"
        self.assertEqual(combine_gates(gates), "FAIL")
        gates["trainer"] = "GO"
        self.assertEqual(combine_gates(gates), "GO")

    def test_tight_control_cannot_promote_a_native_trainer_miss(self) -> None:
        self.assertEqual(
            trainer_status(
                run_called=True,
                logical_steps=3,
                scheduler_steps=0,
                native_failure_count=2,
                tight_failure_count=0,
            ),
            "CONDITIONAL",
        )
        self.assertEqual(
            trainer_status(
                run_called=True,
                logical_steps=3,
                scheduler_steps=0,
                native_failure_count=0,
                tight_failure_count=4,
            ),
            "GO",
        )
        self.assertEqual(
            trainer_status(
                run_called=False,
                logical_steps=0,
                scheduler_steps=0,
                native_failure_count=0,
                tight_failure_count=0,
            ),
            "UNKNOWN",
        )

    def test_official_train_scripts_keep_discriminator_then_generator(self) -> None:
        if not SEMAMBA_TRAIN.is_file() or not MP_TRAIN.is_file():
            self.skipTest("official checkout is not present")
        semamba = source_inventory(SEMAMBA_TRAIN.read_text(encoding="utf-8"), SEMAMBA_ORDER)
        mp = source_inventory(MP_TRAIN.read_text(encoding="utf-8"), MP_ORDER)
        self.assertTrue(semamba["update_order_ok"], semamba)
        self.assertTrue(mp["update_order_ok"], mp)
        self.assertFalse(semamba["mentions_clip_grad"])
        self.assertFalse(mp["mentions_clip_grad"])
        self.assertFalse(semamba["mentions_grad_scaler"])
        self.assertFalse(mp["mentions_grad_scaler"])
        self.assertTrue(semamba["sets_cudnn_benchmark_true"])
        self.assertTrue(mp["sets_cudnn_benchmark_true"])
        self.assertFalse(semamba["passes_weight_decay_keyword"])
        self.assertFalse(mp["passes_weight_decay_keyword"])
        self.assertGreaterEqual(Path(SEMAMBA_TRAIN).stat().st_size, 1)
