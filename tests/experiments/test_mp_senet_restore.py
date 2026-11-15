"""Failure-atomic restore and optimizer-id rules for MP-SENet progress.

The accepted payload shape matches checkpoints written before ``8f54ab8``:
both registered optimizer ids, with ``int`` conversion of their counts.
A payload that omits a registered id is rejected. Extra ids are kept and
are not incremented.
"""

from __future__ import annotations

import unittest

from src.experiments.errors import RuntimeFailure
from src.experiments.official.mp_senet.trainer import MPTrainingProgress


def _progress() -> MPTrainingProgress:
    progress = MPTrainingProgress.initial("optim_g", "optim_d")
    progress.logical_step = 4
    progress.optimizer_updates = {"optim_g": 4, "optim_d": 4}
    progress.scheduler_steps = 2
    progress.phase = "pending_boundary"
    progress.boundary = {"closed": False, "nested": {"n": 1}}
    progress.history = [{"epoch": 1, "inner": {"k": 2}}]
    progress.completed_boundaries = [{"closed": True}]
    progress.step_log = [{"logical_step": 4}]
    progress.best_pesq = 1.25
    progress.best_epoch = 3
    progress.best_logical_step = 4
    progress.stop_reason = "kept"
    return progress


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "best_epoch": 1,
        "best_logical_step": 9,
        "best_pesq": 2.5,
        "boundary": {"closed": False, "nested": {"n": 99}},
        "completed_boundaries": [{"closed": True, "epoch": 9}],
        "experiment_claim": "research",
        "history": [{"epoch": 9}],
        "logical_step": 9,
        "optimizer_updates": {"optim_g": 9, "optim_d": 8},
        "phase": "in_epoch",
        "scheduler_steps": 5,
        "step_log": [{"logical_step": 9}],
        "stop_reason": "resumed",
    }
    payload.update(overrides)
    return payload


class MPSENetRestoreTests(unittest.TestCase):
    def test_failed_restore_keeps_live_containers(self) -> None:
        cases: tuple[tuple[str, dict[str, object], type[BaseException], str], ...] = (
            ("logical_bool", _payload(logical_step=True), RuntimeFailure, "trainer state is invalid"),
            ("logical_str", _payload(logical_step="4"), RuntimeFailure, "trainer state is invalid"),
            ("phase", _payload(phase="pending_scheduler"), RuntimeFailure, "trainer phase 'pending_scheduler' is invalid"),
            ("claim", _payload(experiment_claim="other"), RuntimeFailure, "experiment claim does not match this trainer"),
            ("history", _payload(history="nope"), RuntimeFailure, "trainer history is invalid"),
            ("completed", _payload(completed_boundaries=None), RuntimeFailure, "trainer history is invalid"),
            ("step_log", _payload(step_log={"logical_step": 1}), RuntimeFailure, "trainer history is invalid"),
            ("updates_type", _payload(optimizer_updates=[("optim_g", 1), ("optim_d", 1)]), RuntimeFailure, "trainer state is invalid"),
            ("best_epoch_bool", _payload(best_epoch=True), RuntimeFailure, "best_epoch is invalid"),
            ("best_step_float", _payload(best_logical_step=1.5), RuntimeFailure, "best_logical_step is invalid"),
            ("update_string", _payload(optimizer_updates={"optim_g": 1, "optim_d": "nope"}), ValueError, "invalid literal for int"),
            ("update_none", _payload(optimizer_updates={"optim_g": 1, "optim_d": None}), TypeError, "not 'NoneType'"),
            ("update_dict", _payload(optimizer_updates={"optim_g": 1, "optim_d": {"n": 1}}), TypeError, "not 'dict'"),
            ("boundary_pairs", _payload(boundary=[1, 2]), TypeError, "dictionary update sequence element"),
            ("scheduler", _payload(scheduler_steps="nope"), ValueError, "invalid literal for int"),
            ("best_pesq", _payload(best_pesq={"bad": 1}), TypeError, "not 'dict'"),
            ("missing_id", _payload(optimizer_updates={"optim_g": 3}), RuntimeFailure, "registered optimizer ids"),
            ("foreign_ids", _payload(optimizer_updates={"other_g": 9, "other_d": 8}), RuntimeFailure, "registered optimizer ids"),
            (
                "one_registered_and_spare",
                _payload(optimizer_updates={"optim_g": 3, "spare": 3}),
                RuntimeFailure,
                "registered optimizer ids",
            ),
        )
        for name, payload, error, message in cases:
            with self.subTest(name=name):
                progress = _progress()
                before = progress.export(experiment_claim="research")
                updates = progress.optimizer_updates
                history = progress.history
                history_row = history[0]
                inner = history_row["inner"]
                completed = progress.completed_boundaries
                step_log = progress.step_log
                boundary = progress.boundary
                nested = boundary["nested"]
                with self.assertRaisesRegex(error, message):
                    progress.restore(payload, experiment_claim="research")
                self.assertEqual(progress.export(experiment_claim="research"), before)
                self.assertIs(progress.optimizer_updates, updates)
                self.assertIs(progress.history, history)
                self.assertIs(progress.history[0], history_row)
                self.assertIs(inner, history_row["inner"])
                self.assertEqual(inner["k"], 2)
                self.assertIs(progress.completed_boundaries, completed)
                self.assertIs(progress.step_log, step_log)
                self.assertIs(progress.boundary, boundary)
                self.assertIs(progress.boundary["nested"], nested)
                self.assertEqual(nested["n"], 1)
                self.assertEqual(progress.stop_reason, "kept")

    def test_conversion_error_wins_when_an_earlier_field_would_also_fail(self) -> None:
        """``int`` on optimizer values runs before ``dict(boundary)``, as before."""

        progress = _progress()
        before = progress.export(experiment_claim="research")
        payload = _payload(optimizer_updates={"optim_g": "nope", "optim_d": 1}, boundary=[1, 2])
        with self.assertRaisesRegex(ValueError, "invalid literal for int"):
            progress.restore(payload, experiment_claim="research")
        self.assertEqual(progress.export(experiment_claim="research"), before)
        self.assertEqual(progress.logical_step, 4)

    def test_optimizer_ids_keep_their_names_and_extra_keys_are_not_counted(self) -> None:
        progress = _progress()
        payload = _payload(optimizer_updates={"optim_d": 2, "extra": 7, "optim_g": 4})
        progress.restore(payload, experiment_claim="research")
        self.assertEqual(list(progress.optimizer_updates), ["optim_d", "extra", "optim_g"])
        self.assertEqual(progress.optimizer_updates["optim_g"], 4)
        self.assertEqual(progress.optimizer_updates["optim_d"], 2)
        progress.complete_update()
        self.assertEqual(progress.logical_step, 10)
        self.assertEqual(progress.optimizer_updates, {"optim_d": 3, "extra": 7, "optim_g": 5})

    def test_old_int_conversion_still_accepts_numeric_strings_floats_and_bools(self) -> None:
        progress = _progress()
        progress.restore(
            _payload(optimizer_updates={"optim_g": "8", "optim_d": True, "spare": 1.9}),
            experiment_claim="research",
        )
        self.assertEqual(progress.optimizer_updates, {"optim_g": 8, "optim_d": 1, "spare": 1})
        progress.complete_update()
        self.assertEqual(progress.optimizer_updates, {"optim_g": 9, "optim_d": 2, "spare": 1})

    def test_negative_counter_is_kept_and_missing_scheduler_steps_default_to_zero(self) -> None:
        progress = _progress()
        payload = _payload(optimizer_updates={"optim_g": -1, "optim_d": 0})
        del payload["scheduler_steps"]
        del payload["stop_reason"]
        progress.restore(payload, experiment_claim="research")
        self.assertEqual(progress.optimizer_updates, {"optim_g": -1, "optim_d": 0})
        self.assertEqual(progress.scheduler_steps, 0)
        self.assertEqual(progress.stop_reason, "kept")
        progress.complete_update()
        self.assertEqual(progress.optimizer_updates, {"optim_g": 0, "optim_d": 1})


if __name__ == "__main__":
    unittest.main()
