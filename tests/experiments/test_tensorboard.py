"""TensorBoard events are a mirror of metrics.jsonl."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.experiments.execution.journal import StepJournal
from src.experiments.execution.tensorboard import rebuild_tensorboard, scalar_series


class _Runtime:
    def __init__(self, rank: int, world_size: int = 1) -> None:
        self.rank = rank
        self.world_size = world_size


def _scalars(directory: Path, tag: str) -> list[tuple[int, float]]:
    accumulator = EventAccumulator(str(directory / "tensorboard"), size_guidance={"scalars": 1000})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [(event.step, event.value) for event in accumulator.Scalars(tag)]


class TensorBoardTests(unittest.TestCase):
    def test_events_follow_the_jsonl_and_a_resume_rewrites_them(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        journal = StepJournal(directory, _Runtime(0))
        journal.begin(restored=False, logical_step=0)
        journal.write_metrics(
            {
                "applied": True,
                "logical_step": 1,
                "loss": 0.5,
                "sample_id": "a",
                "terms": {"ri": 0.25, "gan": float("nan")},
            }
        )
        journal.write_metrics(
            {
                "applied": False,
                "logical_step": 2,
                "loss": 0.25,
                "optimizer_updates": {"optim_g": 2},
                "sample_id": "b",
            }
        )
        journal.close()
        self.assertEqual(_scalars(directory, "loss"), [(1, 0.5), (2, 0.25)])
        self.assertEqual(_scalars(directory, "terms/ri"), [(1, 0.25)])
        self.assertEqual(_scalars(directory, "terms/gan"), [])
        self.assertEqual(_scalars(directory, "optimizer_updates/optim_g"), [(2, 2.0)])
        self.assertEqual(_scalars(directory, "sample_id"), [])
        self.assertEqual(_scalars(directory, "applied"), [])

        resumed = StepJournal(directory, _Runtime(0))
        losses = resumed.begin(restored=True, logical_step=1)
        resumed.close()
        self.assertEqual(losses, [0.5])
        self.assertEqual(_scalars(directory, "loss"), [(1, 0.5)])
        self.assertEqual(_scalars(directory, "optimizer_updates/optim_g"), [])

    def test_another_rank_does_not_write_events(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        journal = StepJournal(directory, _Runtime(1, world_size=2))
        journal.begin(restored=False, logical_step=0)
        journal.write_metrics({"logical_step": 1, "loss": 1.0})
        self.assertFalse((directory / "metrics.jsonl").exists())
        self.assertFalse((directory / "tensorboard").exists())

    def test_a_rebuild_matches_the_jsonl_without_keeping_a_writer(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (directory / "metrics.jsonl").write_text(
            '{"logical_step": 3, "loss": 1.5}\n{"logical_step": 4, "loss": 1.25}\n',
            encoding="utf-8",
        )
        (directory / "tensorboard").mkdir()
        (directory / "tensorboard" / "stale").write_text("old", encoding="utf-8")
        rebuild_tensorboard(directory)
        self.assertEqual(_scalars(directory, "loss"), [(3, 1.5), (4, 1.25)])
        self.assertFalse((directory / "tensorboard" / "stale").exists())
        self.assertEqual(scalar_series({"logical_step": 1, "loss": 2, "sample_id": "x"})[1], [("loss", 2.0)])
