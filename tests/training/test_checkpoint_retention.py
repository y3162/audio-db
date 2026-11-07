"""Checkpoint retention deletes only files this run owns, after a successful save."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.training.core.checkpoint import prune_run_checkpoints, save_atomic


def _payload(step: int) -> dict:
    return {
        "format": "audio-db-training-v1",
        "global_step": step,
        "epoch": 0,
        "micro_batches_consumed": step,
        "task": {},
        "rng": {},
        "dataset": {},
        "config": {},
        "batch_plan": {},
    }


class CheckpointRetentionTests(unittest.TestCase):
    def test_failed_save_keeps_the_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step_00000001.pt"
            save_atomic(path, _payload(1), verify_training=True)
            before = path.read_bytes()
            with patch("src.training.core.checkpoint.torch.save", side_effect=RuntimeError("killed")):
                with self.assertRaises(RuntimeError):
                    save_atomic(path, _payload(2), verify_training=True)
            self.assertEqual(path.read_bytes(), before)
            with patch("src.training.core.checkpoint.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    save_atomic(path, _payload(2), verify_training=True)
            self.assertEqual(path.read_bytes(), before)
            names = [item.name for item in path.parent.iterdir()]
            self.assertEqual(names, [path.name])
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["global_step"], 1)

    def test_keeps_best_nbest_sources_and_other_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            other = root / "other"
            run.mkdir()
            other.mkdir()
            owned: set[str] = set()
            for step in range(1, 7):
                name = f"step_{step:08d}.pt"
                (run / name).write_bytes(b"step")
                owned.add(name)
            for epoch in range(1, 7):
                name = f"{epoch}epoch.pt"
                (run / name).write_bytes(f"epoch-{epoch}".encode())
                owned.add(name)
            (run / "valid.loss.ave_5best.pt").write_bytes(b"average")
            (run / "valid.loss.ave_5best.pt.json").write_text(
                json.dumps({"files": [f"{epoch}epoch.pt" for epoch in range(1, 6)]}),
                encoding="utf-8",
            )
            (run / "valid.loss.ave.pt").symlink_to("valid.loss.ave_5best.pt")
            (other / "step_00000001.pt").write_bytes(b"foreign")
            (other / "1epoch.pt").write_bytes(b"foreign-epoch")
            (other / "notes.txt").write_bytes(b"leave me")
            prune_run_checkpoints(
                run,
                owned,
                keep_latest=3,
                keep_best=True,
                best_names={"step_00000001.pt"},
                nbest_names={f"{epoch}epoch.pt" for epoch in range(1, 6)},
                extra_protected={"valid.loss.ave_5best.pt"},
            )
            remaining = {path.name for path in run.iterdir()}
            for step in (1, 4, 5, 6):
                self.assertIn(f"step_{step:08d}.pt", remaining)
            for step in (2, 3):
                self.assertNotIn(f"step_{step:08d}.pt", remaining)
            for epoch in range(1, 6):
                self.assertIn(f"{epoch}epoch.pt", remaining)
                self.assertEqual((run / f"{epoch}epoch.pt").read_bytes(), f"epoch-{epoch}".encode())
            self.assertNotIn("6epoch.pt", remaining)
            self.assertEqual((run / "valid.loss.ave_5best.pt").read_bytes(), b"average")
            self.assertEqual((run / "valid.loss.ave.pt").readlink().as_posix(), "valid.loss.ave_5best.pt")
            self.assertEqual((other / "step_00000001.pt").read_bytes(), b"foreign")
            self.assertEqual((other / "1epoch.pt").read_bytes(), b"foreign-epoch")
            self.assertEqual((other / "notes.txt").read_bytes(), b"leave me")

    def test_symlink_target_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for step in range(1, 6):
                (run / f"step_{step:08d}.pt").write_bytes(f"s{step}".encode())
            (run / "kept.pt").symlink_to("step_00000001.pt")
            prune_run_checkpoints(
                run,
                {f"step_{step:08d}.pt" for step in range(1, 6)},
                keep_latest=3,
                keep_best=False,
                best_names=set(),
                nbest_names=set(),
            )
            self.assertEqual((run / "step_00000001.pt").read_bytes(), b"s1")
            self.assertFalse((run / "step_00000002.pt").exists())
            self.assertTrue((run / "step_00000005.pt").is_file())


if __name__ == "__main__":
    unittest.main()
