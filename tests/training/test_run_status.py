"""Official completion stays closed. A verification cap stays resumable."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.core.batch import plan_batches
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from src.training.se.config import load_training_config
from tests.training.mock_task import LinearPairTask
from tests.training.support import cpu_process, paired_dataset, tiny_config


class _CountingTask(LinearPairTask):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.validations = 0
        self.scheduler_steps = 0

    def validation_report(self, batch):
        del batch
        self.validations += 1
        return {
            "losses": {"generator": {"status": "ok", "value": 1.0, "reason": None}},
            "pesq": {"status": "na", "value": None, "reason": "unused", "count": 0},
            "stoi": {"status": "na", "value": None, "reason": "unused", "count": 0},
        }

    def step_schedulers(self, validation=None):
        self.scheduler_steps += 1
        super().step_schedulers(validation)


class RunStatusTests(unittest.TestCase):
    def _fit(self, checkpoint: Path, dataset, *, max_steps: int, verification: int | None, resume: Path | None,
             validation=None, validation_interval: int = 0, checkpoint_interval: int = 0, task_cls=LinearPairTask):
        config = tiny_config(
            checkpoint, seed=7, loss="mock", max_steps=max_steps,
            global_batch_size=4, micro_batch_size_per_gpu=2,
        )
        if verification is not None:
            config["verification_max_updates"] = verification
        task = task_cls(config, torch.device("cpu"))
        trainer = Trainer(
            task, plan_batches(4, 2, 1), cpu_process(), max_steps=max_steps, seed=7,
            checkpoint_dir=checkpoint, checkpoint_interval=checkpoint_interval,
            validation_interval=validation_interval, config=config,
        )
        trainer.fit(dataset, validation, resume=resume)
        return trainer, task

    def test_verification_cap_resumes_in_the_same_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ckpt"
            dataset = paired_dataset(Path(tmp) / "wav", 8, frames=320, segment_size=80, seed=7)
            first, first_task = self._fit(root, dataset, max_steps=4, verification=2, resume=None)
            self.assertEqual(first.stop_reason, "INTERRUPTED")
            self.assertEqual(first.global_step, 2)
            self.assertFalse((root / "complete.json").is_file())
            marker = json.loads((root / "interrupted.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "INTERRUPTED")
            self.assertEqual(json.loads((root / "status.json").read_text(encoding="utf-8"))["status"], "INTERRUPTED")
            second, second_task = self._fit(root, dataset, max_steps=4, verification=None, resume=Path(first.last_checkpoint))
            self.assertEqual(second.stop_reason, "COMPLETED")
            self.assertEqual(second.global_step, 4)
            self.assertEqual(json.loads((root / "complete.json").read_text(encoding="utf-8"))["status"], "COMPLETED")
            self.assertFalse((root / "interrupted.json").exists())
            continuous, continuous_task = self._fit(
                Path(tmp) / "continuous", dataset, max_steps=4, verification=None, resume=None,
            )
            self.assertEqual(continuous.stop_reason, "COMPLETED")
            self.assertEqual(first.global_step, 2)
            left = continuous_task.state_dict()
            right = second_task.state_dict()
            for name in ("generator", "discriminator", "optim_g", "optim_d", "sched_g", "sched_d"):
                self.assertEqual(left[name].keys(), right[name].keys())
            for key, value in continuous_task.generator_state_dict().items():
                self.assertTrue(torch.equal(value, second_task.generator_state_dict()[key]), key)

    def test_same_verification_cap_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ckpt"
            dataset = paired_dataset(Path(tmp) / "wav", 8, frames=320, segment_size=80, seed=7)
            first, _task = self._fit(root, dataset, max_steps=4, verification=2, resume=None)
            with self.assertRaisesRegex(TrainingError, "verification_max_updates"):
                self._fit(root, dataset, max_steps=4, verification=2, resume=Path(first.last_checkpoint))

    def test_official_max_steps_stays_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ckpt"
            dataset = paired_dataset(Path(tmp) / "wav", 8, frames=320, segment_size=80, seed=7)
            trainer, _task = self._fit(root, dataset, max_steps=2, verification=None, resume=None)
            self.assertEqual(trainer.stop_reason, "COMPLETED")
            with self.assertRaisesRegex(TrainingError, "already complete"):
                self._fit(root, dataset, max_steps=4, verification=None, resume=Path(trainer.last_checkpoint))

    def test_loader_rejects_a_cap_that_meets_the_official_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.yaml"
            path.write_text(
                "seed: 1\ndeterministic: false\nglobal_batch_size: 1\n"
                "micro_batch_size_per_gpu: 1\nmax_steps: 4\nverification_max_updates: 4\n"
                "segment_size: 160\nloss: reconstruction\ncheckpoint_dir: /tmp/audio-db-cap\n"
                "checkpoint_interval: 1\nvalidation_interval: 0\n"
                "optimizer: {name: adamw, lr: 0.001, betas: [0.9, 0.999], weight_decay: 0.0}\n"
                "scheduler: {name: exponential, gamma: 0.99}\n"
                "model: {name: mp_senet, dense_channel: 4, num_tsconformers: 1, n_fft: 32,\n"
                "  hop_size: 8, win_size: 32, compress_factor: 0.3, beta: 2.0, sampling_rate: 16000}\n"
                "data: {database: data/db/voicebank.duckdb, train_set: voicebank_28_train,\n"
                "  validation_set: voicebank_28_valid}\n"
                "runtime: {device: cpu, allow_busy_gpu: false}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TrainingError, "verification_max_updates"):
                load_training_config(path)

    def test_injected_failure_is_recorded_and_does_not_close_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ckpt"
            dataset = paired_dataset(Path(tmp) / "wav", 8, frames=320, segment_size=80, seed=7)
            config = tiny_config(root, seed=7, loss="mock", max_steps=2,
                                 global_batch_size=4, micro_batch_size_per_gpu=2)
            task = LinearPairTask(config, torch.device("cpu"))
            trainer = Trainer(
                task, plan_batches(4, 2, 1), cpu_process(), max_steps=2, seed=7,
                checkpoint_dir=root, checkpoint_interval=1, validation_interval=0, config=config,
            )
            original = task.train_update

            def fail_after(_group, _accumulation):
                original(_group, _accumulation)
                raise RuntimeError("injected failure")

            task.train_update = fail_after
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                trainer.fit(dataset)
            self.assertEqual(trainer.stop_reason, "FAILED")
            self.assertEqual(json.loads((root / "status.json").read_text(encoding="utf-8"))["status"], "FAILED")
            self.assertFalse((root / "complete.json").is_file())

    def test_epoch_scheduler_and_resume_do_not_follow_the_validation_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = paired_dataset(root / "wav", 8, frames=320, segment_size=80, seed=7)
            valid = paired_dataset(root / "valid", 2, frames=320, segment_size=80, seed=8, training=False, name="toy_valid")
            first, first_task = self._fit(
                root / "stop", train, max_steps=6, verification=2, resume=None,
                validation=valid, validation_interval=3, checkpoint_interval=2, task_cls=_CountingTask,
            )
            self.assertEqual(first.stop_reason, "INTERRUPTED")
            self.assertEqual(first_task.scheduler_steps, 1)
            self.assertEqual(first_task.validations, 0)
            self.assertEqual(first_task.sched_g.last_epoch, 1)
            self.assertAlmostEqual(first_task.optim_g.param_groups[0]["lr"], 0.0005)
            resumed, resumed_task = self._fit(
                root / "stop", train, max_steps=6, verification=None,
                resume=Path(first.last_checkpoint), validation=valid, validation_interval=3,
                task_cls=_CountingTask,
            )
            continuous, continuous_task = self._fit(
                root / "continuous", train, max_steps=6, verification=None, resume=None,
                validation=valid, validation_interval=3, task_cls=_CountingTask,
            )
            self.assertEqual(resumed.stop_reason, "COMPLETED")
            self.assertEqual(resumed_task.scheduler_steps, 2)
            self.assertEqual(continuous_task.scheduler_steps, 3)
            self.assertEqual(resumed_task.validations, 4)
            self.assertEqual(continuous_task.validations, 4)
            self.assertEqual(resumed_task.sched_g.last_epoch, continuous_task.sched_g.last_epoch)
            self.assertAlmostEqual(
                resumed_task.optim_g.param_groups[0]["lr"],
                continuous_task.optim_g.param_groups[0]["lr"],
            )
            self.assertAlmostEqual(continuous_task.optim_g.param_groups[0]["lr"], 0.000125)
            for key, value in continuous_task.generator_state_dict().items():
                self.assertTrue(torch.equal(value, resumed_task.generator_state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
