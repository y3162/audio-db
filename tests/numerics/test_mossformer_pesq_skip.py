"""PESQ and non-finite MossFormerGAN updates are skipped on every rank."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

def _prefer_repo_tests() -> None:
    repo = Path(__file__).resolve().parents[2]
    kept = []
    for entry in list(sys.path):
        base = Path(entry) if entry else Path.cwd()
        foreign = base / "tests" / "__init__.py"
        try:
            drop = foreign.is_file() and base.resolve() != repo
        except OSError:
            drop = False
        if not drop:
            kept.append(entry)
    sys.path[:] = kept
    root = str(repo)
    if root not in sys.path:
        sys.path.insert(0, root)


_prefer_repo_tests()

import torch
import torch.distributed as dist

from src.experiments.calc.distributed import ProcessInfo
from src.experiments.official.mossformergan.reference import MossFormerGANTask
from tests.numerics.support import free_tcp_port, paired_dataset, pin_repo_root_for_spawn
from tests.numerics.test_mossformer_syncbn_collective import install_cpu_gloo_syncbn


def _config(root: Path) -> dict:
    return {
        "seed": 7,
        "loss": "official",
        "grad_clip": 10.0,
        "optimizer": {"name": "adamw", "lr": 5e-4, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "mossformergan",
            "variant": "se_16k",
            "sampling_rate": 16000,
            "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
        },
        "checkpoint_dir": str(root),
    }


def _tone(frames: int, frequency: float) -> torch.Tensor:
    time = torch.arange(frames, dtype=torch.float32) / 16000
    return (0.2 * torch.sin(2 * math.pi * frequency * time)).contiguous()


def _batch() -> dict:
    return {
        "clean": _tone(8000, 180).unsqueeze(0),
        "noisy": _tone(8000, 90).unsqueeze(0),
        "case_id": ["toy-case"],
    }


def _clone_weights(task: MossFormerGANTask) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in task.generator.parameters()]


def _same_weights(before: list[torch.Tensor], task: MossFormerGANTask) -> bool:
    after = [parameter.detach() for parameter in task.generator.parameters()]
    return all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def _pesq_worker(rank: int, world: int, port: int, result_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log_path = Path(result_dir) / f"rank{rank}.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            from datetime import timedelta

            dist.init_process_group(backend="gloo", rank=rank, world_size=world, timeout=timedelta(seconds=300))
            task = MossFormerGANTask(_config(Path(result_dir)), torch.device("cpu"))
            task.distribute(ProcessInfo(rank, world, rank, torch.device("cpu"), True))
            # torch 2.6 SyncBatchNorm.forward rejects CPU before its collective.
            install_cpu_gloo_syncbn()
            payload: dict = {
                "sync_discriminator": sum(
                    isinstance(module, torch.nn.SyncBatchNorm) for module in task.discriminator.modules()
                ),
                "validation_requires_all_ranks": task.validation_requires_all_ranks,
            }
            scenarios = (
                ("one_rank", lambda teacher, audio, rate: None if dist.get_rank() == 1 else torch.ones(teacher.shape[0])),
                ("both_ranks", lambda teacher, audio, rate: None),
            )
            for name, metric in scenarios:
                disc_module = (
                    task.discriminator.module
                    if isinstance(task.discriminator, torch.nn.parallel.DistributedDataParallel)
                    else task.discriminator
                )
                before = _clone_weights(task)
                disc_before = disc_module.sigmoid.slope.detach().clone()
                optim_before = copy.deepcopy(task.optim_g.state_dict())
                with patch("src.experiments.official.mossformergan.reference.metric_targets", metric):
                    metrics = task.train_update([_batch()], 1)
                payload[name] = {
                    "update_applied": metrics["update_applied"],
                    "generator": metrics["generator"],
                    "discriminator": metrics["discriminator"],
                    "reason": None if task.last_skip is None else task.last_skip["reason"],
                    "case_ids": None if task.last_skip is None else task.last_skip["case_ids"],
                    "micro_batch_index": None if task.last_skip is None else task.last_skip["micro_batch_index"],
                    "count": None if task.last_skip is None else task.last_skip["count"],
                    "weights_unchanged": _same_weights(before, task)
                    and torch.equal(disc_before, disc_module.sigmoid.slope.detach()),
                    "optimizer_state_unchanged": task.optim_g.state_dict() == optim_before,
                    "skip_log": len(task.skip_log),
                }
            (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
            dist.destroy_process_group()
        except Exception:
            log.write(traceback.format_exc())
            raise


class MossFormerPesqSkipTests(unittest.TestCase):
    def test_pesq_none_skips_without_a_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = MossFormerGANTask(_config(root), torch.device("cpu"))
            data = paired_dataset(root / "wav", 1, frames=8000, segment_size=8000, seed=7)
            batch = data.collate([0])
            before = _clone_weights(task)
            optim_g = task.optim_g.state_dict()
            optim_d = task.optim_d.state_dict()
            with patch("src.experiments.official.mossformergan.reference.metric_targets", return_value=None):
                metrics = task.train_update([batch], 1)
            self.assertEqual(metrics["update_applied"], 0.0)
            self.assertEqual(metrics["generator"], 0.0)
            self.assertEqual(metrics["discriminator"], 0.0)
            self.assertTrue(_same_weights(before, task))
            self.assertEqual(task.optim_g.state_dict(), optim_g)
            self.assertEqual(task.optim_d.state_dict(), optim_d)
            self.assertIsNotNone(task.last_skip)
            self.assertEqual(task.last_skip["reason"], "pesq")
            self.assertEqual(task.last_skip["micro_batch_index"], 0)
            self.assertEqual(task.last_skip["count"], 1)
            self.assertTrue(task.last_skip["case_ids"])
            self.assertEqual(task.skip_log[-1]["reason"], "pesq")
            self.assertTrue(task.generator_ddp_broadcast_buffers)
            self.assertTrue(task.generator_ddp_find_unused_parameters)
            self.assertTrue(task.discriminator_ddp_broadcast_buffers)
            self.assertTrue(task.discriminator_ddp_find_unused_parameters)
            self.assertEqual(
                sum(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in task.generator.modules()),
                0,
            )
            self.assertGreater(
                sum(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in task.discriminator.modules()),
                0,
            )

    def test_nonfinite_generator_loss_skips_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            task = MossFormerGANTask(_config(Path(directory)), torch.device("cpu"))
            before = task.discriminator.sigmoid.slope.detach().clone()
            optim = task.optim_d.state_dict()

            def fake_terms(self, module, discriminator, clean, noisy):
                del module, discriminator
                audio = torch.zeros(clean.shape[0], 8)
                magnitude = torch.rand(clean.shape[0], 1, 201, 19)
                return {
                    "generator": torch.tensor(float("nan")),
                    "teacher": audio,
                    "audio": audio,
                    "label_mag": magnitude,
                    "pred_mag": magnitude,
                }

            batch = {"clean": torch.zeros(1, 32), "noisy": torch.zeros(1, 32), "case_id": ["nan-case"]}
            with patch.object(MossFormerGANTask, "_generator_terms", fake_terms):
                metrics = task.train_update([batch], 1)
            self.assertEqual(metrics["update_applied"], 0.0)
            self.assertEqual(metrics["generator"], 0.0)
            self.assertTrue(torch.equal(before, task.discriminator.sigmoid.slope))
            self.assertEqual(task.optim_d.state_dict(), optim)
            self.assertEqual(task.last_skip["reason"], "non-finite generator loss")
            self.assertEqual(task.last_skip["case_ids"], ["nan-case"])

    def test_later_micro_batch_pesq_failure_does_not_step_the_group(self):
        with tempfile.TemporaryDirectory() as directory:
            task = MossFormerGANTask(_config(Path(directory)), torch.device("cpu"))
            before = task.discriminator.sigmoid.slope.detach().clone()
            calls = {"n": 0}

            def fake_terms(self, module, discriminator, clean, noisy):
                del module
                audio = torch.zeros(clean.shape[0], 8)
                magnitude = torch.rand(clean.shape[0], 1, 201, 19)
                return {
                    "generator": discriminator.sigmoid.slope.reshape(-1)[0] * 0,
                    "teacher": audio,
                    "audio": audio,
                    "label_mag": magnitude,
                    "pred_mag": magnitude,
                }

            def metric(teacher, audio, rate):
                del audio, rate
                calls["n"] += 1
                if calls["n"] == 1:
                    return torch.ones(teacher.shape[0])
                return None

            batch = {"clean": torch.zeros(1, 32), "noisy": torch.zeros(1, 32), "case_id": ["group"]}
            with patch.object(MossFormerGANTask, "_generator_terms", fake_terms):
                with patch("src.experiments.official.mossformergan.reference.metric_targets", metric):
                    metrics = task.train_update([batch, batch], 2)
            self.assertEqual(metrics["update_applied"], 0.0)
            self.assertEqual(calls["n"], 2)
            self.assertTrue(torch.equal(before, task.discriminator.sigmoid.slope))
            self.assertEqual(task.optim_g.state_dict()["state"], {})
            self.assertEqual(task.last_skip["reason"], "pesq")
            self.assertEqual(task.last_skip["micro_batch_index"], 1)
            self.assertEqual(task.last_skip["count"], 1)
            self.assertEqual(task.counters["micro_batches_processed"], 2)
            self.assertEqual(task.counters["micro_batches_succeeded"], 1)
            self.assertEqual(task.counters["micro_batches_failed"], 1)
            self.assertEqual(task.counters["optimizer_updates"], 0)
            self.assertEqual(task.counters["official_step"], 2)
            self.assertEqual(task._accu_count, 1)

    def test_later_success_completes_the_window_after_a_pesq_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            task = MossFormerGANTask(_config(Path(directory)), torch.device("cpu"))
            calls = {"n": 0}

            def fake_terms(self, module, discriminator, clean, noisy):
                del module
                audio = torch.zeros(clean.shape[0], 8)
                magnitude = torch.rand(clean.shape[0], 1, 201, 19)
                return {
                    "generator": discriminator.sigmoid.slope.reshape(-1)[:1].sum(),
                    "teacher": audio,
                    "audio": audio,
                    "label_mag": magnitude,
                    "pred_mag": magnitude,
                }

            def metric(teacher, audio, rate):
                del audio, rate
                calls["n"] += 1
                if calls["n"] == 1:
                    return None
                return torch.ones(teacher.shape[0])

            batch = {"clean": torch.zeros(1, 32), "noisy": torch.zeros(1, 32), "case_id": ["window"]}
            before = task.discriminator.sigmoid.slope.detach().clone()
            with patch.object(MossFormerGANTask, "_generator_terms", fake_terms):
                with patch("src.experiments.official.mossformergan.reference.metric_targets", metric):
                    first = task.train_update([batch, batch], 2)
                    second = task.train_update([batch], 2)
            self.assertEqual(first["update_applied"], 0.0)
            self.assertEqual(second["update_applied"], 1.0)
            self.assertEqual(task.counters["official_step"], 3)
            self.assertEqual(task.counters["optimizer_updates"], 1)
            self.assertEqual(task.counters["micro_batches_processed"], 3)
            self.assertEqual(task.counters["micro_batches_succeeded"], 2)
            self.assertEqual(task.counters["micro_batches_failed"], 1)
            self.assertEqual(task._accu_count, 0)
            self.assertFalse(torch.equal(before, task.discriminator.sigmoid.slope.detach()))
            self.assertNotEqual(task.optim_d.state_dict()["state"], {})

    def test_failure_sequences_keep_the_official_window(self):
        with tempfile.TemporaryDirectory() as directory:
            task = MossFormerGANTask(_config(Path(directory)), torch.device("cpu"))
            initial = copy.deepcopy(task.state_dict())
            slope = task.discriminator.sigmoid.slope.detach().clone()
            sequences = {
                "A": [False, False],
                "B": [True, False, False],
                "C": [False, True, False],
                "D": [False, False, True],
                "E": [True, True],
            }
            expected = {
                "A": (2, 2, 0, 1, 2, 0, 1.0),
                "B": (3, 2, 1, 1, 3, 0, 1.0),
                "C": (3, 2, 1, 1, 3, 0, 1.0),
                "D": (3, 2, 1, 1, 3, 0, 1.0),
                "E": (2, 0, 2, 0, 2, 0, 0.0),
            }

            def fake_terms(self, module, discriminator, clean, noisy):
                del module
                audio = torch.zeros(clean.shape[0], 8)
                magnitude = torch.zeros(clean.shape[0], 1, 201, 19)
                return {
                    "generator": discriminator.sigmoid.slope.reshape(-1)[:1].sum(),
                    "teacher": audio,
                    "audio": audio,
                    "label_mag": magnitude,
                    "pred_mag": magnitude,
                }

            batch = {"clean": torch.zeros(1, 32), "noisy": torch.zeros(1, 32), "case_id": ["schedule"]}
            for name, pattern in sequences.items():
                task.load_state_dict(copy.deepcopy(initial))
                task.optim_g.zero_grad(set_to_none=True)
                task.optim_d.zero_grad(set_to_none=True)
                calls = {"n": 0}

                def metric(teacher, audio, rate, pattern=pattern, calls=calls):
                    del audio, rate
                    failed = pattern[calls["n"]]
                    calls["n"] += 1
                    if failed:
                        return None
                    return torch.ones(teacher.shape[0])

                with patch.object(MossFormerGANTask, "_generator_terms", fake_terms):
                    with patch("src.experiments.official.mossformergan.reference.metric_targets", metric):
                        metrics = task.train_update([batch] * len(pattern), 2)
                processed, succeeded, failed, optim, official, accu, applied = expected[name]
                self.assertEqual(calls["n"], len(pattern), name)
                self.assertEqual(metrics["update_applied"], applied, name)
                self.assertEqual(task.counters["micro_batches_processed"], processed, name)
                self.assertEqual(task.counters["micro_batches_succeeded"], succeeded, name)
                self.assertEqual(task.counters["micro_batches_failed"], failed, name)
                self.assertEqual(task.counters["optimizer_updates"], optim, name)
                self.assertEqual(task.counters["official_step"], official, name)
                self.assertEqual(task._accu_count, accu, name)
                changed = not torch.equal(slope, task.discriminator.sigmoid.slope.detach())
                self.assertEqual(changed, applied == 1.0, name)
                if applied == 0.0:
                    self.assertIsNone(task.discriminator.sigmoid.slope.grad)

    def test_epoch_start_drops_a_partial_window(self):
        with tempfile.TemporaryDirectory() as directory:
            task = MossFormerGANTask(_config(Path(directory)), torch.device("cpu"))

            def fake_terms(self, module, discriminator, clean, noisy):
                del module
                audio = torch.zeros(clean.shape[0], 8)
                magnitude = torch.zeros(clean.shape[0], 1, 201, 19)
                return {
                    "generator": discriminator.sigmoid.slope.reshape(-1)[:1].sum(),
                    "teacher": audio,
                    "audio": audio,
                    "label_mag": magnitude,
                    "pred_mag": magnitude,
                }

            batch = {"clean": torch.zeros(1, 32), "noisy": torch.zeros(1, 32), "case_id": ["partial"]}
            with patch.object(MossFormerGANTask, "_generator_terms", fake_terms):
                with patch("src.experiments.official.mossformergan.reference.metric_targets", return_value=torch.ones(1)):
                    metrics = task.train_update([batch], 2)
            self.assertEqual(metrics["update_applied"], 0.0)
            self.assertEqual(task._accu_count, 1)
            self.assertIsNotNone(task.discriminator.sigmoid.slope.grad)
            kept = task.discriminator.sigmoid.slope.grad.detach().clone()
            task.start_epoch(resume_mid_epoch=True)
            self.assertEqual(task._accu_count, 1)
            self.assertTrue(torch.equal(kept, task.discriminator.sigmoid.slope.grad))
            task.start_epoch(resume_mid_epoch=False)
            self.assertEqual(task._accu_count, 0)
            self.assertIsNone(task.discriminator.sigmoid.slope.grad)
            self.assertIsNone(next(task.generator.parameters()).grad)

    def test_ddp_rank_disagreement_and_both_ranks_skip(self):
        import multiprocessing as mp

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        pin_repo_root_for_spawn()
        with tempfile.TemporaryDirectory() as directory:
            port = free_tcp_port()
            ctx = mp.get_context("spawn")
            processes = [
                ctx.Process(target=_pesq_worker, args=(rank, 2, port, directory))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(400)
            alive = [process for process in processes if process.is_alive()]
            for process in alive:
                process.kill()
            for process in processes:
                process.join()
            logs = []
            for rank in range(2):
                path = Path(directory) / f"rank{rank}.log"
                logs.append(path.read_text(encoding="utf-8") if path.is_file() else "")
            self.assertFalse(alive, "DDP PESQ workers hung\n" + "\n".join(logs))
            codes = [process.exitcode for process in processes]
            self.assertEqual(codes, [0, 0], "\n".join(logs))
            ranks = [json.loads((Path(directory) / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
            for payload in ranks:
                self.assertEqual(payload["sync_discriminator"], 0)
                self.assertFalse(payload["validation_requires_all_ranks"])
            one = [payload["one_rank"] for payload in ranks]
            self.assertEqual([item["update_applied"] for item in one], [0.0, 0.0])
            self.assertEqual(sorted(item["reason"] for item in one), ["peer", "pesq"])
            self.assertTrue(all(item["weights_unchanged"] and item["optimizer_state_unchanged"] for item in one))
            self.assertTrue(all(item["generator"] == 0.0 and item["discriminator"] == 0.0 for item in one))
            self.assertEqual([item["case_ids"] for item in one if item["reason"] == "pesq"], [["toy-case"]])
            both = [payload["both_ranks"] for payload in ranks]
            self.assertEqual([item["reason"] for item in both], ["pesq", "pesq"])
            self.assertEqual([item["update_applied"] for item in both], [0.0, 0.0])
            self.assertTrue(all(item["weights_unchanged"] and item["optimizer_state_unchanged"] for item in both))
            self.assertEqual([item["count"] for item in both], [1, 1])


if __name__ == "__main__":
    unittest.main()
