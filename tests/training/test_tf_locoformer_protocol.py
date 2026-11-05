"""Xavier init, warmup plateau, multi-step match, and checkpoint resume."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.se import build_model
from src.training.core.batch import plan_batches
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from src.training.se.config import load_training_config
from src.training.se.initialization import apply_espnet_initialization
from src.training.se.optim import WarmupReduceLROnPlateau
from src.training.se.tf_locoformer import TFLocoformerTask
from src.training.core.rng import seed_model_init, set_determinism
from tests.se.test_tf_locoformer import OFFICIAL, TINY, _frontend, _official_separator
from tests.training.support import cpu_process, paired_dataset, tiny_config

ROOT = Path(__file__).resolve().parents[2]


def tearDownModule():
    set_determinism(False, cuda=False)


def _task_config(root: Path, **overrides) -> dict:
    architecture = dict(TINY)
    config = tiny_config(
        root,
        loss="official",
        segment_size=512,
        global_batch_size=1,
        micro_batch_size_per_gpu=1,
        max_steps=1,
    )
    config["optimizer"] = {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01}
    config["grad_clip"] = 5.0
    config["scheduler"] = {"name": "none"}
    config["model"] = {
        "name": "tf_locoformer",
        "variant": "dns",
        "stage": None,
        "sampling_rate": 16000,
        "architecture": architecture,
        "stft": {},
        "components": {},
    }
    config.update(overrides)
    return config


class TFLocoformerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_research_yaml_is_separate_from_the_smoke_file(self):
        smoke = load_training_config(ROOT / "configs/training/tf_locoformer.yaml")
        research = load_training_config(ROOT / "configs/training/tf_locoformer_voicebank.yaml")
        self.assertEqual(smoke["max_steps"], 2)
        self.assertEqual(smoke["data"]["train_set"], "libri_demand_smoke_train")
        self.assertTrue(smoke["data"]["rms_normalize"])
        self.assertNotIn("init", smoke["model"])
        self.assertEqual(smoke["scheduler"]["name"], "none")
        self.assertNotIn("chunk_iterator", smoke)
        self.assertEqual(research["data"]["train_set"], "voicebank_28_train")
        self.assertEqual(research["data"]["validation_set"], "voicebank_28_valid")
        self.assertTrue(research["data"]["rms_normalize"])
        self.assertEqual(research["model"]["init"], "xavier_uniform")
        self.assertEqual(research["segment_size"], 32000)
        self.assertEqual(research["global_batch_size"], 4)
        self.assertEqual(research["micro_batch_size_per_gpu"], 4)
        self.assertEqual(research["chunk_iterator"]["chunk_length"], 32000)
        self.assertEqual(research["chunk_iterator"]["num_samples_per_epoch"], 10700)
        self.assertEqual(research["chunk_iterator"]["max_epochs"], 150)
        self.assertEqual(research["chunk_iterator"]["early_stopping_patience"], 10)
        self.assertEqual(research["scheduler"]["name"], "warmup_reduce_on_plateau")
        self.assertEqual(research["scheduler"]["warmup_steps"], 4000)
        self.assertEqual(research["scheduler"]["factor"], 0.5)
        self.assertEqual(research["scheduler"]["patience"], 3)
        self.assertNotEqual(smoke["checkpoint_dir"], research["checkpoint_dir"])

    def test_xavier_matches_the_same_draw_on_the_official_separator(self):
        if not OFFICIAL.is_dir():
            self.skipTest("official checkout missing")
        config = _task_config(Path("unused"), seed=19)
        config["model"]["init"] = "xavier_uniform"
        with seed_model_init(config["seed"]):
            official = _official_separator()
            apply_espnet_initialization(official, "xavier_uniform")
        with seed_model_init(config["seed"]):
            task = TFLocoformerTask(config, torch.device("cpu"))
        for name, left in official.named_parameters():
            right = dict(task.generator.separator.named_parameters())[name]
            self.assertEqual(float((left.detach() - right.detach()).abs().max()), 0.0, name)

    def test_warmup_learning_rate_and_plateau(self):
        weight = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([weight], lr=1e-3)
        schedule = WarmupReduceLROnPlateau(
            optimizer, warmup_steps=4, factor=0.5, patience=3, mode="min"
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        for step in range(1, 5):
            optimizer.step()
            schedule.step_batch()
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3 * step / 4)
        optimizer.step()
        schedule.step_batch()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)
        schedule.step_validation(1.0)
        for _ in range(3):
            schedule.step_validation(1.0)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)
        schedule.step_validation(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)
        restored = WarmupReduceLROnPlateau(optimizer, warmup_steps=4, factor=0.5, patience=3)
        restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.step_num, schedule.step_num)
        self.assertAlmostEqual(restored.optimizer.param_groups[0]["lr"], 5e-4)

    @unittest.skipUnless(OFFICIAL.is_dir(), "official checkout missing")
    def test_five_steps_match_standalone_separator(self):
        config = _task_config(Path("unused"), seed=23)
        config["scheduler"] = {
            "name": "warmup_reduce_on_plateau",
            "warmup_steps": 3,
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 0.0,
            "eps": 1e-8,
        }
        config["model"]["init"] = "xavier_uniform"
        with seed_model_init(config["seed"]):
            official = _official_separator()
            apply_espnet_initialization(official, "xavier_uniform")
        with seed_model_init(config["seed"]):
            task = TFLocoformerTask(config, torch.device("cpu"))
        official_opt = torch.optim.AdamW(official.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        official_sched = WarmupReduceLROnPlateau(
            official_opt, warmup_steps=3, factor=0.5, patience=3, mode="min"
        )
        with tempfile.TemporaryDirectory() as directory:
            data = paired_dataset(Path(directory) / "wav", 5, frames=512, segment_size=512, seed=config["seed"])
            official.train()
            for index in range(5):
                batch = data.collate([index])
                clean = batch["clean"]
                noisy = batch["noisy"]
                official_opt.zero_grad(set_to_none=True)
                wave = _frontend(official, noisy, 64, 32)
                from src.training.se.spectral import multi_resolution_l1

                official_loss = multi_resolution_l1(
                    clean, wave, time_domain_weight=0.5, normalize_variance=False
                )
                official_loss.backward()
                torch.nn.utils.clip_grad_norm_(official.parameters(), 5.0)
                metrics = task.train_update([batch], 1)
                official_opt.step()
                official_sched.step_batch()
                self.assertEqual(float(abs(metrics["mr_l1_tfd"] - float(official_loss.detach()))), 0.0)
                self.assertAlmostEqual(
                    task.optim_g.param_groups[0]["lr"], official_opt.param_groups[0]["lr"]
                )
                for name, left in official.named_parameters():
                    right = dict(task.generator.separator.named_parameters())[name]
                    self.assertEqual(float((left.detach() - right.detach()).abs().max()), 0.0, name)

    def test_zero_variance_batch_does_not_step(self):
        config = _task_config(Path("unused"), seed=29)
        with seed_model_init(config["seed"]):
            task = TFLocoformerTask(config, torch.device("cpu"))
        before = {name: value.detach().clone() for name, value in task.generator.named_parameters()}
        batch = {"clean": torch.zeros(1, 512), "noisy": torch.zeros(1, 512)}
        with self.assertRaises(TrainingError):
            task.train_update([batch], 1)
        for name, value in task.generator.named_parameters():
            self.assertEqual(float((value.detach() - before[name]).abs().max()), 0.0)
        self.assertEqual(task.optim_g.state_dict()["state"], {})

    def test_resume_matches_continuous_twenty_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "wav"
            data = paired_dataset(data_root, 4, frames=512, segment_size=512, seed=31, rewrite=True)

            def run(checkpoint: Path, steps: int, resume: Path | None):
                config = _task_config(checkpoint, seed=31, max_steps=steps, checkpoint_interval=0)
                config["scheduler"] = {
                    "name": "warmup_reduce_on_plateau",
                    "warmup_steps": 4,
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 3,
                    "threshold": 1e-4,
                    "threshold_mode": "rel",
                    "cooldown": 0,
                    "min_lr": 0.0,
                    "eps": 1e-8,
                }
                config["model"]["init"] = "xavier_uniform"
                with seed_model_init(config["seed"]):
                    task = TFLocoformerTask(config, torch.device("cpu"))
                Trainer(
                    task,
                    plan_batches(1, 1, 1),
                    cpu_process(),
                    max_steps=steps,
                    seed=config["seed"],
                    checkpoint_dir=checkpoint,
                    checkpoint_interval=0,
                    validation_interval=0,
                    config=config,
                ).fit(data, resume=resume)
                return task

            continuous = run(root / "continuous", 20, None)
            run(root / "part", 10, None)
            resumed = run(root / "resumed", 20, root / "part" / "step_00000010.pt")
            for name, left in continuous.generator.state_dict().items():
                right = resumed.generator.state_dict()[name]
                self.assertEqual(float((left - right).abs().max()), 0.0, name)
            self.assertEqual(continuous.optim_g.state_dict()["state"].keys(), resumed.optim_g.state_dict()["state"].keys())
            for key, left in continuous.optim_g.state_dict()["state"].items():
                right = resumed.optim_g.state_dict()["state"][key]
                for field, value in left.items():
                    other = right[field]
                    if torch.is_tensor(value):
                        self.assertEqual(float((value - other).abs().max()), 0.0, field)
                    else:
                        self.assertEqual(value, other)
            self.assertEqual(continuous.sched_g.state_dict()["step_num"], resumed.sched_g.state_dict()["step_num"])
            self.assertAlmostEqual(
                continuous.optim_g.param_groups[0]["lr"], resumed.optim_g.param_groups[0]["lr"]
            )
            probe = torch.randn(1, 512)
            with torch.no_grad():
                left = continuous.generator(probe)
                right = resumed.generator(probe)
            self.assertEqual(float((left - right).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
