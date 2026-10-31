"""MP-SENet training step, reproducibility, resume, and the inference path."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.se import load_enhancer
from src.training.core.batch import plan_batches
from src.training.core.checkpoint import assert_output_allowed, load_checkpoint
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from src.training.se.config import load_training_config
from src.training.se.losses import generator_losses, metric_targets, phase_losses
from src.training.se.stft import mag_pha_stft
from src.training.se.mp_senet import MPSENetTask, _require_discriminator_resolution
from tests.training.support import cpu_process, paired_dataset, tiny_config

ROOT = Path(__file__).resolve().parents[2]


def tearDownModule():
    set_determinism(False, cuda=False)


class InitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_same_seed_matches_and_a_different_seed_changes_initialization(self):
        first = _generator_state(11)
        second = _generator_state(11)
        changed = _generator_state(12)
        _assert_equal_tensors(self, first, second)
        self.assertFalse(torch.equal(first["dense_encoder.dense_conv_1.0.weight"], changed["dense_encoder.dense_conv_1.0.weight"]))

    def test_discriminator_initialization_follows_the_same_seed(self):
        config = tiny_config(Path("unused"), loss="official", seed=4)
        left = MPSENetTask(config, torch.device("cpu"))
        right = MPSENetTask(config, torch.device("cpu"))
        other = MPSENetTask(tiny_config(Path("unused"), loss="official", seed=5), torch.device("cpu"))
        _assert_equal_tensors(self, left.discriminator.state_dict(), right.discriminator.state_dict())
        self.assertFalse(_states_equal(left.discriminator.state_dict(), other.discriminator.state_dict()))


class ObjectiveTests(unittest.TestCase):
    def test_generator_weights_match_the_official_combination(self):
        torch.manual_seed(0)
        clean_mag = torch.rand(2, 8, 6)
        generated_mag = torch.rand(2, 8, 6)
        clean_phase = torch.rand(2, 8, 6)
        generated_phase = torch.rand(2, 8, 6)
        clean_complex = torch.rand(2, 8, 6, 2)
        generated_complex = torch.rand(2, 8, 6, 2)
        recomputed = torch.rand(2, 8, 6, 2)
        clean_audio = torch.rand(2, 32)
        generated_audio = torch.rand(2, 32)
        metric = torch.rand(2, 1)
        ones = torch.ones(2)
        losses = generator_losses(
            clean_mag,
            generated_mag,
            clean_phase,
            generated_phase,
            clean_complex,
            generated_complex,
            recomputed,
            clean_audio,
            generated_audio,
            metric,
            ones,
        )
        instantaneous, group_delay, angular = phase_losses(clean_phase, generated_phase)
        expected = (
            F.mse_loss(clean_mag, generated_mag) * 0.9
            + (instantaneous + group_delay + angular) * 0.3
            + F.mse_loss(clean_complex, generated_complex) * 2 * 0.1
            + F.mse_loss(generated_complex, recomputed) * 2 * 0.1
            + F.mse_loss(metric.flatten(), ones) * 0.05
            + F.l1_loss(clean_audio, generated_audio) * 0.2
        )
        self.assertTrue(torch.equal(losses["total"], expected))

    def test_centered_stft_matches_reflect_padding(self):
        waveform = torch.randn(2, 512)
        window = torch.hann_window(64)
        reference = torch.stft(
            waveform, 64, 16, 64, window, center=True, pad_mode="reflect", return_complex=True
        )
        real = torch.view_as_real(reference)
        expected = torch.sqrt(real.pow(2).sum(-1) + 1e-9)
        magnitude, _phase, _complex = mag_pha_stft(waveform, 64, 16, 64, 1.0)
        self.assertTrue(torch.equal(magnitude, expected))

    def test_official_spectrogram_fits_the_discriminator_and_a_tiny_one_does_not(self):
        _require_discriminator_resolution(201, 321)
        with self.assertRaises(TrainingError):
            _require_discriminator_resolution(17, 21)

    def test_metric_targets_score_a_waveform_and_reject_silence(self):
        generator = torch.Generator()
        generator.manual_seed(1)
        time = torch.arange(16000, dtype=torch.float32) / 16000
        clean = 0.2 * torch.sin(2 * torch.pi * 200 * time)
        estimate = clean + 0.01 * torch.randn(16000, generator=generator)
        scored = metric_targets(clean.unsqueeze(0), estimate.unsqueeze(0), 16000)
        self.assertIsNotNone(scored)
        self.assertTrue(torch.isfinite(scored).all())
        self.assertIsNone(metric_targets(torch.zeros(1, 16000), torch.zeros(1, 16000), 16000))


class StepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_generator_and_discriminator_update_in_order_with_finite_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = tiny_config(root / "ckpt")
            model = dict(base["model"])
            model.update(n_fft=64, hop_size=16, win_size=64)
            config = tiny_config(
                root / "ckpt",
                loss="official",
                global_batch_size=2,
                micro_batch_size_per_gpu=2,
                segment_size=512,
                model=model,
            )
            dataset = paired_dataset(root / "wav", 2, frames=512, segment_size=512, seed=config["seed"])
            dataset.set_epoch(0)
            batch = dataset.collate([0, 1])
            task = MPSENetTask(config, torch.device("cpu"))
            generator_before = _clone(task.generator.state_dict())
            discriminator_before = _clone(task.discriminator.state_dict())
            order = []
            snapshots = {}
            real_g = task.optim_g.step
            real_d = task.optim_d.step

            def g_step(*args, **kwargs):
                order.append("g")
                gradients = [parameter.grad for parameter in task.generator.parameters() if parameter.requires_grad]
                self.assertTrue(gradients)
                self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
                self.assertTrue(any(torch.count_nonzero(gradient) for gradient in gradients))
                snapshots["d_at_g"] = _clone(task.discriminator.state_dict())
                return real_g(*args, **kwargs)

            def d_step(*args, **kwargs):
                order.append("d")
                snapshots["g_at_d"] = _clone(task.generator.state_dict())
                return real_d(*args, **kwargs)

            task.optim_g.step = g_step
            task.optim_d.step = d_step
            metrics = task.train_update([batch], 1)
            self.assertEqual(order, ["d", "g"])
            self.assertTrue(all(value == value for value in metrics.values()))
            _assert_equal_tensors(self, snapshots["g_at_d"], generator_before)
            self.assertFalse(_states_equal(snapshots["d_at_g"], discriminator_before))
            _assert_equal_tensors(self, task.discriminator.state_dict(), snapshots["d_at_g"])
            self.assertFalse(_states_equal(task.generator.state_dict(), generator_before))

    def test_reconstruction_overfit_lowers_the_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config(
                root / "ckpt",
                max_steps=20,
                global_batch_size=1,
                micro_batch_size_per_gpu=1,
                optimizer={"name": "adamw", "lr": 0.01, "betas": [0.8, 0.99], "weight_decay": 0.0},
            )
            dataset = paired_dataset(root / "wav", 1, seed=config["seed"])
            fresh = MPSENetTask(config, torch.device("cpu"))
            dataset.set_epoch(0)
            batch = dataset.collate([0])
            before = fresh.validation_metrics(batch)["time"]
            task = MPSENetTask(config, torch.device("cpu"))
            trainer = Trainer(
                task,
                plan_batches(1, 1, 1),
                cpu_process(),
                max_steps=20,
                seed=config["seed"],
                checkpoint_dir=root / "ckpt",
                checkpoint_interval=0,
                validation_interval=0,
                config=config,
            )
            trainer.fit(dataset)
            after = task.validation_metrics(batch)["time"]
            self.assertLess(after, before)

    def test_two_runs_and_a_resumed_run_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "wav"
            continuous = _train(root / "continuous", wav, max_steps=3, resume=None)
            interrupted_dir = root / "interrupted"
            _train(interrupted_dir, wav, max_steps=1, resume=None)
            resumed = _train(
                interrupted_dir,
                wav,
                max_steps=3,
                resume=interrupted_dir / "step_00000001.pt",
            )
            _assert_equal_tensors(self, continuous.generator.state_dict(), resumed.generator.state_dict())
            _assert_equal_tensors(self, continuous.optim_g.state_dict(), resumed.optim_g.state_dict())
            _assert_equal_tensors(self, continuous.sched_g.state_dict(), resumed.sched_g.state_dict())
            repeated = _train(root / "repeat", wav, max_steps=3, resume=None)
            _assert_equal_tensors(self, continuous.generator.state_dict(), repeated.generator.state_dict())
            digest = _tensor_hash(continuous.generator.state_dict())
            print(f"generator_sha256={digest}", flush=True)
            payload = load_checkpoint(root / "continuous" / "step_00000003.pt", torch.device("cpu"))
            self.assertEqual(payload["global_step"], 3)
            self.assertEqual(payload["batch_plan"]["accumulation_steps"], 2)
            self.assertIn("rng", payload)
            self.assertIn("environment", payload)
            self.assertEqual(payload["dataset"]["case_ids_sha256"], resumed_dataset_hash(wav))

    def test_trained_generator_loads_through_the_inference_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config(root / "ckpt", max_steps=1, global_batch_size=1, micro_batch_size_per_gpu=1)
            dataset = paired_dataset(root / "wav", 1, seed=config["seed"])
            task = MPSENetTask(config, torch.device("cpu"))
            Trainer(
                task,
                plan_batches(1, 1, 1),
                cpu_process(),
                max_steps=1,
                seed=config["seed"],
                checkpoint_dir=root / "ckpt",
                checkpoint_interval=0,
                validation_interval=0,
                config=config,
            ).fit(dataset)
            enhancer = load_enhancer("mp_senet", root / "ckpt" / "generator.pt", device="cpu", config=config["model"])
            waveform = torch.randn(1000)
            enhanced = enhancer.enhance(waveform, 16000)
            self.assertEqual(tuple(enhanced.shape), (1000,))
            self.assertTrue(torch.isfinite(enhanced).all())

    def test_published_checkpoint_directories_are_refused(self):
        with self.assertRaises(TrainingError):
            assert_output_allowed(ROOT / "data/checkpoints/not-a-real-run.pt")
        with self.assertRaises(TrainingError):
            assert_output_allowed(ROOT / "data/db/training-checkpoint.pt")
        with self.assertRaises(TrainingError):
            assert_output_allowed(ROOT / "data/staging/checkpoint.pt")

    def test_checked_in_config_uses_the_official_optimizer(self):
        config = load_training_config(ROOT / "configs/training/mp_senet.yaml")
        self.assertEqual(config["optimizer"]["lr"], 0.0005)
        self.assertEqual(config["optimizer"]["betas"], [0.8, 0.99])
        self.assertEqual(config["optimizer"]["weight_decay"], 0.01)
        self.assertEqual(config["scheduler"]["gamma"], 0.99)
        self.assertEqual(config["best_checkpoint_start_epoch"], 40)
        self.assertEqual(config["loss"], "official")
        self.assertEqual(config["model"]["n_fft"], 400)
        self.assertTrue(config["deterministic"])


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_short_audio_metrics_are_na_and_best_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config(
                root / "ckpt",
                max_steps=1,
                global_batch_size=1,
                micro_batch_size_per_gpu=1,
                best_checkpoint_start_epoch=0,
            )
            dataset = paired_dataset(root / "wav", 1, seed=config["seed"])
            task = MPSENetTask(config, torch.device("cpu"))
            task.eval_mode()
            report = task.validation_report(dataset.collate([0]))
            self.assertEqual(report["pesq"]["status"], "na")
            self.assertIsNone(report["pesq"]["value"])
            self.assertEqual(report["stoi"]["status"], "na")
            self.assertIsNone(report["stoi"]["value"])
            self.assertEqual(report["losses"]["time"]["status"], "ok")
            self.assertGreater(report["losses"]["time"]["value"], 0.0)
            self.assertEqual(report["losses"]["discriminator"]["status"], "na")

            scores = iter((2.5, 1.0))

            def fake_report(_batch):
                score = next(scores)
                return {
                    "losses": {
                        "generator": {"status": "ok", "value": 1.0, "reason": None},
                        "discriminator": {"status": "na", "value": None, "reason": "test"},
                        "time": {"status": "ok", "value": 0.2, "reason": None},
                    },
                    "pesq": {"status": "ok", "value": score, "reason": None},
                    "stoi": {"status": "na", "value": None, "reason": "short"},
                }

            task.validation_report = fake_report
            validation = paired_dataset(
                root / "wav",
                1,
                seed=config["seed"],
                role="validation",
                name="toy_valid",
                training=False,
            )
            trainer = Trainer(
                task,
                plan_batches(1, 1, 1),
                cpu_process(),
                max_steps=2,
                seed=config["seed"],
                checkpoint_dir=root / "ckpt",
                checkpoint_interval=1,
                validation_interval=1,
                config=config,
            )
            trainer.fit(dataset, validation)
            best = json.loads((root / "ckpt" / "best_generator.json").read_text(encoding="utf-8"))
            self.assertEqual(best["pesq"], 2.5)
            self.assertEqual(best["global_step"], 1)
            self.assertEqual(best["model"]["name"], "mp_senet")
            enhancer = load_enhancer(
                "mp_senet",
                root / "ckpt" / "best_generator.pt",
                device="cpu",
                config=config["model"],
            )
            enhanced = enhancer.enhance(torch.randn(1000), 16000)
            self.assertEqual(tuple(enhanced.shape), (1000,))
            lines = [
                json.loads(line)
                for line in (root / "ckpt" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            train_steps = [row["global_step"] for row in lines if row["type"] == "train"]
            self.assertEqual(train_steps, [1, 2])
            self.assertEqual(sum(row["type"] == "validation" for row in lines), 2)
            self.assertEqual(lines[0]["type"], "run")
            self.assertEqual(lines[0]["scheduler"], "epoch")

            with (root / "ckpt" / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "train", "global_step": 9}) + "\n")
            resumed = MPSENetTask(config, torch.device("cpu"))
            again = Trainer(
                resumed,
                plan_batches(1, 1, 1),
                cpu_process(),
                max_steps=3,
                seed=config["seed"],
                checkpoint_dir=root / "ckpt",
                checkpoint_interval=1,
                validation_interval=0,
                config=config,
            )
            again.fit(dataset, resume=root / "ckpt" / "step_00000002.pt")
            repeated = [
                json.loads(line)
                for line in (root / "ckpt" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["global_step"] for row in repeated if row["type"] == "train"], [1, 2, 3])
            self.assertNotIn(9, [row.get("global_step") for row in repeated])
            self.assertEqual(again.best_pesq, 2.5)


def _train(checkpoint_dir: Path, wav: Path, *, max_steps: int, resume: Path | None) -> MPSENetTask:
    config = tiny_config(checkpoint_dir, max_steps=max_steps)
    dataset = paired_dataset(wav, 8, seed=config["seed"])
    task = MPSENetTask(config, torch.device("cpu"))
    trainer = Trainer(
        task,
        plan_batches(config["global_batch_size"], config["micro_batch_size_per_gpu"], 1),
        cpu_process(),
        max_steps=max_steps,
        seed=config["seed"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=0,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset, resume=resume)
    return task


def resumed_dataset_hash(wav: Path) -> str:
    return paired_dataset(wav, 8, seed=7).identity()["case_ids_sha256"]


def _generator_state(seed: int) -> dict:
    task = MPSENetTask(tiny_config(Path("unused"), seed=seed), torch.device("cpu"))
    return task.generator.state_dict()


def _clone(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {key: _clone(value) for key, value in state.items()}
    if isinstance(state, list):
        return [_clone(value) for value in state]
    return state


def _states_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _states_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            _states_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _assert_equal_tensors(test: unittest.TestCase, left, right) -> None:
    test.assertTrue(_states_equal(left, right))


def _tensor_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
