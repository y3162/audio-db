"""One-step training checks for the SE models other than the MP-SENet foundation tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.se import build_model, load_enhancer
from src.training.core.rng import seed_model_init, set_determinism
from src.training.errors import TrainingError
from src.training.se.deepfilternet import DeepFilterNetTask
from src.training.se.mossformergan import MossFormerGANTask
from src.training.se.semambapp import SEMambaTask
from src.training.se.stages import StagedTask
from src.training.se.tf_gridnet import TFGridNetTask
from tests.training.support import paired_dataset, tiny_config


def tearDownModule():
    set_determinism(False, cuda=False)


def _base(root: Path, name: str, **model) -> dict:
    config = tiny_config(
        root / "ckpt",
        loss="official",
        segment_size=model.pop("segment_size", 800),
        global_batch_size=1,
        micro_batch_size_per_gpu=1,
        max_steps=1,
    )
    config["optimizer"]["lr"] = 1e-3
    config["model"] = {
        "name": name,
        "variant": model.pop("variant", None),
        "stage": model.pop("stage", None),
        "sampling_rate": model.pop("sampling_rate", 16000),
        "architecture": model.pop("architecture", {}),
        "stft": model.pop("stft", {}),
        "components": model.pop("components", {}),
    }
    config["grad_clip"] = model.pop("grad_clip", None)
    config["scheduler"] = model.pop("scheduler", {"name": "exponential", "gamma": 0.99})
    return config


def _updated(task, dataset) -> dict[str, float]:
    before = [parameter.detach().clone() for parameter in task.generator.parameters() if parameter.requires_grad]
    metrics = task.train_update([dataset.collate([0])], 1)
    for value in metrics.values():
        if value != value or value in (float("inf"), float("-inf")):
            raise AssertionError(metrics)
    after = [parameter.detach() for parameter in task.generator.parameters() if parameter.requires_grad]
    if not any(not torch.equal(left, right) for left, right in zip(before, after, strict=True)):
        raise AssertionError("generator weights did not change")
    grads = [parameter.grad for parameter in task.generator.parameters() if parameter.grad is not None]
    if not grads or not all(torch.isfinite(grad).all() for grad in grads):
        raise AssertionError("generator gradients are missing or non-finite")
    return metrics


class GridNetOfficialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_ms_snsd_official_loss_reloads(self):
        architecture = {
            "n_fft": 32,
            "stride": 16,
            "n_layers": 1,
            "lstm_hidden_units": 8,
            "attn_n_head": 1,
            "attn_approx_qk_dim": 8,
            "emb_dim": 8,
            "emb_ks": 1,
            "emb_hs": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base(root, "tf_gridnet", variant="ms-snsd", architecture=architecture, segment_size=800)
            config["scheduler"] = {"name": "plateau", "factor": 0.7, "patience": 1}
            with seed_model_init(config["seed"]):
                task = TFGridNetTask(config, torch.device("cpu"))
            data = paired_dataset(root / "wav", 1, frames=800, segment_size=800, seed=config["seed"])
            metrics = _updated(task, data)
            self.assertIn("mr_l1_tfd", metrics)
            path = root / "separator.pt"
            torch.save(task.export_for_inference(), path)
            enhancer = load_enhancer("tf_gridnet", {"model": str(path)}, device="cpu", config=architecture, variant="ms-snsd")
            enhanced = enhancer.enhance(torch.randn(800), 16000)
            self.assertEqual(tuple(enhanced.shape), (800,))
            self.assertTrue(torch.isfinite(enhanced).all())
            restored = TFGridNetTask(config, torch.device("cpu"))
            restored.load_state_dict(task.state_dict())
            self.assertTrue(torch.equal(next(restored.generator.parameters()), next(task.generator.parameters())))

    def test_urgent24_step_is_finite(self):
        architecture = {
            "n_layers": 1,
            "lstm_hidden_units": 8,
            "attn_n_head": 1,
            "attn_qk_output_channel": 2,
            "emb_dim": 8,
            "emb_ks": 1,
            "emb_hs": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base(
                root,
                "tf_gridnet",
                variant="urgent24",
                architecture=architecture,
                segment_size=1024,
                scheduler={"name": "steplr", "gamma": 0.5, "step_size": 2},
            )
            task = TFGridNetTask(config, torch.device("cpu"))
            data = paired_dataset(root / "wav", 1, frames=1024, segment_size=1024, seed=config["seed"])
            _updated(task, data)
            before = task.learning_rates()["generator"]
            task.step_schedulers(None)
            task.step_schedulers(None)
            self.assertAlmostEqual(task.learning_rates()["generator"], before * 0.5)

    def test_ms_snsd_official_width_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base(
                root,
                "tf_gridnet",
                variant="ms-snsd",
                architecture={},
                segment_size=1024,
                scheduler={"name": "plateau", "factor": 0.7, "patience": 1},
            )
            task = TFGridNetTask(config, torch.device("cpu"))
            self.assertEqual(len(task.generator.blocks), 4)
            self.assertEqual(task.generator.n_fft, 512)
            data = paired_dataset(root / "wav", 1, frames=1024, segment_size=1024, seed=config["seed"])
            metrics = _updated(task, data)
            self.assertIn("mr_l1_tfd", metrics)
            path = root / "separator.pt"
            torch.save(task.export_for_inference(), path)
            enhancer = load_enhancer("tf_gridnet", {"model": str(path)}, device="cpu", variant="ms-snsd")
            enhanced = enhancer.enhance(torch.randn(1024), 16000)
            self.assertEqual(tuple(enhanced.shape), (1024,))
            self.assertTrue(torch.isfinite(enhanced).all())
            restored = TFGridNetTask(config, torch.device("cpu"))
            restored.load_state_dict(task.state_dict())
            self.assertTrue(torch.equal(next(restored.generator.parameters()), next(task.generator.parameters())))


class SEMambaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_official_step_is_finite_and_reloads(self):
        architecture = {
            "hid_feature": 8,
            "num_tfmamba": 1,
            "d_state": 4,
            "d_conv": 2,
            "expand": 2,
            "n_fft": 400,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base(
                root,
                "semambapp",
                architecture=architecture,
                stft={"hop": 100, "win": 400, "disc_windows": [256, 512]},
                segment_size=2048,
            )
            task = SEMambaTask(config, torch.device("cpu"))
            data = paired_dataset(root / "wav", 1, frames=2048, segment_size=2048, seed=config["seed"])
            metrics = _updated(task, data)
            self.assertIn("discriminator", metrics)
            path = root / "generator.pt"
            torch.save(task.export_for_inference(), path)
            enhancer = load_enhancer("semambapp", path, device="cpu", config=architecture)
            enhanced = enhancer.enhance(torch.randn(2048), 16000)
            self.assertTrue(torch.isfinite(enhanced).all())


class DeepFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_shrinking_emb_layers_fails_in_erb_decoder(self):
        with self.assertRaises(ValueError):
            build_model(
                "deepfilternet3",
                {
                    "sr": 16000,
                    "fft_size": 128,
                    "hop_size": 64,
                    "nb_erb": 16,
                    "nb_df": 16,
                    "conv_ch": 16,
                    "emb_hidden_dim": 32,
                    "df_hidden_dim": 32,
                    "enc_lin_groups": 8,
                    "lin_groups": 8,
                    "emb_num_layers": 1,
                    "df_num_layers": 1,
                },
            )

    def test_valid_reduction_trains(self):
        architecture = {
            "sr": 16000,
            "fft_size": 128,
            "hop_size": 64,
            "nb_erb": 16,
            "nb_df": 16,
            "conv_ch": 16,
            "emb_hidden_dim": 32,
            "df_hidden_dim": 32,
            "enc_lin_groups": 8,
            "lin_groups": 8,
            "emb_num_layers": 2,
            "df_num_layers": 1,
            "df_order": 5,
            "conv_lookahead": 0,
            "df_lookahead": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base(root, "deepfilternet3", architecture=architecture, segment_size=1024, grad_clip=1.0)
            task = DeepFilterNetTask(config, torch.device("cpu"))
            data = paired_dataset(root / "wav", 1, frames=1024, segment_size=1024, seed=config["seed"])
            _updated(task, data)
            path = root / "model.pt"
            torch.save(task.export_for_inference(), path)
            enhancer = load_enhancer("deepfilternet3", path, device="cpu", config=architecture)
            waveform = torch.randn(1024)
            first = enhancer.enhance(waveform, 16000)
            second = enhancer.enhance(waveform, 16000)
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.isfinite(first).all())

    def test_official_width_step_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base(root, "deepfilternet3", architecture={}, segment_size=4800, sampling_rate=48000, grad_clip=1.0)
            config["optimizer"] = {"name": "adamw", "lr": 0.001, "betas": [0.9, 0.999], "weight_decay": 1e-12, "amsgrad": True}
            task = DeepFilterNetTask(config, torch.device("cpu"))
            self.assertEqual(task.generator.params.conv_ch, 64)
            self.assertEqual(task.generator.params.emb_num_layers, 3)
            again = DeepFilterNetTask(config, torch.device("cpu"))
            self.assertTrue(torch.equal(next(task.generator.parameters()), next(again.generator.parameters())))
            data = paired_dataset(root / "wav", 1, frames=4800, segment_size=4800, seed=config["seed"])
            metrics = _updated(task, data)
            self.assertTrue(all(value == value and abs(value) != float("inf") for value in metrics.values()))
            path = root / "model.pt"
            torch.save(task.export_for_inference(), path)
            enhancer = load_enhancer("deepfilternet3", path, device="cpu")
            waveform = torch.randn(4800)
            first = enhancer.enhance(waveform, 48000)
            second = enhancer.enhance(waveform, 48000)
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.isfinite(first).all())
            restored = DeepFilterNetTask(config, torch.device("cpu"))
            restored.load_state_dict(task.state_dict())
            self.assertTrue(torch.equal(next(restored.generator.parameters()), next(task.generator.parameters())))


class MossFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_official_width_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base(
                root,
                "mossformergan",
                variant="se_16k",
                architecture={"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
                segment_size=8000,
                grad_clip=10.0,
            )
            task = MossFormerGANTask(config, torch.device("cpu"))
            data = paired_dataset(root / "wav", 1, frames=8000, segment_size=8000, seed=config["seed"])
            metrics = _updated(task, data)
            self.assertGreater(task.learning_rates()["discriminator"], task.learning_rates()["generator"])
            self.assertIn("discriminator", metrics)
            path = root / "model.pt"
            torch.save(task.export_for_inference(), path)
            enhancer = load_enhancer(
                "mossformergan",
                path,
                device="cpu",
                config={"n_fft": 400, "num_channel": 64},
                variant="se_16k",
            )
            enhanced = enhancer.enhance(torch.randn(8000), 16000)
            self.assertTrue(torch.isfinite(enhanced).all())


class StagedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_dewavlm_without_teacher_or_architecture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("unipase", "stupase"):
                config = _base(root, name, stage="dewavlm")
                with self.assertRaises(TrainingError) as caught:
                    StagedTask(config, torch.device("cpu"))
                self.assertIn("WavLM-Large", str(caught.exception))

    def test_reduced_dewavlm_cfm_and_vocoder_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dewavlm = _base(
                root,
                "stupase",
                stage="dewavlm",
                architecture={"wavlm": {"output_layer": 1, "encoder_layers": 1, "encoder_embed_dim": 32}},
                segment_size=640,
            )
            task = StagedTask(dewavlm, torch.device("cpu"))
            again = StagedTask(dewavlm, torch.device("cpu"))
            self.assertTrue(torch.equal(next(task.generator.parameters()), next(again.generator.parameters())))
            data = paired_dataset(root / "dw", 1, frames=640, segment_size=640, seed=dewavlm["seed"])
            metrics = _updated(task, data)
            self.assertIn("dewavlm", metrics)
            restored = StagedTask(dewavlm, torch.device("cpu"))
            restored.load_state_dict(task.state_dict())
            self.assertTrue(torch.equal(next(restored.generator.parameters()), next(task.generator.parameters())))

    def test_unipase_adapter_vocoder_and_postnet_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = _base(
                root,
                "unipase",
                stage="adapter",
                architecture={"input_channels": 16, "dim": 32, "intermediate_dim": 64, "num_layers": 1, "output_channels": 16, "disc_dims": [8, 16]},
                segment_size=512,
            )
            task = StagedTask(adapter, torch.device("cpu"))
            data = paired_dataset(root / "ad", 1, frames=512, segment_size=512, seed=adapter["seed"])
            metrics = _updated(task, data)
            self.assertIn("discriminator", metrics)
            restored = StagedTask(adapter, torch.device("cpu"))
            restored.load_state_dict(task.state_dict())
            self.assertTrue(torch.equal(next(restored.generator.parameters()), next(task.generator.parameters())))

            vocoder = _base(
                root,
                "unipase",
                stage="vocoder",
                architecture={"input_channels": 16, "dim": 32, "intermediate_dim": 64, "num_layers": 1, "num_res": 2, "num_attn": 1, "n_fft": 64, "hop_length": 16},
                segment_size=512,
            )
            vocoder_task = StagedTask(vocoder, torch.device("cpu"))
            vocoder_data = paired_dataset(root / "uv", 1, frames=512, segment_size=512, seed=vocoder["seed"])
            self.assertIn("discriminator", _updated(vocoder_task, vocoder_data))

            postnet = _base(
                root,
                "unipase",
                stage="postnet",
                architecture={"n_layers": 1, "lstm_hidden_units": 16, "attn_n_head": 2, "attn_qk_output_channel": 2, "emb_dim": 16, "emb_ks": 4, "emb_hs": 4},
                segment_size=2048,
                sampling_rate=16000,
            )
            post = StagedTask(postnet, torch.device("cpu"))
            post_data = paired_dataset(root / "pn", 1, frames=2048, segment_size=2048, seed=postnet["seed"])
            self.assertIn("discriminator", _updated(post, post_data))

    def test_stupase_vocoder_and_cfm_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocoder = _base(
                root,
                "stupase",
                stage="vocoder",
                architecture={
                    "input_channels": 16,
                    "dim": 32,
                    "intermediate_dim": 64,
                    "num_layers": 1,
                    "num_res": 2,
                    "num_attn": 1,
                    "n_fft": 64,
                    "hop_length": 16,
                },
                segment_size=512,
            )
            task = StagedTask(vocoder, torch.device("cpu"))
            data = paired_dataset(root / "voc", 1, frames=512, segment_size=512, seed=vocoder["seed"])
            metrics = _updated(task, data)
            self.assertIn("discriminator", metrics)
            cfm = _base(
                root,
                "stupase",
                stage="cfm",
                architecture={"dit": {"dim": 32, "depth": 1, "heads": 4, "dim_head": 8, "mel_dim": 16, "text_in_dim": 16, "ff_mult": 2}, "mel": {"n_fft": 64, "hop_length": 16, "win_length": 64, "n_mel_channels": 16, "target_sample_rate": 16000}},
                segment_size=512,
            )
            flow = StagedTask(cfm, torch.device("cpu"))
            flow_data = paired_dataset(root / "flow", 1, frames=512, segment_size=512, seed=cfm["seed"])
            flow_metrics = _updated(flow, flow_data)
            self.assertIn("flow_matching", flow_metrics)


if __name__ == "__main__":
    unittest.main()
