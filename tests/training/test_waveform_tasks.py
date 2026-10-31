"""Waveform L1 steps for models that are not MP-SENet.

The objective is time L1. It is not the official GAN or multi-stage loss.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from src.se import load_enhancer
from src.training.core.rng import set_determinism
from src.training.se.waveform import WaveformTask
from tests.training.support import paired_dataset, tiny_config


def tearDownModule():
    set_determinism(False, cuda=False)


def _config(root: Path, name: str, variant: str | None, architecture: dict, stft: dict | None = None) -> dict:
    config = tiny_config(root / "ckpt", loss="waveform_l1", segment_size=128)
    config["model"] = {
        "name": name,
        "variant": variant,
        "architecture": architecture,
        "stft": stft or {},
    }
    return config


class WaveformTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_tf_gridnet_trains_and_reloads_for_inference(self):
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
            config = _config(root, "tf_gridnet", "ms-snsd", architecture)
            left = WaveformTask(config, torch.device("cpu"))
            right = WaveformTask(config, torch.device("cpu"))
            self.assertTrue(torch.equal(next(left.generator.parameters()), next(right.generator.parameters())))
            frozen = next(parameter for parameter in left.generator.parameters() if parameter.requires_grad)
            frozen.requires_grad_(False)
            before = frozen.detach().clone()
            dataset = paired_dataset(root / "wav", 2, frames=800, segment_size=800, seed=config["seed"])
            metrics = left.train_update([dataset.collate([0, 1])], 1)
            self.assertTrue(all(value == value and abs(value) != float("inf") for value in metrics.values()))
            self.assertTrue(torch.equal(frozen, before))
            trained = [
                parameter
                for parameter in left.generator.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(trained)
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in trained))
            path = root / "weights.pt"
            torch.save(
                {"separator." + item: value for item, value in left.generator_state_dict().items()},
                path,
            )
            enhancer = load_enhancer(
                "tf_gridnet",
                {"model": str(path)},
                device="cpu",
                config=architecture,
                variant="ms-snsd",
            )
            enhanced = enhancer.enhance(torch.randn(800), 16000)
            self.assertEqual(tuple(enhanced.shape), (800,))
            self.assertTrue(torch.isfinite(enhanced).all())
