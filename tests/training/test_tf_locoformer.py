"""One official training step for TF-Locoformer against the standalone network."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.se import build_model, load_enhancer
from src.training.core.rng import seed_model_init, set_determinism
from src.training.se.spectral import multi_resolution_l1
from src.training.se.tf_locoformer import TFLocoformerTask
from tests.se.test_tf_locoformer import OFFICIAL, TINY, _frontend, _official_separator
from tests.training.support import paired_dataset, tiny_config


def tearDownModule():
    set_determinism(False, cuda=False)


class TFLocoformerTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    @unittest.skipUnless(OFFICIAL.is_dir(), "official checkout missing")
    def test_one_step_matches_standalone_separator(self):
        torch.manual_seed(11)
        official = _official_separator()
        architecture = dict(TINY)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config(root / "ckpt", loss="official", segment_size=512, global_batch_size=1, micro_batch_size_per_gpu=1, max_steps=1)
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
            with seed_model_init(config["seed"]):
                task = TFLocoformerTask(config, torch.device("cpu"))
            task.generator.separator.load_state_dict(official.state_dict(), strict=True)
            data = paired_dataset(root / "wav", 1, frames=512, segment_size=512, seed=config["seed"])
            batch = data.collate([0])
            clean = batch["clean"]
            noisy = batch["noisy"]
            official.train()
            official_opt = torch.optim.AdamW(official.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
            official_wave = _frontend(official, noisy, 64, 32)
            official_loss = multi_resolution_l1(clean, official_wave, time_domain_weight=0.5, normalize_variance=False)
            official_opt.zero_grad(set_to_none=True)
            official_loss.backward()
            torch.nn.utils.clip_grad_norm_(official.parameters(), 5.0)
            metrics = task.train_update([batch], 1)
            official_opt.step()
            self.assertEqual(float(abs(metrics["mr_l1_tfd"] - float(official_loss.detach()))), 0.0)
            for name, left in official.named_parameters():
                right = dict(task.generator.separator.named_parameters())[name]
                self.assertEqual(float((left.detach() - right.detach()).abs().max()), 0.0)
            path = root / "separator.pt"
            torch.save(task.export_for_inference(), path)
            enhancer = load_enhancer("tf_locoformer", {"model": str(path)}, device="cpu", config=architecture, variant="dns")
            reloaded = build_model("tf_locoformer", architecture, variant="dns")
            reloaded.load_state_dict(task.generator.state_dict(), strict=True)
            with torch.no_grad():
                wave = reloaded(noisy)
                heard = enhancer.enhance(noisy[0], 16000)
            self.assertEqual(float((wave[0] - heard).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
