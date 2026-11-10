"""Kept MossFormerGAN weights follow ClearerVoice ``train.py`` construction.

Official ``network_wrapper`` builds two ``MossFormerGAN_SE_16K`` modules, and
``train.py`` builds two wrappers. With ``mode='train'`` the RNG order is
G1, D1, G2, D2, G3, D3, G4, D4. Learning keeps G1 and D4.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "src/se/official/clearervoice-studio/train/speech_enhancement"
SEED = 777


def _batchnorm_count(module) -> int:
    import torch

    kinds = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)
    return sum(isinstance(child, kinds) for child in module.modules())


def _param_sum(module) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for parameter in module.parameters():
            total = total + parameter.detach().double().sum().cpu()
    return float(total)


def _state_max_abs(left, right) -> dict:
    import torch

    left_state = left.state_dict()
    right_state = right.state_dict()
    only_left = sorted(set(left_state) - set(right_state))
    only_right = sorted(set(right_state) - set(left_state))
    if only_left or only_right:
        return {
            "key_mismatch": True,
            "only_audio_db": only_left[:8],
            "only_official": only_right[:8],
        }
    max_abs = 0.0
    worst = None
    for key, value in left_state.items():
        other = right_state[key]
        if tuple(value.shape) != tuple(other.shape):
            return {"key_mismatch": True, "shape": key, "audio_db": list(value.shape), "official": list(other.shape)}
        diff = float((value.detach().cpu() - other.detach().cpu()).abs().max())
        if diff > max_abs:
            max_abs = diff
            worst = key
    return {"key_mismatch": False, "max_abs": max_abs, "worst": worst}


def compare_official() -> dict:
    import numpy as np
    import torch

    from src.experiments.calc.mossformer_discriminator import Discriminator
    from src.experiments.calc.mossformergan import MossFormerGANTask

    result: dict = {"status": "ok", "seed": SEED}
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        rng_before_import = torch.random.get_rng_state().clone()
        sys.path.insert(0, str(OFFICIAL))
        from networks import network_wrapper

        rng_after_import = torch.random.get_rng_state().clone()
        result["official_import_changed_rng"] = not torch.equal(rng_before_import, rng_after_import)
        args = SimpleNamespace(network="MossFormerGAN_SE_16K", mode="train", fft_len=400)
        first = network_wrapper(args)
        official_generator = first.se_network.to(torch.device("cpu"))
        del first
        second = network_wrapper(args)
        official_discriminator = second.discriminator.to(torch.device("cpu"))
        del second
    except Exception as exc:
        result["status"] = "import_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    config = {
        "seed": SEED,
        "loss": "official",
        "grad_clip": 10.0,
        "optimizer": {"name": "adamw", "lr": 5e-4, "betas": [0.9, 0.999], "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "model": {
            "name": "mossformergan",
            "variant": "se_16k",
            "sampling_rate": 16000,
            "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
        },
    }
    task = MossFormerGANTask(config, torch.device("cpu"))
    generator_diff = _state_max_abs(task.generator, official_generator)
    discriminator_diff = _state_max_abs(task.discriminator, official_discriminator)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    first_discriminator = Discriminator(ndf=16)
    result.update(
        {
            "audio_db_generator_param_sum": _param_sum(task.generator),
            "audio_db_discriminator_param_sum": _param_sum(task.discriminator),
            "official_generator_param_sum": _param_sum(official_generator),
            "official_discriminator_param_sum": _param_sum(official_discriminator),
            "first_discriminator_param_sum": _param_sum(first_discriminator),
            "generator": generator_diff,
            "discriminator": discriminator_diff,
            "generator_batchnorm_modules": _batchnorm_count(task.generator),
            "official_generator_batchnorm_modules": _batchnorm_count(official_generator),
            "kept_discriminator_differs_from_first_consumer": not torch.equal(
                task.discriminator.cnn.conv1.weight.detach().cpu(),
                first_discriminator.cnn.conv1.weight.detach().cpu(),
            ),
        }
    )
    return result


class MossFormerOfficialInitTests(unittest.TestCase):
    def test_kept_generator_and_discriminator_match_official_train_py(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONHASHSEED"] = "0"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--compare"],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        line = next((item for item in completed.stdout.splitlines() if item.startswith("RESULT ")), "")
        self.assertTrue(line, f"comparison produced no result\n{completed.stdout}\n{completed.stderr}")
        result = json.loads(line[len("RESULT ") :])
        if result.get("status") == "import_error":
            self.fail(f"official import failed: {result.get('error')}\n{completed.stderr}")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(result["generator"]["key_mismatch"], result["generator"])
        self.assertFalse(result["discriminator"]["key_mismatch"], result["discriminator"])
        self.assertEqual(result["generator"]["max_abs"], 0.0, result)
        self.assertEqual(result["discriminator"]["max_abs"], 0.0, result)
        self.assertEqual(result["generator_batchnorm_modules"], 0, result)
        self.assertEqual(result["official_generator_batchnorm_modules"], 0, result)
        self.assertTrue(result["kept_discriminator_differs_from_first_consumer"], result)
        self.assertNotEqual(result["audio_db_discriminator_param_sum"], result["first_discriminator_param_sum"])


if __name__ == "__main__":
    if "--compare" in sys.argv:
        print("RESULT " + json.dumps(compare_official()))
    else:
        unittest.main()
