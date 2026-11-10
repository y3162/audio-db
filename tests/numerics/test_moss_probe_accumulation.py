import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import torch

from tests.numerics.four_process.real_models import MossFormerGAN
from tests.numerics.four_process.compare import _batch, digest


class MossProbeAccumulationTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_stft_uses_official_cpu_window(self):
        from src.experiments.calc.mossformergan import _stft
        torch.manual_seed(7)
        wave = torch.randn(1, 8000, device="cuda")
        window = torch.hamming_window(400, periodic=False).to(wave.device)
        expected = torch.view_as_real(torch.stft(wave, 400, 100, 400, window, center=True, return_complex=True))
        torch.testing.assert_close(_stft(wave, 400, 100, 400), expected, rtol=0, atol=0)

    def test_official_width_options(self):
        from tests.numerics.four_process.adapters import MPSENet
        from tests.numerics.four_process.compare import identity
        from tests.numerics.four_process.real_models import SEMambaPP
        mp = MPSENet()
        mp.configure({"official_width": True})
        self.assertEqual((mp.hyper().dense_channel, mp.hyper().num_tsconformers), (64, 4))
        semamba = SEMambaPP()
        semamba.configure({"official_width": True})
        arch = semamba._arch()
        self.assertEqual(tuple(arch[k] for k in ("hid_feature", "num_tfmamba", "d_state", "d_conv")), (48, 4, 16, 4))
        semamba.configure({"scheduler_step": "epoch"})
        self.assertFalse(semamba.step_every_update)
        semamba.configure({"scheduler_step": "update"})
        self.assertTrue(semamba.step_every_update)
        self.assertEqual(identity({}), identity({"official_width": False}))
        self.assertNotEqual(identity({}), identity({"official_width": True}))

    def test_recipe_steps_use_two_distinct_recorded_inputs(self):
        waves = [torch.tensor([float(i)]) for i in range(4)]
        items = [{"case_id": str(i), "clean_sha256": digest(w), "noisy_sha256": digest(w)}
                 for i, w in enumerate(waves)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"kind": "recipe", "items": items, "clean": waves, "noisy": waves}, root / "input.pt")
            clean, noisy, selected = _batch(root, {"model": "mossformergan"}, 1, None)
        self.assertEqual(clean.flatten().tolist(), [2, 3])
        self.assertTrue(torch.equal(clean, noisy))
        self.assertEqual([item["case_id"] for item in selected], ["2", "3"])

    def test_old_single_batch_probe_cannot_resume_as_accumulation_two(self):
        from tests.numerics.four_process.compare import worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model": "mossformergan", "probe": {"update": "single batch"}}))
            with self.assertRaisesRegex(RuntimeError, "before accumulation=2"):
                worker(root, "official_a")

    def test_old_semamba_probe_cannot_resume_with_changed_scheduler_and_losses(self):
        from tests.numerics.four_process.compare import worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "model": "semambapp", "probe": {"aggregation": "interleave real/fake"},
            }))
            with self.assertRaisesRegex(RuntimeError, "mel, MRD and scheduler fixes"):
                worker(root, "official_a")

    def test_two_microbatches_are_clipped_before_each_update(self):
        adapter = MossFormerGAN()
        adapter.clip = 0.1
        modules = {name: torch.nn.Linear(1, 1, bias=False) for name in ("generator", "discriminator")}
        for module in modules.values():
            module.weight.data.fill_(1)
        optimizers = {name: torch.optim.SGD(module.parameters(), lr=0.1) for name, module in modules.items()}
        calls = []

        def losses(generator, discriminator, noisy, teacher, rate):
            calls.append(noisy.detach().clone())
            value = generator(noisy)
            return value, value.sum() * 100, discriminator(noisy).sum() * 100

        inputs = torch.tensor([[1.0], [-2.0]])
        with patch("tests.numerics.four_process.real_models._independent_moss_losses", side_effect=losses):
            result = adapter.train_step("independent", modules, optimizers, {}, inputs, inputs, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.forward.shape[0], 2)
        # Opposite gradients are clipped independently at each micro-batch.
        self.assertAlmostEqual(modules["generator"].weight.item(), 1.01, places=5)
        self.assertAlmostEqual(modules["discriminator"].weight.item(), 1.01, places=5)


if __name__ == "__main__":
    unittest.main()
