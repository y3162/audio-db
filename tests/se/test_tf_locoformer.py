"""TF-Locoformer matches the standalone official separator and the DNS frontend."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

from src.se import available_enhancers, build_model, load_enhancer
from src.experiments.calc.spectral import multi_resolution_l1

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "src/se/official/tf-locoformer"
TINY = dict(
    num_spk=1,
    n_layers=1,
    emb_dim=16,
    norm_type="rmsgroupnorm",
    num_groups=4,
    tf_order="ft",
    n_heads=2,
    flash_attention=False,
    attention_dim=16,
    pos_enc="rope",
    ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
    ffn_hidden_dim=[16, 16],
    conv1d_kernel=4,
    conv1d_shift=1,
    dropout=0.0,
    eps=1.0e-5,
    n_fft=64,
    hop_length=32,
    normalize_variance=True,
)


def _official_separator():
    root = str(OFFICIAL)
    if root not in sys.path:
        sys.path.insert(0, root)
    from standalone.tflocoformer_separator import TFLocoformerSeparator

    options = {key: value for key, value in TINY.items() if key not in ("n_fft", "hop_length", "normalize_variance")}
    return TFLocoformerSeparator(**options)


def _frontend(separator, waveform, n_fft, hop):
    scale = torch.std(waveform, dim=1, keepdim=True)
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform / scale, n_fft, hop, n_fft, window, center=True, return_complex=True
    ).transpose(1, 2)
    enhanced = separator(spectrum)[:, 0].transpose(1, 2)
    restored = torch.istft(enhanced, n_fft, hop, n_fft, window, center=True, length=waveform.shape[-1])
    return restored * scale


class TFLocoformerMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_registry(self):
        self.assertIn("tf_locoformer", available_enhancers())

    @unittest.skipUnless(OFFICIAL.is_dir(), "official checkout missing")
    def test_separator_and_waveform_match_official(self):
        torch.manual_seed(3)
        official = _official_separator()
        independent = build_model("tf_locoformer", TINY, variant="dns")
        independent.separator.load_state_dict(official.state_dict(), strict=True)
        spectrum = torch.complex(torch.randn(2, 12, 33), torch.randn(2, 12, 33))
        official.train()
        independent.separator.train()
        left = official(spectrum)
        right = independent.separator(spectrum)
        self.assertEqual(float((left - right).abs().max()), 0.0)
        waveform = torch.randn(2, 512)
        official_wave = _frontend(official, waveform, 64, 32)
        independent_wave = independent(waveform)
        self.assertEqual(float((official_wave - independent_wave).abs().max()), 0.0)
        clean = torch.randn(2, 512)
        official_loss = multi_resolution_l1(clean, official_wave, time_domain_weight=0.5, normalize_variance=False)
        independent_loss = multi_resolution_l1(clean, independent_wave, time_domain_weight=0.5, normalize_variance=False)
        self.assertEqual(float((official_loss - independent_loss).abs()), 0.0)
        official_loss.backward()
        independent_loss.backward()
        saw_grad = False
        for name, left_value in official.named_parameters():
            right_value = dict(independent.separator.named_parameters())[name]
            if left_value.grad is None or right_value.grad is None:
                self.assertIsNone(left_value.grad)
                self.assertIsNone(right_value.grad)
                continue
            saw_grad = True
            self.assertEqual(float((left_value.grad - right_value.grad).abs().max()), 0.0)
        self.assertTrue(saw_grad)

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(5)
        model = build_model("tf_locoformer", TINY, variant="dns").eval()
        path = Path(self.id().replace(".", "_") + ".pt")
        try:
            payload = {"separator." + key: value.detach() for key, value in model.separator.state_dict().items()}
            torch.save(payload, path)
            enhancer = load_enhancer("tf_locoformer", {"model": str(path)}, device="cpu", config=TINY, variant="dns")
            noisy = torch.randn(400)
            enhanced = enhancer.enhance(noisy, 16000)
            self.assertEqual(tuple(enhanced.shape), tuple(noisy.shape))
            self.assertTrue(torch.isfinite(enhanced).all())
            again = enhancer.enhance(noisy, 16000)
            self.assertEqual(float((enhanced - again).abs().max()), 0.0)
        finally:
            path.unlink(missing_ok=True)

    def test_forward_rejects_zero_variance_and_keeps_quiet_inputs_finite(self):
        torch.manual_seed(9)
        model = build_model("tf_locoformer", TINY, variant="dns").eval()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 512))
        with self.assertRaises(ValueError):
            model(torch.ones(1, 512))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 512) * 1e-12)
        quiet = torch.randn(1, 512) * 1e-3
        with torch.no_grad():
            enhanced = model(quiet)
        self.assertTrue(torch.isfinite(enhanced).all())
        partial = torch.zeros(1, 512)
        partial[:, 200:280] = 0.1
        with torch.no_grad():
            heard = model(partial)
        self.assertTrue(torch.isfinite(heard).all())

    def test_enhance_boundary_conditions(self):
        torch.manual_seed(13)
        model = build_model("tf_locoformer", dict(TINY), variant="dns").eval()
        path = Path(self.id().replace(".", "_") + ".pt")
        try:
            payload = {"separator." + key: value.detach() for key, value in model.separator.state_dict().items()}
            torch.save(payload, path)
            enhancer = load_enhancer("tf_locoformer", {"model": str(path)}, device="cpu", config=TINY, variant="dns")
            for rate in (8000, 16000, 48000):
                waveform = torch.randn(rate // 2)
                enhanced = enhancer.enhance(waveform, rate)
                self.assertEqual(tuple(enhanced.shape), tuple(waveform.shape))
                self.assertEqual(enhanced.dtype, waveform.dtype)
                self.assertTrue(torch.isfinite(enhanced).all())
            short = torch.randn(33)
            enhanced = enhancer.enhance(short, 16000)
            self.assertEqual(tuple(enhanced.shape), (33,))
            with self.assertRaises(ValueError):
                enhancer.enhance(torch.randn(32), 16000)
            self.assertEqual(float(enhancer.enhance(torch.zeros(400), 16000).abs().max()), 0.0)
            self.assertEqual(float(enhancer.enhance(torch.full((400,), 0.2), 16000).abs().max()), 0.0)
            for bad in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    enhancer.enhance(torch.tensor([bad, 0.0, 1.0]), 16000)
            doubled = torch.randn(400, dtype=torch.float64)
            heard = enhancer.enhance(doubled, 16000)
            self.assertEqual(heard.dtype, torch.float64)
            batch = torch.randn(3, 300)
            lengths = torch.tensor([300, 200, 80])
            out, returned = enhancer.enhance_batch(batch, 16000, lengths)
            self.assertEqual(tuple(out.shape), tuple(batch.shape))
            self.assertTrue(torch.equal(returned, lengths))
            self.assertTrue(torch.isfinite(out).all())
            reloaded = load_enhancer("tf_locoformer", {"model": str(path)}, device="cpu", config=TINY, variant="dns")
            left = enhancer.enhance(batch[0], 16000)
            right = reloaded.enhance(batch[0], 16000)
            self.assertEqual(float((left - right).abs().max()), 0.0)
        finally:
            path.unlink(missing_ok=True)

    def test_cuda_enhance_when_a_gpu_is_free(self):
        device = _free_cuda()
        if device is None:
            self.skipTest("no CUDA GPU under 1024 MiB")
        torch.manual_seed(15)
        model = build_model("tf_locoformer", TINY, variant="dns").eval()
        path = Path(self.id().replace(".", "_") + ".pt")
        try:
            payload = {"separator." + key: value.detach() for key, value in model.separator.state_dict().items()}
            torch.save(payload, path)
            enhancer = load_enhancer(
                "tf_locoformer", {"model": str(path)}, device=f"cuda:{device}", config=TINY, variant="dns"
            )
            waveform = torch.randn(1600)
            enhanced = enhancer.enhance(waveform, 16000)
            self.assertEqual(tuple(enhanced.shape), tuple(waveform.shape))
            self.assertTrue(torch.isfinite(enhanced).all())
            self.assertEqual(enhanced.device.type, "cpu")
            again = enhancer.enhance(waveform, 16000)
            self.assertEqual(float((enhanced - again).abs().max()), 0.0)
        finally:
            path.unlink(missing_ok=True)


def _free_cuda():
    if not torch.cuda.is_available():
        return None
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        if total - free < 1024 * 1024 * 1024:
            return index
    return None
