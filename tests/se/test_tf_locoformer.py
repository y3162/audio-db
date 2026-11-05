"""TF-Locoformer matches the standalone official separator and the DNS frontend."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

from src.se import available_enhancers, build_model, load_enhancer
from src.training.se.spectral import multi_resolution_l1

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
