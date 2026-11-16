"""Load an ASR-guided SEMamba++ v1 checkpoint into the existing enhancer API.

A state dict without the fusion layers is the official network and does not
load Parakeet. A state dict with those layers loads the frozen TDT v2 model
at enhance time and fuses the configured encoder layer.
"""

from __future__ import annotations

import torch
import torchaudio.functional as AF

from src.asr.models.parakeet import TDT_V2_MODEL_ID
from src.models.core.audio import Enhancer
from src.models.core.checkpoint import checkpoint_paths
from src.models.core.config import load_config
from src.models.semambapp.inference import (
    COMPRESS,
    HOP,
    N_FFT,
    SAMPLE_RATE,
    WIN,
    _enhance_16k,
)
from src.models.semambapp.network import SEMambaPP
from src.research.asr_guided_se.semambapp.v1.config import ASR_HOP
from src.research.asr_guided_se.semambapp.v1.network import GuidedSEMambaPP

_DEFAULTS = dict(
    hid_feature=48,
    num_tfmamba=4,
    d_state=16,
    d_conv=4,
    expand=2,
    norm_epsilon=1e-5,
    input_channel=2,
    output_channel=1,
    n_fft=N_FFT,
    scan_backend="reference",
)
_OPTIONAL = {"encoder_layer"}


def build(config, variant=None):
    if variant not in (None, "v1"):
        raise ValueError(f"unsupported ASR-guided SEMamba++ variant: {variant!r}")
    values = _read_config(config)
    return SEMambaPP(values)


def load(checkpoint, device, config, variant):
    if variant not in (None, "v1"):
        raise ValueError(f"unsupported ASR-guided SEMamba++ variant: {variant!r}")
    values = _read_config(config)
    path = checkpoint_paths(checkpoint, ("generator",))["generator"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "generator" in state and isinstance(state["generator"], dict):
        state = state["generator"]
    if not isinstance(state, dict):
        raise ValueError("ASR-guided SEMamba++ checkpoint has no generator state")
    guided = any(key.startswith("asr_proj.") or key.startswith("asr_fuse.") for key in state)
    device = torch.device(device)
    if not guided:
        model = SEMambaPP(values)
        model.load_state_dict(state, strict=True)
        return _PlainEnhancer(model.to(device).eval(), device)
    if "encoder_layer_index" not in state:
        raise ValueError("guided checkpoint is missing encoder_layer_index")
    saved_layer = int(state["encoder_layer_index"])
    if "encoder_layer" in load_config(config) and int(values.get("encoder_layer", saved_layer)) != saved_layer:
        raise ValueError("encoder_layer does not match the checkpoint")
    weight = state.get("asr_proj.weight")
    if weight is None:
        raise ValueError("guided checkpoint is missing asr_proj.weight")
    encoder_dim = int(weight.shape[1])
    model = GuidedSEMambaPP(values, encoder_dim, saved_layer)
    model.load_state_dict(state, strict=True)
    return _GuidedEnhancer(model.to(device).eval(), device, saved_layer)


def _read_config(config) -> dict:
    values = dict(_DEFAULTS)
    raw = load_config(config)
    unknown = set(raw) - set(_DEFAULTS) - _OPTIONAL
    if unknown:
        raise ValueError(f"unknown ASR-guided SEMamba++ config keys: {sorted(unknown)}")
    values.update(raw)
    if int(values["n_fft"]) != N_FFT:
        raise ValueError(f"SEMamba++ n_fft must be {N_FFT}, got {values['n_fft']!r}")
    if "encoder_layer" in values:
        layer = values["encoder_layer"]
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError("encoder_layer must be an integer >= 0")
    return values


class _PlainEnhancer(Enhancer):
    def __init__(self, model, device):
        self.model, self.device = model, device

    def _enhance_one(self, waveform, sample_rate):
        return _enhance_waveform(self.model, waveform, sample_rate, self.device, prepare=None)


class _GuidedEnhancer(Enhancer):
    def __init__(self, model, device, encoder_layer: int):
        self.model = model
        self.device = device
        self.encoder_layer = int(encoder_layer)
        self._asr = None

    def _enhance_one(self, waveform, sample_rate):
        return _enhance_waveform(self.model, waveform, sample_rate, self.device, prepare=self._prepare)

    def _prepare(self, audio: torch.Tensor) -> None:
        asr = self._load_asr()
        lengths = torch.tensor([audio.shape[0]], device=audio.device, dtype=torch.long)
        with torch.no_grad():
            hidden, encoded_length = asr.encode_layer(audio.view(1, -1), lengths, self.encoder_layer)
        self.model.set_pending(hidden.detach(), encoded_length.detach())

    def _load_asr(self):
        if self._asr is None:
            from src.asr.models.parakeet_tdt import ParakeetTDT

            asr = ParakeetTDT(TDT_V2_MODEL_ID).to(self.device).eval()
            if int(asr.samples_per_encoder_frame) != ASR_HOP:
                raise RuntimeError(
                    f"ASR samples per encoder frame must be {ASR_HOP}, got {asr.samples_per_encoder_frame}"
                )
            if self.encoder_layer >= int(asr.encoder_layers):
                raise RuntimeError("encoder_layer is outside the ASR encoder")
            self._asr = asr
        return self._asr


def _enhance_waveform(model, waveform, sample_rate, device, prepare):
    original_device, original_dtype, length = waveform.device, waveform.dtype, waveform.numel()
    audio = waveform.to(device=device, dtype=torch.float32)
    if sample_rate != SAMPLE_RATE:
        audio = AF.resample(audio, sample_rate, SAMPLE_RATE)
    if not torch.any(audio):
        return waveform.clone()
    if audio.numel() <= N_FFT // 2:
        raise ValueError(
            f"input is too short for reflect-padded STFT (need > {N_FFT // 2} samples at {SAMPLE_RATE} Hz)"
        )
    peak = audio.abs().max().clamp_min(1e-9)
    with torch.inference_mode():
        if prepare is not None:
            prepare(audio / peak)
        try:
            enhanced = _enhance_16k(model, audio / peak) * peak
        finally:
            clear = getattr(model, "clear_pending", None)
            if callable(clear):
                clear()
    if sample_rate != SAMPLE_RATE:
        enhanced = AF.resample(enhanced, SAMPLE_RATE, sample_rate)
    if enhanced.numel() < length:
        enhanced = torch.nn.functional.pad(enhanced, (0, length - enhanced.numel()))
    return enhanced[:length].to(device=original_device, dtype=original_dtype)
