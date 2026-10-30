"""Build and run DeepFilterNet3 without libdf."""
import configparser
from pathlib import Path

import torch
import torchaudio.functional as AF

from ...core.audio import Enhancer
from ...core.checkpoint import checkpoint_paths
from ...core.config import load_config
from .dsp import DFStream, erb_band_widths, rectangular_erb
from .model import DFN3Params, DfNet

_FIELDS = {name: type(value) for name, value in DFN3Params().__dict__.items()}


def _cast(name, value):
    expected = _FIELDS[name]
    if expected is tuple:
        return _pair(value)
    if expected is bool:
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes')
        return bool(value)
    return expected(value)


def params_from(config):
    values = {}
    if config is None:
        pass
    elif isinstance(config, (str, Path)) and Path(config).suffix.lower() == '.ini':
        parser = configparser.ConfigParser()
        parser.read(config)
        flat = {}
        for section in parser.sections():
            flat.update(parser[section])
        aliases = {
            'sr': 'sr', 'fft_size': 'fft_size', 'hop_size': 'hop_size', 'nb_erb': 'nb_erb',
            'nb_df': 'nb_df', 'norm_tau': 'norm_tau', 'lsnr_max': 'lsnr_max', 'lsnr_min': 'lsnr_min',
            'min_nb_erb_freqs': 'min_nb_freqs', 'df_order': 'df_order', 'df_lookahead': 'df_lookahead',
            'conv_lookahead': 'conv_lookahead', 'conv_ch': 'conv_ch', 'emb_hidden_dim': 'emb_hidden_dim',
            'emb_num_layers': 'emb_num_layers', 'enc_linear_groups': 'enc_lin_groups',
            'linear_groups': 'lin_groups', 'df_hidden_dim': 'df_hidden_dim', 'df_num_layers': 'df_num_layers',
            'df_gru_skip': 'df_gru_skip', 'df_pathway_kernel_size_t': 'df_pathway_kernel_size_t',
            'enc_concat': 'enc_concat', 'conv_kernel': 'conv_kernel', 'convt_kernel': 'convt_kernel',
            'conv_kernel_inp': 'conv_kernel_inp', 'emb_gru_skip': 'emb_gru_skip',
            'df_n_iter': 'df_n_iter', 'mask_pf': 'mask_pf', 'pf_beta': 'pf_beta',
            'lsnr_dropout': 'lsnr_dropout',
        }
        unknown = []
        for key, value in flat.items():
            if key in aliases:
                values[aliases[key]] = value
            elif key in _FIELDS:
                values[key] = value
            # Official ini files also contain training, loss, and unused architecture flags.
    else:
        raw = load_config(config)
        extra = set(raw) - set(_FIELDS)
        if extra:
            raise ValueError(f'unknown DeepFilterNet3 config keys: {sorted(extra)}')
        values = raw
    params = DFN3Params()
    for name, value in values.items():
        setattr(params, name, _cast(name, value))
    params.conv_kernel = _pair(params.conv_kernel)
    params.convt_kernel = _pair(params.convt_kernel)
    params.conv_kernel_inp = _pair(params.conv_kernel_inp)
    return params


def _pair(value):
    if isinstance(value, str):
        value = [part for part in value.replace(' ', '').split(',') if part]
    return tuple(int(part) for part in value)


def build(config, variant=None):
    if variant not in (None, 'deepfilternet3'):
        raise ValueError(f'unsupported DeepFilterNet3 variant: {variant!r}')
    params = params_from(config)
    widths = erb_band_widths(params.sr, params.fft_size, params.nb_erb, params.min_nb_freqs)
    erb = rectangular_erb(widths, params.sr, inverse=False)
    inverse = rectangular_erb(widths, params.sr, inverse=True)
    return DfNet(erb, inverse, params)


def load(checkpoint, device, config, variant):
    model = build(config, variant)
    path = checkpoint_paths(checkpoint, ('model',))['model']
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f'expected a DeepFilterNet3 state dictionary in {path}')
    model.load_state_dict(state, strict=True)
    device = torch.device(device)
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    return DFEnhancer(model, model.params, device)


class DFEnhancer(Enhancer):
    def __init__(self, model, params, device):
        self.model, self.params, self.device = model, params, device
        self.stream = DFStream(
            params.sr, params.fft_size, params.hop_size, params.nb_erb, params.nb_df,
            params.min_nb_freqs, params.norm_tau, device)

    def reset_state(self):
        """Drop STFT overlap and normalization state so the next call is independent."""
        self.stream.reset()

    def _enhance_one(self, waveform, sample_rate):
        self.reset_state()
        original_device, original_dtype, length = waveform.device, waveform.dtype, waveform.numel()
        audio = waveform.to(device=self.device, dtype=torch.float32)
        if sample_rate != self.params.sr:
            audio = AF.resample(audio, sample_rate, self.params.sr)
        enhanced = self._enhance_native(audio)
        if sample_rate != self.params.sr:
            enhanced = AF.resample(enhanced, self.params.sr, sample_rate)
        if enhanced.numel() < length:
            enhanced = torch.nn.functional.pad(enhanced, (0, length - enhanced.numel()))
        return enhanced[:length].to(device=original_device, dtype=original_dtype)

    def _enhance_native(self, audio):
        params = self.params
        original = audio.shape[-1]
        padded = torch.nn.functional.pad(audio, (0, params.fft_size)).unsqueeze(0)
        spec, erb, feat = self.stream.features(self.stream.analyze(padded))
        with torch.inference_mode():
            enhanced = self.model(spec, erb, feat)[0]
        waveform = self.stream.synthesize(torch.view_as_complex(enhanced.squeeze(1)))
        delay = params.fft_size - params.hop_size
        return waveform[0, delay:original + delay]
