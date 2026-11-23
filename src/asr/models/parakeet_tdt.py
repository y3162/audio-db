"""Frozen ``nvidia/parakeet-tdt-0.6b-v2`` encode, token times, and TDT loss.

The text API stays in ``parakeet.py``. This module is imported by the research
path when that model is constructed. Importing ``src.asr`` does not import it.

``train(True)`` does not unfreeze the checkpoint. Dropout stays off. The
decoder is put in train mode only while TDT loss is enabled, so the joint
backward reaches the waveform, and every dropout module stays in eval.
``TDTLossNumba`` is built on the first request. Its blank, durations, sigma,
and omega are read from the checkpoint. There is no second TDT implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.asr.models.parakeet import TDT_V2_MODEL_ID, apply_tdt_v2_decoding

_PATCHED = False


@dataclass(frozen=True)
class Token:
    token_id: int
    token: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class EncodeResult:
    encoded: torch.Tensor
    encoded_length: torch.Tensor
    layer_outputs: dict[int, torch.Tensor]


@dataclass(frozen=True)
class Recognition:
    text: str
    encoded_length: int
    samples_per_encoder_frame: int
    tokens: tuple[Token, ...]


def _local_nemo(model_id: str) -> str:
    """Return the cached ``.nemo`` file. A missing cache is an error, not a download."""

    filename = model_id.split("/")[-1] + ".nemo"
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=model_id, filename=filename, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"{model_id} is not available as a local {filename}. Refusing to download it."
        ) from exc


def clip_encoded_time(
    encoded: torch.Tensor,
    encoded_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make encoded T equal ``max(encoded_length)`` so the TDT length check passes."""

    if encoded.ndim != 3:
        raise ValueError(f"encoded must be [B, D, T], got {tuple(encoded.shape)}")
    time = encoded.size(-1)
    clipped_length = encoded_length.to(dtype=torch.int64).clamp(min=0, max=time)
    cut = int(clipped_length.max().item()) if clipped_length.numel() else 0
    if time != cut:
        encoded = encoded[:, :, :cut]
    return encoded, clipped_length


def install_normalize_patch() -> None:
    """Replace NeMo's in-place std update on the TDT gradient path.

    ``x_std += CONSTANT`` after ``sqrt`` breaks autograd into the waveform.
    The replacement adds the constant out of place. Other normalize modes call
    the original function. The patch is installed when TDT loss is enabled,
    not when ``src.asr`` is imported and not for CTC, RNN-T, or Whisper.
    """

    global _PATCHED
    if _PATCHED:
        return
    from nemo.collections.asr.parts.preprocessing import features as nemo_features

    original = nemo_features.normalize_batch

    def _normalize_batch_without_inplace(x, seq_len, normalize_type):
        if normalize_type != "per_feature":
            return original(x, seq_len, normalize_type)
        batch_size = x.shape[0]
        max_time = x.shape[2]
        if (
            torch.cuda.is_available()
            and not torch.cuda.is_current_stream_capturing()
            and torch.any(seq_len == 1).item()
        ):
            raise ValueError(
                "normalize_batch with `per_feature` normalize_type received a tensor of length 1. "
                "This will result in torch.std() returning nan."
            )
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
        x_mean_denominator = valid_mask.sum(axis=1)
        x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)
        x_std = torch.sqrt(
            torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
            / (x_mean_denominator.unsqueeze(1) - 1.0)
        )
        x_std = x_std + nemo_features.CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std

    nemo_features.normalize_batch = _normalize_batch_without_inplace
    _PATCHED = True


def configure_numba_for_tdt() -> dict[str, Any]:
    """Relax NeMo's Numba CUDA matrix only when this host fails the strict check.

    The flag is process-wide. It is set from TDT loss construction. CTC, RNN-T,
    and Whisper do not call this. When CUDA is absent the flag is left alone;
    a CPU TDT kernel that cannot be built still raises instead of selecting
    another loss.
    """

    from nemo.core.utils import numba_utils

    strict_before = bool(numba_utils.is_numba_compat_strict())
    relaxed = False
    reason = "strict Numba CUDA check left unchanged"
    if strict_before and torch.cuda.is_available():
        supported = bool(numba_utils.numba_cuda_is_supported(numba_utils.__NUMBA_MINIMUM_VERSION__))
        if not supported:
            numba_utils.set_numba_compat_strictness(False)
            relaxed = True
            supported_relaxed = bool(
                numba_utils.numba_cuda_is_supported(numba_utils.__NUMBA_MINIMUM_VERSION__)
            )
            if supported_relaxed:
                reason = "strict CUDA compatibility matrix failed; relaxed for this process TDT path"
            else:
                reason = "Numba CUDA is unavailable after relaxing the compatibility matrix"
    return {
        "strict_before": strict_before,
        "relaxed": relaxed,
        "strict_after": bool(numba_utils.is_numba_compat_strict()),
        "reason": reason,
    }


class ParakeetTDT(nn.Module):
    """One frozen Parakeet TDT v2 checkpoint."""

    def __init__(self, model_id: str = TDT_V2_MODEL_ID) -> None:
        super().__init__()
        if model_id != TDT_V2_MODEL_ID:
            raise ValueError(f"ParakeetTDT only loads {TDT_V2_MODEL_ID}, got {model_id}")
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

        self.model_id = model_id
        self._filterbank = FilterbankFeatures
        self.model = nemo_asr.models.ASRModel.restore_from(_local_nemo(model_id), map_location="cpu")
        self.model.preprocessor.featurizer.use_grads = True
        self.effective_decoding = apply_tdt_v2_decoding(self.model)
        self._read_tdt_checkpoint()
        hop = int(self.model.preprocessor.featurizer.hop_length)
        subsampling = int(self.model.encoder.subsampling_factor)
        self.samples_per_encoder_frame = hop * subsampling
        self.encoder_dim = int(self.model.encoder.d_model)
        self.encoder_layers = int(len(self.model.encoder.layers))
        self._tdt_loss = None
        self._deterministic_tdt_loss = None
        self._numba: dict[str, Any] | None = None
        self._tdt_enabled = False
        self.eval()

    def _read_tdt_checkpoint(self) -> None:
        loss = getattr(self.model, "loss", None)
        blank = getattr(loss, "_blank", None)
        tdt_kwargs = getattr(getattr(self.model, "cfg", None), "loss", None)
        tdt_kwargs = getattr(tdt_kwargs, "tdt_kwargs", None)
        missing = [
            name
            for name, present in (
                ("blank", blank is not None),
                ("durations", tdt_kwargs is not None and "durations" in tdt_kwargs),
                ("sigma", tdt_kwargs is not None and "sigma" in tdt_kwargs),
                ("omega", tdt_kwargs is not None and "omega" in tdt_kwargs),
            )
            if not present
        ]
        if missing:
            raise RuntimeError(
                "Parakeet TDT v2 checkpoint is missing "
                + ", ".join(missing)
                + "; refusing to hardcode TDT settings"
            )
        self.blank_id = int(blank)
        self.durations = [int(duration) for duration in tdt_kwargs.durations]
        self.sigma = float(tdt_kwargs.sigma)
        self.omega = float(tdt_kwargs.omega)
        if not self.durations:
            raise RuntimeError("Parakeet TDT v2 durations are empty")

    def train(self, mode: bool = True) -> nn.Module:
        """Stay in eval with frozen parameters. ``mode`` does not unfreeze them."""

        del mode
        super().train(False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self._tdt_enabled:
            self._set_decoder_backward_enabled(True)
        else:
            self.model.decoder.eval()
        return self

    def enable_tdt_loss(self) -> None:
        """Build ``TDTLossNumba`` from the checkpoint. A second call is a no-op."""

        if self._tdt_enabled:
            return
        from nemo.collections.asr.parts.numba.rnnt_loss import TDTLossNumba

        self._numba = configure_numba_for_tdt()
        install_normalize_patch()
        kwargs = {
            "blank": self.blank_id,
            "durations": list(self.durations),
            "reduction": "none",
            "sigma": self.sigma,
            "fastemit_lambda": 0.0,
            "clamp": -1.0,
        }
        try:
            self._tdt_loss = TDTLossNumba(**kwargs, omega=self.omega)
            self._deterministic_tdt_loss = (
                self._tdt_loss if self.omega == 0.0 else TDTLossNumba(**kwargs, omega=0.0)
            )
        except Exception as exc:
            raise RuntimeError(
                "TDTLossNumba could not be built. Another TDT loss is not substituted."
            ) from exc
        self._tdt_enabled = True
        self.train(False)

    def software_fingerprint(self) -> dict[str, Any]:
        import numba

        import nemo

        device = next(self.model.parameters()).device
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        return {
            "model_id": self.model_id,
            "torch": torch.__version__,
            "nemo": nemo.__version__,
            "numba": numba.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu,
            "decoding": dict(self.effective_decoding),
            "samples_per_encoder_frame": self.samples_per_encoder_frame,
            "blank": self.blank_id,
            "durations": list(self.durations),
            "sigma": self.sigma,
            "omega": self.omega,
            "tdt_loss_enabled": self._tdt_enabled,
            "numba_compat": None if self._numba is None else dict(self._numba),
        }

    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> EncodeResult:
        self._install_deterministic_stft()
        layer_indices = self._resolve_layer_indices(layers)
        batch_size = int(waveforms.shape[0])
        valid = lengths > 0
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            encoded = waveforms.new_zeros(batch_size, self.encoder_dim, 0) + waveforms.sum() * 0
            encoded_length = torch.zeros(batch_size, device=lengths.device, dtype=torch.int64)
            return EncodeResult(
                encoded=encoded,
                encoded_length=encoded_length,
                layer_outputs={index: encoded for index in layer_indices},
            )
        idx = None
        waveforms_valid = waveforms
        lengths_valid = lengths
        if n_valid != batch_size:
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms_valid = waveforms.index_select(0, idx)
            lengths_valid = lengths.index_select(0, idx)
        features, feature_lengths = self._filterbank.forward(
            self.model.preprocessor.featurizer,
            waveforms_valid.to(dtype=torch.float32),
            lengths_valid,
        )
        layer_outputs_valid: dict[int, torch.Tensor] = {}
        handles = []

        def capture_layer(layer_index: int):
            def hook(_module, _args, out):
                layer_outputs_valid[layer_index] = _layer_tensor(out, self.encoder_dim)

            return hook

        try:
            for layer_index in layer_indices:
                handles.append(self.model.encoder.layers[layer_index].register_forward_hook(capture_layer(layer_index)))
            encoded_valid, encoded_length_valid = self.model.encoder(
                audio_signal=features,
                length=feature_lengths,
            )
        finally:
            for handle in handles:
                handle.remove()
        encoded_length_valid = encoded_length_valid.to(dtype=torch.int64)
        encoded_valid, encoded_length_valid = clip_encoded_time(encoded_valid, encoded_length_valid)
        cut = encoded_valid.size(-1)
        for layer_index, layer_tensor in list(layer_outputs_valid.items()):
            if layer_tensor.size(-1) != cut:
                layer_outputs_valid[layer_index] = layer_tensor[:, :, :cut]
        missing = [index for index in layer_indices if index not in layer_outputs_valid]
        if missing:
            raise RuntimeError(f"encoder layers did not produce outputs: {missing}")
        if idx is None:
            return EncodeResult(
                encoded=encoded_valid,
                encoded_length=encoded_length_valid,
                layer_outputs=layer_outputs_valid,
            )
        encoded = _scatter_batch(encoded_valid, idx, batch_size)
        encoded_length = encoded_length_valid.new_zeros(batch_size).scatter(0, idx, encoded_length_valid)
        layer_outputs = {
            layer_index: _scatter_batch(layer_tensor, idx, batch_size)
            for layer_index, layer_tensor in layer_outputs_valid.items()
        }
        return EncodeResult(encoded=encoded, encoded_length=encoded_length, layer_outputs=layer_outputs)

    def recognize(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> list[Recognition]:
        with torch.no_grad():
            encoded = self.encode(waveforms, lengths)
        return self.recognize_encoded(encoded.encoded, encoded.encoded_length)

    def recognize_encoded(
        self,
        encoded: torch.Tensor,
        encoded_length: torch.Tensor,
    ) -> list[Recognition]:
        batch_size = int(encoded.shape[0])
        valid = encoded_length > 0
        if not bool(valid.any()):
            return [self._empty_recognition() for _ in range(batch_size)]
        idx = None
        encoded_valid = encoded
        encoded_length_valid = encoded_length
        if not bool(valid.all()):
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            encoded_valid = encoded.index_select(0, idx)
            encoded_length_valid = encoded_length.index_select(0, idx)
        encoded_lengths = encoded_length_valid.detach().cpu().tolist()
        decoder_training = bool(self.model.decoder.training)
        self.model.decoder.eval()
        with torch.no_grad():
            predictions = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded_valid.detach(),
                encoded_lengths=encoded_length_valid,
                return_hypotheses=True,
            )
        self._set_decoder_backward_enabled(decoder_training and self._tdt_enabled)
        timed = [
            self._timed_recognition(prediction, int(encoded_lengths[i]))
            for i, prediction in enumerate(predictions)
        ]
        if idx is None:
            return timed
        results = [self._empty_recognition() for _ in range(batch_size)]
        for source, destination in enumerate(idx.tolist()):
            results[destination] = timed[source]
        return results

    def loss_from_ids(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        token_ids: Sequence[Sequence[int]],
        *,
        deterministic_tdt: bool = False,
    ) -> torch.Tensor:
        """Per-sample TDT NLL. Empty targets and zero lengths stay connected to ``waveforms``."""

        if not self._tdt_enabled or self._tdt_loss is None or self._deterministic_tdt_loss is None:
            raise RuntimeError("TDT loss is not enabled")
        batch_size = int(waveforms.shape[0])
        if len(token_ids) != batch_size:
            raise ValueError(f"token_ids length {len(token_ids)} != batch {batch_size}")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        nonempty = [len(ids) > 0 for ids in token_ids]
        valid = (lengths > 0) & torch.tensor(nonempty, device=lengths.device, dtype=torch.bool)
        if int(valid.sum().item()) == 0:
            return waveforms.new_zeros(batch_size) + waveforms.sum() * 0
        idx = None
        token_id_lists = [list(ids) for ids in token_ids]
        waveforms_valid = waveforms
        lengths_valid = lengths
        if int(valid.sum().item()) != batch_size:
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms_valid = waveforms.index_select(0, idx)
            lengths_valid = lengths.index_select(0, idx)
            token_id_lists = [token_id_lists[int(i)] for i in idx.tolist()]
        target_lengths_cpu = torch.tensor([len(ids) for ids in token_id_lists], dtype=torch.int64)
        max_length = int(target_lengths_cpu.max().item())
        targets_cpu = torch.zeros((len(token_id_lists), max_length), dtype=torch.int64)
        for row, ids in enumerate(token_id_lists):
            if ids:
                targets_cpu[row, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
        target_lengths = target_lengths_cpu.to(device=waveforms.device)
        targets = targets_cpu.to(device=waveforms.device)
        self._set_decoder_backward_enabled(True)
        encoded = self.encode(waveforms_valid, lengths_valid)
        decoder_outputs, _, _ = self.model.decoder(targets=targets, target_length=target_lengths)
        logits = self.model.joint.joint(
            encoded.encoded.transpose(1, 2),
            decoder_outputs.transpose(1, 2),
        )
        tdt_loss = self._deterministic_tdt_loss if deterministic_tdt else self._tdt_loss
        nll_valid = tdt_loss(
            acts=logits,
            labels=targets.to(dtype=torch.int64),
            act_lens=encoded.encoded_length.to(dtype=torch.int64),
            label_lens=target_lengths.to(dtype=torch.int64),
        )
        if idx is None:
            return nll_valid
        return nll_valid.new_zeros(batch_size).scatter(0, idx, nll_valid)

    def ids_to_text(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        return self.model.tokenizer.ids_to_text(list(token_ids))

    def _resolve_layer_indices(self, layers: Sequence[int] | None) -> list[int]:
        if not layers:
            return []
        resolved: list[int] = []
        seen: set[int] = set()
        for index in layers:
            layer_index = int(index)
            if layer_index < 0:
                layer_index += self.encoder_layers
            if not 0 <= layer_index < self.encoder_layers:
                raise ValueError(
                    f"encoder layer index {index} is out of range for {self.encoder_layers} layers"
                )
            if layer_index not in seen:
                seen.add(layer_index)
                resolved.append(layer_index)
        return resolved

    def _empty_recognition(self) -> Recognition:
        return Recognition(
            text="",
            encoded_length=0,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=(),
        )

    def _timed_recognition(self, prediction: Any, encoded_length: int) -> Recognition:
        token_ids = _as_ints(getattr(prediction, "y_sequence", None))
        starts = _as_ints(getattr(prediction, "timestamp", None))
        raw_durations = getattr(prediction, "token_duration", None)
        durations = _as_ints(raw_durations)
        # NeMo 2.2.1 batched_hyps_to_hypotheses encodes an all-zero row as [].
        if (
            isinstance(raw_durations, list)
            and not raw_durations
            and self.effective_decoding["tdt_include_token_duration"]
            and len(token_ids) == len(starts)
        ):
            durations = [0] * len(token_ids)
        if not (len(token_ids) == len(starts) == len(durations)):
            raise RuntimeError(
                "y_sequence/timestamp/token_duration lengths differ: "
                f"{len(token_ids)}, {len(starts)}, {len(durations)}"
            )
        kept: list[tuple[int, int, int]] = []
        for token_id, start, duration in zip(token_ids, starts, durations, strict=True):
            if token_id >= self.blank_id:
                continue
            start_offset = max(int(start), 0)
            if start_offset >= encoded_length:
                continue
            end_offset = min(start_offset + max(int(duration), 1), encoded_length)
            kept.append((token_id, start_offset, end_offset))
        token_texts = self.model.decoding.decode_ids_to_tokens([token_id for token_id, _, _ in kept])
        if len(token_texts) != len(kept):
            raise RuntimeError("token text count does not match the kept token ids")
        tokens = tuple(
            Token(token_id=token_id, token=str(token_text), start_offset=start_offset, end_offset=end_offset)
            for (token_id, start_offset, end_offset), token_text in zip(kept, token_texts, strict=True)
        )
        text = getattr(prediction, "text", None)
        if not isinstance(text, str):
            text = self.ids_to_text([token.token_id for token in tokens])
        return Recognition(
            text=text,
            encoded_length=encoded_length,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=tokens,
        )

    def _install_deterministic_stft(self) -> None:
        """Match centered STFT without CUDA reflection-pad backward.

        NeMo's featurizer uses ``torch.stft(..., center=True)`` unless
        ``exact_pad`` is set. That backward calls ``reflection_pad1d``, which
        has no deterministic CUDA implementation. An index reflection plus
        ``center=False`` matches the centered spectrum and can propagate
        gradients while deterministic algorithms stay on. ``src.asr`` does not
        import the experiment STFT helper, so the pad lives here.
        """

        featurizer = self.model.preprocessor.featurizer
        if getattr(featurizer, "_audio_db_deterministic_stft", False):
            return
        if not torch.are_deterministic_algorithms_enabled():
            return
        if bool(getattr(featurizer, "exact_pad", False)):
            raise RuntimeError("Parakeet featurizer exact_pad still uses reflection pad")
        original = featurizer.stft

        def stft(x: torch.Tensor) -> torch.Tensor:
            if not x.is_cuda:
                return original(x)
            pad = int(featurizer.n_fft) // 2
            framed = _index_reflect(x, pad)
            return torch.stft(
                framed,
                n_fft=int(featurizer.n_fft),
                hop_length=int(featurizer.hop_length),
                win_length=int(featurizer.win_length),
                center=False,
                window=featurizer.window.to(dtype=torch.float),
                return_complex=True,
            )

        featurizer.stft = stft
        featurizer._audio_db_deterministic_stft = True

    def _set_decoder_backward_enabled(self, enabled: bool) -> None:
        if enabled:
            self.model.decoder.train()
            for module in self.model.decoder.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            return
        self.model.decoder.eval()


def _index_reflect(waveform: torch.Tensor, pad: int) -> torch.Tensor:
    """Reflect ``pad`` samples on each side by indexing, not ``reflection_pad1d``."""

    if pad == 0:
        return waveform
    length = int(waveform.shape[-1])
    if length <= pad:
        raise RuntimeError(f"waveform length {length} cannot be reflect-padded by {pad} samples")
    left = waveform[..., 1 : pad + 1].flip(-1)
    right = waveform[..., -pad - 1 : -1].flip(-1)
    return torch.cat((left, waveform, right), dim=-1)


def _layer_tensor(out: Any, encoder_dim: int) -> torch.Tensor:
    value = out[0] if isinstance(out, tuple) else out
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise RuntimeError(f"encoder layer output must be a rank-3 tensor, got {type(value).__name__}")
    if value.shape[-1] == encoder_dim:
        return value.transpose(1, 2)
    if value.shape[1] == encoder_dim:
        return value
    raise RuntimeError(
        f"encoder layer output shape {tuple(value.shape)} does not contain d_model {encoder_dim}"
    )


def _scatter_batch(valid: torch.Tensor, idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    index = idx.view(-1, *([1] * (valid.ndim - 1))).expand_as(valid)
    shape = (batch_size, *valid.shape[1:])
    return valid.new_zeros(shape).scatter(0, index, valid)


def _as_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        if value.numel() == 0:
            return []
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]
