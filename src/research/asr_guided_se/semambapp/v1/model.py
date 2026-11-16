"""Research generator. Features off keeps the official SEMamba++ module and weights."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from src.experiments.calc.rng import mix_seed, seed_model_init
from src.experiments.errors import BuildError
from src.models.semambapp.model import SEMambaPPGenerator, network_config
from src.research.asr_guided_se.semambapp.v1.config import ASR_HOP, STFT_HOP, GuidanceV1
from src.research.asr_guided_se.semambapp.v1.network import GuidedSEMambaPP


class GuidedSEMambaPPGenerator(SEMambaPPGenerator):
    """Official generator, with ASR fusion only when that switch is on.

    The ASR object is stored in a list so it is not a registered submodule and
    does not enter ``parameters()`` or the generator optimizer.
    """

    def __init__(self, *, guidance: GuidanceV1, asr: Any = None, **kwargs: Any) -> None:
        if not isinstance(guidance, GuidanceV1):
            raise BuildError("guidance must be GuidanceV1")
        super().__init__(**kwargs)
        self.guidance = guidance
        self._asr_holder: list[Any] = []
        self._batch_lookup: Callable[[torch.Tensor], Any] | None = None
        if guidance.asr_features_enabled:
            self._install_features(asr)

    @property
    def asr(self) -> Any | None:
        return self._asr_holder[0] if self._asr_holder else None

    def bind_batch_lookup(self, lookup: Callable[[torch.Tensor], Any] | None) -> None:
        self._batch_lookup = lookup

    def train(self, mode: bool = True) -> GuidedSEMambaPPGenerator:
        super().train(mode)
        for asr in self._asr_holder:
            asr.eval()
            for parameter in asr.parameters():
                parameter.requires_grad_(False)
        return self

    def forward(self, clean: torch.Tensor, noisy: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.guidance.asr_features_enabled:
            return super().forward(clean, noisy)
        asr = self.asr
        if asr is None:
            raise RuntimeError("ASR features are enabled but the frozen ASR is missing")
        lengths = self._content_lengths(clean)
        with torch.no_grad():
            hidden, encoded_length = asr.encode_layer(noisy, lengths, self.guidance.encoder_layer)
        self.generator.set_pending(hidden.detach(), encoded_length.detach())
        try:
            return super().forward(clean, noisy)
        finally:
            self.generator.clear_pending()

    def _install_features(self, asr: Any) -> None:
        if asr is None:
            raise BuildError("ASR features require the frozen Parakeet component")
        if int(asr.samples_per_frame) != ASR_HOP:
            raise BuildError(
                f"ASR samples per encoder frame must be {ASR_HOP}, got {asr.samples_per_frame}"
            )
        if int(self.hop) != STFT_HOP:
            raise BuildError(f"SEMamba++ hop must be {STFT_HOP} when ASR features are on, got {self.hop}")
        layers = int(asr.encoder_layers)
        if self.guidance.encoder_layer >= layers:
            raise BuildError(f"encoder_layer {self.guidance.encoder_layer} is outside 0..{layers - 1}")
        official_state = self.generator.state_dict()
        with seed_model_init(mix_seed("asr-guidance-scratch", self.seed)):
            guided = GuidedSEMambaPP(
                network_config(self.architecture),
                int(asr.encoder_dim),
                self.guidance.encoder_layer,
            )
        missing, unexpected = guided.load_state_dict(official_state, strict=False)
        if unexpected or any(not _guidance_key(key) for key in missing):
            raise BuildError(f"ASR fusion state copy is incomplete: missing {missing}, unexpected {unexpected}")
        with seed_model_init(mix_seed("asr-guidance-v1", self.seed, self.guidance.encoder_layer)):
            guided.asr_proj.reset_parameters()
            guided.asr_fuse.reset_parameters()
        guided.encoder_layer_index.fill_(int(self.guidance.encoder_layer))
        self.generator = guided
        self._asr_holder = [asr]
        asr.eval()
        for parameter in asr.parameters():
            parameter.requires_grad_(False)

    def _content_lengths(self, clean: torch.Tensor) -> torch.Tensor:
        batch = self._batch_lookup(clean) if self._batch_lookup is not None else None
        spans = None if batch is None else getattr(batch, "spans", None)
        if spans:
            if len(spans) != clean.shape[0]:
                raise RuntimeError("ASR feature spans do not match the batch")
            values = [int(span[1]) for span in spans]
            return torch.tensor(values, device=clean.device, dtype=torch.long)
        return torch.full((clean.shape[0],), int(clean.shape[-1]), device=clean.device, dtype=torch.long)


def _guidance_key(key: str) -> bool:
    return key == "encoder_layer_index" or key.startswith("asr_proj.") or key.startswith("asr_fuse.")
