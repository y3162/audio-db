"""Test data module that feeds trainers from the common audio-pair contract.

This is not a dataset adapter. It either synthesizes waveforms or copies one
pair already produced by a dataset adapter. Trainers see only the contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.data.pair import AudioPairData

ROOT = Path(__file__).resolve().parents[2]


class AudioPairBridge:
    """YAML-constructable stand-in whose identity declares rate and teacher."""

    def __init__(
        self,
        *,
        sample_rate: int,
        teacher: str,
        length: int,
        count: int = 2,
        seed: int = 0,
        origin: str = "synthetic",
    ) -> None:
        if origin == "synthetic":
            generator = np.random.default_rng(seed)
            pairs = tuple(
                (
                    (generator.standard_normal(length) * 0.05).astype(np.float32),
                    (generator.standard_normal(length) * 0.05).astype(np.float32),
                )
                for _ in range(count)
            )
            rate = sample_rate
            kind = teacher
        else:
            noisy, target, rate, kind = _adapter_pair(origin, length)
            if rate != sample_rate or kind != teacher:
                raise ValueError(f"{origin} produced {rate} Hz {kind}, not {sample_rate} Hz {teacher}")
            pairs = tuple((noisy, target) for _ in range(count))
        self._inner = AudioPairData(pairs, sample_rate=rate, teacher=kind, batch_size=1)
        self.origin = origin

    @property
    def epoch(self) -> int:
        return self._inner.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._inner.epoch = value

    @property
    def update_index(self) -> int:
        return self._inner.update_index

    @update_index.setter
    def update_index(self, value: int) -> None:
        self._inner.update_index = value

    @property
    def group_size(self) -> int:
        return self._inner.group_size

    @property
    def plan_hash(self) -> str:
        return self._inner.plan_hash

    def setup(self, runtime: object) -> None:
        self._inner.setup(runtime)

    def identity(self) -> dict[str, object]:
        identity = self._inner.identity()
        identity["origin"] = self.origin
        return identity

    def train_epoch(self, epoch: int, *, start_update: int):
        return self._inner.train_epoch(epoch, start_update=start_update)

    def validation_epoch(self, epoch: int):
        return self._inner.validation_epoch(epoch)

    def state_dict(self) -> dict[str, object]:
        return self._inner.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._inner.load_state_dict(state)

    def close(self) -> None:
        self._inner.close()


def _adapter_pair(origin: str, length: int) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Render one real pair and crop it to the model step length."""

    import torch

    from src.experiments.data.chunk import ChunkProtocol, ChunkSampler
    from src.experiments.data.fixed_pair import (
        FixedPairProtocol,
        MicroBatcher,
        OrderSampler,
        SnapshotDataModule,
        SnapshotSource,
    )

    class _Runtime:
        device = torch.device("cpu")
        world_size = 1
        rank = 0

    if origin == "voicebank":
        source = SnapshotSource("mp_senet.vb28.ondemand.official@1", ROOT / "data/db/snapshots-voicebank-float-v1", "train")
        protocol = FixedPairProtocol(source)
        module = SnapshotDataModule(
            source,
            OrderSampler(source, protocol, seed=0),
            protocol,
            MicroBatcher(1, 1),
        )
    elif origin == "ms_snsd":
        source = SnapshotSource("tf_gridnet.ms_snsd.espnet_recipe@1", ROOT / "data/db/snapshots", "train")
        protocol = ChunkProtocol(source)
        sampler = ChunkSampler(
            source,
            seed=0,
            num_samples_per_epoch=1,
            chunk_length=max(length, 8000),
            shift_ratio=0.5,
            num_cache_chunks=4,
            shuffle=False,
            global_batch=1,
        )
        module = SnapshotDataModule(source, sampler, protocol, MicroBatcher(1, 1))
    else:
        raise ValueError(f"unknown audio pair origin {origin}")
    module.setup(_Runtime())
    try:
        batch = module.train_epoch(0, start_update=0).batch(0)[0]
        noisy = batch.inputs[0].detach().cpu().numpy()
        target = batch.targets[0].detach().cpu().numpy()
        valid = int(batch.lengths[0]) if batch.lengths is not None else int(noisy.shape[0])
        noisy = np.ascontiguousarray(noisy[:valid], dtype=np.float32)
        target = np.ascontiguousarray(target[:valid], dtype=np.float32)
        if noisy.shape[0] < length or target.shape[0] < length:
            raise ValueError(f"{origin} pair is shorter than {length}")
        return noisy[:length], target[:length], int(batch.sample_rate), str(batch.teacher_kind)
    finally:
        module.close()
