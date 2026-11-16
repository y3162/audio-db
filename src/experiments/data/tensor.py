"""In-memory supervised pairs for linear experiments.

Tensors are a pure function of the constructor arguments. Reading a batch does
not advance a cursor, so the same update index returns the same tensors.

``batch_size`` is the number of examples one rank consumes in one micro-batch.
The global micro-batch is ``batch_size * world_size``. This class slices that
block by rank. A second sampler must not shard the same pairs again.
"""

from __future__ import annotations

import hashlib

import torch

from src.experiments.data.sample import Batch
from src.experiments.errors import StateError


class EpochView:
    """Updates available at one epoch, addressed by update index."""

    def __init__(self, data: TensorPairData, epoch: int, start_update: int, updates: int) -> None:
        self.updates = updates
        self.plan_hash = data.plan_hash
        self._data = data
        self._epoch = epoch
        self._start = start_update

    def batch(self, update_index: int) -> list[Batch]:
        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise StateError(f"update index {update_index!r} is not an integer")
        if update_index < self._start:
            raise StateError(f"update index {update_index} is before start_update {self._start}")
        if update_index < 0 or update_index >= self.updates:
            raise StateError(f"update index {update_index} is outside 0..{self.updates - 1}")
        stride = self._data.batch_size * self._data.world_size
        block = update_index * stride
        start = block + self._data.rank * self._data.batch_size
        end = min(start + self._data.batch_size, block + stride, self._data.examples)
        if start >= end:
            return []
        sample_ids = [f"ex-{index}" for index in range(start, end)]
        sample_id = ",".join(sample_ids)
        return [
            Batch(
                inputs=self._data.inputs[start:end],
                targets=self._data.targets[start:end],
                sample_id=sample_id,
                realization_id=f"epoch-{self._epoch}:{sample_id}",
            )
        ]


class TensorPairData:
    """Fixed random pairs of shape ``[examples, features]``.

    The private generator does not consume the process seed used for module
    initialization. ``seed`` defaults to 0 so the published linear experiment
    is deterministic without an extra data argument.
    """

    def __init__(
        self,
        features: int,
        examples: int,
        batch_size: int,
        seed: int = 0,
        target_features: int | None = None,
        drop_last: bool = False,
    ) -> None:
        self.features = _positive("features", features)
        self.examples = _positive("examples", examples)
        self.batch_size = _positive("batch_size", batch_size)
        self.seed = _non_negative("seed", seed)
        self.target_features = self.features if target_features is None else _positive("target_features", target_features)
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last must be a boolean")
        self.drop_last = drop_last
        if self.examples < self.batch_size:
            raise ValueError("examples must be at least batch_size")
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.inputs = torch.randn(self.examples, self.features, generator=generator)
        self.targets = torch.randn(self.examples, self.target_features, generator=generator)
        self.epoch = 0
        self.update_index = 0
        self.world_size = 1
        self.rank = 0
        self.plan_hash = self._plan_hash()
        self._closed = False

    @property
    def global_batch_size(self) -> int:
        """Examples in one micro-batch across every rank, before accumulation."""

        return self.batch_size * self.world_size

    def identity(self) -> dict[str, object]:
        return {
            "class": "src.experiments.data.TensorPairData",
            "features": self.features,
            "target_features": self.target_features,
            "examples": self.examples,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "world_size": self.world_size,
        }

    def setup(self, runtime: object) -> None:
        device = getattr(runtime, "device", torch.device("cpu"))
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)
        world_size = getattr(runtime, "world_size", 1)
        rank = getattr(runtime, "rank", 0)
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise StateError("runtime world_size must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= world_size:
            raise StateError("runtime rank is outside this world")
        self.world_size = world_size
        self.rank = rank
        self.plan_hash = self._plan_hash()

    def train_epoch(self, epoch: int, *, start_update: int) -> EpochView:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError(f"epoch {epoch!r} is not a valid index")
        stride = self.global_batch_size
        if self.drop_last:
            updates = self.examples // stride
        else:
            updates = (self.examples + stride - 1) // stride
        if isinstance(start_update, bool) or not isinstance(start_update, int):
            raise StateError(f"start_update {start_update!r} is not an integer")
        if start_update < 0 or start_update > updates:
            raise StateError(f"start_update {start_update} is outside 0..{updates}")
        return EpochView(self, epoch, start_update, updates)

    def state_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "update_index": self.update_index,
            "plan_hash": self.plan_hash,
            "world_size": self.world_size,
            "identity": self.identity(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        epoch = state.get("epoch")
        update_index = state.get("update_index")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError("data epoch is invalid")
        if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
            raise StateError("data update_index is invalid")
        saved_identity = state.get("identity")
        saved_world = state.get("world_size")
        if saved_identity == self.identity() and saved_world == self.world_size:
            if state.get("plan_hash") != self.plan_hash:
                raise StateError("data plan_hash does not match this data module")
            self.epoch = epoch
            self.update_index = update_index
            return
        if not _same_data_except_rank_width(saved_identity, self.identity()):
            raise StateError("data state does not match this data module")
        if not isinstance(saved_identity, dict) or not isinstance(saved_world, int) or isinstance(saved_world, bool):
            raise StateError("data state does not match this data module")
        saved_batch = saved_identity.get("batch_size")
        if (
            isinstance(saved_batch, bool)
            or not isinstance(saved_batch, int)
            or saved_batch * saved_world != self.batch_size * self.world_size
        ):
            raise StateError("global batch is not maintained")
        finished = tensor_epoch_updates(saved_identity, saved_world)
        if update_index != 0 and (finished is None or update_index != finished):
            raise StateError("world size can change only at an epoch boundary")
        if finished is not None and finished > 0 and update_index == finished:
            epoch += 1
            update_index = 0
        self.epoch = epoch
        self.update_index = update_index

    def close(self) -> None:
        self._closed = True

    def _plan_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.inputs.detach().cpu().numpy().tobytes())
        digest.update(self.targets.detach().cpu().numpy().tobytes())
        digest.update(str(self.batch_size).encode("utf-8"))
        digest.update(str(self.world_size).encode("utf-8"))
        digest.update(b"drop_last" if self.drop_last else b"keep_tail")
        return digest.hexdigest()


def tensor_epoch_updates(identity: dict[str, object], world_size: int) -> int | None:
    """Return how many updates the saved identity had in one epoch."""

    examples = identity.get("examples")
    batch = identity.get("batch_size")
    drop_last = identity.get("drop_last", False)
    if isinstance(examples, bool) or not isinstance(examples, int) or examples < 1:
        return None
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        return None
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        return None
    stride = batch * world_size
    if drop_last is True:
        return examples // stride
    if drop_last is False:
        return (examples + stride - 1) // stride
    return None


def _same_data_except_rank_width(saved: object, live: object) -> bool:
    if not isinstance(saved, dict) or not isinstance(live, dict):
        return False
    left = dict(saved)
    right = dict(live)
    left.pop("batch_size", None)
    left.pop("world_size", None)
    right.pop("batch_size", None)
    right.pop("world_size", None)
    return left == right


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return value
