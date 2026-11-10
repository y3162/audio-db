"""Records produced by the real recipe dataset and the real Trainer.

The dataset is not replaced. A thin wrapper only watches whether collation
consumes the global random generators.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from src.data.recipe import case_id_for
from src.evaluation.audio.render import render_recipe
from src.experiments.calc.batch import BatchPlan
from src.experiments.calc.distributed import ProcessInfo
from src.experiments.calc.dataset import RecipeDataset, _mono, _normalize
from src.experiments.calc.sampler import crop_start, epoch_indices
from src.experiments.calc.errors import TrainingError


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _numpy_token(state: tuple) -> tuple:
    kind, keys, pos, has_gauss, cached = state
    return (kind, keys.tobytes(), int(pos), int(has_gauss), float(cached))


class AuditedDataset:
    """``RecipeDataset`` plus a check that collation does not draw global random numbers."""

    def __init__(self, dataset: RecipeDataset) -> None:
        self.dataset = dataset
        self.rng_draws = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def identity(self) -> dict[str, Any]:
        return self.dataset.identity()

    def collate(self, indices: list[int]) -> dict[str, Any]:
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        numpy_state = _numpy_token(np.random.get_state())
        batch = self.dataset.collate(indices)
        if (
            not torch.equal(torch_state, torch.random.get_rng_state())
            or random.getstate() != python_state
            or _numpy_token(np.random.get_state()) != numpy_state
        ):
            self.rng_draws += 1
        batch["audit"] = describe_batch(self.dataset, batch, indices)
        return batch


def describe_batch(dataset: RecipeDataset, batch: dict[str, Any], indices: list[int]) -> list[dict[str, Any]]:
    rows = []
    for position, index in enumerate(indices):
        example = dataset.examples[index]
        clean = batch["clean"][position]
        noisy = batch["noisy"][position]
        recipe_hash = case_id_for(example.recipe)
        full_noisy, full_clean = _full_pair(dataset, example)
        segment = dataset.segment_size
        if dataset.training and segment is not None and int(full_noisy.numel()) >= segment:
            start = crop_start(example.case_id, dataset.seed, dataset.epoch, int(full_noisy.numel()), segment)
            expected_noisy = full_noisy[start : start + segment]
            expected_clean = full_clean[start : start + segment]
        elif dataset.training and segment is not None:
            start = 0
            pad = segment - int(full_noisy.numel())
            expected_noisy = F.pad(full_noisy, (0, pad))
            expected_clean = F.pad(full_clean, (0, pad))
        else:
            start = 0
            expected_noisy, expected_clean = full_noisy, full_clean
        rows.append(
            {
                "epoch": dataset.epoch,
                "case_id": example.case_id,
                "recipe_sha256": recipe_hash,
                "recipe_matches_case": recipe_hash == example.case_id,
                "utterance_id": _utterance_id(example.recipe),
                "crop_start": int(start),
                "clean_length": int(clean.numel()),
                "noisy_length": int(noisy.numel()),
                "full_length": int(full_noisy.numel()),
                "clean_sha256": tensor_sha256(clean),
                "noisy_sha256": tensor_sha256(noisy),
                "pair_aligned": bool(torch.equal(clean, expected_clean) and torch.equal(noisy, expected_noisy)),
                "position": position,
                "seed": dataset.seed,
            }
        )
    return rows


def _full_pair(dataset: RecipeDataset, example) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = render_recipe(example.recipe, dataset.root)
    if rendered.reference is None:
        raise TrainingError(f"case {example.case_id} has no teacher waveform")
    noisy, clean = _normalize(_mono(rendered.observed), _mono(rendered.reference))
    return noisy, clean


def _utterance_id(recipe: dict[str, Any]) -> str:
    sources = recipe.get("sources") or {}
    for key in ("clean", "noisy"):
        path = (sources.get(key) or {}).get("path")
        if isinstance(path, str) and path:
            return path.rsplit("/", 1)[-1]
    return ""


def source_frame_count(dataset: RecipeDataset, example) -> int | None:
    path = (example.recipe.get("sources") or {}).get("noisy", {}).get("path")
    if not isinstance(path, str):
        return None
    resolved = (dataset.root / path).resolve()
    if not resolved.is_file():
        return None
    return int(torchaudio.info(str(resolved)).num_frames)


def optimizer_groups(
    size: int,
    seed: int,
    plan: BatchPlan,
    *,
    steps: int,
) -> list[list[list[int]]]:
    """Sample indexes for each optimizer step, grouped by rank.

    The grouping follows ``Trainer.fit``: one epoch permutation, then
    ``accumulation_steps`` micro-batches on each rank.
    """
    if steps < 1:
        raise TrainingError("steps must be positive")
    epoch = 0
    skip = [0 for _ in range(plan.world_size)]
    grouped: list[list[list[int]]] = []
    produced = 0
    while produced < steps:
        per_rank = [
            epoch_indices(size, seed, epoch, plan.world_size, rank) for rank in range(plan.world_size)
        ]
        micro = plan.micro_batch_size
        accumulation = plan.accumulation_steps
        n_batches = len(per_rank[0]) // micro
        usable = (n_batches // accumulation) * accumulation
        if usable < accumulation:
            raise TrainingError("epoch does not contain one optimizer step")
        if any(skip[rank] == usable for rank in range(plan.world_size)):
            epoch += 1
            skip = [0 for _ in range(plan.world_size)]
            continue
        cursor = skip[0]
        while cursor < usable and produced < steps:
            ranks: list[list[int]] = []
            for rank, indices in enumerate(per_rank):
                chosen: list[int] = []
                for offset in range(accumulation):
                    start = (cursor + offset) * micro
                    chosen.extend(indices[start : start + micro])
                ranks.append(chosen)
            grouped.append(ranks)
            cursor += accumulation
            produced += 1
        epoch += 1
        skip = [0 for _ in range(plan.world_size)]
    return grouped


class SupplyTask:
    """Trainer task that records batches and does not update a model."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def distribute(self, process: ProcessInfo) -> None:
        del process

    def train_mode(self) -> None:
        return None

    def eval_mode(self) -> None:
        return None

    def learning_rates(self) -> dict[str, float]:
        return {"generator": 0.0, "discriminator": 0.0}

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        del validation

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {"loss": "supply"}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loss") != "supply":
            raise TrainingError("checkpoint is not a supply recorder")

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        del accumulation
        rows = []
        for batch in batches:
            rows.extend(batch["audit"])
        self.records.append({"rows": rows})
        return {"generator": 0.0, "discriminator": 0.0}
