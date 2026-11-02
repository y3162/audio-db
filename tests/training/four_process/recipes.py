"""RecipeDataset batches for the four-process probe.

Waveforms come from the training loader, including its load-time resampling.
This module does not open the production database and does not resample again.
"""
import torch

from src.data.recipe import case_id_for
from src.evaluation.audio.render import render_recipe
from src.training.data.dataset import _match_rate, _mono, _normalize, load_recipe_dataset
from src.training.data.sampler import crop_start, epoch_indices
from tests.training.four_process.compare import digest


def load_probe_dataset(cfg):
    if PathName(cfg["database"]) == "resource.duckdb":
        raise RuntimeError("refusing to train from resource.duckdb")
    return load_recipe_dataset(
        cfg["database"],
        cfg["dataset"],
        role="train",
        seed=int(cfg["seed"]),
        sample_rate=int(cfg["sample_rate"]),
        segment_size=int(cfg["segment_size"]),
        training=True,
    )


def PathName(path):
    from pathlib import Path
    return Path(path).name


def planned_indices(dataset, cfg):
    count = max(int(cfg["steps"]), int(cfg.get("prepare_steps") or 0)) * (2 if cfg["model"] == "mossformergan" else 1)
    order = epoch_indices(len(dataset), int(cfg["seed"]), int(cfg["epoch"]), 1, 0)
    if count > len(order):
        raise RuntimeError(f"requested {count} steps but the dataset only yields {len(order)}")
    return order[:count]


def describe_item(dataset, index):
    example = dataset.examples[index]
    if case_id_for(example.recipe) != example.case_id:
        raise RuntimeError(f"case_id does not match the recipe hash: {example.case_id}")
    rendered = render_recipe(example.recipe, dataset.root)
    noisy = _mono(rendered.observed)
    clean = _mono(rendered.reference)
    noisy = _match_rate(noisy, rendered.sample_rate, dataset.sample_rate, example.case_id)
    clean = _match_rate(clean, rendered.sample_rate, dataset.sample_rate, example.case_id)
    noisy, clean = _normalize(noisy, clean)
    start = 0
    if int(noisy.numel()) >= int(dataset.segment_size):
        start = crop_start(
            example.case_id,
            dataset.seed,
            dataset.epoch,
            int(noisy.numel()),
            int(dataset.segment_size),
        )
    loaded_clean, loaded_noisy, loaded_id = dataset._load(index)
    return {
        "case_id": example.case_id,
        "recipe_sha256": example.case_id,
        "sample_rate": int(dataset.sample_rate),
        "file_sample_rate": int(rendered.sample_rate),
        "sources": {name: dict(spec)["path"] for name, spec in example.recipe["sources"].items()},
        "source_sha256": {name: dict(spec)["sha256"] for name, spec in example.recipe["sources"].items()},
        "clean_length": int(clean.numel()),
        "noisy_length": int(noisy.numel()),
        "crop_start": int(start),
        "segment_size": int(dataset.segment_size),
        "batch_index": int(index),
        "clean_sha256": digest(loaded_clean),
        "noisy_sha256": digest(loaded_noisy),
        "loaded_case_id": loaded_id,
    }, loaded_clean, loaded_noisy
