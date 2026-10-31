"""Dispatch a dataset config to a concrete loader."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.evaluation.config import DatasetConfig
from src.evaluation.datasets.duckdb import load_duckdb
from src.evaluation.datasets.synthetic import make_synthetic_samples
from src.evaluation.errors import DatasetError
from src.evaluation.sample import AudioSample


@dataclass
class DatasetBundle:
    samples: list[AudioSample]
    noise_pool: list[AudioSample] = field(default_factory=list)


def load_dataset(config: DatasetConfig, experiment_seed: int) -> DatasetBundle:
    if config.source == "synthetic":
        seed = experiment_seed if config.seed is None else config.seed
        return DatasetBundle(
            samples=make_synthetic_samples(
                config.count,
                config.sample_rate,
                config.seconds,
                seed,
            )
        )
    if config.source == "duckdb":
        if not config.path:
            raise DatasetError("DuckDB path is required")
        if config.set_name:
            from src.evaluation.datasets.evaldb import EvaluationDB

            database = EvaluationDB(config.path, corpora_root=config.corpora_root or "data/corpora")
            dataset = database.load_dataset(config.set_name)
            return DatasetBundle(samples=list(dataset))
        corpora_root = config.corpora_root or "data/corpora"
        samples = load_duckdb(
            config.path,
            filters=config.filters,
            limit=config.limit,
            role=config.role,
            corpora_root=corpora_root,
        )
        noise_pool: list[AudioSample] = []
        if config.noise_filters:
            noise_pool = load_duckdb(
                config.path,
                filters=config.noise_filters,
                limit=config.noise_limit,
                role="noise",
                corpora_root=corpora_root,
            )
        return DatasetBundle(samples=samples, noise_pool=noise_pool)
    raise DatasetError(f"unknown dataset source: {config.source}")
