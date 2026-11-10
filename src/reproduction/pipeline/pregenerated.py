"""Open an MS-SNSD snapshot for the ESPnet chunk iterator.

Training reconstructs PCM from the realization document. A snapshot that still
points at stored wavs is refused here, including ``tf_gridnet.ms_snsd.smoke@1``.
That dataset remains in the database as history.
"""

from __future__ import annotations

from pathlib import Path

from src.reproduction.errors import ReproductionError
from src.reproduction.paths import DEFAULT_CORPORA, DEFAULT_DERIVED, DEFAULT_GENERATED, DEFAULT_SNAPSHOTS
from src.reproduction.pipeline.training_data import find_snapshot_dir
from src.reproduction.snapshot_reader import load_snapshot


def require_on_demand(dataset: dict) -> None:
    """Refuse a snapshot that still points at stored or pre-resampled wavs."""

    delivery = (dataset.get("rule_json") or {}).get("delivery")
    name = f"{dataset.get('name')}@{dataset.get('version')}"
    if delivery != "on_demand":
        raise ReproductionError(
            f"dataset {name} is not delivery on_demand; "
            "stored wav and pre-resampled wav training is disabled"
        )


def open_chunk_datasets(
    *,
    dataset: str,
    load_workers: int = 1,
    verify_on_read: bool = False,
    snapshot_root: str | Path | None = None,
    derived_root: str | Path | None = None,
    corpora_root: str | Path | None = None,
    generated_root: str | Path | None = None,
):
    """Train and valid views of one on-demand snapshot."""
    root = Path(DEFAULT_SNAPSHOTS if snapshot_root is None else snapshot_root)
    loaded = load_snapshot(find_snapshot_dir(root, dataset))
    require_on_demand(loaded.dataset)
    from src.reproduction.pipeline.ondemand import open_ondemand_datasets

    return open_ondemand_datasets(
        loaded,
        load_workers=load_workers,
        verify_on_read=verify_on_read,
        derived_root=DEFAULT_DERIVED if derived_root is None else derived_root,
        corpora_root=DEFAULT_CORPORA if corpora_root is None else corpora_root,
        generated_root=DEFAULT_GENERATED if generated_root is None else generated_root,
    )
