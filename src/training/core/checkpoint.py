"""Atomic training checkpoints.

A checkpoint is written to a temporary file in the destination directory and
then renamed. A killed writer leaves the previous checkpoint in place.
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from src.evaluation.envinfo import REPO_ROOT
from src.training.errors import TrainingError

_PROTECTED = (
    "data/checkpoints",
    "data/corpora",
    "data/db",
    "data/staging",
)


def assert_output_allowed(path: Path) -> None:
    """Refuse writes into published checkpoints, corpora, the database, or staging."""
    resolved = path.resolve()
    for relative in _PROTECTED:
        protected = (REPO_ROOT / relative).resolve()
        if resolved == protected or _is_relative_to(resolved, protected):
            raise TrainingError(f"refusing to write training output under {relative}: {path}")


def save_atomic(path: Path, payload: dict[str, Any], *, verify_training: bool = False) -> None:
    assert_output_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_output_allowed(path.parent)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if verify_training:
            restored = load_checkpoint(temporary, torch.device("cpu"))
            for key in ("global_step", "epoch", "micro_batches_consumed"):
                if key not in restored or restored[key] != payload[key]:
                    raise TrainingError(f"checkpoint verification failed: {key}")
            for key in ("task", "rng", "dataset", "config", "batch_plan"):
                if key not in restored:
                    raise TrainingError(f"checkpoint verification failed: missing {key}")
            _verify_rank_rng(payload.get("rng_by_rank"), restored.get("rng_by_rank"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_rank_rng(original: Any, restored: Any) -> None:
    """Reject a checkpoint whose per-rank RNG did not round-trip."""
    if original is None and restored is None:
        return
    if not isinstance(original, list) or not isinstance(restored, list) or len(original) != len(restored):
        raise TrainingError("checkpoint verification failed: rng_by_rank")
    if len(original) == 0:
        raise TrainingError("checkpoint verification failed: rng_by_rank empty")
    for index, (left, right) in enumerate(zip(original, restored, strict=True)):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise TrainingError(f"checkpoint verification failed: rng_by_rank[{index}]")
        for key in ("python", "numpy", "torch"):
            if key not in right:
                raise TrainingError(f"checkpoint verification failed: rng_by_rank[{index}].{key}")
        if not torch.equal(left["torch"].cpu(), right["torch"].cpu()):
            raise TrainingError(f"checkpoint verification failed: rng_by_rank[{index}].torch")
        if "cuda" in left or "cuda" in right:
            if "cuda" not in left or "cuda" not in right:
                raise TrainingError(f"checkpoint verification failed: rng_by_rank[{index}].cuda")
            if int(left["cuda_device"]) != int(right["cuda_device"]):
                raise TrainingError(f"checkpoint verification failed: rng_by_rank[{index}].cuda_device")
            if not torch.equal(left["cuda"].cpu(), right["cuda"].cpu()):
                raise TrainingError(f"checkpoint verification failed: rng_by_rank[{index}].cuda")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise TrainingError(f"checkpoint {path} is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("format") != "audio-db-training-v1":
        raise TrainingError(f"checkpoint {path} is not an audio-db training checkpoint")
    return payload


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


_STEP_NAME = re.compile(r"^step_(\d+)\.pt$")
_EPOCH_NAME = re.compile(r"^(\d+)epoch\.pt$")


def prune_run_checkpoints(
    directory: Path,
    owned: set[str],
    *,
    keep_latest: int,
    keep_best: bool,
    best_names: set[str],
    nbest_names: set[str],
    extra_protected: set[str] | None = None,
) -> list[str]:
    """Delete step and epoch files this run owns, after a successful save.

    The latest ``keep_latest`` resumable ``step_*.pt`` files stay. Best files,
    n-best average inputs, symlink targets, and names in ``extra_protected``
    stay as well. Files outside ``directory`` and files this run does not own
    are left in place. A symlink itself is not removed here.
    """
    if keep_latest < 0:
        raise TrainingError("keep_latest must be zero or positive")
    root = directory.resolve()
    if not root.is_dir():
        return []
    symlink_targets: set[Path] = set()
    for child in root.iterdir():
        if not child.is_symlink():
            continue
        try:
            symlink_targets.add(child.resolve())
        except OSError:
            continue
    steps: list[tuple[int, str]] = []
    for child in root.iterdir():
        match = _STEP_NAME.match(child.name)
        if match and child.is_file() and not child.is_symlink():
            steps.append((int(match.group(1)), child.name))
    steps.sort()
    protected = {name for _, name in steps[-keep_latest:]} if keep_latest else set()
    protected.update(nbest_names)
    protected.update(extra_protected or set())
    if keep_best:
        protected.update(best_names)
    deleted: list[str] = []
    for name in sorted(owned):
        if not (_STEP_NAME.match(name) or _EPOCH_NAME.match(name)):
            continue
        if name in protected:
            continue
        path = root / name
        if path.is_symlink() or not path.exists():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.parent != root:
            continue
        if resolved in symlink_targets or not resolved.is_file():
            continue
        resolved.unlink()
        deleted.append(name)
    return deleted
