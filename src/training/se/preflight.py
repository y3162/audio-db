"""Execution-condition check for a training YAML.

Preflight reads the config and snapshot metadata. It does not build a model,
step an optimizer, render waveforms, or write a waveform cache. A PASS result
means the launch conditions that this module can see are consistent. It is not
a claim that an official reproduction succeeded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from src.evaluation.envinfo import REPO_ROOT, git_metadata
from src.evaluation.provenance import content_sha256
from src.training.core.batch import plan_batches
from src.training.errors import TrainingError
from src.training.se.config import load_training_config

NOTE = "この preflight の PASS は実行条件の確認であり、公式再現の成功ではない。"

RESUME_ALLOWLIST = frozenset({"logging"})

SMOKE_CAP_PATHS = (
    ("verification_max_updates",),
    ("data", "max_updates_per_epoch"),
    ("data", "validation_max_batches"),
    ("chunk_iterator", "max_updates"),
    ("chunk_iterator", "validation_max_updates"),
)

_CHECKPOINT_BYTES = {
    "tf_gridnet": 31_071_895,
    "tf_locoformer": 179_910_164,
    "mp_senet": 29_689_087,
    "mossformergan": 136_454_907,
}

_OFFICIAL_CLAIMS = frozenset({"official_faithful", "official_algorithm_fixed_seed"})
_RESEARCH_CLAIMS = frozenset({"research_checkpoint_selection"})


def resume_config_mismatch(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Keys that refuse a resume. ``logging`` is the only allowed difference.

    ``checkpoint_dir`` is compared. A checkpoint from another directory does
    not resume unless that path is unchanged.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return ["config"]
    mismatched = []
    for key in sorted((set(saved) | set(current)) - RESUME_ALLOWLIST):
        if saved.get(key) != current.get(key):
            mismatched.append(key)
    return mismatched


def smoke_cap_keys(config: dict[str, Any]) -> list[str]:
    found = []
    for path in SMOKE_CAP_PATHS:
        cursor: Any = config
        present = True
        for part in path:
            if not isinstance(cursor, dict) or part not in cursor:
                present = False
                break
            cursor = cursor[part]
        if present:
            found.append(".".join(path))
    return found


def claim_conflict(claim: str | None, dataset: str | None) -> str | None:
    """Official and research_valid_fixed claims cannot be mixed."""
    name = dataset or ""
    research_data = "research_valid_fixed" in name
    if claim in _OFFICIAL_CLAIMS and research_data:
        return "official claim is mixed with the research_valid_fixed dataset"
    if claim in _RESEARCH_CLAIMS and not research_data:
        return "research_checkpoint_selection is mixed with a dataset that is not research_valid_fixed"
    return None


def run_preflight(
    config_path: str | Path,
    *,
    world_size: int | None = None,
    expect_dataset: str | None = None,
    expect_snapshot: str | None = None,
    resume: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> int:
    """Write ``preflight.json`` and ``preflight.md``. Return 0 only for PASS."""
    source = Path(config_path)
    destination = Path(output_dir) if output_dir is not None else source.parent / f"{source.stem}-preflight"
    report = build_preflight_report(
        source,
        world_size=world_size,
        expect_dataset=expect_dataset,
        expect_snapshot=expect_snapshot,
        resume=None if resume is None else Path(resume),
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "preflight.md").write_text(_markdown(report), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


def build_preflight_report(
    config_path: Path,
    *,
    world_size: int | None,
    expect_dataset: str | None,
    expect_snapshot: str | None,
    resume: Path | None,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        config = load_training_config(config_path)
    except TrainingError as exc:
        return _report(failures=[f"config: {exc}"], checks={"config_path": str(config_path)})
    data = config.get("data") or {}
    dataset_name = data.get("dataset")
    profile = config.get("experiment_profile")
    claim = config.get("experiment_claim")
    checks: dict[str, Any] = {
        "config_path": str(config_path),
        "model_name": (config.get("model") or {}).get("name"),
        "experiment_profile": profile,
        "experiment_claim": claim,
        "dataset_name": dataset_name,
        "allow_test_selection": data.get("allow_test_selection"),
        "seed": config.get("seed"),
        "optimizer": config.get("optimizer"),
        "scheduler": config.get("scheduler"),
        "loss": config.get("loss"),
        "max_steps": config.get("max_steps"),
        "max_epochs": (config.get("chunk_iterator") or {}).get("max_epochs"),
        "global_batch_size": config.get("global_batch_size"),
        "micro_batch_size_per_gpu": config.get("micro_batch_size_per_gpu"),
        "required_world_size": config.get("required_world_size"),
        "actual_world_size": world_size,
        "checkpoint_interval": config.get("checkpoint_interval"),
        "validation_interval": config.get("validation_interval"),
        "keep_nbest": (config.get("chunk_iterator") or {}).get("keep_nbest"),
        "checkpoint_retention": config.get("checkpoint_retention"),
        "checkpoint_dir": config.get("checkpoint_dir"),
        "resume": None if resume is None else str(resume),
    }
    checks.update(_implementation_identity(config))
    if profile == "full":
        caps = smoke_cap_keys(config)
        if caps:
            failures.append("full profile still has smoke caps: " + ", ".join(caps))
        if config.get("required_world_size") is None:
            failures.append("full profile is missing required_world_size")
    if world_size is not None:
        required = config.get("required_world_size")
        if required is None:
            failures.append("world size was passed but required_world_size is missing")
        elif int(world_size) != int(required):
            failures.append(
                f"world size {world_size} does not match required_world_size {required}"
            )
    conflict = claim_conflict(claim if isinstance(claim, str) else None, dataset_name if isinstance(dataset_name, str) else None)
    if conflict:
        failures.append(conflict)
    if "expected_global_batch" in config and config["expected_global_batch"] != config["global_batch_size"]:
        failures.append(
            f"global_batch_size {config['global_batch_size']} does not match "
            f"expected_global_batch {config['expected_global_batch']}"
        )
    plan_world = config.get("required_world_size") if config.get("required_world_size") is not None else world_size
    if plan_world is not None:
        try:
            plan = plan_batches(
                int(config["global_batch_size"]),
                int(config["micro_batch_size_per_gpu"]),
                int(plan_world),
            )
        except TrainingError as exc:
            failures.append(f"global batch cannot be formed: {exc}")
            checks["accumulation_steps"] = None
        else:
            checks["micro_batch_size"] = plan.micro_batch_size
            checks["accumulation_steps"] = plan.accumulation_steps
            checks["planned_global_batch_size"] = plan.global_batch_size
    else:
        checks["accumulation_steps"] = None
        checks["batch_plan"] = "UNKNOWN"
    _retention_failures(config, failures)
    _output_dir_failures(config, resume, failures)
    _disk_check(config, checks, failures)
    if data.get("pipeline") == "reproduction" and isinstance(dataset_name, str):
        _dataset_checks(
            dataset_name,
            bool(data.get("allow_test_selection")),
            claim if isinstance(claim, str) else None,
            expect_dataset,
            expect_snapshot,
            checks,
            failures,
        )
    elif profile == "full":
        failures.append("full profile has no reproduction dataset id")
        checks["source_access"] = "UNKNOWN"
    if resume is not None:
        _resume_checks(resume, config, checks, failures)
    checks["checkpoint_frequency"] = _checkpoint_frequency(config)
    checks["validation_frequency"] = _validation_frequency(config)
    return _report(failures=failures, checks=checks)


def _report(*, failures: list[str], checks: dict[str, Any]) -> dict[str, Any]:
    status = "FAIL" if failures else "PASS"
    return {
        "status": status,
        "note": NOTE,
        "failures": failures,
        "checks": checks,
    }


def _implementation_identity(config: dict[str, Any]) -> dict[str, Any]:
    info = git_metadata()
    digest = hashlib.sha256()
    for relative in (
        "src/training/core/trainer.py",
        "src/training/core/checkpoint.py",
        "src/training/se/config.py",
        "src/training/se/run.py",
        "src/training/se/cli.py",
        "src/training/se/preflight.py",
    ):
        path = REPO_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"")
    config_hash = content_sha256(config)
    return {
        "git_commit": info.get("commit"),
        "git_dirty": info.get("dirty"),
        "config_sha256": config_hash,
        "experiment_id": config_hash,
        "implementation_sha256": digest.hexdigest(),
    }


def _retention_failures(config: dict[str, Any], failures: list[str]) -> None:
    retention = config.get("checkpoint_retention")
    keep_nbest = int((config.get("chunk_iterator") or {}).get("keep_nbest") or 0)
    if not isinstance(retention, dict):
        return
    if not retention.get("keep_best", True):
        failures.append("retention cannot keep best")
    if keep_nbest > 0 and not retention.get("protect_nbest", True):
        failures.append(f"retention cannot keep the {keep_nbest} n-best checkpoints required by keep_nbest")


def _output_dir_failures(config: dict[str, Any], resume: Path | None, failures: list[str]) -> None:
    raw = config.get("checkpoint_dir")
    if not isinstance(raw, str) or not raw:
        failures.append("checkpoint_dir is missing")
        return
    directory = Path(raw)
    if not directory.exists() or resume is not None:
        return
    entries = [path for path in directory.iterdir() if not path.name.startswith(".")]
    if entries:
        failures.append(f"output directory exists and is non-empty: {directory}")


def _disk_check(config: dict[str, Any], checks: dict[str, Any], failures: list[str]) -> None:
    model = (config.get("model") or {}).get("name")
    retention = config.get("checkpoint_retention") if isinstance(config.get("checkpoint_retention"), dict) else {}
    keep_latest = int(retention.get("keep_latest", 3))
    keep_best = 1 if retention.get("keep_best", True) else 0
    nbest = int((config.get("chunk_iterator") or {}).get("keep_nbest") or 0)
    average = 1 if nbest else 0
    final = 1
    file_count = keep_latest + keep_best + nbest + average + final
    each = retention.get("bytes_per_checkpoint")
    if each is None and isinstance(model, str):
        each = _CHECKPOINT_BYTES.get(model)
    checks["retained_file_estimate"] = file_count
    checks["bytes_per_checkpoint"] = each
    if each is None:
        checks["disk"] = "UNKNOWN"
        if config.get("experiment_profile") == "full":
            failures.append("checkpoint size is UNKNOWN; disk check cannot PASS")
        return
    estimate = int(file_count) * int(each)
    checks["disk_bytes_estimate"] = estimate
    directory = Path(str(config.get("checkpoint_dir") or "."))
    probe = directory
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    free = shutil.disk_usage(probe).free
    checks["disk_bytes_free"] = free
    checks["disk_probe"] = str(probe)
    if free < estimate:
        checks["disk"] = "FAIL"
        failures.append(
            f"free disk space {free} bytes is below the retained-checkpoint estimate {estimate} bytes"
        )
    else:
        checks["disk"] = "PASS"


def _dataset_checks(
    dataset_name: str,
    allow_test_selection: bool,
    claim: str | None,
    expect_dataset: str | None,
    expect_snapshot: str | None,
    checks: dict[str, Any],
    failures: list[str],
) -> None:
    if expect_dataset is not None and expect_dataset != dataset_name:
        failures.append(f"dataset {dataset_name} does not match --expect-dataset {expect_dataset}")
    from src.reproduction.errors import ReproductionError

    try:
        meta = _read_snapshot_metadata(dataset_name)
    except (ReproductionError, OSError, RuntimeError, ValueError, KeyError) as exc:
        checks["dataset_id"] = None
        checks["snapshot_hash"] = None
        checks["protocol_id"] = None
        checks["ready"] = None
        checks["source_access"] = "UNKNOWN"
        checks["counts"] = None
        failures.append(f"data gap: snapshot metadata for {dataset_name} could not be read: {exc}")
        return
    checks.update(meta)
    if expect_snapshot is not None and expect_snapshot != meta.get("snapshot_hash"):
        failures.append(
            f"snapshot hash {meta.get('snapshot_hash')} does not match --expect-snapshot {expect_snapshot}"
        )
    if meta.get("selection_uses_test") is True and not allow_test_selection:
        failures.append("selection_uses_test is true but data.allow_test_selection is false")
    source = meta.get("source_access")
    if source == "UNKNOWN":
        failures.append("source access is UNKNOWN")
    elif source == "blocked" and claim in _OFFICIAL_CLAIMS:
        failures.append("official materials are blocked but the claim says the run is official")
    elif source == "blocked":
        failures.append("dataset source access is blocked")
    if meta.get("ready") is not True:
        failures.append(f"dataset declared_status is {meta.get('declared_status')}")


def _read_snapshot_metadata(dataset_name: str) -> dict[str, Any]:
    """Read dataset.json, manifest.json, and case-set counts. No waveforms."""
    from src.reproduction.errors import ReproductionError
    from src.reproduction.paths import DEFAULT_SNAPSHOTS
    from src.reproduction.pipeline.training_data import find_snapshot_dir
    from src.reproduction.snapshot_reader import items_from_arrays

    try:
        directory = find_snapshot_dir(Path(DEFAULT_SNAPSHOTS), dataset_name)
    except ReproductionError:
        raise
    dataset_path = directory / "dataset.json"
    manifest_path = directory / "manifest.json"
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"snapshot metadata is unreadable: {directory}") from exc
    if not isinstance(dataset, dict) or not isinstance(manifest, dict):
        raise RuntimeError(f"snapshot metadata is not an object: {directory}")
    declared = dataset.get("declared_status")
    if declared == "blocked":
        source_access = "blocked"
    elif declared == "ready":
        source_access = "ready"
    else:
        source_access = "UNKNOWN"
    counts = {"train": None, "valid": None, "test": None}
    items_path = directory / "case_set_items.npz"
    try:
        import numpy as np

        with np.load(items_path, allow_pickle=False) as handle:
            arrays = {key: handle[key] for key in handle.files}
        items = items_from_arrays(arrays)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RuntimeError(f"snapshot case sets are unreadable: {items_path}") from exc
    by_set: dict[str, int] = {}
    for row in items:
        key = row.get("case_set_id")
        if isinstance(key, str):
            by_set[key] = by_set.get(key, 0) + 1
    for binding in dataset.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        slot = str(binding.get("slot") or "")
        case_set = binding.get("case_set_id")
        role = slot.split(".", 1)[0]
        if role in counts and isinstance(case_set, str):
            counts[role] = by_set.get(case_set, 0)
    return {
        "dataset_id": dataset.get("dataset_id"),
        "snapshot_hash": manifest.get("snapshot_hash"),
        "snapshot_directory": str(directory),
        "protocol_id": dataset.get("protocol_id"),
        "declared_status": declared,
        "ready": declared == "ready",
        "source_access": source_access,
        "selection_uses_test": dataset.get("selection_uses_test"),
        "dataset_profile": dataset.get("profile"),
        "dataset_claim_label": dataset.get("claim_label"),
        "counts": counts,
    }


def _resume_checks(resume: Path, config: dict[str, Any], checks: dict[str, Any], failures: list[str]) -> None:
    from src.training.core.checkpoint import load_checkpoint

    try:
        payload = load_checkpoint(resume, _cpu_device())
    except TrainingError as exc:
        failures.append(f"resume target is unreadable: {exc}")
        return
    mismatches = resume_config_mismatch(payload.get("config") or {}, config)
    if mismatches:
        failures.append("resume config mismatch: " + ", ".join(mismatches))
    saved = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    current_hash = checks.get("snapshot_hash")
    saved_hash = saved.get("snapshot_hash")
    if current_hash and saved_hash and saved_hash != current_hash:
        failures.append("resume dataset snapshot does not match the current snapshot")
    saved_name = saved.get("dataset_name") or saved.get("name")
    current_name = checks.get("dataset_name")
    if isinstance(saved_name, str) and isinstance(current_name, str) and saved_name not in (current_name, current_name.split("@", 1)[0]):
        failures.append(f"resume dataset {saved_name} does not match {current_name}")


def _cpu_device():
    import torch

    return torch.device("cpu")


def _checkpoint_frequency(config: dict[str, Any]) -> str:
    interval = int(config.get("checkpoint_interval") or 0)
    if config.get("chunk_iterator") and interval == 0:
        return "epoch end"
    if interval == 0:
        return "max_steps or managed stop only"
    return f"every {interval} optimizer updates"


def _validation_frequency(config: dict[str, Any]) -> str:
    interval = int(config.get("validation_interval") or 0)
    if config.get("chunk_iterator"):
        if interval == 0:
            return "epoch end"
        return f"epoch end and every {interval} optimizer updates"
    if interval == 0:
        return "no step validation"
    return f"every {interval} optimizer updates"


def _markdown(report: dict[str, Any]) -> str:
    checks = report.get("checks") or {}
    lines = [
        "# preflight",
        "",
        NOTE,
        "",
        f"- status: {report.get('status')}",
        f"- model: {checks.get('model_name')}",
        f"- dataset: {checks.get('dataset_name')}",
        f"- dataset id: {checks.get('dataset_id')}",
        f"- snapshot hash: {checks.get('snapshot_hash')}",
        f"- protocol: {checks.get('protocol_id')}",
        f"- ready: {checks.get('ready')}",
        f"- source access: {checks.get('source_access')}",
        f"- counts: {checks.get('counts')}",
        f"- selection_uses_test: {checks.get('selection_uses_test')}",
        f"- allow_test_selection: {checks.get('allow_test_selection')}",
        f"- claim: {checks.get('experiment_claim')}",
        f"- profile: {checks.get('experiment_profile')}",
        f"- required world size: {checks.get('required_world_size')}",
        f"- actual world size: {checks.get('actual_world_size')}",
        f"- global batch: {checks.get('global_batch_size')}",
        f"- micro batch: {checks.get('micro_batch_size')}",
        f"- accumulation: {checks.get('accumulation_steps')}",
        f"- optimizer: {checks.get('optimizer')}",
        f"- scheduler: {checks.get('scheduler')}",
        f"- epochs: {checks.get('max_epochs')}",
        f"- update total: {checks.get('max_steps')}",
        f"- checkpoint frequency: {checks.get('checkpoint_frequency')}",
        f"- validation frequency: {checks.get('validation_frequency')}",
        f"- n-best: {checks.get('keep_nbest')}",
        f"- retention: {checks.get('checkpoint_retention')}",
        f"- output dir: {checks.get('checkpoint_dir')}",
        f"- experiment id: {checks.get('experiment_id')}",
        f"- resume: {checks.get('resume')}",
        f"- config hash: {checks.get('config_sha256')}",
        f"- implementation hash: {checks.get('implementation_sha256')}",
        f"- git: {checks.get('git_commit')} dirty={checks.get('git_dirty')}",
        f"- disk: {checks.get('disk')} estimate={checks.get('disk_bytes_estimate')} free={checks.get('disk_bytes_free')}",
        "",
        "## failures",
        "",
    ]
    failures = report.get("failures") or []
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
