"""Create the eval schema and register the sets listed in configs/db/init.yaml.

This command is idempotent. It does not call corpus ingest and does not
recreate corpora.audio_files. Missing optional inputs are reported and skipped.
A required input that is missing or invalid stops the command.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.evaluation.errors import DatasetError

DEFAULT_CONFIG = Path("configs/db/init.yaml")
_TOP_LEVEL = {"database", "corpora_root", "manifests", "corpus_sets"}
_MANIFEST_KEYS = {"path", "set", "required", "description", "root"}
_CORPUS_KEYS = {
    "name",
    "required",
    "description",
    "role",
    "filters",
    "query",
    "params",
    "limit",
    "noise_filters",
    "noise_query",
    "noise_params",
    "noise_limit",
    "snr_db",
    "seed",
    "protocol",
    "max_seconds",
    "prevent_clip",
    "hard_clip",
}


def initialize(config_path: str | Path = DEFAULT_CONFIG) -> str:
    raw = _load(Path(config_path))
    database = Path(raw["database"])
    corpora_root = raw.get("corpora_root", "data/corpora")
    from src.evaluation.datasets.evaldb import EvaluationDB

    db = EvaluationDB(database, corpora_root=corpora_root)
    before = db.corpora_fingerprint()
    db.ensure_schema()
    lines = [f"database: {database}"]
    if before is None:
        lines.append("corpora.audio_files: not present")
    else:
        lines.append(f"corpora.audio_files: unchanged ({before[0]} rows)")
    for item in raw.get("manifests") or []:
        lines.append(_import_manifest(db, item))
    for item in raw.get("corpus_sets") or []:
        lines.append(_import_corpus_set(db, item))
    after = db.corpora_fingerprint()
    if before != after:
        raise DatasetError("corpora.audio_files changed during db-init")
    text = "\n".join(lines)
    print(text)
    return text


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    config_path = DEFAULT_CONFIG if not args else Path(args[0])
    if len(args) > 1:
        print("error: pass at most one config path", file=sys.stderr)
        return 2
    try:
        initialize(config_path)
    except (DatasetError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _import_manifest(db, item: dict[str, Any]) -> str:
    path = Path(item["path"])
    set_name = item["set"]
    required = bool(item.get("required", False))
    if not path.is_file():
        message = f"manifest {path}: skipped (file not found)"
        if required:
            raise DatasetError(message.replace("skipped", "required"))
        return message
    try:
        report = db.import_manifest(
            path,
            set_name,
            root=item.get("root"),
            description=item.get("description"),
        )
    except DatasetError as exc:
        if required:
            raise DatasetError(f"manifest {path}: {exc}") from exc
        return f"manifest {path}: skipped ({exc})"
    return f"manifest {path} -> {set_name}: {report}"


def _import_corpus_set(db, item: dict[str, Any]) -> str:
    name = item["name"]
    required = bool(item.get("required", False))
    kwargs = {
        key: item[key]
        for key in (
            "description",
            "role",
            "filters",
            "query",
            "params",
            "limit",
            "noise_filters",
            "noise_query",
            "noise_params",
            "noise_limit",
            "snr_db",
            "seed",
            "protocol",
            "max_seconds",
            "prevent_clip",
            "hard_clip",
        )
        if key in item
    }
    try:
        report = db.create_set_from_corpus(name, **kwargs)
    except DatasetError as exc:
        if required:
            raise DatasetError(f"set {name}: {exc}") from exc
        return f"set {name}: skipped ({exc})"
    return f"set {name}: {report}"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetError(f"db-init config not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise DatasetError("YAML configuration requires PyYAML") from exc
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"invalid YAML config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError("db-init config must be a mapping")
    unknown = sorted(set(raw) - _TOP_LEVEL)
    if unknown:
        raise DatasetError(f"unknown fields in db-init config: {', '.join(unknown)}")
    if "database" not in raw or not isinstance(raw["database"], str) or not raw["database"].strip():
        raise DatasetError("database must be a path")
    _check_list(raw.get("manifests", []), "manifests", _MANIFEST_KEYS, ("path", "set"))
    _check_list(raw.get("corpus_sets", []), "corpus_sets", _CORPUS_KEYS, ("name",))
    return raw


def _check_list(items: Any, where: str, allowed: set[str], required: tuple[str, ...]) -> None:
    if not isinstance(items, list):
        raise DatasetError(f"{where} must be a list")
    for index, item in enumerate(items):
        label = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise DatasetError(f"{label} must be a mapping")
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise DatasetError(f"unknown fields in {label}: {', '.join(unknown)}")
        missing = [key for key in required if key not in item]
        if missing:
            raise DatasetError(f"missing fields in {label}: {', '.join(missing)}")
        if "required" in item and not isinstance(item["required"], bool):
            raise DatasetError(f"{label}.required must be a boolean")


if __name__ == "__main__":
    raise SystemExit(main())
