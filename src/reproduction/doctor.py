"""Read-only status for the reproduction database.

This command does not create or modify a database. A missing reproduction
database is reported as absent. An incomplete or incompatible one is a failure.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
from pathlib import Path

from src.reproduction.db.schema import assert_ready
from src.reproduction.errors import ReproductionError
from src.reproduction.paths import DEFAULT_CORPORA, DEFAULT_DB, DEFAULT_GENERATED, DEFAULT_RESOURCE_DB, ROOT

_LISTS = ROOT / "configs" / "data" / "lists"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def add(self, status: str, name: str, detail: str, *, required: bool) -> None:
        self.rows.append((status, name, detail, required))
        print(f"{status}\t{name}\t{detail}")

    def failed(self) -> bool:
        return any(status == "FAIL" and required for status, _name, _detail, required in self.rows)


def run_doctor(
    *,
    database: str | Path = DEFAULT_DB,
    resource_db: str | Path = DEFAULT_RESOURCE_DB,
    models: str = "",
    corpora: str | Path = DEFAULT_CORPORA,
    generated: str | Path = DEFAULT_GENERATED,
) -> int:
    report = Report()
    _duckdb(report)
    _reproduction_db(report, Path(database))
    _resource_db(report, Path(resource_db))
    _directory(report, Path(corpora), "corpora")
    _generated(report, Path(generated))
    _libraries(report)
    _lists(report, models)
    return 1 if report.failed() else 0


def _duckdb(report: Report) -> None:
    try:
        import duckdb
    except ImportError:
        report.add("FAIL", "duckdb", "not installed", required=True)
        return
    report.add("PASS", "duckdb", duckdb.__version__, required=True)


def _reproduction_db(report: Report, path: Path) -> None:
    if not path.exists():
        report.add("NOT_TESTED", "reproduction.duckdb", "absent", required=False)
        return
    before = _sha256(path)
    names = _names(path.parent)
    try:
        import duckdb

        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        report.add("FAIL", "reproduction.duckdb", f"cannot open read-only: {exc}", required=True)
        return
    try:
        assert_ready(connection)
        ready = True
        message = "ready"
    except ReproductionError as exc:
        ready = False
        message = str(exc)
    finally:
        connection.close()
    if _sha256(path) != before or _names(path.parent) != names:
        report.add("FAIL", "reproduction.duckdb", "read-only open changed the database directory", required=True)
        return
    report.add("PASS" if ready else "FAIL", "reproduction.duckdb", message, required=True)


def _resource_db(report: Report, path: Path) -> None:
    if not path.is_file():
        report.add("FAIL", "resource.duckdb", "absent", required=True)
        return
    before = _sha256(path)
    names = _names(path.parent)
    try:
        import duckdb

        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        report.add("FAIL", "resource.duckdb", f"cannot open read-only: {exc}", required=True)
        return
    try:
        schemas = {
            row[0]
            for row in connection.execute(
                "SELECT schema_name FROM information_schema.schemata"
            ).fetchall()
        }
    except Exception as exc:
        report.add("FAIL", "resource.duckdb", f"cannot read schemas: {exc}", required=True)
        return
    finally:
        connection.close()
    if _sha256(path) != before or _names(path.parent) != names:
        report.add("FAIL", "resource.duckdb", "read-only open changed the database directory", required=True)
        return
    corpora = "present" if "corpora" in schemas else "absent"
    evaluation = "present" if "eval" in schemas else "absent"
    report.add("PASS" if corpora == "present" else "NOT_TESTED", "resource.corpora", corpora, required=False)
    if evaluation == "present":
        report.add("PASS", "resource.eval", "present", required=False)
    else:
        report.add("NOT_TESTED", "resource.eval", "absent; leakage overlap is UNKNOWN", required=False)


def _directory(report: Report, path: Path, name: str) -> None:
    if path.is_dir():
        report.add("PASS", name, str(path), required=False)
    else:
        report.add("NOT_TESTED", name, "absent", required=False)


def _generated(report: Report, path: Path) -> None:
    if not path.is_dir():
        report.add("NOT_TESTED", "generated", "absent", required=False)
        return
    usage = shutil.disk_usage(path)
    report.add("PASS", "generated", f"free_bytes={usage.free}", required=False)


def _libraries(report: Report) -> None:
    for name in ("numpy", "soundfile", "librosa", "soxr", "torch"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            report.add("NOT_TESTED", name, "not installed", required=False)
            continue
        report.add("PASS", name, getattr(module, "__version__", "unknown"), required=False)


def _lists(report: Report, models: str) -> None:
    selected = [item.strip() for item in models.split(",") if item.strip()]
    if not selected:
        report.add("NOT_TESTED", "official_lists", "MODELS unset", required=False)
        return
    for model in selected:
        directory = _LISTS / model
        files = sorted(path.name for path in directory.glob("*") if path.is_file()) if directory.is_dir() else []
        if not files:
            report.add("FAIL", f"official_lists.{model}", "missing", required=True)
        else:
            report.add("PASS", f"official_lists.{model}", ",".join(files), required=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {path.name for path in directory.iterdir()}
