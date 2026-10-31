"""Named evaluation sets stored in DuckDB.

Manifests and corpus queries are import paths. After registration, samples are
read from the ``eval`` schema. Audio paths are stored relative to the database
file so the checkout can move. This module does not create or modify
``corpora.audio_files``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data.db.eval_schema import create_eval_schema
from src.data.models.audio_file import AUDIO_FILE_COLUMNS
from src.evaluation.audio.mixing import PROTOCOL_RMS_V1, crop_start
from src.evaluation.datasets.duckdb import fetch_audio_files
from src.evaluation.datasets.manifest import load_manifest
from src.evaluation.errors import DatasetError
from src.evaluation.identity import canonical_snr, derive_seed, sha256_json
from src.evaluation.sample import AudioSample

SET_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
KINDS = ("clean", "noisy", "paired", "synthetic")
CORPUS_ROLES = ("clean", "noisy", "synthetic")
_WRITE_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|copy|attach|pragma)\b",
    re.IGNORECASE,
)
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_MIX_METADATA = ("mix_seed", "mix_protocol", "prevent_clip", "hard_clip", "max_seconds", "noise_start")

CASE_COLUMNS = (
    "case_id",
    "kind",
    "dataset",
    "split",
    "sample_rate",
    "transcript",
    "speaker_id",
    "snr_db",
    "audio_path",
    "clean_path",
    "noisy_path",
    "noise_path",
    "corpus_path",
    "mix_protocol",
    "mix_seed",
    "noise_start",
    "max_seconds",
    "prevent_clip",
    "hard_clip",
    "metadata",
    "content_hash",
)


@dataclass
class RegistrationReport:
    set_name: str
    inserted: int
    unchanged: int
    set_status: str

    def __str__(self) -> str:
        return f"inserted {self.inserted}, unchanged {self.unchanged}, {self.set_status}"


@dataclass
class EvaluationDataset:
    """Path-only samples in a fixed order. Waveforms are loaded by the caller."""

    name: str
    _samples: list[AudioSample]

    def __iter__(self):
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> AudioSample:
        return self._samples[index]


@dataclass
class _Case:
    case_id: str
    kind: str
    dataset: str
    split: str | None = None
    sample_rate: int | None = None
    transcript: str | None = None
    speaker_id: str | None = None
    snr_db: float | None = None
    audio_path: str | None = None
    clean_path: str | None = None
    noisy_path: str | None = None
    noise_path: str | None = None
    corpus_path: str | None = None
    mix_protocol: str | None = None
    mix_seed: int | None = None
    noise_start: int | None = None
    max_seconds: float | None = None
    prevent_clip: bool | None = None
    hard_clip: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        return sha256_json(asdict(self))


class EvaluationDB:
    def __init__(
        self,
        path: str | Path,
        *,
        corpora_root: str | Path = "data/corpora",
        project_root: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        root = Path(corpora_root)
        self.corpora_root = root if root.is_absolute() else self.project_root / root

    def ensure_schema(self) -> None:
        """Create the eval schema. Existing corpora tables are left as they are."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            create_eval_schema(connection)
        finally:
            connection.close()

    def corpora_fingerprint(self) -> tuple[Any, ...] | None:
        """Identity of corpora.audio_files, or None when that table is absent."""
        if not self.path.is_file():
            return None
        connection = self._connect(read_only=True)
        try:
            exists = connection.execute(
                """
                SELECT count(*) FROM information_schema.tables
                WHERE table_schema = 'corpora' AND table_name = 'audio_files'
                """
            ).fetchone()[0]
            if not exists:
                return None
            count, low, high, mixed = connection.execute(
                """
                SELECT count(*), min(path), max(path), bit_xor(hash(path))
                FROM corpora.audio_files
                """
            ).fetchone()
            return (int(count), low, high, None if mixed is None else int(mixed))
        finally:
            connection.close()

    def stats(self) -> dict[str, int]:
        connection = self._connect(read_only=True)
        try:
            cases, sets, items = (
                connection.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM eval.cases),
                        (SELECT count(*) FROM eval.sets),
                        (SELECT count(*) FROM eval.set_items)
                    """
                ).fetchone()
            )
        except Exception as exc:
            raise DatasetError(f"evaluation schema is missing in {self.path}") from exc
        finally:
            connection.close()
        return {"cases": int(cases), "sets": int(sets), "items": int(items)}

    def import_manifest(
        self,
        path: str | Path,
        set_name: str,
        *,
        root: str | Path | None = None,
        description: str | None = None,
    ) -> RegistrationReport:
        """Validate a manifest and register its rows as one named set."""
        samples = load_manifest(path, root)
        cases = [self._case_from_manifest(sample) for sample in samples]
        return self._register(cases, set_name, description)

    def register_samples(
        self,
        samples: list[AudioSample],
        set_name: str,
        *,
        description: str | None = None,
    ) -> RegistrationReport:
        """Register samples that were already resolved to audio files."""
        if not samples:
            raise DatasetError(f"evaluation set {set_name} has no cases")
        cases = [self._case_from_manifest(sample) for sample in samples]
        return self._register(cases, set_name, description)

    def create_set_from_corpus(
        self,
        name: str,
        *,
        role: str = "clean",
        filters: dict[str, Any] | None = None,
        query: str | None = None,
        params: list[Any] | None = None,
        limit: int | None = None,
        noise_filters: dict[str, Any] | None = None,
        noise_query: str | None = None,
        noise_params: list[Any] | None = None,
        noise_limit: int | None = None,
        snr_db: float | None = None,
        seed: int = 0,
        protocol: str = PROTOCOL_RMS_V1,
        max_seconds: float | None = None,
        prevent_clip: bool = False,
        hard_clip: bool = False,
        description: str | None = None,
    ) -> RegistrationReport:
        """Register a named set from corpora.audio_files. No manifest is read."""
        if role not in CORPUS_ROLES:
            raise DatasetError(f"corpus role must be one of: {', '.join(CORPUS_ROLES)}")
        if query and filters:
            raise DatasetError("pass filters or query, not both")
        if noise_query and noise_filters:
            raise DatasetError("pass noise_filters or noise_query, not both")
        if protocol != PROTOCOL_RMS_V1:
            raise DatasetError(f"unknown mixing protocol: {protocol}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DatasetError("seed must be an integer")
        clean_limit = _optional_seconds(max_seconds, "max_seconds")
        self.ensure_schema()
        connection = self._connect(read_only=True)
        try:
            rows = self._select_corpus(connection, filters=filters, query=query, params=params, limit=limit)
            if role != "synthetic":
                cases = [self._case_from_corpus_row(row, role) for row in rows]
            else:
                if snr_db is None:
                    raise DatasetError("synthetic corpus sets require snr_db")
                if isinstance(snr_db, bool) or not isinstance(snr_db, (int, float)):
                    raise DatasetError("snr_db must be a number")
                noises = self._select_corpus(
                    connection,
                    filters=noise_filters,
                    query=noise_query,
                    params=noise_params,
                    limit=noise_limit,
                )
                if not noises:
                    raise DatasetError("synthetic corpus sets require at least one noise file")
                cases = [
                    self._synthetic_corpus_case(
                        row,
                        noises,
                        snr_db=float(snr_db),
                        seed=seed,
                        protocol=protocol,
                        max_seconds=clean_limit,
                        prevent_clip=bool(prevent_clip),
                        hard_clip=bool(hard_clip),
                    )
                    for row in rows
                ]
        finally:
            connection.close()
        return self._register(cases, name, description)

    def load_dataset(self, name: str) -> EvaluationDataset:
        """Return the named set in registered order, without decoding audio."""
        _check_set_name(name)
        if not self.path.is_file():
            raise DatasetError(f"DuckDB not found: {self.path}. Run make db-init before evaluation.")
        columns = ", ".join(f"c.{column}" for column in CASE_COLUMNS)
        connection = self._connect(read_only=True)
        try:
            try:
                cursor = connection.execute(
                    f"""
                    SELECT {columns}
                    FROM eval.set_items AS i
                    JOIN eval.cases AS c ON c.case_id = i.case_id
                    WHERE i.set_name = ?
                    ORDER BY i.position
                    """,
                    [name],
                )
            except Exception as exc:
                raise DatasetError(
                    f"evaluation schema is missing in {self.path}. Run make db-init."
                ) from exc
            names = [item[0] for item in cursor.description]
            rows = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
        finally:
            connection.close()
        if not rows:
            raise DatasetError(f"evaluation set not found: {name}")
        return EvaluationDataset(name, [self._sample_from_row(row) for row in rows])

    def _register(
        self,
        cases: list[_Case],
        set_name: str,
        description: str | None,
    ) -> RegistrationReport:
        _check_set_name(set_name)
        if not cases:
            raise DatasetError(f"evaluation set {set_name} has no cases")
        seen: set[str] = set()
        for case in cases:
            if case.case_id in seen:
                raise DatasetError(f"duplicate case id in import: {case.case_id}")
            seen.add(case.case_id)
        self.ensure_schema()
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            inserted = 0
            unchanged = 0
            for case in cases:
                state = self._put_case(connection, case)
                inserted += int(state == "inserted")
                unchanged += int(state == "unchanged")
            set_status = self._put_set(connection, set_name, description, [case.case_id for case in cases])
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return RegistrationReport(set_name, inserted, unchanged, set_status)

    def _put_case(self, connection, case: _Case) -> str:
        digest = case.content_hash()
        row = connection.execute(
            "SELECT content_hash FROM eval.cases WHERE case_id = ?",
            [case.case_id],
        ).fetchone()
        if row is None:
            payload = asdict(case)
            payload["metadata"] = json.dumps(case.metadata, sort_keys=True, ensure_ascii=False)
            payload["content_hash"] = digest
            placeholders = ", ".join("?" for _ in CASE_COLUMNS)
            connection.execute(
                f"INSERT INTO eval.cases ({', '.join(CASE_COLUMNS)}) VALUES ({placeholders})",
                [payload[name] for name in CASE_COLUMNS],
            )
            return "inserted"
        if row[0] != digest:
            raise DatasetError(f"case {case.case_id} is already registered with different content")
        return "unchanged"

    def _put_set(self, connection, name: str, description: str | None, case_ids: list[str]) -> str:
        existing = connection.execute("SELECT 1 FROM eval.sets WHERE name = ?", [name]).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO eval.sets (name, description) VALUES (?, ?)",
                [name, description],
            )
            for position, case_id in enumerate(case_ids):
                connection.execute(
                    "INSERT INTO eval.set_items (set_name, position, case_id) VALUES (?, ?, ?)",
                    [name, position, case_id],
                )
            return "created"
        rows = connection.execute(
            "SELECT case_id FROM eval.set_items WHERE set_name = ? ORDER BY position",
            [name],
        ).fetchall()
        current = [item[0] for item in rows]
        if current != case_ids:
            raise DatasetError(
                f"evaluation set {name} is already registered with different cases or order"
            )
        return "unchanged"

    def _case_from_manifest(self, sample: AudioSample) -> _Case:
        kind = _manifest_kind(sample)
        metadata = {key: value for key, value in sample.metadata.items() if key not in _MIX_METADATA}
        clean = self._store_audio(sample.clean_path)
        noisy = self._store_audio(sample.noisy_path)
        noise = self._store_audio(sample.noise_path)
        audio = self._store_audio(sample.audio_path)
        if kind == "clean" and clean is None:
            clean = audio
            audio = None
        corpus_path = self._corpus_key(clean or noisy or audio or noise)
        corpus_row = self._corpus_row(corpus_path)
        transcript = sample.transcript
        speaker_id = sample.speaker_id
        sample_rate = sample.sample_rate
        if corpus_row is not None:
            if transcript is None:
                transcript = corpus_row.get("transcript")
            if speaker_id is None:
                speaker_id = corpus_row.get("speaker_id")
            if sample_rate is None and corpus_row.get("sample_rate") is not None:
                sample_rate = int(corpus_row["sample_rate"])
        case = _Case(
            case_id=sample.utterance_id,
            kind=kind,
            dataset=sample.dataset,
            split=sample.split,
            sample_rate=sample_rate,
            transcript=transcript,
            speaker_id=None if speaker_id is None else str(speaker_id),
            snr_db=None if kind != "synthetic" else canonical_snr(sample.snr_db),
            audio_path=audio,
            clean_path=clean,
            noisy_path=noisy,
            noise_path=noise if kind == "synthetic" else None,
            corpus_path=corpus_path,
            metadata=metadata,
        )
        if kind == "synthetic":
            self._attach_mix(case, sample.metadata, seed_parts=("manifest", sample.utterance_id))
        elif case.sample_rate is None:
            case.sample_rate = _rate_of(self._absolute(clean or noisy or audio))
        return case

    def _case_from_corpus_row(self, row: dict[str, Any], role: str) -> _Case:
        stored = self._store_audio(str(self.corpora_root / row["path"]))
        kind = "noisy" if role == "noisy" else "clean"
        metadata = _corpus_metadata(row)
        return _Case(
            case_id=f"corpus:{row['path']}",
            kind=kind,
            dataset=row["corpus"],
            split=row.get("subset") or row.get("eval_split"),
            sample_rate=int(row["sample_rate"]) if row.get("sample_rate") is not None else None,
            transcript=row.get("transcript"),
            speaker_id=row.get("speaker_id"),
            clean_path=stored if kind == "clean" else None,
            noisy_path=stored if kind == "noisy" else None,
            corpus_path=row["path"],
            metadata=metadata,
        )

    def _synthetic_corpus_case(
        self,
        row: dict[str, Any],
        noises: list[dict[str, Any]],
        *,
        snr_db: float,
        seed: int,
        protocol: str,
        max_seconds: float | None,
        prevent_clip: bool,
        hard_clip: bool,
    ) -> _Case:
        offset = derive_seed(seed, "noise-pair", row["path"]) % len(noises)
        noise_row = noises[offset]
        clean_stored = self._store_audio(str(self.corpora_root / row["path"]))
        noise_stored = self._store_audio(str(self.corpora_root / noise_row["path"]))
        snr = canonical_snr(snr_db)
        mix_seed = derive_seed(
            "eval-mix",
            seed,
            clean_stored,
            noise_stored,
            snr,
            protocol,
            max_seconds,
        )
        case = _Case(
            case_id=(
                f"synthetic:{row['path']}|{noise_row['path']}|{snr}|{mix_seed}|{protocol}"
            ),
            kind="synthetic",
            dataset=row["corpus"],
            split=row.get("subset") or row.get("eval_split"),
            sample_rate=int(row["sample_rate"]) if row.get("sample_rate") is not None else None,
            transcript=row.get("transcript"),
            speaker_id=row.get("speaker_id"),
            snr_db=snr,
            clean_path=clean_stored,
            noise_path=noise_stored,
            corpus_path=row["path"],
            mix_protocol=protocol,
            mix_seed=mix_seed,
            max_seconds=max_seconds,
            prevent_clip=prevent_clip,
            hard_clip=hard_clip,
            metadata=_corpus_metadata(row),
        )
        case.noise_start = self._noise_start(case)
        return case

    def _attach_mix(self, case: _Case, metadata: dict[str, Any], *, seed_parts: tuple[object, ...]) -> None:
        protocol = metadata.get("mix_protocol", PROTOCOL_RMS_V1)
        if protocol != PROTOCOL_RMS_V1:
            raise DatasetError(f"{case.case_id} has unknown mix protocol {protocol!r}")
        supplied = metadata.get("mix_seed")
        if supplied is None:
            mix_seed = derive_seed(*seed_parts, case.clean_path, case.noise_path, case.snr_db, protocol)
        elif isinstance(supplied, bool) or not isinstance(supplied, int):
            raise DatasetError(f"{case.case_id} mix_seed must be an integer")
        else:
            mix_seed = supplied
        case.mix_protocol = protocol
        case.mix_seed = mix_seed
        case.max_seconds = _optional_seconds(metadata.get("max_seconds"), f"{case.case_id} max_seconds")
        prevent_clip = _optional_bool(metadata.get("prevent_clip"), f"{case.case_id} prevent_clip")
        hard_clip = _optional_bool(metadata.get("hard_clip"), f"{case.case_id} hard_clip")
        case.prevent_clip = False if prevent_clip is None else prevent_clip
        case.hard_clip = False if hard_clip is None else hard_clip
        case.noise_start = self._noise_start(case)
        if case.sample_rate is None:
            case.sample_rate = _rate_of(self._absolute(case.clean_path))

    def _noise_start(self, case: _Case) -> int | None:
        clean_frames, clean_rate = _frame_count(self._absolute(case.clean_path), case.max_seconds)
        noise_frames, noise_rate = _frame_count(self._absolute(case.noise_path), None)
        if clean_rate != noise_rate:
            return None
        return crop_start(noise_frames, clean_frames, int(case.mix_seed))

    def _select_corpus(self, connection, *, filters, query, params, limit) -> list[dict[str, Any]]:
        try:
            if query:
                rows = _fetch_sql(connection, query, params or [])
            else:
                rows = fetch_audio_files(connection, filters or {}, limit)
        except DatasetError as exc:
            if "does not exist" in str(exc):
                raise DatasetError("corpora.audio_files is not available") from exc
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "does not exist" in message or "not found" in message:
                raise DatasetError("corpora.audio_files is not available") from exc
            raise DatasetError(f"corpus query failed: {exc}") from exc
        if query and limit is not None:
            rows = rows[: int(limit)]
        if not rows:
            raise DatasetError("corpus query returned no rows")
        return rows

    def _sample_from_row(self, row: dict[str, Any]) -> AudioSample:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        if not isinstance(metadata, dict):
            raise DatasetError(f"case {row['case_id']} metadata is not an object")
        metadata["eval_kind"] = row["kind"]
        if row["corpus_path"]:
            metadata["corpus_path"] = row["corpus_path"]
        if row["kind"] == "synthetic":
            metadata["mix"] = {
                "protocol": row["mix_protocol"],
                "seed": int(row["mix_seed"]),
                "noise_start": None if row["noise_start"] is None else int(row["noise_start"]),
                "max_seconds": None if row["max_seconds"] is None else float(row["max_seconds"]),
                "prevent_clip": bool(row["prevent_clip"]),
                "hard_clip": bool(row["hard_clip"]),
            }
        return AudioSample(
            utterance_id=row["case_id"],
            dataset=row["dataset"],
            split=row["split"],
            audio_path=self._resolve(row["audio_path"]),
            sample_rate=None if row["sample_rate"] is None else int(row["sample_rate"]),
            transcript=row["transcript"],
            clean_path=self._resolve(row["clean_path"]),
            noisy_path=self._resolve(row["noisy_path"]),
            noise_path=self._resolve(row["noise_path"]),
            snr_db=None if row["snr_db"] is None else float(row["snr_db"]),
            speaker_id=row["speaker_id"],
            metadata=metadata,
        )

    def _store_audio(self, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if not path.is_file():
            raise DatasetError(f"audio file not found: {path}")
        self._ensure_inside(path)
        relative = Path(os.path.relpath(path, self.path.parent)).as_posix()
        if relative.startswith("/"):
            raise DatasetError(f"refusing to store an absolute audio path: {path}")
        return relative

    def _ensure_inside(self, path: Path) -> None:
        # Lexical check, so a corpus symlink under data/corpora stays portable.
        audio = os.path.normpath(path if path.is_absolute() else self.project_root / path)
        roots = (os.path.normpath(self.project_root), os.path.normpath(self.corpora_root))
        for root in roots:
            try:
                Path(audio).relative_to(root)
                return
            except ValueError:
                continue
        raise DatasetError(
            f"audio is outside the project and corpora roots: {path}. "
            "Place it inside the project so the database can store a relative path."
        )

    def _corpus_key(self, stored: str | None) -> str | None:
        if stored is None:
            return None
        audio = os.path.normpath(self._absolute(stored))
        root = os.path.normpath(self.corpora_root)
        try:
            relative = Path(audio).relative_to(root)
        except ValueError:
            return None
        text = relative.as_posix()
        if text.startswith(".."):
            return None
        return text

    def _corpus_row(self, corpus_path: str | None) -> dict[str, Any] | None:
        if not corpus_path or not self.path.is_file():
            return None
        connection = self._connect(read_only=True)
        try:
            columns = ", ".join(AUDIO_FILE_COLUMNS)
            try:
                cursor = connection.execute(
                    f"SELECT {columns} FROM corpora.audio_files WHERE path = ?",
                    [corpus_path],
                )
            except Exception:
                return None
            row = cursor.fetchone()
            if row is None:
                return None
            names = [item[0] for item in cursor.description]
            return dict(zip(names, row, strict=True))
        finally:
            connection.close()

    def _absolute(self, stored: str | None) -> Path:
        if stored is None:
            raise DatasetError("audio path is missing")
        path = Path(stored)
        if not path.is_absolute():
            path = self.path.parent / path
        return path

    def _resolve(self, stored: str | None) -> str | None:
        if stored is None:
            return None
        return self._absolute(stored).as_posix()

    def _connect(self, *, read_only: bool = False):
        try:
            import duckdb
        except ImportError as exc:
            raise DatasetError("duckdb is not installed in this environment") from exc
        try:
            return duckdb.connect(str(self.path), read_only=read_only)
        except duckdb.Error as exc:
            raise DatasetError(f"failed to open DuckDB: {exc}") from exc


def _manifest_kind(sample: AudioSample) -> str:
    clean = bool(sample.clean_path)
    noisy = bool(sample.noisy_path)
    noise = bool(sample.noise_path)
    audio = bool(sample.audio_path)
    if clean and noisy and noise:
        raise DatasetError(f"{sample.utterance_id} has both noisy_path and noise_path")
    if clean and noisy:
        return "paired"
    if clean and noise:
        if sample.snr_db is None:
            raise DatasetError(f"{sample.utterance_id} has clean and noise paths but no snr_db")
        return "synthetic"
    if noisy:
        return "noisy"
    if clean or audio:
        return "clean"
    raise DatasetError(f"{sample.utterance_id} has no evaluation audio")


def _corpus_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in ("file_name", "utterance_id", "track_type", "noise_type", "scene_code"):
        if row.get(key) is not None:
            metadata[key] = row[key]
    return metadata


def _fetch_sql(connection, query: str, params: list[Any]) -> list[dict[str, Any]]:
    text = _read_only_select(query)
    if not _ORDER_BY.search(text):
        text = f"SELECT * FROM ({text}) AS audio_files_query ORDER BY path"
    try:
        cursor = connection.execute(text, params)
    except Exception as exc:
        raise DatasetError(f"corpus query failed: {exc}") from exc
    names = [item[0] for item in cursor.description]
    if "path" not in names or "corpus" not in names:
        raise DatasetError("corpus query must return path and corpus")
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _read_only_select(query: str) -> str:
    text = query.strip()
    if text.endswith(";"):
        text = text[:-1].strip()
    if not text.lower().startswith("select") or ";" in text:
        raise DatasetError("query must be a single SELECT")
    if "corpora.audio_files" not in text.lower():
        raise DatasetError("query must read corpora.audio_files")
    found = _WRITE_SQL.search(text)
    if found:
        raise DatasetError(f"query must be read-only, found {found.group(1).lower()}")
    return text


def _check_set_name(name: str) -> None:
    if not isinstance(name, str) or SET_NAME_RE.fullmatch(name) is None:
        raise DatasetError("set name must use letters, digits, '.', '_' or '-', up to 128 characters")


def _optional_seconds(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise DatasetError(f"{where} must be a positive number")
    return float(value)


def _optional_bool(value: Any, where: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise DatasetError(f"{where} must be a boolean")
    return value


def _frame_count(path: Path, max_seconds: float | None) -> tuple[int, int]:
    try:
        import torchaudio
    except ImportError as exc:
        raise DatasetError("torchaudio is required to register synthetic audio") from exc
    if not path.is_file():
        raise DatasetError(f"audio file not found: {path}")
    info = torchaudio.info(str(path))
    frames = int(info.num_frames)
    rate = int(info.sample_rate)
    if frames < 1 or rate < 1:
        raise DatasetError(f"audio has no frames: {path}")
    if max_seconds is not None:
        frames = min(frames, max(1, int(max_seconds * rate)))
    return frames, rate


def _rate_of(path: Path) -> int:
    _, rate = _frame_count(path, None)
    return rate
