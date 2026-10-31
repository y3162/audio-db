"""Named evaluation sets stored in DuckDB.

Manifests and corpus queries are import paths. After registration, samples are
read from the ``eval`` schema. Generation is described only by the recipe JSON.
Audio paths are stored relative to the database file. This module does not
create or modify ``corpora.audio_files``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.db.eval_schema import create_eval_schema
from src.data.models.audio_file import AUDIO_FILE_COLUMNS
from src.data.recipe import (
    canonical_recipe,
    case_content_hash,
    case_id_for,
    sha256_file,
)
from src.data.recipe import json_hash as recipe_json_hash
from src.evaluation.audio.mixing import PEAK_LIMIT
from src.evaluation.audio.render import prepare_noise_start
from src.evaluation.datasets.duckdb import fetch_audio_files
from src.evaluation.datasets.manifest import load_manifest
from src.evaluation.errors import DatasetError
from src.evaluation.identity import canonical_snr, derive_seed
from src.evaluation.sample import AudioSample

SET_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
KINDS = ("clean", "noisy", "paired", "synthetic")
CORPUS_ROLES = ("clean", "noisy", "synthetic")
_WRITE_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|copy|attach|pragma)\b",
    re.IGNORECASE,
)
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_BUILDER_KEYS = {"mix_seed", "mix_protocol", "prevent_clip", "hard_clip", "max_seconds", "noise_start"}

CASE_COLUMNS = (
    "case_id",
    "kind",
    "dataset",
    "split",
    "sample_rate",
    "transcript",
    "speaker_id",
    "recipe_json",
    "content_hash",
    "metadata",
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
    recipe: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        return case_content_hash(
            {
                "kind": self.kind,
                "dataset": self.dataset,
                "split": self.split,
                "sample_rate": self.sample_rate,
                "transcript": self.transcript,
                "speaker_id": self.speaker_id,
                "recipe": self.recipe,
                "metadata": self.metadata,
            }
        )


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
            try:
                create_eval_schema(connection)
            except ValueError as exc:
                raise DatasetError(str(exc)) from exc
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

    def register_recipe(
        self,
        recipe: dict[str, Any],
        set_name: str,
        *,
        dataset: str,
        kind: str | None = None,
        split: str | None = None,
        transcript: str | None = None,
        speaker_id: str | None = None,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> RegistrationReport:
        """Register one case from a recipe document."""
        try:
            document = canonical_recipe(recipe)
        except ValueError as exc:
            raise DatasetError(str(exc)) from exc
        self._check_source_files(document)
        case = self._make_case(
            kind=kind or _kind_for_recipe(document),
            dataset=dataset,
            split=split,
            transcript=transcript,
            speaker_id=speaker_id,
            recipe=document,
            metadata=dict(metadata or {}),
            source_id=source_id or case_id_for(document),
        )
        return self._register([case], set_name, description)

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
        protocol: str = "rms_v1",
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
        if protocol != "rms_v1":
            raise DatasetError(f"unknown mixing protocol: {protocol}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DatasetError("seed must be an integer")
        if hard_clip:
            raise DatasetError("hard clipping is not a recipe operation")
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
                        max_seconds=clean_limit,
                        prevent_clip=bool(prevent_clip),
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

    def export_staging(self, staging: str | Path):
        from src.data.staging import export_database

        return export_database(self.path, staging)

    def _register(self, cases: list[_Case], set_name: str, description: str | None) -> RegistrationReport:
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
            connection.execute(
                f"""
                INSERT INTO eval.cases ({", ".join(CASE_COLUMNS)})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?::JSON, ?, ?::JSON)
                """,
                [
                    case.case_id,
                    case.kind,
                    case.dataset,
                    case.split,
                    case.sample_rate,
                    case.transcript,
                    case.speaker_id,
                    json.dumps(case.recipe, ensure_ascii=False, sort_keys=True),
                    digest,
                    json.dumps(case.metadata, ensure_ascii=False, sort_keys=True),
                ],
            )
            return "inserted"
        if row[0] != digest and not _same_registered_case(connection, case):
            source = case.metadata.get("source_id", case.case_id)
            raise DatasetError(f"case {case.case_id} ({source}) is already registered with different content")
        return "unchanged"

    def _put_set(self, connection, name: str, description: str | None, case_ids: list[str]) -> str:
        digest = recipe_json_hash({"case_ids": case_ids, "description": description})
        existing = connection.execute(
            "SELECT content_hash FROM eval.sets WHERE name = ?",
            [name],
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO eval.sets (name, description, content_hash) VALUES (?, ?, ?)",
                [name, description, digest],
            )
            for position, case_id in enumerate(case_ids):
                connection.execute(
                    "INSERT INTO eval.set_items (set_name, position, case_id) VALUES (?, ?, ?)",
                    [name, position, case_id],
                )
            return "created"
        if existing[0] != digest:
            raise DatasetError(
                f"evaluation set {name} is already registered with different cases, order, or description"
            )
        return "unchanged"

    def _case_from_manifest(self, sample: AudioSample) -> _Case:
        kind = _manifest_kind(sample)
        metadata = {key: value for key, value in sample.metadata.items() if key not in _BUILDER_KEYS}
        clean = self._store_audio(sample.clean_path)
        noisy = self._store_audio(sample.noisy_path)
        noise = self._store_audio(sample.noise_path)
        audio = self._store_audio(sample.audio_path)
        if kind == "clean" and clean is None:
            clean = audio
        corpus_path = self._corpus_key(clean or noisy or audio or noise)
        if corpus_path:
            metadata["corpus_path"] = corpus_path
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
        primary = clean or noisy or audio
        if sample_rate is None:
            sample_rate = _rate_of(self._absolute(primary))
        else:
            file_rate = _rate_of(self._absolute(primary))
            if int(sample_rate) != file_rate:
                raise DatasetError(
                    f"{sample.utterance_id} sample_rate {sample_rate} does not match the file ({file_rate})"
                )
            sample_rate = int(sample_rate)
        if kind == "synthetic":
            recipe = self._synthetic_recipe(
                clean,
                noise,
                snr_db=canonical_snr(sample.snr_db),
                metadata=sample.metadata,
                seed_parts=("manifest", sample.utterance_id),
                sample_rate=sample_rate,
            )
        elif kind == "paired":
            recipe = self._recorded_recipe(sample_rate, "dry_clean", clean=clean, noisy=noisy)
        elif kind == "noisy":
            recipe = self._recorded_recipe(sample_rate, "none", noisy=noisy)
        else:
            recipe = self._recorded_recipe(sample_rate, "dry_clean", clean=clean)
        return self._make_case(
            kind=kind,
            dataset=sample.dataset,
            split=sample.split,
            transcript=transcript,
            speaker_id=None if speaker_id is None else str(speaker_id),
            recipe=recipe,
            metadata=metadata,
            source_id=sample.utterance_id,
            sample_rate=sample_rate,
        )

    def _case_from_corpus_row(self, row: dict[str, Any], role: str) -> _Case:
        stored = self._store_audio(str(self.corpora_root / row["path"]))
        kind = "noisy" if role == "noisy" else "clean"
        rate = int(row["sample_rate"]) if row.get("sample_rate") is not None else _rate_of(self._absolute(stored))
        reference = "none" if kind == "noisy" else "dry_clean"
        recipe = self._recorded_recipe(
            rate,
            reference,
            clean=stored if kind == "clean" else None,
            noisy=stored if kind == "noisy" else None,
        )
        return self._make_case(
            kind=kind,
            dataset=row["corpus"],
            split=row.get("subset") or row.get("eval_split"),
            transcript=row.get("transcript"),
            speaker_id=None if row.get("speaker_id") is None else str(row["speaker_id"]),
            recipe=recipe,
            metadata=_corpus_metadata(row),
            source_id=row["path"],
            sample_rate=rate,
            corpus_path=row["path"],
        )

    def _synthetic_corpus_case(
        self,
        row: dict[str, Any],
        noises: list[dict[str, Any]],
        *,
        snr_db: float,
        seed: int,
        max_seconds: float | None,
        prevent_clip: bool,
    ) -> _Case:
        offset = derive_seed(seed, "noise-pair", row["path"]) % len(noises)
        noise_row = noises[offset]
        clean_stored = self._store_audio(str(self.corpora_root / row["path"]))
        noise_stored = self._store_audio(str(self.corpora_root / noise_row["path"]))
        rate = int(row["sample_rate"]) if row.get("sample_rate") is not None else _rate_of(self._absolute(clean_stored))
        snr = canonical_snr(snr_db)
        # ``rms_v1`` is the historical seed input. Using the operation name would
        # change noise crops that are already stored in production recipes.
        mix_seed = derive_seed("eval-mix", seed, clean_stored, noise_stored, snr, "rms_v1", max_seconds)
        recipe = self._synthetic_recipe(
            clean_stored,
            noise_stored,
            snr_db=snr,
            metadata={"mix_seed": mix_seed, "max_seconds": max_seconds, "prevent_clip": prevent_clip},
            seed_parts=(),
            sample_rate=rate,
        )
        return self._make_case(
            kind="synthetic",
            dataset=row["corpus"],
            split=row.get("subset") or row.get("eval_split"),
            transcript=row.get("transcript"),
            speaker_id=None if row.get("speaker_id") is None else str(row["speaker_id"]),
            recipe=recipe,
            metadata=_corpus_metadata(row),
            source_id=row["path"],
            sample_rate=rate,
            corpus_path=row["path"],
        )

    def _synthetic_recipe(
        self,
        clean: str | None,
        noise: str | None,
        *,
        snr_db: float | None,
        metadata: dict[str, Any],
        seed_parts: tuple[object, ...],
        sample_rate: int,
    ) -> dict[str, Any]:
        if clean is None or noise is None or snr_db is None:
            raise DatasetError("synthetic cases require clean, noise, and snr_db")
        protocol = metadata.get("mix_protocol", "rms_v1")
        if protocol != "rms_v1":
            raise DatasetError(f"unknown mix protocol {protocol!r}")
        if _optional_bool(metadata.get("hard_clip"), "hard_clip"):
            raise DatasetError("hard clipping is not a recipe operation")
        supplied = metadata.get("mix_seed")
        if supplied is None:
            mix_seed = derive_seed(*seed_parts, clean, noise, snr_db, protocol)
        elif isinstance(supplied, bool) or not isinstance(supplied, int):
            raise DatasetError("mix_seed must be an integer")
        else:
            mix_seed = supplied
        max_seconds = _optional_seconds(metadata.get("max_seconds"), "max_seconds")
        prevent_clip = _optional_bool(metadata.get("prevent_clip"), "prevent_clip")
        supplied_start = metadata.get("noise_start")
        if supplied_start is None:
            noise_start = prepare_noise_start(
                self._absolute(clean),
                self._absolute(noise),
                sample_rate,
                mix_seed,
                max_seconds,
            )
        elif isinstance(supplied_start, bool) or not isinstance(supplied_start, int) or supplied_start < 0:
            raise DatasetError("noise_start must be a non-negative integer")
        else:
            noise_start = supplied_start
        sources = {"clean": self._source_entry(clean), "noise": self._source_entry(noise)}
        operations: list[dict[str, Any]] = [
            {
                "type": "additive_noise",
                "version": 1,
                "source": "noise",
                "snr_db": snr_db,
                "noise_start": noise_start,
                "noise_start_unit": "sample",
                "seed": mix_seed,
            }
        ]
        if prevent_clip:
            operations.append({"type": "peak_normalize", "version": 1, "limit": PEAK_LIMIT})
        output: dict[str, Any] = {"sample_rate": sample_rate, "reference": "dry_clean"}
        if max_seconds is not None:
            output["max_seconds"] = max_seconds
        return canonical_recipe(
            {"recipe_version": 1, "sources": sources, "operations": operations, "output": output}
        )

    def _recorded_recipe(
        self,
        sample_rate: int,
        reference: str,
        *,
        clean: str | None = None,
        noisy: str | None = None,
    ) -> dict[str, Any]:
        sources = {}
        if clean is not None:
            sources["clean"] = self._source_entry(clean)
        if noisy is not None:
            sources["noisy"] = self._source_entry(noisy)
        return canonical_recipe(
            {
                "recipe_version": 1,
                "sources": sources,
                "operations": [],
                "output": {"sample_rate": sample_rate, "reference": reference},
            }
        )

    def _make_case(
        self,
        *,
        kind: str,
        dataset: str,
        split: str | None,
        transcript: str | None,
        speaker_id: str | None,
        recipe: dict[str, Any],
        metadata: dict[str, Any],
        source_id: str,
        sample_rate: int | None = None,
        corpus_path: str | None = None,
    ) -> _Case:
        if kind not in KINDS:
            raise DatasetError(f"unknown case kind: {kind}")
        document = canonical_recipe(recipe)
        rate = int(document["output"]["sample_rate"])
        if sample_rate is not None and int(sample_rate) != rate:
            raise DatasetError("sample_rate does not match the recipe")
        stored = dict(metadata)
        stored["source_id"] = source_id
        if corpus_path:
            stored["corpus_path"] = corpus_path
        return _Case(
            case_id=case_id_for(document),
            kind=kind,
            dataset=dataset,
            split=split,
            sample_rate=rate,
            transcript=transcript,
            speaker_id=speaker_id,
            recipe=document,
            metadata=stored,
        )

    def _source_entry(self, stored: str) -> dict[str, str]:
        path = self._absolute(stored)
        if not path.is_file():
            raise DatasetError(f"audio file not found: {path}")
        return {"path": stored, "sha256": sha256_file(path)}

    def _check_source_files(self, recipe: dict[str, Any]) -> None:
        for name, source in recipe["sources"].items():
            path = self._absolute(source["path"])
            self._ensure_inside(path)
            if not path.is_file():
                raise DatasetError(f"audio file not found: {path}")
            digest = sha256_file(path)
            if digest != source["sha256"]:
                raise DatasetError(f"source {name} changed: {source['path']}")

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
        recipe = _json_object(row["recipe_json"], f"case {row['case_id']} recipe")
        metadata = _json_object(row["metadata"], f"case {row['case_id']} metadata")
        try:
            recipe = canonical_recipe(recipe)
        except ValueError as exc:
            raise DatasetError(f"case {row['case_id']} recipe is invalid: {exc}") from exc
        if case_id_for(recipe) != row["case_id"]:
            raise DatasetError(f"case {row['case_id']} does not match its recipe")
        metadata = dict(metadata)
        metadata["eval_kind"] = row["kind"]
        metadata["recipe"] = recipe
        metadata["reference"] = recipe["output"]["reference"]
        metadata["audio_root"] = self.path.parent.as_posix()
        sources = recipe["sources"]
        snr = None
        for operation in recipe["operations"]:
            if operation["type"] == "additive_noise":
                snr = float(operation["snr_db"])
        return AudioSample(
            utterance_id=row["case_id"],
            dataset=row["dataset"],
            split=row["split"],
            sample_rate=None if row["sample_rate"] is None else int(row["sample_rate"]),
            transcript=row["transcript"],
            clean_path=self._resolve(_source_path(sources, "clean")),
            noisy_path=self._resolve(_source_path(sources, "noisy")),
            noise_path=self._resolve(_source_path(sources, "noise")),
            snr_db=snr,
            speaker_id=None if row["speaker_id"] is None else str(row["speaker_id"]),
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
        return Path(os.path.normpath(path))

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


def _kind_for_recipe(recipe: dict[str, Any]) -> str:
    if recipe["operations"]:
        return "synthetic"
    names = set(recipe["sources"])
    if names == {"noisy"}:
        return "noisy"
    if names == {"clean", "noisy"}:
        return "paired"
    if names == {"clean"}:
        return "clean"
    raise DatasetError("recipe kind could not be inferred")


def _source_path(sources: dict[str, Any], name: str) -> str | None:
    item = sources.get(name)
    if not isinstance(item, dict):
        return None
    path = item.get("path")
    return None if not isinstance(path, str) else path


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


def _json_object(value: Any, where: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where} is not JSON") from exc
    if not isinstance(value, dict):
        raise DatasetError(f"{where} must be an object")
    return value


def _same_registered_case(connection, case: _Case) -> bool:
    """True when the stored row matches this case aside from migration provenance.

    The one-time conversion keeps the previous case id in ``legacy_case_id`` and
    copies it into ``source_id``. A later ``db-init`` does not write that key
    and uses the current source id. The recipe and the other fields must match.
    The stored row is left as it is.
    """
    row = connection.execute(
        """
        SELECT kind, dataset, split, sample_rate, transcript, speaker_id,
               recipe_json::VARCHAR, metadata::VARCHAR
        FROM eval.cases WHERE case_id = ?
        """,
        [case.case_id],
    ).fetchone()
    if row is None:
        return False
    kind, dataset, split, sample_rate, transcript, speaker_id, recipe_json, metadata_json = row
    speaker = None if speaker_id is None else str(speaker_id)
    if (
        kind != case.kind
        or dataset != case.dataset
        or split != case.split
        or sample_rate != case.sample_rate
        or transcript != case.transcript
        or speaker != case.speaker_id
    ):
        return False
    try:
        stored_recipe = canonical_recipe(_json_object(recipe_json, "stored recipe"))
    except Exception:
        return False
    if stored_recipe != case.recipe:
        return False
    stored_meta = _json_object(metadata_json, "stored metadata")
    return _metadata_matches_registered(stored_meta, case.metadata)


def _metadata_matches_registered(stored: dict[str, Any], incoming: dict[str, Any]) -> bool:
    left = dict(stored)
    legacy = left.pop("legacy_case_id", None)
    right = dict(incoming)
    if left == right:
        return True
    if legacy is None or left.get("source_id") != legacy:
        return False
    left["source_id"] = right.get("source_id")
    return left == right


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
