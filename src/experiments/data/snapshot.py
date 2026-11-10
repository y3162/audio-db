"""Read a reproduction snapshot with numpy and JSON only.

This module must not import DuckDB. Learning environments such as
``.venvs/deepfilternet`` do not have it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import SCHEMA_VERSION, digest, generated_rel_path

FORMAT = "reproduction_snapshot_v1"
PART_NAMES = (
    "dataset",
    "assets",
    "cases",
    "case_set_items",
    "material_set_members",
    "docs",
    "generated",
)


@dataclass(frozen=True)
class LoadedSnapshot:
    directory: Path
    snapshot_hash: str
    dataset: dict[str, Any]
    assets: list[dict[str, Any]]
    cases: list[dict[str, Any]]
    case_set_items: list[dict[str, Any]]
    material_set_members: list[dict[str, Any]]
    docs: list[dict[str, Any]]
    generated: list[dict[str, Any]]
    manifest: dict[str, Any]

    def parts(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "assets": self.assets,
            "cases": self.cases,
            "case_set_items": self.case_set_items,
            "material_set_members": self.material_set_members,
            "docs": self.docs,
            "generated": self.generated,
        }


def snapshot_hash(parts: dict[str, Any]) -> str:
    logical = {name: digest(parts[name]) for name in PART_NAMES}
    return digest(logical)


def file_hashes(parts: dict[str, Any]) -> dict[str, str]:
    return {name: digest(parts[name]) for name in PART_NAMES}


def pack_strings(values: list[str | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.empty(len(values), dtype=np.int64)
    ends = np.empty(len(values), dtype=np.int64)
    blob = bytearray()
    for index, value in enumerate(values):
        if value is None:
            starts[index] = -1
            ends[index] = -1
            continue
        if not isinstance(value, str):
            raise ReproductionError("snapshot string columns must be text or null")
        raw = value.encode("utf-8")
        starts[index] = len(blob)
        blob.extend(raw)
        ends[index] = len(blob)
    data = np.frombuffer(bytes(blob), dtype=np.uint8).copy() if blob else np.zeros(0, dtype=np.uint8)
    return data, starts, ends


def unpack_strings(data: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> list[str | None]:
    raw = data.tobytes()
    values: list[str | None] = []
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if int(start) < 0:
            values.append(None)
        else:
            values.append(raw[int(start) : int(end)].decode("utf-8"))
    return values


def pack_ints(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def pack_optional_ints(values: list[int | None]) -> tuple[np.ndarray, np.ndarray]:
    present = np.asarray([value is not None for value in values], dtype=np.bool_)
    stored = [0 if value is None else int(value) for value in values]
    return np.asarray(stored, dtype=np.int64), present


def unpack_optional_ints(stored: np.ndarray, present: np.ndarray) -> list[int | None]:
    return [int(value) if flag else None for value, flag in zip(stored.tolist(), present.tolist(), strict=True)]


def arrays_from_assets(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    asset_id = pack_strings([row["asset_id"] for row in rows])
    rel_path = pack_strings([row["rel_path"] for row in rows])
    sha = pack_strings([row["sha256"] for row in rows])
    integrity = pack_strings([row["integrity"] for row in rows])
    return {
        "asset_id_data": asset_id[0],
        "asset_id_start": asset_id[1],
        "asset_id_end": asset_id[2],
        "rel_path_data": rel_path[0],
        "rel_path_start": rel_path[1],
        "rel_path_end": rel_path[2],
        "sample_rate": pack_ints([row["sample_rate"] for row in rows]),
        "frames": pack_ints([row["frames"] for row in rows]),
        "channels": pack_ints([row["channels"] for row in rows]),
        "sha256_data": sha[0],
        "sha256_start": sha[1],
        "sha256_end": sha[2],
        "integrity_data": integrity[0],
        "integrity_start": integrity[1],
        "integrity_end": integrity[2],
    }


def assets_from_arrays(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    ids = unpack_strings(arrays["asset_id_data"], arrays["asset_id_start"], arrays["asset_id_end"])
    rels = unpack_strings(arrays["rel_path_data"], arrays["rel_path_start"], arrays["rel_path_end"])
    shas = unpack_strings(arrays["sha256_data"], arrays["sha256_start"], arrays["sha256_end"])
    integrity = unpack_strings(arrays["integrity_data"], arrays["integrity_start"], arrays["integrity_end"])
    rates = [int(value) for value in arrays["sample_rate"].tolist()]
    frames = [int(value) for value in arrays["frames"].tolist()]
    channels = [int(value) for value in arrays["channels"].tolist()]
    rows = []
    for index, asset_id in enumerate(ids):
        rows.append(
            {
                "asset_id": asset_id,
                "rel_path": rels[index],
                "sample_rate": rates[index],
                "frames": frames[index],
                "channels": channels[index],
                "sha256": shas[index],
                "integrity": integrity[index],
            }
        )
    return rows


def arrays_from_cases(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    frames, frames_present = pack_optional_ints([row["frames"] for row in rows])
    gen_index, gen_present = pack_optional_ints([row["gen_index"] for row in rows])
    packed = {
        "sample_rate": pack_ints([row["sample_rate"] for row in rows]),
        "frames": frames,
        "frames_present": frames_present,
        "gen_index": gen_index,
        "gen_index_present": gen_present,
    }
    for name in (
        "case_id",
        "doc_kind",
        "case_role",
        "protocol_id",
        "teacher_kind",
        "speaker_id",
        "utterance_key",
        "clean_asset_id",
        "noisy_asset_id",
        "noise_asset_id",
        "build_id",
    ):
        data, start, end = pack_strings([row[name] for row in rows])
        packed[f"{name}_data"] = data
        packed[f"{name}_start"] = start
        packed[f"{name}_end"] = end
    return packed


def cases_from_arrays(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    columns = {
        name: unpack_strings(arrays[f"{name}_data"], arrays[f"{name}_start"], arrays[f"{name}_end"])
        for name in (
            "case_id",
            "doc_kind",
            "case_role",
            "protocol_id",
            "teacher_kind",
            "speaker_id",
            "utterance_key",
            "clean_asset_id",
            "noisy_asset_id",
            "noise_asset_id",
            "build_id",
        )
    }
    frames = unpack_optional_ints(arrays["frames"], arrays["frames_present"])
    gen_index = unpack_optional_ints(arrays["gen_index"], arrays["gen_index_present"])
    rates = [int(value) for value in arrays["sample_rate"].tolist()]
    rows = []
    for index in range(len(rates)):
        rows.append(
            {
                "case_id": columns["case_id"][index],
                "doc_kind": columns["doc_kind"][index],
                "case_role": columns["case_role"][index],
                "protocol_id": columns["protocol_id"][index],
                "teacher_kind": columns["teacher_kind"][index],
                "sample_rate": rates[index],
                "frames": frames[index],
                "speaker_id": columns["speaker_id"][index],
                "utterance_key": columns["utterance_key"][index],
                "clean_asset_id": columns["clean_asset_id"][index],
                "noisy_asset_id": columns["noisy_asset_id"][index],
                "noise_asset_id": columns["noise_asset_id"][index],
                "build_id": columns["build_id"][index],
                "gen_index": gen_index[index],
            }
        )
    return rows


def arrays_from_items(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    case_set = pack_strings([row["case_set_id"] for row in rows])
    case_id = pack_strings([row["case_id"] for row in rows])
    return {
        "case_set_id_data": case_set[0],
        "case_set_id_start": case_set[1],
        "case_set_id_end": case_set[2],
        "position": pack_ints([row["position"] for row in rows]),
        "case_id_data": case_id[0],
        "case_id_start": case_id[1],
        "case_id_end": case_id[2],
    }


def items_from_arrays(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    sets = unpack_strings(arrays["case_set_id_data"], arrays["case_set_id_start"], arrays["case_set_id_end"])
    cases = unpack_strings(arrays["case_id_data"], arrays["case_id_start"], arrays["case_id_end"])
    positions = [int(value) for value in arrays["position"].tolist()]
    return [
        {"case_set_id": sets[index], "position": positions[index], "case_id": cases[index]}
        for index in range(len(positions))
    ]


def arrays_from_members(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    material = pack_strings([row["material_set_id"] for row in rows])
    asset = pack_strings([row["asset_id"] for row in rows])
    group = pack_strings([row["group_key"] for row in rows])
    return {
        "material_set_id_data": material[0],
        "material_set_id_start": material[1],
        "material_set_id_end": material[2],
        "position": pack_ints([row["position"] for row in rows]),
        "asset_id_data": asset[0],
        "asset_id_start": asset[1],
        "asset_id_end": asset[2],
        "group_key_data": group[0],
        "group_key_start": group[1],
        "group_key_end": group[2],
    }


def members_from_arrays(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    sets = unpack_strings(
        arrays["material_set_id_data"], arrays["material_set_id_start"], arrays["material_set_id_end"]
    )
    assets = unpack_strings(arrays["asset_id_data"], arrays["asset_id_start"], arrays["asset_id_end"])
    groups = unpack_strings(arrays["group_key_data"], arrays["group_key_start"], arrays["group_key_end"])
    positions = [int(value) for value in arrays["position"].tolist()]
    return [
        {
            "material_set_id": sets[index],
            "position": positions[index],
            "asset_id": assets[index],
            "group_key": groups[index],
        }
        for index in range(len(positions))
    ]


def generated_path(generated_root: Path, row: dict[str, Any]) -> Path:
    rel = generated_rel_path(
        generator=row["generator"],
        build_id=row["build_id"],
        stream=row["stream"],
        gen_index=int(row["gen_index"]),
        shard_size=int(row["shard_size"]),
    )
    if rel != row["rel_path"]:
        raise ReproductionError(f"generated path does not match {row.get('asset_id')}")
    return Path(generated_root) / rel


def load_snapshot(directory: str | Path) -> LoadedSnapshot:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ReproductionError(f"snapshot manifest is missing: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ReproductionError("snapshot manifest must be an object")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("format") != FORMAT:
        raise ReproductionError("snapshot format is not reproduction_snapshot_v1")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReproductionError("snapshot schema_version is not 1")
    dataset = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    docs = _read_jsonl(directory / "docs.jsonl")
    generated = json.loads((directory / "generated.json").read_text(encoding="utf-8"))
    if not isinstance(generated, list):
        raise ReproductionError("generated.json must be a list")
    assets = assets_from_arrays(_load_npz(directory / "assets.npz"))
    cases = cases_from_arrays(_load_npz(directory / "cases.npz"))
    items = items_from_arrays(_load_npz(directory / "case_set_items.npz"))
    members = members_from_arrays(_load_npz(directory / "material_set_members.npz"))
    parts = {
        "dataset": dataset,
        "assets": assets,
        "cases": cases,
        "case_set_items": items,
        "material_set_members": members,
        "docs": docs,
        "generated": generated,
    }
    logical = file_hashes(parts)
    declared = manifest.get("files")
    if declared != logical:
        raise ReproductionError("snapshot file hashes do not match the logical rows")
    digest_value = snapshot_hash(parts)
    if manifest.get("snapshot_hash") != digest_value:
        raise ReproductionError("snapshot_hash does not match the logical rows")
    for row in generated:
        expected = generated_rel_path(
            generator=row["generator"],
            build_id=row["build_id"],
            stream=row["stream"],
            gen_index=int(row["gen_index"]),
            shard_size=int(row["shard_size"]),
        )
        if expected != row["rel_path"]:
            raise ReproductionError(f"generated asset path mismatch: {row.get('asset_id')}")
    return LoadedSnapshot(
        directory=directory,
        snapshot_hash=digest_value,
        dataset=dataset,
        assets=assets,
        cases=cases,
        case_set_items=items,
        material_set_members=members,
        docs=docs,
        generated=generated,
        manifest=manifest,
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReproductionError(f"snapshot file is missing: {path.name}")
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReproductionError(f"snapshot file is missing: {path.name}")
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ReproductionError("docs.jsonl rows must be objects")
        rows.append(item)
    return rows
