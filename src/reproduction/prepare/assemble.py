"""Build the VoiceBank fixed-pair graph shared by MP-SENet and MossFormerGAN.

Recorded-pair ``case_id`` is the recipe hash. The same 16 kHz files therefore
produce one case row. ``cases.protocol_id`` is ``mp_senet.fixed_pair`` when
that protocol is present. Each Dataset still carries its own protocol, and
training must read that protocol for normalization, crop, and order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.recipe import canonical_recipe
from src.reproduction.db.graph import Graph, logical_bindings
from src.reproduction.db.validate import compute_selection_uses_test, validate_graph
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import (
    case_content_hash,
    case_set_id_for,
    catalog_id_for,
    dataset_id_for,
    digest,
    protocol_id_for,
    recipe_case_id,
    validate_deviations,
    validate_official_ref,
    validate_rule,
)
from src.reproduction.prepare.lists import holdout_split, speaker_of

SHARED_PROTOCOL_NAME = "mp_senet.fixed_pair"
DATASET_FILES = (
    "mp_senet.vb28.official.yaml",
    "mp_senet.vb28.safe.yaml",
    "mossformergan.vb28.official.yaml",
    "mossformergan.vb28.safe.yaml",
)
PROTOCOL_FILES = (
    "mp_senet.fixed_pair.yaml",
    "mossformergan_se_16k.fixed_pair.yaml",
)


def load_protocols(config_root: Path) -> list[dict[str, Any]]:
    return [load_protocol(config_root / "protocols" / name) for name in PROTOCOL_FILES]


def load_protocol(path: Path) -> dict[str, Any]:
    document = _yaml(path)
    rule = validate_rule(document["algorithm"], document["rule"])
    official = validate_official_ref(document["official_ref"])
    deviations = validate_deviations(document.get("deviations") or [])
    parent = document.get("parent_protocol_id")
    protocol_id = protocol_id_for(
        algorithm=document["algorithm"],
        algorithm_version=int(document["algorithm_version"]),
        kind=document["kind"],
        rule=rule,
        deviation_class=document["deviation_class"],
        parent=parent,
    )
    return {
        "protocol_id": protocol_id,
        "name": document["name"],
        "version": int(document["version"]),
        "kind": document["kind"],
        "algorithm": document["algorithm"],
        "algorithm_version": int(document["algorithm_version"]),
        "rule_json": rule,
        "official_ref_json": official,
        "deviation_class": document["deviation_class"],
        "parent_protocol_id": parent,
        "deviations_json": deviations,
    }


def load_dataset_specs(config_root: Path) -> list[dict[str, Any]]:
    return [_dataset_spec(_yaml(config_root / "datasets" / name)) for name in DATASET_FILES]


def load_layout(config_root: Path) -> dict[str, Any]:
    document = _yaml(config_root / "voicebank_vb28.yaml")
    speakers = document.get("holdout_speakers")
    if not isinstance(speakers, list) or not all(isinstance(item, str) for item in speakers):
        raise ReproductionError("holdout_speakers must be a list of strings")
    document["holdout_speakers"] = tuple(speakers)
    for key in (
        "corpus",
        "train_clean_subset",
        "train_noisy_subset",
        "test_clean_subset",
        "test_noisy_subset",
        "train_text_subset",
        "test_text_subset",
        "derived_rel_root",
    ):
        if not isinstance(document.get(key), str) or not document[key]:
            raise ReproductionError(f"voicebank layout is missing {key}")
    document["source_rate"] = int(document["source_rate"])
    document["target_rate"] = int(document["target_rate"])
    return document


def build_voicebank_graph(
    *,
    protocols: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    list_meta: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    holdout: tuple[str, ...],
    utterances: dict[str, dict[str, Any]],
    catalog_source_kind: str,
    source_db_sha256: str | None,
    resource_db: str | None,
) -> Graph:
    """Create protocols, shared cases, case sets, and the requested datasets."""
    by_name = {row["name"]: row for row in protocols}
    if len(by_name) != len(protocols):
        raise ReproductionError("protocol names must be unique")
    protocol_for_cases = _case_protocol_id(by_name)
    cases = {
        utterance_id: _case(utterances[utterance_id], protocol_for_cases)
        for utterance_id in _required_ids(datasets, list_meta, train_ids, test_ids, holdout)
    }
    case_sets = []
    dataset_rows = []
    for spec in datasets:
        model_lists = list_meta[spec["list_model"]]
        bindings = []
        for slot in ("train", "valid", "test"):
            role = spec["bindings"][slot]
            ids = _ids_for(role, model_lists, train_ids, test_ids, holdout)
            expected = spec["expect"][slot]
            if len(ids) != expected:
                raise ReproductionError(
                    f"{spec['name']} {slot} has {len(ids)} utterances, expected {expected}"
                )
            case_set = _case_set(
                name=_case_set_name(spec["name"], slot),
                split=slot,
                role=role,
                ids=ids,
                cases=cases,
                list_meta=model_lists,
                holdout=holdout,
            )
            case_sets.append(case_set)
            bindings.append(
                {
                    "slot": f"{slot}.cases",
                    "case_set_id": case_set["case_set_id"],
                    "material_set_id": None,
                }
            )
        dataset_rows.append(
            _dataset(
                spec,
                by_name[spec["protocol"]]["protocol_id"],
                bindings,
                case_sets,
                {row["case_id"]: row for row in cases.values()},
            )
        )
    corpus_assets, derived_assets = _assets(utterances, cases)
    catalog, catalog_rows = _catalog(corpus_assets, source_kind=catalog_source_kind, source_db_sha256=source_db_sha256)
    for asset in corpus_assets:
        asset["catalog_id"] = catalog["catalog_id"]
    graph = Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: catalog_rows},
        derivations=[utterances[next(iter(cases))]["derivation"]],
        assets=corpus_assets + derived_assets,
        protocols=list(protocols),
        cases=list(cases.values()),
        case_sets=case_sets,
        datasets=dataset_rows,
    )
    validate_graph(graph, resource_db=resource_db)
    return graph


def make_corpus_asset(
    *,
    rel_path: str,
    subset: str,
    speaker_id: str,
    utterance_id: str,
    sample_rate: int,
    frames: int,
    channels: int,
    size_bytes: int,
    sha256: str,
    role: str,
) -> dict[str, Any]:
    return {
        "asset_id": f"corpora/{rel_path}",
        "storage_root": "corpora",
        "rel_path": rel_path,
        "corpus": "VoiceBank-DEMAND",
        "subset": subset,
        "speaker_id": speaker_id,
        "utterance_key": utterance_id,
        "sample_rate": sample_rate,
        "frames": frames,
        "channels": channels,
        "sample_format": "PCM_16",
        "size_bytes": size_bytes,
        "sha256": sha256,
        "integrity": "sha256",
        "catalog_id": None,
        "derivation_id": None,
        "derived_from": None,
        "build_id": None,
        "extra_json": {"official_id": utterance_id, "role": role},
    }


def make_derived_asset(
    *,
    rel_path: str,
    subset: str,
    speaker_id: str,
    utterance_id: str,
    frames: int,
    size_bytes: int,
    sha256: str,
    derivation_id: str,
    derived_from: str,
    role: str,
    source_frames: int,
    source_rate: int,
    clipped_samples: int,
) -> dict[str, Any]:
    return {
        "asset_id": f"derived/{rel_path}",
        "storage_root": "derived",
        "rel_path": rel_path,
        "corpus": "VoiceBank-DEMAND",
        "subset": subset,
        "speaker_id": speaker_id,
        "utterance_key": utterance_id,
        "sample_rate": 16000,
        "frames": frames,
        "channels": 1,
        "sample_format": "PCM_16",
        "size_bytes": size_bytes,
        "sha256": sha256,
        "integrity": "sha256",
        "catalog_id": None,
        "derivation_id": derivation_id,
        "derived_from": derived_from,
        "build_id": None,
        "extra_json": {
            "official_id": utterance_id,
            "role": role,
            "source_frames": source_frames,
            "source_sample_rate": source_rate,
            "clipped_samples": clipped_samples,
        },
    }


def _dataset_spec(document: dict[str, Any]) -> dict[str, Any]:
    bindings = document["bindings"]
    expect = document["expect"]
    if set(bindings) != {"train", "valid", "test"} or set(expect) != {"train", "valid", "test"}:
        raise ReproductionError(f"dataset {document.get('name')} must bind train, valid, and test")
    deviations = validate_deviations(document["deviations"])
    return {
        "name": document["name"],
        "version": int(document["version"]),
        "protocol": document["protocol"],
        "profile": document["profile"],
        "material_origin": document["material_origin"],
        "claim_label": document["claim_label"],
        "list_model": document["list_model"],
        "selection_note": document["selection_note"],
        "bindings": {slot: bindings[slot] for slot in ("train", "valid", "test")},
        "expect": {slot: int(expect[slot]) for slot in ("train", "valid", "test")},
        "deviations": deviations,
    }


def _required_ids(datasets, list_meta, train_ids, test_ids, holdout) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for spec in datasets:
        model_lists = list_meta[spec["list_model"]]
        for slot in ("train", "valid", "test"):
            for utterance_id in _ids_for(spec["bindings"][slot], model_lists, train_ids, test_ids, holdout):
                if utterance_id not in seen:
                    seen.add(utterance_id)
                    found.append(utterance_id)
    return found


def _ids_for(role: str, model_lists: dict[str, Any], train_ids: list[str], test_ids: list[str], holdout) -> list[str]:
    """Use the pinned list for this model when it is the canonical id sequence.

    MP-SENet and MossFormerGAN currently name the same ids in the same order.
    The model list is still what selects membership, so a future difference
    is not silently replaced by the other model's order.
    """
    model_train = model_lists["train"]["ids"]
    model_test = model_lists["test"]["ids"]
    if role == "official_train":
        _same_ids("train", model_train, train_ids)
        return list(model_train)
    if role == "official_test":
        _same_ids("test", model_test, test_ids)
        return list(model_test)
    if role in ("safe_train", "safe_valid"):
        _same_ids("train", model_train, train_ids)
        train, valid = holdout_split(model_train, holdout)
        return train if role == "safe_train" else valid
    raise ReproductionError(f"unknown dataset role: {role}")


def _same_ids(split: str, left: list[str], right: list[str]) -> None:
    if left != right:
        raise ReproductionError(f"{split} list ids differ from the resolved VoiceBank ids")


def _case_set_name(dataset_name: str, slot: str) -> str:
    prefix, _, suffix = dataset_name.rpartition(".")
    if not prefix or suffix not in ("official", "safe"):
        raise ReproductionError(f"dataset name must end in .official or .safe: {dataset_name}")
    return f"{prefix}.{slot}.{suffix}"


def _case_set(name, split, role, ids, cases, list_meta, holdout) -> dict[str, Any]:
    case_ids = []
    for utterance_id in ids:
        case = cases.get(utterance_id)
        if case is None:
            raise ReproductionError(f"missing case for {utterance_id}")
        if speaker_of(utterance_id) != case["speaker_id"]:
            raise ReproductionError(f"{utterance_id} speaker does not match its case")
        case_ids.append(case["case_id"])
    rule = _rule(role, list_meta, holdout)
    origin = "official" if role.startswith("official") else "derived_from_official"
    return {
        "case_set_id": case_set_id_for(
            name=name, version=1, split_role=split, rule=rule, case_ids=case_ids
        ),
        "name": name,
        "version": 1,
        "split_role": split,
        "origin": origin,
        "selection_rule_json": rule,
        "provenance_json": _provenance(role, split, list_meta),
        "member_count": len(case_ids),
        "list_sha256": digest(case_ids),
        "items": [{"position": index, "case_id": case_id} for index, case_id in enumerate(case_ids)],
    }


def _rule(role: str, list_meta: dict[str, Any], holdout: tuple[str, ...]) -> dict[str, Any]:
    if role == "official_train":
        return {"rule": "official_filelist", "file": Path(list_meta["train"]["file"]).name}
    if role == "official_test":
        return {"rule": "official_filelist", "file": Path(list_meta["test"]["file"]).name}
    speakers = list(holdout)
    if role == "safe_train":
        return {"rule": "subset_exclude_speakers", "exclude": speakers}
    if role == "safe_valid":
        return {"rule": "speaker_holdout", "speakers": speakers}
    raise ReproductionError(f"unknown dataset role: {role}")


def _provenance(role: str, split: str, list_meta: dict[str, Any]) -> dict[str, Any]:
    side = "test" if role == "official_test" else "train"
    body = {
        "source": list_meta["source"],
        "repo": list_meta["repo"],
        "commit": list_meta["commit"],
        "path": list_meta[side]["path"],
        "file": list_meta[side]["file"],
        "sha256": list_meta[side]["sha256"],
    }
    if role == "safe_train":
        body["source"] = list_meta["source"] + " speaker holdout"
        body["note"] = "official training order with holdout speakers removed; no random seed"
    elif role == "safe_valid":
        body["source"] = list_meta["source"] + " speaker holdout"
        body["note"] = "official training order restricted to holdout speakers; no random seed"
    elif split == "valid" and role == "official_test":
        body["note"] = "official test list used as validation and checkpoint selection"
    return body


def _dataset(spec, protocol_id, bindings, case_sets, cases) -> dict[str, Any]:
    encoded = logical_bindings(bindings)
    case_set_index = {row["case_set_id"]: row for row in case_sets}
    provisional = {
        "name": spec["name"],
        "declared_status": "ready",
        "bindings": encoded,
    }
    selection = compute_selection_uses_test(provisional, case_set_index, {}, cases)
    if spec["profile"] == "leakage_safe" and selection:
        raise ReproductionError(f"{spec['name']} is leakage_safe but selection uses test")
    if spec["profile"] == "official_faithful" and not selection:
        raise ReproductionError(f"{spec['name']} is official_faithful but selection does not use test")
    dataset_id = dataset_id_for(
        protocol_id=protocol_id,
        bindings=encoded,
        profile=spec["profile"],
        material_origin=spec["material_origin"],
        claim_label=spec["claim_label"],
        selection_uses_test=selection,
        declared_status="ready",
    )
    return {
        "dataset_id": dataset_id,
        "name": spec["name"],
        "version": spec["version"],
        "protocol_id": protocol_id,
        "profile": spec["profile"],
        "material_origin": spec["material_origin"],
        "claim_label": spec["claim_label"],
        "declared_status": "ready",
        "blocking_json": [],
        "selection_uses_test": selection,
        "selection_note": spec["selection_note"],
        "deviations_json": spec["deviations"],
        "bindings": encoded,
    }


def _case(utterance: dict[str, Any], protocol_id: str) -> dict[str, Any]:
    clean = utterance["clean"]["derived"]
    noisy = utterance["noisy"]["derived"]
    if clean["frames"] != noisy["frames"] or clean["sample_rate"] != noisy["sample_rate"]:
        raise ReproductionError(f"{utterance['utterance_id']} clean and noisy derivatives differ in shape")
    document = canonical_recipe(
        {
            "recipe_version": 1,
            "sources": {
                "clean": {"path": "../" + clean["asset_id"], "sha256": clean["sha256"]},
                "noisy": {"path": "../" + noisy["asset_id"], "sha256": noisy["sha256"]},
            },
            "operations": [],
            "output": {"sample_rate": 16000, "reference": "dry_clean"},
        }
    )
    row = {
        "case_id": recipe_case_id(document),
        "doc_kind": "recipe_v1",
        "doc_json": document,
        "case_role": "recorded_pair",
        "protocol_id": protocol_id,
        "clean_asset_id": clean["asset_id"],
        "noisy_asset_id": noisy["asset_id"],
        "noise_asset_id": None,
        "teacher_kind": "dry_clean",
        "sample_rate": 16000,
        "frames": int(noisy["frames"]),
        "speaker_id": utterance["speaker_id"],
        "utterance_key": utterance["utterance_id"],
        "transcript": utterance.get("transcript"),
        "build_id": None,
        "gen_index": None,
        "content_hash": "",
    }
    row["content_hash"] = case_content_hash(row)
    return row


def _case_protocol_id(protocols_by_name: dict[str, dict[str, Any]]) -> str:
    chosen = protocols_by_name.get(SHARED_PROTOCOL_NAME)
    if chosen is None:
        if len(protocols_by_name) != 1:
            raise ReproductionError("shared VoiceBank cases need the mp_senet.fixed_pair protocol")
        chosen = next(iter(protocols_by_name.values()))
    return chosen["protocol_id"]


def _assets(utterances: dict[str, dict[str, Any]], cases: dict[str, dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    corpus = []
    derived = []
    seen: set[str] = set()
    for utterance_id in cases:
        utterance = utterances[utterance_id]
        for side in ("clean", "noisy"):
            for kind in ("corpus", "derived"):
                asset = utterance[side][kind]
                if asset["asset_id"] in seen:
                    raise ReproductionError(f"duplicate asset {asset['asset_id']}")
                seen.add(asset["asset_id"])
                if kind == "corpus":
                    corpus.append(asset)
                else:
                    derived.append(asset)
    return corpus, derived


def _catalog(assets: list[dict[str, Any]], *, source_kind: str, source_db_sha256: str | None) -> tuple[dict, list[dict]]:
    rows = [
        {
            "path": asset["rel_path"],
            "corpus": asset["corpus"],
            "subset": asset["subset"],
            "sample_rate": asset["sample_rate"],
            "frames": asset["frames"],
            "channels": asset["channels"],
            "speaker_id": asset["speaker_id"],
            "utterance_id": asset["utterance_key"],
        }
        for asset in assets
    ]
    ordered = sorted(rows, key=lambda row: row["path"])
    counts: dict[str, int] = {}
    for row in ordered:
        counts[row["corpus"]] = counts.get(row["corpus"], 0) + 1
    catalog = {
        "catalog_id": catalog_id_for(ordered),
        "source_kind": source_kind,
        "corpora_json": counts,
        "row_count": len(ordered),
        "source_db_sha256": source_db_sha256,
    }
    return catalog, ordered


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReproductionError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ReproductionError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ReproductionError(f"{path} must be a mapping")
    return document
