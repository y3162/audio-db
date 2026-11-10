"""Invariants for a reproduction graph.

SQL CHECKs cover single-row constraints. Cross-table rules are checked here
before a registration transaction starts.
"""

from __future__ import annotations

from typing import Any

from src.data.recipe import RecipeError, canonical_recipe
from src.experiments.data.db.graph import Graph, index_rows
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import (
    ALLOWED_SLOTS,
    ALGORITHMS,
    DEVIATION_CLASSES,
    REQUIRED_SLOTS,
    asset_ids_in,
    build_id_for,
    CLAIM_LABELS,
    case_content_hash,
    case_set_id_for,
    catalog_id_for,
    check_name,
    check_optional_sha256,
    check_rel_path,
    check_sha256,
    check_segment,
    dataset_id_for,
    derivation_id_for,
    digest,
    generated_rel_path,
    material_members_key,
    material_set_id_for,
    non_negative_int,
    positive_int,
    protocol_id_for,
    recipe_case_id,
    require_object,
    rule_mentions,
    validate_algorithm,
    validate_deviations,
    validate_official_ref,
    validate_provenance,
    validate_realization,
    validate_rule,
    validate_selection_rule,
    weakest_integrity,
)

CATALOG_FIELDS = (
    "path",
    "corpus",
    "subset",
    "sample_rate",
    "frames",
    "channels",
    "speaker_id",
    "utterance_id",
)


def validate_graph(
    graph: Graph,
    *,
    resource_db: str | None = None,
    require_verified: set[str] | None = None,
) -> None:
    """Raise when the graph cannot be registered.

    ``resource_db`` is optional. When it is omitted, eval-set overlap is UNKNOWN.
    ``require_verified`` lists generated asset ids that must have a verified shard.
    ``None`` requires that check for every generated asset.
    """
    assets = index_rows(graph.assets, "asset_id", "assets")
    protocols = index_rows(graph.protocols, "protocol_id", "protocols")
    cases = index_rows(graph.cases, "case_id", "cases")
    case_sets = index_rows(graph.case_sets, "case_set_id", "case_sets")
    materials = index_rows(graph.materials, "material_set_id", "materials")
    datasets = index_rows(graph.datasets, "dataset_id", "datasets")
    builds = index_rows(graph.builds, "build_id", "builds")
    catalogs = index_rows(graph.catalogs, "catalog_id", "catalogs")
    derivations = index_rows(graph.derivations, "derivation_id", "derivations")
    _catalogs(graph, catalogs)
    _derivations(derivations)
    _builds(builds, materials)
    _protocols(protocols)
    _assets(assets, catalogs, derivations, builds, graph.verified, require_verified)
    _materials(materials, assets)
    _cases(cases, assets, protocols, builds)
    _case_sets(case_sets, cases)
    for dataset in datasets.values():
        _dataset(dataset, protocols, case_sets, materials, cases, assets)
        report = leak_report(dataset, case_sets, materials, cases, assets, resource_db=resource_db)
        if report["status"] == "FAIL":
            raise ReproductionError(
                f"dataset {dataset['name']} failed leak_check: {report['reason']}"
            )
    for audit in graph.audits:
        if audit["audit_id"] != audit_identity(audit):
            raise ReproductionError(f"audit {audit.get('audit_id')} does not match its report")
        if audit["status"] != audit["report_json"]["status"]:
            raise ReproductionError(f"audit {audit['audit_id']} status does not match its report")
    index_rows(graph.runs, "run_id", "runs")


def audit_identity(audit: dict[str, Any]) -> str:
    from src.experiments.data.ids import audit_id_for

    return audit_id_for(
        subject_kind=audit["subject_kind"],
        subject_id=audit["subject_id"],
        audit_kind=audit["audit_kind"],
        report=audit["report_json"],
    )


def compute_selection_uses_test(
    dataset: dict[str, Any],
    case_sets: dict[str, dict[str, Any]],
    materials: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> bool:
    if dataset["declared_status"] != "ready":
        return False
    groups = _split_assets(dataset, case_sets, materials, cases)
    valid_cases = _case_ids(dataset, case_sets, "valid")
    test_cases = _case_ids(dataset, case_sets, "test")
    if valid_cases & test_cases:
        return True
    if groups["valid"] & groups["test"]:
        return True
    test_sets = {
        binding["case_set_id"]
        for binding in dataset["bindings"]
        if binding["slot"] == "test.cases" and binding.get("case_set_id")
    }
    for binding in dataset["bindings"]:
        if binding["slot"] != "valid.cases" or not binding.get("case_set_id"):
            continue
        rule = case_sets[binding["case_set_id"]]["selection_rule_json"]
        if any(rule_mentions(rule, target) for target in test_sets):
            return True
    return False


def leak_report(
    dataset: dict[str, Any],
    case_sets: dict[str, dict[str, Any]],
    materials: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    *,
    resource_db: str | None,
) -> dict[str, Any]:
    groups = _split_assets(dataset, case_sets, materials, cases)
    components: list[dict[str, Any]] = []
    pairs = (("train", "valid"), ("train", "test"))
    if dataset["profile"] == "leakage_safe":
        pairs = pairs + (("valid", "test"),)
    for left, right in pairs:
        overlap = sorted(groups[left] & groups[right])
        components.append(
            {
                "check": f"{left}_{right}_assets",
                "status": "FAIL" if overlap else "PASS",
                "overlap": overlap,
            }
        )
        components.append(_speaker_overlap(left, right, groups[left], groups[right], assets))
        components.append(_sha_overlap(left, right, groups[left], groups[right], assets))
    if dataset["profile"] == "leakage_safe":
        components.append(_eval_overlap(groups, assets, resource_db))
    status = _worst(component["status"] for component in components)
    reasons = [component["check"] for component in components if component["status"] == "FAIL"]
    return {
        "status": status,
        "profile": dataset["profile"],
        "reason": "overlap" if reasons else status.lower(),
        "failed": reasons,
        "components": components,
    }


def _worst(statuses) -> str:
    rank = {"PASS": 0, "WARN": 1, "UNKNOWN": 2, "FAIL": 3}
    return max(statuses, key=lambda item: rank[item], default="PASS")


def _dataset(dataset, protocols, case_sets, materials, cases, assets) -> None:
    check_name(dataset["name"], "dataset name")
    positive_int(dataset["version"], "dataset version")
    protocol = protocols.get(dataset["protocol_id"])
    if protocol is None:
        raise ReproductionError(f"dataset {dataset['name']} references a missing protocol")
    bindings = list(dataset["bindings"])
    if dataset["declared_status"] == "ready":
        if dataset["blocking_json"] != []:
            raise ReproductionError(f"dataset {dataset['name']} is ready but blocking_json is not empty")
        _bindings(dataset, protocol, bindings, case_sets, materials)
    else:
        if bindings:
            raise ReproductionError(f"dataset {dataset['name']} is {dataset['declared_status']} but has bindings")
        if not dataset["blocking_json"]:
            raise ReproductionError(f"dataset {dataset['name']} needs blocking_json")
    computed = compute_selection_uses_test(dataset, case_sets, materials, cases)
    if bool(dataset["selection_uses_test"]) != computed:
        raise ReproductionError(
            f"dataset {dataset['name']} selection_uses_test does not match the bindings "
            f"(computed {computed})"
        )
    if dataset["profile"] == "leakage_safe" and computed:
        raise ReproductionError(f"dataset {dataset['name']} is leakage_safe but selection uses test")
    _claim(dataset, protocol, materials, case_sets, cases, assets)
    expected = dataset_id_for(
        protocol_id=dataset["protocol_id"],
        bindings=bindings,
        profile=dataset["profile"],
        material_origin=dataset["material_origin"],
        claim_label=dataset["claim_label"],
        selection_uses_test=bool(dataset["selection_uses_test"]),
        declared_status=dataset["declared_status"],
    )
    if dataset["dataset_id"] != expected:
        raise ReproductionError(f"dataset {dataset['name']} id does not match its definition")
    for case_id in _bound_case_ids(dataset, case_sets):
        if cases[case_id]["teacher_kind"] == "none":
            raise ReproductionError(f"dataset {dataset['name']} includes a case with teacher none")


def _bindings(dataset, protocol, bindings, case_sets, materials) -> None:
    kind = protocol["kind"]
    slots = [binding["slot"] for binding in bindings]
    if len(slots) != len(set(slots)):
        raise ReproductionError(f"dataset {dataset['name']} repeats a slot")
    for slot in slots:
        if slot not in ALLOWED_SLOTS[kind]:
            raise ReproductionError(f"dataset {dataset['name']} slot {slot} is not used by {kind}")
    for slot in REQUIRED_SLOTS[kind]:
        if slot not in slots:
            raise ReproductionError(f"dataset {dataset['name']} is missing slot {slot}")
    for binding in bindings:
        slot = binding["slot"]
        case_set_id = binding.get("case_set_id")
        material_set_id = binding.get("material_set_id")
        if slot.endswith(".cases"):
            if not case_set_id or material_set_id:
                raise ReproductionError(f"slot {slot} requires a case set only")
            case_set = case_sets.get(case_set_id)
            if case_set is None:
                raise ReproductionError(f"slot {slot} references a missing case set")
            if case_set["split_role"] != slot.split(".", 1)[0]:
                raise ReproductionError(f"case set {case_set['name']} split does not match {slot}")
        else:
            if not material_set_id or case_set_id:
                raise ReproductionError(f"slot {slot} requires a material set only")
            material = materials.get(material_set_id)
            if material is None:
                raise ReproductionError(f"slot {slot} references a missing material set")
            split, role = slot.split(".", 1)
            if material["split_role"] != split or material["role"] != role:
                raise ReproductionError(f"material set {material['name']} does not match {slot}")


def _claim(dataset, protocol, materials, case_sets, cases, assets) -> None:
    label = dataset["claim_label"]
    profile = dataset["profile"]
    origin = dataset["material_origin"]
    deviation = protocol["deviation_class"]
    if profile == "leakage_safe" and label != "leakage_safe_comparison":
        raise ReproductionError("leakage_safe requires claim_label leakage_safe_comparison")
    if label == "leakage_safe_comparison" and profile != "leakage_safe":
        raise ReproductionError("leakage_safe_comparison requires profile leakage_safe")
    if label == "official_reproduction":
        if profile != "official_faithful":
            raise ReproductionError("official_reproduction requires profile official_faithful")
        if origin not in ("official", "derived_from_official"):
            raise ReproductionError("official_reproduction requires official material_origin")
        if deviation != "official_effective":
            raise ReproductionError("official_reproduction requires deviation_class official_effective")
        if dataset["declared_status"] == "ready":
            pass
        elif dataset["declared_status"] in ("blocked", "unknown"):
            if dataset["bindings"]:
                raise ReproductionError("blocked official_reproduction cannot bind materials")
        else:
            raise ReproductionError("official_reproduction status is unknown")
        for asset_id in _bound_assets(dataset, case_sets, materials, cases):
            asset = assets[asset_id]
            if asset["integrity"] != "sha256" or not asset["sha256"]:
                raise ReproductionError(f"official_reproduction asset {asset_id} is missing sha256")
        if origin == "derived_from_official" and not dataset["deviations_json"]:
            raise ReproductionError("derived official_reproduction must record deviations_json")
    if label == "minimal_repair_reproduction" and deviation != "minimal_launch_repair":
        raise ReproductionError("minimal_repair_reproduction requires minimal_launch_repair")
    if label == "protocol_conformant_substitute_materials" and origin not in ("substitute", "mixed"):
        raise ReproductionError("substitute-material claim requires substitute or mixed origin")
    if label == "espnet_public_se_recipe_reproduction":
        if protocol["algorithm"] not in (
            "pregenerated.ms_snsd_tfgridnet",
            "pregenerated.dns_tflocoformer",
        ):
            raise ReproductionError(
                "espnet_public_se_recipe_reproduction requires the MS-SNSD or DNS TF-Locoformer protocol"
            )
    if label == "official_recipe_scaled_down" and protocol["kind"] != "pregenerated":
        raise ReproductionError("official_recipe_scaled_down requires a pregenerated protocol")
    if label == "research_checkpoint_selection":
        if profile != "official_faithful":
            raise ReproductionError("research_checkpoint_selection requires profile official_faithful")
        if dataset["selection_uses_test"]:
            raise ReproductionError("research_checkpoint_selection cannot select on test")
        if deviation != "research_improvement":
            raise ReproductionError("research_checkpoint_selection requires deviation_class research_improvement")
    if label not in CLAIM_LABELS:
        raise ReproductionError(f"unknown claim_label: {label}")


def _protocols(protocols: dict[str, dict[str, Any]]) -> None:
    for protocol in protocols.values():
        check_name(protocol["name"], "protocol name")
        positive_int(protocol["version"], "protocol version")
        spec = validate_algorithm(protocol["algorithm"], protocol["algorithm_version"], protocol["kind"])
        if spec["kind"] != protocol["kind"]:
            raise ReproductionError("protocol kind does not match the algorithm")
        validate_rule(protocol["algorithm"], protocol["rule_json"])
        validate_official_ref(protocol["official_ref_json"])
        validate_deviations(protocol["deviations_json"])
        if protocol["deviation_class"] not in DEVIATION_CLASSES:
            raise ReproductionError("unknown deviation_class")
        parent = protocol.get("parent_protocol_id")
        if protocol["deviation_class"] != "official_effective" and not parent:
            raise ReproductionError(f"protocol {protocol['name']} needs parent_protocol_id")
        if parent is not None and parent not in protocols:
            raise ReproductionError(f"protocol {protocol['name']} references a missing parent")
        expected = protocol_id_for(
            algorithm=protocol["algorithm"],
            algorithm_version=protocol["algorithm_version"],
            kind=protocol["kind"],
            rule=protocol["rule_json"],
            deviation_class=protocol["deviation_class"],
            parent=parent,
        )
        if protocol["protocol_id"] != expected:
            raise ReproductionError(f"protocol {protocol['name']} id does not match its rule")
    _parent_cycles(protocols)


def _parent_cycles(protocols: dict[str, dict[str, Any]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(protocol_id: str) -> None:
        if protocol_id in visited:
            return
        if protocol_id in visiting:
            raise ReproductionError(f"protocol parent cycle at {protocol_id}")
        visiting.add(protocol_id)
        parent = protocols[protocol_id].get("parent_protocol_id")
        if parent:
            walk(parent)
        visiting.remove(protocol_id)
        visited.add(protocol_id)

    for protocol_id in protocols:
        walk(protocol_id)


def _assets(assets, catalogs, derivations, builds, verified, require_verified: set[str] | None) -> None:
    for asset in assets.values():
        rel = check_rel_path(asset["rel_path"], "rel_path")
        if asset["storage_root"] not in ("corpora", "generated", "derived"):
            raise ReproductionError(f"unknown storage_root on {asset.get('asset_id')}")
        if asset["asset_id"] != f"{asset['storage_root']}/{rel}":
            raise ReproductionError(f"asset_id does not match the path: {asset['asset_id']}")
        positive_int(asset["sample_rate"], "sample_rate")
        positive_int(asset["frames"], "frames")
        positive_int(asset["channels"], "channels")
        check_optional_sha256(asset.get("sha256"), "sha256")
        if not isinstance(asset.get("extra_json"), dict):
            raise ReproductionError(f"asset {asset['asset_id']} extra_json must be an object")
        if asset["integrity"] == "sha256":
            if not asset.get("sha256") or asset.get("size_bytes") is None:
                raise ReproductionError(f"asset {asset['asset_id']} integrity sha256 needs sha256 and size_bytes")
        elif asset["integrity"] == "size":
            if asset.get("sha256") is not None or asset.get("size_bytes") is None:
                raise ReproductionError(f"asset {asset['asset_id']} integrity size needs size_bytes and no sha256")
        elif asset["integrity"] == "catalog":
            if asset.get("sha256") is not None:
                raise ReproductionError(f"asset {asset['asset_id']} integrity catalog cannot store sha256")
        else:
            raise ReproductionError(f"unknown integrity on {asset['asset_id']}")
        if asset["storage_root"] == "corpora":
            if asset.get("catalog_id") not in catalogs:
                raise ReproductionError(f"asset {asset['asset_id']} needs a catalog snapshot")
        if asset["storage_root"] == "derived":
            if asset.get("derivation_id") not in derivations or asset.get("derived_from") not in assets:
                raise ReproductionError(f"derived asset {asset['asset_id']} needs derivation and parent")
        if asset["storage_root"] == "generated":
            needed = require_verified is None or asset["asset_id"] in require_verified
            _generated_asset(asset, builds, verified, require_verified=needed)


def _generated_asset(asset, builds, verified, *, require_verified: bool) -> None:
    build = builds.get(asset.get("build_id"))
    if build is None:
        raise ReproductionError(f"generated asset {asset['asset_id']} needs a build")
    extra = asset["extra_json"]
    for key in ("stream", "gen_index", "shard_id"):
        if key not in extra:
            raise ReproductionError(f"generated asset {asset['asset_id']} is missing extra_json.{key}")
    gen_index = non_negative_int(extra["gen_index"], "extra_json.gen_index")
    stream = check_segment(extra["stream"], "stream")
    if not isinstance(extra["shard_id"], str) or not extra["shard_id"]:
        raise ReproductionError(f"generated asset {asset['asset_id']} shard_id is empty")
    rel = generated_rel_path(
        generator=build["generator"],
        build_id=build["build_id"],
        stream=stream,
        gen_index=gen_index,
        shard_size=build["shard_size"],
    )
    if asset["rel_path"] != rel:
        raise ReproductionError(f"generated asset {asset['asset_id']} path does not match the build")
    if asset["integrity"] != "sha256":
        raise ReproductionError(f"generated asset {asset['asset_id']} requires sha256")
    if not require_verified:
        return
    marker = verified.get(build["build_id"])
    if marker is None:
        raise ReproductionError(
            f"generated build {build['build_id']} has no verified/complete.json; refusing incomplete generated data"
        )
    shard_ids = marker.get("shard_ids")
    if not isinstance(shard_ids, list) or extra["shard_id"] not in shard_ids:
        raise ReproductionError(f"generated asset {asset['asset_id']} is not in a verified shard")


def _cases(cases, assets, protocols, builds) -> None:
    for case in cases.values():
        protocol = protocols.get(case["protocol_id"])
        if protocol is None:
            raise ReproductionError(f"case {case['case_id']} references a missing protocol")
        doc = case["doc_json"]
        if case["doc_kind"] == "recipe_v1":
            try:
                canonical = canonical_recipe(doc)
            except RecipeError as exc:
                raise ReproductionError(str(exc)) from exc
            if canonical != doc:
                raise ReproductionError(f"case {case['case_id']} recipe is not canonical")
            if recipe_case_id(doc) != case["case_id"]:
                raise ReproductionError(f"case {case['case_id']} does not match its recipe")
            if case["sample_rate"] != doc["output"]["sample_rate"]:
                raise ReproductionError(f"case {case['case_id']} sample_rate does not match the recipe")
            if case["teacher_kind"] != doc["output"]["reference"]:
                raise ReproductionError(f"case {case['case_id']} teacher does not match the recipe")
        elif case["doc_kind"] == "realization_v1":
            document = validate_realization(doc)
            if digest(document) != case["case_id"]:
                raise ReproductionError(f"case {case['case_id']} does not match its realization")
            if document["algorithm"] != protocol["algorithm"]:
                raise ReproductionError(f"case {case['case_id']} algorithm does not match its protocol")
            if document["sample_rate"] != case["sample_rate"]:
                raise ReproductionError(f"case {case['case_id']} sample_rate does not match the realization")
            if document["output"]["teacher"] != case["teacher_kind"]:
                raise ReproductionError(f"case {case['case_id']} teacher does not match the realization")
        else:
            raise ReproductionError(f"unknown doc_kind on {case['case_id']}")
        if case["content_hash"] != case_content_hash(case):
            raise ReproductionError(f"case {case['case_id']} content_hash does not match")
        if case["case_role"] == "recorded_pair":
            _recorded_sources(case, assets)
        elif case["case_role"] == "generated":
            _generated_case(case, assets, builds, protocol)
        elif case["case_role"] == "online_snapshot":
            if any(case.get(key) is not None for key in ("clean_asset_id", "noisy_asset_id", "noise_asset_id", "build_id", "gen_index")):
                raise ReproductionError(f"online snapshot {case['case_id']} must not store audio columns")
        else:
            raise ReproductionError(f"unknown case_role on {case['case_id']}")
        for key in ("clean_asset_id", "noisy_asset_id", "noise_asset_id"):
            if case.get(key) is not None and case[key] not in assets:
                raise ReproductionError(f"case {case['case_id']} references a missing asset {case[key]}")
        for asset_id in asset_ids_in(doc):
            if asset_id not in assets:
                raise ReproductionError(f"case {case['case_id']} document references a missing asset {asset_id}")


def _recorded_sources(case, assets) -> None:
    clean = assets[case["clean_asset_id"]]
    noisy = assets[case["noisy_asset_id"]]
    sources = case["doc_json"]["sources"]
    for name, asset in (("clean", clean), ("noisy", noisy)):
        source = sources.get(name)
        if source is None:
            raise ReproductionError(f"recorded case {case['case_id']} is missing source {name}")
        if source["path"] != "../" + asset["asset_id"] or source["sha256"] != asset["sha256"]:
            raise ReproductionError(f"recorded case {case['case_id']} source {name} does not match the asset")
    if case.get("speaker_id") and clean.get("speaker_id") and case["speaker_id"] != clean["speaker_id"]:
        raise ReproductionError(f"recorded case {case['case_id']} speaker does not match the clean asset")


def _generated_case(case, assets, builds, protocol) -> None:
    if case.get("build_id") not in builds or case.get("gen_index") is None:
        raise ReproductionError(f"generated case {case['case_id']} is missing build_id or gen_index")
    delivery = (protocol.get("rule_json") or {}).get("delivery")
    if delivery == "on_demand":
        if any(case.get(key) for key in ("clean_asset_id", "noisy_asset_id", "noise_asset_id")):
            raise ReproductionError(
                f"on-demand case {case['case_id']} must not store a generated waveform"
            )
        if int(case["frames"]) != int(case["doc_json"]["output"]["length"]):
            raise ReproductionError(f"on-demand case {case['case_id']} frames do not match the plan")
        if int(case["gen_index"]) != int(case["doc_json"]["draws"]["gen_index"]):
            raise ReproductionError(f"on-demand case {case['case_id']} gen_index does not match the plan")
        return
    if not case.get("clean_asset_id") or not case.get("noisy_asset_id"):
        raise ReproductionError(f"generated case {case['case_id']} needs clean and noisy assets")
    for key in ("clean_asset_id", "noisy_asset_id", "noise_asset_id"):
        if case.get(key) is None:
            continue
        asset = assets[case[key]]
        if asset["storage_root"] != "generated" or asset.get("build_id") != case["build_id"]:
            raise ReproductionError(f"generated case {case['case_id']} asset {asset['asset_id']} is not from the build")
        if int(asset["extra_json"]["gen_index"]) != int(case["gen_index"]):
            raise ReproductionError(f"generated case {case['case_id']} gen_index does not match {asset['asset_id']}")


def _materials(materials, assets) -> None:
    for material in materials.values():
        check_name(material["name"], "material set name")
        positive_int(material["version"], "material set version")
        validate_selection_rule(material["selection_rule_json"])
        validate_provenance(material["provenance_json"], origin=material["origin"])
        members = sorted(material["members"], key=lambda item: item["position"])
        if not members:
            raise ReproductionError(f"material set {material['name']} has no members")
        keys = []
        integrities = []
        seconds = 0.0
        seen = set()
        for position, member in enumerate(members):
            if member["position"] != position:
                raise ReproductionError(f"material set {material['name']} positions must be 0..n-1")
            asset = assets.get(member["asset_id"])
            if asset is None:
                raise ReproductionError(f"material set {material['name']} references a missing asset")
            if member["asset_id"] in seen:
                raise ReproductionError(f"material set {material['name']} repeats {member['asset_id']}")
            seen.add(member["asset_id"])
            keys.append(material_members_key(asset["asset_id"], asset.get("sha256"), asset.get("size_bytes")))
            integrities.append(asset["integrity"])
            seconds += asset["frames"] / asset["sample_rate"]
        if material["member_count"] != len(members):
            raise ReproductionError(f"material set {material['name']} member_count does not match")
        if float(f"{material['total_seconds']:.6f}") != float(f"{seconds:.6f}"):
            raise ReproductionError(f"material set {material['name']} total_seconds does not match")
        if material["list_sha256"] != digest(keys):
            raise ReproductionError(f"material set {material['name']} list_sha256 does not match")
        if material["integrity_min"] != weakest_integrity(integrities):
            raise ReproductionError(f"material set {material['name']} integrity_min does not match")
        expected = material_set_id_for(
            name=material["name"],
            version=material["version"],
            role=material["role"],
            split_role=material["split_role"],
            rule=material["selection_rule_json"],
            members=keys,
        )
        if material["material_set_id"] != expected:
            raise ReproductionError(f"material set {material['name']} id does not match its members")


def _case_sets(case_sets, cases) -> None:
    for case_set in case_sets.values():
        check_name(case_set["name"], "case set name")
        positive_int(case_set["version"], "case set version")
        validate_selection_rule(case_set["selection_rule_json"])
        validate_provenance(case_set["provenance_json"], origin=case_set["origin"])
        items = sorted(case_set["items"], key=lambda item: item["position"])
        if not items:
            raise ReproductionError(f"case set {case_set['name']} has no items")
        case_ids = []
        seen = set()
        for position, item in enumerate(items):
            if item["position"] != position:
                raise ReproductionError(f"case set {case_set['name']} positions must be 0..n-1")
            if item["case_id"] not in cases:
                raise ReproductionError(f"case set {case_set['name']} references a missing case")
            if item["case_id"] in seen:
                raise ReproductionError(f"case set {case_set['name']} repeats {item['case_id']}")
            seen.add(item["case_id"])
            case_ids.append(item["case_id"])
        if case_set["member_count"] != len(case_ids):
            raise ReproductionError(f"case set {case_set['name']} member_count does not match")
        if case_set["list_sha256"] != digest(case_ids):
            raise ReproductionError(f"case set {case_set['name']} list_sha256 does not match")
        expected = case_set_id_for(
            name=case_set["name"],
            version=case_set["version"],
            split_role=case_set["split_role"],
            rule=case_set["selection_rule_json"],
            case_ids=case_ids,
        )
        if case_set["case_set_id"] != expected:
            raise ReproductionError(f"case set {case_set['name']} id does not match its cases")


def _builds(builds, materials) -> None:
    for build in builds.values():
        check_segment(build["generator"], "generator")
        positive_int(build["generator_version"], "generator_version")
        positive_int(build["shard_size"], "shard_size")
        if isinstance(build["seed"], bool) or not isinstance(build["seed"], int):
            raise ReproductionError("build seed must be an integer")
        if not isinstance(build["params_json"], dict) or not isinstance(build["inputs_json"], dict):
            raise ReproductionError(f"build {build['build_id']} params and inputs must be objects")
        for slot, material_set_id in sorted(build["inputs_json"].items()):
            if material_set_id not in materials:
                raise ReproductionError(f"build {build['build_id']} input {slot} is missing")
        expected = build_id_for(
            generator=build["generator"],
            generator_version=build["generator_version"],
            params=build["params_json"],
            seed=build["seed"],
            inputs=build["inputs_json"],
        )
        if build["build_id"] != expected:
            raise ReproductionError(f"build {build['build_id']} id does not match its definition")


def _derivations(derivations) -> None:
    for derivation in derivations.values():
        spec = derivation["spec_json"]
        require_object(spec, "spec_json")
        expected = derivation_id_for(
            kind=derivation["kind"],
            tool=spec["tool"],
            tool_version=spec["tool_version"],
            params=spec["params"],
        )
        if derivation["derivation_id"] != expected:
            raise ReproductionError("derivation id does not match its spec")


def _catalogs(graph: Graph, catalogs) -> None:
    for catalog in catalogs.values():
        rows = graph.catalog_rows.get(catalog["catalog_id"])
        if rows is None:
            continue
        ordered = sorted(rows, key=lambda row: row["path"])
        if catalog["catalog_id"] != catalog_id_for(ordered):
            raise ReproductionError("catalog id does not match its rows")
        if catalog["row_count"] != len(ordered):
            raise ReproductionError("catalog row_count does not match")
        counts: dict[str, int] = {}
        for row in ordered:
            for field in CATALOG_FIELDS:
                if field not in row:
                    raise ReproductionError(f"catalog row is missing {field}")
            counts[row["corpus"]] = counts.get(row["corpus"], 0) + 1
        if catalog["corpora_json"] != counts:
            raise ReproductionError("catalog corpora_json does not match its rows")
        check_optional_sha256(catalog.get("source_db_sha256"), "source_db_sha256")


def _case_ids(dataset, case_sets, split: str) -> set[str]:
    found = set()
    for binding in dataset["bindings"]:
        if binding["slot"] == f"{split}.cases" and binding.get("case_set_id"):
            items = case_sets[binding["case_set_id"]]["items"]
            found.update(item["case_id"] for item in items)
    return found


def _bound_case_ids(dataset, case_sets) -> set[str]:
    found = set()
    for binding in dataset["bindings"]:
        if binding.get("case_set_id"):
            found.update(item["case_id"] for item in case_sets[binding["case_set_id"]]["items"])
    return found


def _bound_assets(dataset, case_sets, materials, cases) -> set[str]:
    return set().union(*_split_assets(dataset, case_sets, materials, cases).values())


def _split_assets(dataset, case_sets, materials, cases) -> dict[str, set[str]]:
    groups = {"train": set(), "valid": set(), "test": set()}
    for binding in dataset["bindings"]:
        split = binding["slot"].split(".", 1)[0]
        if binding.get("material_set_id"):
            groups[split].update(member["asset_id"] for member in materials[binding["material_set_id"]]["members"])
        if binding.get("case_set_id"):
            for item in case_sets[binding["case_set_id"]]["items"]:
                case = cases[item["case_id"]]
                for key in ("clean_asset_id", "noisy_asset_id", "noise_asset_id"):
                    if case.get(key):
                        groups[split].add(case[key])
                groups[split].update(asset_ids_in(case["doc_json"]))
    return groups


def _speaker_overlap(left_name: str, right_name: str, left: set[str], right: set[str], assets) -> dict[str, Any]:
    def speakers(ids: set[str]) -> tuple[set[str], bool]:
        found = set()
        missing = False
        for asset_id in ids:
            speaker = assets[asset_id].get("speaker_id")
            if speaker:
                found.add(speaker)
            else:
                missing = True
        return found, missing

    left_speakers, left_missing = speakers(left)
    right_speakers, right_missing = speakers(right)
    overlap = sorted(left_speakers & right_speakers)
    if overlap:
        status = "FAIL"
    elif left_missing or right_missing:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "check": f"{left_name}_{right_name}_speakers",
        "status": status,
        "overlap": overlap,
        "speaker_missing": left_missing or right_missing,
    }


def _sha_overlap(left_name: str, right_name: str, left: set[str], right: set[str], assets) -> dict[str, Any]:
    def hashes(ids: set[str]) -> tuple[set[str], bool]:
        found = set()
        missing = False
        for asset_id in ids:
            value = assets[asset_id].get("sha256")
            if value:
                found.add(value)
            else:
                missing = True
        return found, missing

    left_hash, left_missing = hashes(left)
    right_hash, right_missing = hashes(right)
    overlap = sorted(left_hash & right_hash)
    if overlap:
        status = "FAIL"
    elif left_missing or right_missing:
        status = "UNKNOWN"
    else:
        status = "PASS"
    return {"check": f"{left_name}_{right_name}_sha256", "status": status, "overlap": overlap}


def _eval_overlap(groups, assets, resource_db: str | None) -> dict[str, Any]:
    if resource_db is None:
        return {
            "check": "eval_sets",
            "status": "UNKNOWN",
            "reason": "resource database was not opened",
        }
    observed = _read_eval_hashes(resource_db)
    if observed is None:
        return {"check": "eval_sets", "status": "UNKNOWN", "reason": "eval schema is absent"}
    ours = set()
    for ids in groups.values():
        for asset_id in ids:
            value = assets[asset_id].get("sha256")
            if value:
                ours.add(value)
    overlap = sorted(ours & observed)
    return {
        "check": "eval_sets",
        "status": "FAIL" if overlap else "PASS",
        "overlap": overlap,
        "reason": "compared",
    }


def _read_eval_hashes(resource_db: str) -> set[str] | None:
    import duckdb

    path = resource_db
    try:
        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        raise ReproductionError(f"cannot open resource database read-only: {exc}") from exc
    try:
        present = connection.execute(
            """
            SELECT count(*) FROM information_schema.schemata
            WHERE schema_name = 'eval'
            """
        ).fetchone()[0]
        if not present:
            return None
        columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'eval' AND table_name = 'cases'
                """
            ).fetchall()
        }
        if "recipe_json" not in columns:
            return None
        rows = connection.execute("SELECT recipe_json::VARCHAR FROM eval.cases").fetchall()
    finally:
        connection.close()
    found = set()
    for (payload,) in rows:
        if not payload:
            continue
        document = json_loads(payload)
        sources = document.get("sources") if isinstance(document, dict) else None
        if not isinstance(sources, dict):
            continue
        for source in sources.values():
            if isinstance(source, dict) and isinstance(source.get("sha256"), str):
                found.add(source["sha256"])
    return found


def json_loads(payload: str):
    import json

    return json.loads(payload)
