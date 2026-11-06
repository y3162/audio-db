"""Independence judgments over checkpoint provenance JSON."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.provenance import (
    UNKNOWN,
    apply_independence,
    canonical_bytes,
    is_confirmed_independent,
    judge_checkpoint,
    judge_independence,
    load_provenance,
)
from src.evaluation.results.summary import build_summary
from src.reproduction.paths import ROOT

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
PARENT = "d" * 64
OTHER = "e" * 64


def _row(role: str, sha: str, *, source: str | None = None, lineage: str = "source", case_set: str = "cases") -> dict:
    return {
        "role": role,
        "slot": "valid.cases" if role == "selection" else f"{role}.cases",
        "case_set_id": case_set,
        "asset_id": f"derived/{role}/{sha[:8]}",
        "sha256": sha,
        "source_sha256": source,
        "lineage": lineage,
    }


def _document(materials: list[dict], **extra) -> dict:
    document = {
        "schema_version": 1,
        "format": "audio-db-data-provenance-v1",
        "pipeline": "reproduction",
        "run_id": "1" * 64,
        "model_name": "mp_senet",
        "dataset_id": "2" * 64,
        "dataset_name": "example",
        "dataset_version": 1,
        "snapshot_hash": "3" * 64,
        "protocol_id": "protocol",
        "profile": "leakage_safe",
        "claim_label": "leakage_safe_comparison",
        "selection_uses_test": False,
        "materials": materials,
        "sets": [],
    }
    document.update(extra)
    return document


def _eval(sha: str, *, source: str | None = None, lineage: str = "source", name: str = "same-name") -> dict:
    return {
        "set_name": name,
        "sha256": sha,
        "source_sha256": source,
        "lineage": lineage,
    }


class ProvenanceJudgmentTests(unittest.TestCase):
    def test_selection_overlap_is_false_and_records_the_set(self) -> None:
        document = _document(
            [
                _row("train", OTHER),
                _row("selection", SHA_A, case_set="selected-cases"),
                _row("test", SHA_A, case_set="selected-cases"),
            ],
            selection_uses_test=True,
            profile="official_faithful",
            claim_label="official_reproduction",
        )
        judged = judge_independence(document, [_eval(SHA_A, name="voicebank_test")])
        self.assertIs(judged["independent"], False)
        self.assertFalse(is_confirmed_independent(judged["independent"]))
        self.assertEqual(judged["checkpoint_selection_overlap"], ["voicebank_test"])
        self.assertIn(SHA_A, judged["overlap_sha256"])
        self.assertIn("selected-cases", judged["overlap_case_sets"])
        self.assertIs(judged["selection_uses_test"], True)

    def test_disjoint_attested_materials_are_independent(self) -> None:
        document = _document(
            [
                _row("train", SHA_A),
                _row("selection", SHA_B, case_set="valid"),
                _row("test", SHA_C, case_set="test"),
            ]
        )
        judged = judge_independence(document, [_eval(OTHER, name="same-name")])
        self.assertIs(judged["independent"], True)
        self.assertTrue(is_confirmed_independent(judged["independent"]))
        self.assertEqual(judged["checkpoint_selection_overlap"], [])
        self.assertIs(judged["split_independent"], True)

    def test_same_name_and_different_content_is_not_overlap(self) -> None:
        document = _document(
            [
                _row("train", SHA_A),
                _row("selection", SHA_B, case_set="same-name"),
                _row("test", SHA_C),
            ]
        )
        judged = judge_independence(document, [_eval(OTHER, name="same-name")])
        self.assertIs(judged["independent"], True)
        self.assertEqual(judged["checkpoint_selection_overlap"], [])

    def test_derivative_of_the_same_source_overlaps(self) -> None:
        document = _document(
            [
                _row("train", SHA_A),
                _row("selection", SHA_B, source=PARENT, lineage="derived"),
                _row("test", OTHER),
            ]
        )
        judged = judge_independence(
            document,
            [_eval("f" * 64, source=PARENT, lineage="derived", name="resampled")],
        )
        self.assertIs(judged["independent"], False)
        self.assertEqual(judged["checkpoint_selection_overlap"], ["resampled"])
        self.assertIn(PARENT, judged["overlap_sha256"])

    def test_parent_file_overlaps_a_derivative_used_for_selection(self) -> None:
        document = _document(
            [
                _row("train", SHA_A),
                _row("selection", SHA_B, source=PARENT, lineage="derived"),
                _row("test", OTHER),
            ]
        )
        judged = judge_independence(document, [_eval(PARENT, name="original")])
        self.assertIs(judged["independent"], False)

    def test_unspecified_lineage_stays_unknown_without_a_hash_hit(self) -> None:
        document = _document(
            [
                _row("train", SHA_A),
                _row("selection", SHA_B),
                _row("test", SHA_C),
            ]
        )
        judged = judge_independence(document, [_eval(OTHER, lineage="unspecified")])
        self.assertEqual(judged["independent"], UNKNOWN)
        self.assertIsNone(judged["checkpoint_selection_overlap"])
        self.assertFalse(is_confirmed_independent(judged["independent"]))

    def test_unknown_lineage_does_not_become_independent(self) -> None:
        document = _document(
            [
                _row("train", SHA_A, lineage="unknown"),
                _row("selection", SHA_B, lineage="unknown"),
                _row("test", SHA_C, lineage="unknown"),
            ]
        )
        judged = judge_independence(document, [_eval(OTHER)])
        self.assertEqual(judged["independent"], UNKNOWN)
        self.assertFalse(is_confirmed_independent(judged["independent"]))

    def test_missing_and_corrupt_provenance_are_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"weights")
            missing = judge_checkpoint(checkpoint, [_eval(OTHER)])
            self.assertEqual(missing["independent"], UNKNOWN)
            self.assertIsNone(missing["checkpoint_selection_overlap"])
            sidecar = checkpoint.with_suffix(".data_provenance.json")
            sidecar.write_text("{", encoding="utf-8")
            corrupt = judge_checkpoint(checkpoint, [_eval(OTHER)])
            self.assertEqual(corrupt["independent"], UNKNOWN)
            self.assertIn("unreadable", corrupt["reason"])

    def test_legacy_document_is_unknown(self) -> None:
        judged = judge_independence(
            {"schema_version": 1, "format": "audio-db-data-provenance-v1", "pipeline": "legacy", "materials": []},
            [_eval(OTHER)],
        )
        self.assertEqual(judged["independent"], UNKNOWN)
        self.assertFalse(is_confirmed_independent(judged["independent"]))

    def test_official_flag_is_kept_when_evaluation_materials_are_absent(self) -> None:
        document = _document(
            [_row("train", SHA_A), _row("selection", SHA_B), _row("test", SHA_B)],
            selection_uses_test=True,
            profile="official_faithful",
            claim_label="official_reproduction",
        )
        judged = judge_independence(document, None)
        self.assertEqual(judged["independent"], UNKNOWN)
        self.assertEqual(judged["comparison"], "NOT_TESTED")
        self.assertIs(judged["selection_uses_test"], True)
        self.assertIs(judged["split_independent"], False)
        self.assertFalse(is_confirmed_independent(judged["independent"]))

    def test_safe_split_is_not_an_external_independence_claim(self) -> None:
        document = _document(
            [_row("train", SHA_A), _row("selection", SHA_B), _row("test", SHA_C)]
        )
        judged = judge_independence(document, None)
        self.assertIs(judged["split_independent"], True)
        self.assertEqual(judged["independent"], UNKNOWN)
        self.assertEqual(judged["comparison"], "NOT_TESTED")

    def test_summary_does_not_treat_unknown_as_independent(self) -> None:
        record = {"status": "ok"}
        apply_independence(record, judge_independence(None, None, load_problem="data_provenance.json is missing"))
        summary = build_summary([record])
        self.assertEqual(summary["data_independence"]["independent"], UNKNOWN)
        self.assertIs(summary["data_independence"]["unknown_is_not_independent"], True)
        self.assertFalse(is_confirmed_independent(summary["data_independence"]["independent"]))

    def test_checkpoint_and_directory_provenance_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"weights")
            left = _document([_row("train", SHA_A), _row("selection", SHA_B), _row("test", SHA_C)])
            right = _document([_row("train", OTHER), _row("selection", SHA_B), _row("test", SHA_C)])
            (root / "data_provenance.json").write_bytes(canonical_bytes(left))
            checkpoint.with_suffix(".data_provenance.json").write_bytes(canonical_bytes(right))
            document, problem = load_provenance(checkpoint)
            self.assertIsNone(document)
            self.assertIn("disagree", problem or "")

    def test_round_trip_bytes_are_stable(self) -> None:
        document = _document([_row("train", SHA_A), _row("selection", SHA_B), _row("test", SHA_C)])
        self.assertEqual(canonical_bytes(document), canonical_bytes(json.loads(canonical_bytes(document))))

    def test_evaluation_package_does_not_import_reproduction(self) -> None:
        offenders = []
        for path in (ROOT / "src" / "evaluation").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                if any(name == "src.reproduction" or name.startswith("src.reproduction.") for name in modules):
                    offenders.append(str(path))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
