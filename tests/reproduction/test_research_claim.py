"""Research checkpoint-selection claim, without touching production databases."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb

from src.reproduction.db.registry import initialize_database, read_graph, register_staging, table_hashes
from src.reproduction.db.schema import ensure_dataset_claim_labels
from src.reproduction.db.snapshot import export_snapshot
from src.reproduction.db.staging import write_graph
from src.reproduction.ids import CLAIM_LABELS, digest
from src.reproduction.prepare.research_valid_fixed import (
    locate_reproduction_database,
    register_research_validation_dataset,
)
from src.reproduction.snapshot_reader import load_snapshot
from tests.reproduction.support import fixed_pair_graph


class ResearchClaimTests(unittest.TestCase):
    def test_claim_is_named_and_an_empty_root_stays_blocked(self):
        self.assertIn("research_checkpoint_selection", CLAIM_LABELS)
        with tempfile.TemporaryDirectory() as directory:
            status = locate_reproduction_database(directory)
        self.assertEqual(status["registration"], "BLOCKED")
        self.assertEqual(status["found"], [])

    def test_old_check_gains_the_label_without_rewriting_rows(self):
        from src.reproduction.db.schema import _STATEMENTS

        with tempfile.TemporaryDirectory() as directory:
            connection = duckdb.connect(str(Path(directory) / "old.duckdb"))
            for statement in _STATEMENTS:
                connection.execute(statement.replace(",\n                'research_checkpoint_selection'", ""))
            connection.execute(
                """
                INSERT INTO repro.protocols VALUES (
                    'p', 'proto', 1, 'fixed_pair', 'fixed_pair.mp_senet', 1,
                    '{}', '{"repo": "synthetic"}', 'official_effective', NULL, '[]'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO repro.datasets VALUES (
                    'parent', 'synthetic.parent', 1, 'p', 'official_faithful', 'official',
                    'official_reproduction', 'ready', '[]', false, NULL, '[]'
                )
                """
            )
            before = connection.execute(
                "SELECT dataset_id, name, claim_label, profile FROM repro.datasets"
            ).fetchall()
            ensure_dataset_claim_labels(connection)
            after = connection.execute(
                "SELECT dataset_id, name, claim_label, profile FROM repro.datasets"
            ).fetchall()
            self.assertEqual(before, after)
            connection.execute(
                """
                INSERT INTO repro.datasets VALUES (
                    'child', 'synthetic.child', 1, 'p', 'official_faithful', 'derived_from_official',
                    'research_checkpoint_selection', 'ready', '[]', false, NULL, '[]'
                )
                """
            )
            labels = connection.execute(
                "SELECT claim_label FROM repro.datasets ORDER BY dataset_id"
            ).fetchall()
            connection.close()
            self.assertEqual(labels, [("research_checkpoint_selection",), ("official_reproduction",)])

    def test_child_dataset_is_idempotent_and_leaves_the_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpora = root / "corpora"
            database = root / "reproduction.duckdb"
            staging = root / "staging"
            initialize_database(database)
            parent = fixed_pair_graph(corpora, safe=True)
            write_graph(staging / "parent", parent)
            register_staging(staging / "parent", database)
            before = table_hashes(database)
            parent_graph = read_graph(database)
            parent_row = next(row for row in parent_graph.datasets if row["name"] == "synthetic.safe")
            valid_id = next(item["case_set_id"] for item in parent_row["bindings"] if item["slot"] == "valid.cases")
            valid_set = next(row for row in parent_graph.case_sets if row["case_set_id"] == valid_id)
            selected = sorted(item["case_id"] for item in valid_set["items"])
            parent_fingerprint = digest(parent_row)
            first = register_research_validation_dataset(
                database,
                parent_name="synthetic.safe",
                child_name="synthetic.safe.research",
                selected_case_ids=selected,
                staging=staging / "child",
            )
            second = register_research_validation_dataset(
                database,
                parent_name="synthetic.safe",
                child_name="synthetic.safe.research",
                selected_case_ids=selected,
                staging=staging / "child-again",
            )
            self.assertEqual(first["dataset_id"], second["dataset_id"])
            self.assertEqual(first["claim_label"], "research_checkpoint_selection")
            self.assertTrue(second["inserted"])
            self.assertTrue(all(value == 0 for value in second["inserted"].values()))
            reread = read_graph(database)
            stored_parent = next(row for row in reread.datasets if row["dataset_id"] == parent_row["dataset_id"])
            self.assertEqual(digest(stored_parent), parent_fingerprint)
            self.assertEqual(stored_parent["claim_label"], "leakage_safe_comparison")
            self.assertNotEqual(before["datasets"], table_hashes(database)["datasets"])
            child = next(row for row in reread.datasets if row["name"] == "synthetic.safe.research")
            self.assertEqual(
                [row["name"] for row in reread.datasets].count("synthetic.safe.research"),
                1,
            )
            self.assertFalse(child["selection_uses_test"])
            self.assertNotEqual(child["claim_label"], "official_reproduction")
            snapshots = root / "snapshots"
            first_hash = export_snapshot(
                database,
                "synthetic.safe.research@1",
                snapshots,
                allow_unknown_leak_check=True,
                corpora=corpora,
                generated=root / "generated",
                derived=root / "derived",
            )
            second_hash = export_snapshot(
                database,
                "synthetic.safe.research@1",
                snapshots,
                allow_unknown_leak_check=True,
                corpora=corpora,
                generated=root / "generated",
                derived=root / "derived",
            )
            self.assertEqual(first_hash, second_hash)
            loaded = load_snapshot(snapshots / child["dataset_id"])
            self.assertEqual(loaded.snapshot_hash, first_hash)
            self.assertEqual(loaded.dataset["claim_label"], "research_checkpoint_selection")
            valid_items = [
                row for row in loaded.case_set_items if row["case_set_id"] == first["valid_case_set_id"]
            ]
            self.assertEqual(len(valid_items), len(selected))
            self.assertEqual(sorted(row["case_id"] for row in valid_items), selected)


if __name__ == "__main__":
    unittest.main()
