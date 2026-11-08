"""Reconstruction refuses a missing snapshot and does not create a database."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.reproduction.errors import ReproductionError
from src.reproduction.prepare.reconstruct_accept import (
    ORIGIN,
    PUBLISHED_LIST_SHA256,
    commit_reconstructed_database,
)


class ReconstructAcceptTests(unittest.TestCase):
    def test_origin_label_and_published_list_digest(self):
        self.assertEqual(ORIGIN, "reconstructed_from_verified_snapshot")
        self.assertEqual(len(PUBLISHED_LIST_SHA256), 64)

    def test_commit_does_not_create_a_database_when_the_snapshot_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "reproduction.duckdb"
            with self.assertRaises(ReproductionError):
                commit_reconstructed_database(
                    snapshot_dir=root / "missing-snapshot",
                    database=database,
                    staging=root / "staging",
                    snapshots=root / "snapshots",
                    corpora=root / "corpora",
                    resource_db=root / "resource.duckdb",
                    generated=root / "generated",
                    derived=root / "derived",
                    list_path=root / "list.json",
                )
            self.assertFalse(database.exists())

    def test_commit_does_not_replace_an_existing_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "reproduction.duckdb"
            database.write_bytes(b"existing")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            with self.assertRaises(ReproductionError):
                commit_reconstructed_database(
                    snapshot_dir=snapshot,
                    database=database,
                    staging=root / "staging",
                    snapshots=root / "snapshots",
                    corpora=root / "corpora",
                    resource_db=root / "resource.duckdb",
                    generated=root / "generated",
                    derived=root / "derived",
                    list_path=root / "list.json",
                )
            self.assertEqual(database.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
