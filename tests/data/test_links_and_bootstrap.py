import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import duckdb

from src.data.bootstrap import MANAGED_CORPORA, bootstrap, replace_production, validate_database
from src.data.corpora_catalog import CATALOG
from src.data.db.eval_schema import create_eval_schema
from src.data.db.schema import create_corpora_schema
from src.data.link_corpora import link_corpora
from src.data.recipe import json_hash

EXCLUDED = {"LibriLM", "VCTK-RVA", "VoxCelebMeta", "cmudict", "_ingest", "FSD50K", "SpeechCommands"}


def _seed_db(path: Path, corpus: str | None = None) -> None:
    connection = duckdb.connect(str(path))
    create_corpora_schema(connection)
    create_eval_schema(connection)
    for name in ([corpus] if corpus else sorted(MANAGED_CORPORA)):
        connection.execute(
            "INSERT INTO corpora.audio_files (path, corpus, file_name, sample_rate, frames, channels) VALUES (?, ?, 'a.flac', 16000, 16000, 1)",
            [f"{name}/a.flac", name],
        )
    connection.execute("INSERT INTO eval.sets VALUES (?, ?, ?)",
                       ["fixture", "fixture", json_hash({"case_ids": [], "description": "fixture"})])
    connection.close()


class LinkAndBootstrapTests(unittest.TestCase):
    def test_catalog_is_the_twelve(self) -> None:
        names = {spec.source_name for spec in CATALOG}
        self.assertTrue(EXCLUDED.isdisjoint(names))
        self.assertEqual(len(CATALOG), 12)

    def test_link_create_existing_and_conflicts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            corpora = root / "corpora"
            for spec in CATALOG:
                (source / spec.source_name).mkdir(parents=True)
            first = link_corpora(source, corpora)
            self.assertTrue(any(line.startswith("linked ") for line in first))
            again = link_corpora(source, corpora)
            self.assertTrue(all(line.startswith("ok ") for line in again))
            for name in EXCLUDED:
                self.assertFalse((corpora / name).exists())
            wrong = source / "other"
            wrong.mkdir()
            link = corpora / CATALOG[0].link_name
            link.unlink()
            link.symlink_to(wrong)
            with self.assertRaises(FileExistsError):
                link_corpora(source, corpora)
            link.unlink()
            link.mkdir()
            with self.assertRaises(FileExistsError):
                link_corpora(source, corpora)

    def test_bootstrap_refuses_existing_output(self) -> None:
        with TemporaryDirectory() as tmp:
            database = Path(tmp) / "next.duckdb"
            database.write_bytes(b"x")
            with self.assertRaises(SystemExit):
                bootstrap(database, Path(tmp), rebuild=False)

    def test_missing_corpora_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.duckdb"
            _seed_db(path, "LibriSpeech")
            with self.assertRaises(ValueError):
                validate_database(path)

    def test_rebuild_existing_requires_backup(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.duckdb"
            path.write_bytes(b"original")
            with self.assertRaises(ValueError):
                bootstrap(path, Path(tmp), rebuild=True)
            self.assertEqual(path.read_bytes(), b"original")

    def test_rebuild_replaces_from_empty_and_retains_backup(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, backup = root / "old.duckdb", root / "backup.duckdb"
            _seed_db(path)
            connection = duckdb.connect(str(path))
            connection.execute("INSERT INTO eval.sets VALUES (?, ?, ?)",
                               ["old", "old", json_hash({"case_ids": [], "description": "old"})])
            connection.close()
            old = path.read_bytes()
            def ingest():
                _seed_db(Path(sys.argv[1]))
            with patch("src.data.bootstrap.link_corpora", return_value=[]), \
                 patch("src.data.bootstrap.ingest_main", side_effect=ingest), \
                 patch("src.data.bootstrap.initialize"):
                bootstrap(path, root, rebuild=True, backup=backup)
            self.assertEqual(backup.read_bytes(), old)
            connection = duckdb.connect(str(path), read_only=True)
            self.assertEqual(connection.execute("SELECT name FROM eval.sets").fetchall(), [("fixture",)])
            connection.close()
            self.assertEqual(list(root.glob(".old.duckdb.*")), [])

    def test_invalid_recipe_and_dangling_reference_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / "invalid.duckdb"
            _seed_db(invalid)
            connection = duckdb.connect(str(invalid))
            connection.execute("INSERT INTO eval.cases VALUES (?, 'clean', 'toy', NULL, 16000, NULL, NULL, ?, ?, ?)",
                               ["wrong", '{}', "wrong", '{}'])
            connection.close()
            with self.assertRaises(ValueError):
                validate_database(invalid)
            dangling = root / "dangling.duckdb"
            _seed_db(dangling)
            connection = duckdb.connect(str(dangling))
            connection.execute("DROP TABLE eval.set_items")
            connection.execute("CREATE TABLE eval.set_items (set_name VARCHAR, position INTEGER, case_id VARCHAR)")
            connection.execute("INSERT INTO eval.set_items VALUES ('fixture', 0, 'missing')")
            connection.close()
            with self.assertRaises(ValueError):
                validate_database(dangling)

    def test_replace_validates_and_keeps_production_on_bad_input(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            production = root / "resource.duckdb"
            good = root / "good.duckdb"
            bad = root / "bad.duckdb"
            _seed_db(production)
            _seed_db(good)
            bad.write_bytes(b"not-a-database")
            before = production.read_bytes()
            with self.assertRaises(Exception):
                replace_production(bad, root / "backup.duckdb", production)
            self.assertEqual(production.read_bytes(), before)
            self.assertFalse((root / "backup.duckdb").exists())
            backup = root / "backup.duckdb"
            replace_production(good, backup, production)
            validate_database(production)
            self.assertTrue(backup.is_file())
            alien = root / "alien.duckdb"
            _seed_db(alien, corpus="FSD50K")
            with self.assertRaises(ValueError):
                replace_production(alien, root / "backup-2.duckdb", production)
            validate_database(production)
