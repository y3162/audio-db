import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from src.data.bootstrap import bootstrap, replace_production, validate_database
from src.data.corpora_catalog import CATALOG
from src.data.db.eval_schema import create_eval_schema
from src.data.db.schema import create_corpora_schema
from src.data.link_corpora import link_corpora

EXCLUDED = {"LibriLM", "VCTK-RVA", "VoxCelebMeta", "cmudict", "_ingest", "FSD50K", "SpeechCommands"}


def _seed_db(path: Path, corpus: str = "LibriSpeech") -> None:
    connection = duckdb.connect(str(path))
    create_corpora_schema(connection)
    create_eval_schema(connection)
    connection.execute(
        """
        INSERT INTO corpora.audio_files (path, corpus, file_name, sample_rate, frames, channels)
        VALUES ('LibriSpeech/a.flac', ?, 'a.flac', 16000, 16000, 1)
        """,
        [corpus],
    )
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
