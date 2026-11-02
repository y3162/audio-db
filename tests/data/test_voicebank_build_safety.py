from pathlib import Path
import tempfile
import unittest

from src.data.voicebank_sets import build


class VoiceBankBuildSafetyTests(unittest.TestCase):
    def test_existing_staging_or_database_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            database = root / "voicebank.duckdb"
            staging.mkdir()
            marker = staging / "metadata.json"
            marker.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build(root / "corpora", staging, database)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")
            marker.unlink()
            staging.rmdir()
            database.write_bytes(b"existing database")
            with self.assertRaises(FileExistsError):
                build(root / "corpora", staging, database)
            self.assertEqual(database.read_bytes(), b"existing database")
            self.assertFalse(staging.exists())


if __name__ == "__main__":
    unittest.main()
