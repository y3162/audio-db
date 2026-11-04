import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.setup.dotenv import emit_unset, parse_env


class DotenvTests(unittest.TestCase):
    def test_quotes_comments_and_export(self) -> None:
        values = parse_env('\n'.join([
            "# comment",
            "export AUDIO_DB_DATASET_ROOT=\"/data/my sets\"",
            "UV_CACHE_DIR='/tmp/cache'",
            "EMPTY=",
            "NOTE=value # ignored",
        ]))
        self.assertEqual(values["AUDIO_DB_DATASET_ROOT"], "/data/my sets")
        self.assertEqual(values["UV_CACHE_DIR"], "/tmp/cache")
        self.assertNotIn("EMPTY", values)
        self.assertEqual(values["NOTE"], "value")

    def test_rejects_command_substitution(self) -> None:
        with self.assertRaises(ValueError):
            parse_env("AUDIO_DB_DATASET_ROOT=$(pwd)")

    def test_existing_environment_wins(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('AUDIO_DB_DATASET_ROOT="/from/file"\nOTHER="/from/file"\n')
            text = emit_unset(path, {"AUDIO_DB_DATASET_ROOT": "/from/shell"})
        self.assertNotIn("AUDIO_DB_DATASET_ROOT", text)
        self.assertIn("OTHER=", text)


if __name__ == "__main__":
    unittest.main()
