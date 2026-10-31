"""CLI dry-run, limit, and summarize."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.evaluation.cli import main

ROOT = Path(__file__).resolve().parents[2]


class CliTests(unittest.TestCase):
    def test_smoke_dry_run_and_execute(self):
        with self._temporary_dir() as directory:
            config = ROOT / "configs/evaluation/smoke.yaml"
            dry_out = io.StringIO()
            with redirect_stdout(dry_out):
                code = main(["--config", str(config), "--dry-run", "--output", directory, "--run-id", "cli"])
            self.assertEqual(code, 0)
            self.assertIn("ready: True", dry_out.getvalue())
            self.assertIn("mock", dry_out.getvalue())
            self.assertFalse((Path(directory) / "cli").exists())
            run_out = io.StringIO()
            with redirect_stdout(run_out):
                code = main(
                    [
                        "--config",
                        str(config),
                        "--output",
                        directory,
                        "--run-id",
                        "cli",
                        "--limit",
                        "1",
                        "--seed",
                        "0",
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("wer_aggregation: micro", run_out.getvalue())
            self.assertTrue((Path(directory) / "cli" / "results.jsonl").is_file())
            summary_out = io.StringIO()
            with redirect_stdout(summary_out):
                code = main(["--summarize", str(Path(directory) / "cli")])
            self.assertEqual(code, 0)
            self.assertIn("records:", summary_out.getvalue())

    def test_missing_config_and_bad_model_filter(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--dry-run"])
        self.assertEqual(caught.exception.code, 2)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "--config",
                    str(ROOT / "configs/evaluation/smoke.yaml"),
                    "--models",
                    "not-a-model",
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("not-a-model", stderr.getvalue())

    def _temporary_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()
