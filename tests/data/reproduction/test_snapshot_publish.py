"""Readers must not open an export that has not been renamed into place."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from src.experiments.data.errors import ReproductionError
from src.experiments.data.fixed_pair import SnapshotSource
from src.experiments.data.fixed_pair_stream import find_snapshot_dir
from src.experiments.errors import DataError


_SELECTOR = "semambapp.gsr.smoke@1"
_DOCUMENT = {"name": "semambapp.gsr.smoke", "version": 1}


def _write_dataset(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dataset.json").write_text(json.dumps(_DOCUMENT), encoding="utf-8")


class SnapshotPublishTests(unittest.TestCase):
    def test_partial_directory_is_invisible_until_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / ".dataset.partial"
            _write_dataset(partial)
            _write_dataset(root / "dataset.partial")
            with self.assertRaises(ReproductionError) as stream:
                find_snapshot_dir(root, _SELECTOR)
            self.assertIn("found 0", str(stream.exception))
            source = SnapshotSource(_SELECTOR, root, "train")
            with self.assertRaises(DataError) as resolved:
                source._resolve()
            self.assertIn("matched 0", str(resolved.exception))

            final = root / "dataset"
            os.replace(partial, final)
            self.assertEqual(find_snapshot_dir(root, _SELECTOR), final)
            self.assertEqual(source._resolve(), final)

    def test_reader_does_not_attach_while_the_partial_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / ".dataset.partial"
            started = threading.Event()
            release = threading.Event()

            def publish() -> None:
                _write_dataset(partial)
                started.set()
                self.assertTrue(release.wait(5))
                os.replace(partial, root / "dataset")

            thread = threading.Thread(target=publish)
            thread.start()
            self.assertTrue(started.wait(5))
            try:
                with self.assertRaises(ReproductionError) as caught:
                    find_snapshot_dir(root, _SELECTOR)
                self.assertIn("found 0", str(caught.exception))
            finally:
                release.set()
                thread.join()
            self.assertEqual(find_snapshot_dir(root, _SELECTOR).name, "dataset")
