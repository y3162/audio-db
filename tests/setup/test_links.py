import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.corpora_catalog import CATALOG
from src.data.link_corpora import link_corpora


class LinkSetupTests(unittest.TestCase):
    def test_dry_run_does_not_create_links(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            corpora = root / "corpora"
            for spec in CATALOG:
                (source / spec.source_name).mkdir(parents=True)
            lines = link_corpora(source, corpora, dry_run=True)
            self.assertTrue(any(line.startswith("would-link ") for line in lines))
            self.assertFalse(corpora.exists())

    def test_second_run_is_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            corpora = root / "corpora"
            for spec in CATALOG:
                (source / spec.source_name).mkdir(parents=True)
            link_corpora(source, corpora)
            again = link_corpora(source, corpora)
            self.assertTrue(all(line.startswith("ok ") for line in again))

    def test_only_and_partial_leave_other_names_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "LibriSpeech").mkdir(parents=True)
            lines = link_corpora(source, root / "corpora", only=["LibriSpeech"], allow_partial=True)
            self.assertTrue(any("LibriSpeech" in line for line in lines))
            self.assertTrue((root / "corpora" / "LibriSpeech").is_symlink())
            self.assertFalse((root / "corpora" / "DEMAND").exists())

    def test_conflict_is_not_replaced_when_partial(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            corpora = root / "corpora"
            for spec in CATALOG:
                (source / spec.source_name).mkdir(parents=True)
            blocked = corpora / "LibriSpeech"
            blocked.parent.mkdir()
            blocked.mkdir()
            with self.assertRaises(FileExistsError):
                link_corpora(source, corpora, allow_partial=True)
            self.assertTrue(blocked.is_dir())
            self.assertFalse((corpora / "DEMAND").exists())

    def test_override_beats_dataset_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "other" / "LibriSpeech"
            other.mkdir(parents=True)
            (root / "source" / "LibriSpeech").mkdir(parents=True)
            link_corpora(
                root / "source",
                root / "corpora",
                only=["LibriSpeech"],
                overrides={"LibriSpeech": other},
            )
            self.assertEqual((root / "corpora" / "LibriSpeech").resolve(), other.resolve())


if __name__ == "__main__":
    unittest.main()
