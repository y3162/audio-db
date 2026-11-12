import unittest
from pathlib import Path

from src.data.constants import CORPORA_ROOT
from src.data.corpora.paths import common_path_fields, rel_corpus_path, subset_under


class CorpusPathTests(unittest.TestCase):
    def test_fields_match_pathlib_for_nested_and_flat_names(self) -> None:
        samples = [
            Path("data/corpora/DNS2020/datasets/noise/a.wav"),
            Path("data/corpora/LibriSpeech/train-clean-100/19/198/19-198-0001.flac"),
            Path("data/corpora/Babble/noise.wav"),
            Path("data/corpora/VoiceBank-DEMAND/clean_testset_wav/p232_001.wav"),
            Path("data/corpora/X/dir/foo.bar.wav"),
            Path("data/corpora/X/dir/.hidden.wav"),
            Path("data/corpora/X/dir/no_suffix"),
        ]
        for path in samples:
            rel = path.relative_to(CORPORA_ROOT).as_posix()
            rel_dir = path.parent.relative_to(CORPORA_ROOT).as_posix()
            extension = path.suffix[1:] if path.suffix else None
            self.assertEqual(rel_corpus_path(path), rel)
            self.assertEqual(
                common_path_fields(path, "corpus", "subset"),
                (rel, "corpus", "subset", rel_dir, path.name, extension),
            )

    def test_subset_is_the_first_directory_under_the_corpus(self) -> None:
        root = Path("data/corpora/DNS2020")
        nested = root / "datasets" / "noise" / "a.wav"
        flat = root / "only.wav"
        self.assertEqual(subset_under(nested, root), "datasets")
        self.assertEqual(subset_under(flat, root), None)
        self.assertEqual(nested.relative_to(root).parts[0], "datasets")
