"""Unsupported model names must fail at the public SE entry points."""
import unittest

from src.se import available_enhancers, build_model, load_enhancer


class RemovedModelsTests(unittest.TestCase):
    def test_removed_models_are_rejected(self):
        for name in ("unipase", "uni-pase", "stupase", "stu-pase"):
            with self.subTest(name=name):
                self.assertNotIn(name, available_enhancers())
                with self.assertRaisesRegex(ValueError, "Unknown or not yet implemented model"):
                    build_model(name)
                with self.assertRaisesRegex(ValueError, "Unknown or not yet implemented model"):
                    load_enhancer(name, checkpoint="unused.pt")
