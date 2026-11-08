"""ComponentFactory creates classes without a central registry."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.experiments.errors import BuildError
from src.experiments.factory import ComponentFactory, ComponentSpec
from tests.experiments.support import Marker

ROOT = Path(__file__).resolve().parents[2]


class FactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        Marker.created = 0

    def test_creates_class_that_factory_source_does_not_name(self) -> None:
        source = (ROOT / "src/experiments/factory.py").read_text(encoding="utf-8")
        self.assertNotIn("Marker", source)
        created = ComponentFactory().create(
            ComponentSpec("tests.experiments.support.Marker", {"value": 4})
        )
        self.assertIsInstance(created, Marker)
        self.assertEqual(created.value, 4)
        self.assertEqual(Marker.created, 1)

    def test_constructor_kwargs_reach_linear(self) -> None:
        module = ComponentFactory().create(
            ComponentSpec("torch.nn.Linear", {"in_features": 3, "out_features": 5})
        )
        self.assertIsInstance(module, torch.nn.Linear)
        self.assertEqual(tuple(module.weight.shape), (5, 3))

    def test_missing_module_is_build_error(self) -> None:
        with self.assertRaises(BuildError) as caught:
            ComponentFactory().create(ComponentSpec("src.experiments._missing_module.Missing", {}))
        self.assertIn("src.experiments._missing_module.Missing", str(caught.exception))

    def test_missing_class_is_build_error(self) -> None:
        with self.assertRaises(BuildError) as caught:
            ComponentFactory().create(ComponentSpec("torch.nn.NotARealClass", {}))
        self.assertIn("torch.nn.NotARealClass", str(caught.exception))

    def test_constructor_failure_is_build_error(self) -> None:
        with self.assertRaises(BuildError) as caught:
            ComponentFactory().create(ComponentSpec("torch.nn.Linear", {}))
        self.assertIn("in_features", str(caught.exception))
        self.assertNotIsInstance(caught.exception, TypeError)


if __name__ == "__main__":
    unittest.main()
