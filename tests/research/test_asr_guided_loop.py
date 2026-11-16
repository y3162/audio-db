"""One narrow CPU step of the research trainer against the official SEMamba++ step.

The network is the test architecture (hid 8, one TF-Mamba). It is not the
official-width model and it does not load Parakeet.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from src.experiments.api import build_experiment, load_experiment
from src.experiments.errors import RuntimeFailure
from src.models.semambapp.trainer import SEMambaUpdate
from src.research.asr_guided_se.semambapp.v1.trainer import GuidedUpdate
from tests.experiments.test_semambapp import (
    _adam_gap,
    _assert_same_checkpoint,
    _metric_lines,
    _random_batch,
    _state_gap,
    _torchrun,
    _yaml,
)
from tests.research.fake_asr import FakeParakeet


def _research_yaml(directory: Path, **kwargs) -> Path:
    features = bool(kwargs.pop("features", False))
    error = bool(kwargs.pop("error", False))
    tdt = bool(kwargs.pop("tdt", False))
    alpha = float(kwargs.pop("alpha", 1.0 if error else 0.0))
    layer = int(kwargs.pop("layer", 3))
    kwargs.setdefault("claim", "research")
    path = _yaml(directory, **kwargs)
    text = path.read_text(encoding="utf-8")
    guidance = f"""  guidance:
    role: other
    class: src.research.asr_guided_se.semambapp.v1.config.GuidanceV1
    kwargs:
      asr_features_enabled: {str(features).lower()}
      encoder_layer: {layer}
      error_weight_enabled: {str(error).lower()}
      error_weight_alpha: {alpha}
      tdt_loss_enabled: {str(tdt).lower()}
"""
    text = text.replace("components:\n  generator:", "components:\n" + guidance + "  generator:", 1)
    text = text.replace(
        "class: src.models.semambapp.models.SEMambaPPGenerator",
        "class: src.research.asr_guided_se.semambapp.v1.model.GuidedSEMambaPPGenerator",
        1,
    )
    text = text.replace(
        "class: src.models.semambapp.trainer.SEMambaPPTrainer",
        "class: src.research.asr_guided_se.semambapp.v1.trainer.GuidedSEMambaPPTrainer",
        1,
    )
    text = text.replace(
        "claim_label: research\n",
        "claim_label: research\n      experiment_claim: research\n",
        1,
    )
    inject = "      guidance: {ref: guidance}\n"
    generator_inject = """    prepare:
      trainable: true
      distribute: ddp
    inject:
      guidance: {ref: guidance}
"""
    if features:
        generator_inject += "      asr: {ref: asr}\n"
        inject += "      asr: {ref: asr}\n"
    uses_asr = features or error or tdt
    if uses_asr and not features:
        inject += "      asr: {ref: asr}\n"
    text = text.replace(
        """    prepare:
      trainable: true
      distribute: ddp
  mrd:""",
        generator_inject + "  mrd:",
        1,
    )
    if uses_asr:
        text = text.replace(
            "  trainer:\n",
            f"""  asr:
    role: other
    class: tests.research.fake_asr.FakeParakeet
    kwargs:
      model_id: nvidia/parakeet-tdt-0.6b-v2
      tdt_loss: {str(tdt).lower()}
  trainer:
""",
            1,
        )
    text = text.replace(
        "discriminator_objective: {ref: discriminator_objective}\n",
        "discriminator_objective: {ref: discriminator_objective}\n" + inject,
        1,
    )
    path.write_text(text, encoding="utf-8")
    return path


def _component(experiment, name):
    value = experiment.get(name)
    return getattr(value, "state_module", value)


class ResearchLoopTest(unittest.TestCase):
    def test_all_off_step_matches_the_official_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = build_experiment(load_experiment(_yaml(root / "official", steps=1, count=1)))
            research = build_experiment(
                load_experiment(_research_yaml(root / "research", steps=1, count=1))
            )
            try:
                self.assertIsInstance(research.get("trainer")._update, SEMambaUpdate)
                self.assertNotIsInstance(research.get("trainer")._update, GuidedUpdate)
                self.assertEqual(
                    _state_gap(
                        official.get("generator").state_module.generator,
                        research.get("generator").state_module.generator,
                    ),
                    0.0,
                )
                batch = _random_batch()
                other = replace(batch, inputs=batch.inputs.clone(), targets=batch.targets.clone())
                old = official.get("trainer").official_step([batch], official._graph.runtime)
                new = research.get("trainer").official_step([other], research._graph.runtime)
                for key, value in old.items():
                    self.assertIn(key, new)
                    if isinstance(value, float):
                        self.assertEqual(value, new[key], key)
                self.assertNotIn("tdt", new)
                self.assertEqual(
                    _state_gap(official.get("generator").state_module, research.get("generator").state_module),
                    0.0,
                )
                self.assertEqual(_state_gap(official.get("mrd").state_module, research.get("mrd").state_module), 0.0)
                self.assertEqual(_state_gap(official.get("cqt").state_module, research.get("cqt").state_module), 0.0)
                self.assertEqual(
                    _adam_gap(
                        official.get("optim_g"),
                        [parameter for group in official.get("optim_g").param_groups for parameter in group["params"]],
                        research.get("optim_g"),
                        [parameter for group in research.get("optim_g").param_groups for parameter in group["params"]],
                    ),
                    0.0,
                )
            finally:
                official.close()
                research.close()

    def test_features_stay_out_of_the_optimizer_and_error_tdt_uses_the_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = build_experiment(
                load_experiment(_research_yaml(root / "features", steps=1, count=1, features=True))
            )
            guided = build_experiment(
                load_experiment(_research_yaml(root / "guided", steps=1, count=1, error=True, tdt=True))
            )
            try:
                asr = _component(features, "asr")
                self.assertIsInstance(asr, FakeParakeet)
                self.assertIs(features.get("generator").state_module.asr, asr)
                self.assertIsInstance(features.get("trainer")._update, GuidedUpdate)
                before = asr.source.detach().clone()
                features.get("trainer").official_step([_random_batch()], features._graph.runtime)
                self.assertEqual(asr.seen_layers, [3])
                self.assertEqual(asr.loss_calls, 0)
                self.assertTrue(torch.equal(before, asr.source))
                self.assertFalse(asr.source.requires_grad)
                self.assertIsNone(asr.source.grad)
                owned = {
                    id(parameter)
                    for group in features.get("optim_g").param_groups
                    for parameter in group["params"]
                }
                self.assertNotIn(id(asr.source), owned)

                guided_asr = _component(guided, "asr")
                missing = _random_batch()
                with self.assertRaises(RuntimeFailure):
                    guided.get("trainer").official_step([missing], guided._graph.runtime)
                self.assertEqual(guided_asr.loss_calls, 0)
                spanned = replace(missing, spans=((0, 2048, 2048, 2048),))
                result = guided.get("trainer").official_step([spanned], guided._graph.runtime)
                self.assertGreater(guided_asr.loss_calls, 0)
                self.assertGreaterEqual(guided_asr.recognize_calls, 2)
                self.assertIn("tdt", result)
                self.assertFalse(guided_asr.source.requires_grad)
                self.assertIsNone(guided_asr.source.grad)
                self.assertTrue(torch.equal(guided_asr.source, torch.ones_like(guided_asr.source)))
            finally:
                features.close()
                guided.close()

    def test_resume_matches_a_continuous_research_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = root / "continuous"
            interrupted = root / "mid"
            build_experiment(
                load_experiment(_research_yaml(continuous, steps=2, count=4, validation_interval=0))
            ).run()
            first = build_experiment(
                load_experiment(
                    _research_yaml(interrupted, steps=2, count=4, validation_interval=0, stop_after=1)
                )
            )
            self.assertEqual(first.run().stop_reason, "interrupted")
            second = build_experiment(
                load_experiment(_research_yaml(interrupted, steps=2, count=4, validation_interval=0))
            )
            self.assertEqual(second.run(resume=interrupted / "archive/step_00000001.pt").stop_reason, "completed")
            _assert_same_checkpoint(continuous / "archive/step_00000002.pt", interrupted / "archive/step_00000002.pt")
            self.assertEqual(_metric_lines(continuous), _metric_lines(interrupted))

    def test_cpu_gloo_all_off_agrees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "gloo"
            completed = _torchrun(
                _research_yaml(target, steps=1, count=2, world=2, validation_interval=0)
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-2500:])
            payload = torch.load(target / "archive/step_00000001.pt", map_location="cpu", weights_only=False)
            self.assertTrue(payload["parameter_ranks_agree"])
            self.assertEqual(payload["world_size"], 2)
            self.assertEqual(payload["states"]["trainer"]["claim_label"], "research")


if __name__ == "__main__":
    unittest.main()
