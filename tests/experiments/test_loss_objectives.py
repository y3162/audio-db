"""YAML loss graphs: official numbers, rejection, and one research term.

The official Python functions stay the oracle. These tests build the same
graphs from experiment YAML and from the public objective helpers, then run
one real TF-GridNet trainer step with an added waveform term.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from src.experiments.calc.losses import generator_signal_losses
from src.experiments.config import config_sha256, load_experiment
from src.experiments.core.factory import ComponentFactory, ComponentSpec
from src.experiments.calc.errors import TrainingError
from src.experiments.errors import BuildError, ConfigError
from src.experiments.losses import (
    ContextTerm,
    MagnitudeMSE,
    MultiResolutionL1Loss,
    WaveformL1,
    WeightedObjective,
)
from src.experiments.official.deepfilternet3.losses import official_deepfilternet_objective
from src.experiments.official.deepfilternet3.model import DeepFilterNet3Model
from src.experiments.official.deepfilternet3.trainer import component_losses
from src.experiments.training.official_objectives import (
    MP_METRIC,
    MP_SIGNAL,
    MOSS_GENERATOR,
    SEMAMBA_GENERATOR,
    mp_senet_signal_objective,
    require_mossformer,
    require_mp_senet,
)
from src.experiments.api import build_experiment
from tests.experiments.test_deepfilternet3 import _ARCH, _df_stream
from tests.experiments.test_tf_gridnet_training import _scripted_text

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/configs/experiments"
OFFICIAL = ROOT / "configs/reproduction/experiments/official"

_FIXTURES = {
    "mp_senet": FIX / "mp_senet_gan.yaml",
    "mossformer": FIX / "mossformergan_se_16k.yaml",
    "mossformer_leakage_safe": FIX / "mossformergan_se_16k_leakage_safe.yaml",
    "deepfilternet3": FIX / "deepfilternet3.yaml",
    "semamba": FIX / "semambapp_gsr.yaml",
    "tf_gridnet": FIX / "tf_gridnet_ms_snsd.yaml",
    "tf_locoformer": FIX / "tf_locoformer_dns.yaml",
}


def _refs(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(item.ref for item in value)
    return (value.ref,)


def _build_losses(config) -> dict[str, torch.nn.Module]:
    factory = ComponentFactory()
    built: dict[str, torch.nn.Module] = {}
    pending = [component_id for component_id, spec in config.components.items() if spec.role == "loss"]
    while pending:
        progressed = False
        for component_id in list(pending):
            spec = config.components[component_id]
            if any(ref not in built for value in spec.inject.values() for ref in _refs(value)):
                continue
            kwargs = dict(spec.kwargs)
            for name, value in spec.inject.items():
                kwargs[name] = [built[ref] for ref in _refs(value)] if isinstance(value, tuple) else built[value.ref]
            built[component_id] = factory.create(ComponentSpec(spec.class_path, kwargs))
            pending.remove(component_id)
            progressed = True
        if not progressed:
            raise AssertionError(f"unresolved loss components: {pending}")
    return built


def _leaf(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(*shape, requires_grad=True)


class ObjectiveAlgebraTests(unittest.TestCase):
    def test_weighted_sum_keeps_order_sign_and_a_zero_weight(self) -> None:
        first = ContextTerm("first")
        second = ContextTerm("second")
        third = ContextTerm("third")
        objective = WeightedObjective(
            [first, second, third],
            {"third": -2.0, "first": 0.0, "second": 0.5},
        )
        context = {
            "first": torch.tensor(3.0),
            "second": torch.tensor(4.0),
            "third": torch.tensor(5.0),
        }
        result = objective(context)
        self.assertEqual(result.records[0][0], "first")
        self.assertEqual(result.records[1][0], "second")
        self.assertEqual(result.records[2][0], "third")
        self.assertEqual(float(result.total), 0.0 * 3.0 + 0.5 * 4.0 + -2.0 * 5.0)
        self.assertEqual(float(result.terms["first"]), 3.0)

    def test_unit_weight_does_not_multiply(self) -> None:
        estimate = _leaf((1, 64))
        target = torch.randn(1, 64)
        term = MultiResolutionL1Loss(name="mr_l1")
        result = WeightedObjective([term], {"mr_l1": 1.0})({"estimate": estimate, "target": target})
        self.assertIs(result.total, result.terms["mr_l1"])

    def test_internal_scale_changes_the_term_before_the_weight(self) -> None:
        from src.experiments.losses import ScaledComplexMSE

        value = torch.randn(2, 4, 4, 2)
        context = {"estimate": value, "target": torch.zeros_like(value)}
        narrow = ScaledComplexMSE("complex", 2.0, "estimate", "target")(context)
        wide = ScaledComplexMSE("complex", 4.0, "estimate", "target")(context)
        self.assertTrue(torch.equal(wide, narrow * 2))

    def test_rejects_bad_weights_duplicate_names_and_a_missing_context(self) -> None:
        term = ContextTerm("only")
        with self.assertRaises(BuildError):
            WeightedObjective([term], {"only": float("nan")})
        with self.assertRaises(BuildError):
            WeightedObjective([term], {"other": 1.0})
        with self.assertRaises(BuildError):
            WeightedObjective([term, ContextTerm("only")], {"only": 1.0})
        with self.assertRaises(BuildError):
            WeightedObjective([term], {"only": True})
        with self.assertRaises(ValueError):
            MagnitudeMSE("magnitude", "estimate", "target")({})
        finite = WeightedObjective([term], {"only": 1.0}, check_finite=True)
        with self.assertRaises(TrainingError):
            finite({"only": torch.tensor(float("nan"))})

    def test_adding_a_term_changes_the_total_and_a_zero_weight_does_not(self) -> None:
        base = ContextTerm("base")
        extra = ContextTerm("extra")
        context = {"base": torch.tensor(1.5), "extra": torch.tensor(4.0)}
        official = WeightedObjective([base], {"base": 1.0})(context).total
        added = WeightedObjective([base, extra], {"base": 1.0, "extra": 0.25})(context).total
        silent = WeightedObjective([base, extra], {"base": 1.0, "extra": 0.0})(context).total
        self.assertNotEqual(float(added), float(official))
        self.assertEqual(float(silent), float(official))
        extra_leaf = torch.tensor(4.0, requires_grad=True)
        silent_grad = WeightedObjective([base, ContextTerm("extra")], {"base": 1.0, "extra": 0.0})(
            {"base": torch.tensor(1.5), "extra": extra_leaf}
        )
        silent_grad.total.backward()
        self.assertEqual(float(extra_leaf.grad), 0.0)


class OfficialGraphTests(unittest.TestCase):
    def test_yaml_declares_the_published_graphs(self) -> None:
        documents = {name: load_experiment(path) for name, path in _FIXTURES.items()}
        documents["mp_senet_production"] = load_experiment(OFFICIAL / "mp_senet_gan.yaml")
        documents["mossformer_production"] = load_experiment(OFFICIAL / "mossformergan_se_16k.yaml")
        documents["gridnet_production"] = load_experiment(OFFICIAL / "tf_gridnet_ms_snsd.yaml")
        for name in ("mp_senet", "mp_senet_production"):
            config = documents[name]
            signal = config.components["signal_objective"]
            self.assertEqual(signal.kwargs["weights"], {item[0]: item[1] for item in MP_SIGNAL})
            self.assertEqual(
                _refs(signal.inject["terms"]),
                ("magnitude_loss", "phase_loss", "complex_loss", "consistency_loss", "time_loss"),
            )
            self.assertEqual(config.components["metric_objective"].kwargs["weights"], {"metric": MP_METRIC[0][1]})
            self.assertEqual(
                config.components["discriminator_objective"].class_path,
                "src.experiments.losses.MPSENetDiscriminatorObjective",
            )
            self.assertEqual(config.components["phase_loss"].kwargs["mode"], "diff")
        for name in ("mossformer", "mossformer_leakage_safe", "mossformer_production"):
            config = documents[name]
            self.assertEqual(
                config.components["generator_objective"].kwargs["weights"],
                {item[0]: item[1] for item in MOSS_GENERATOR},
            )
            self.assertEqual(
                _refs(config.components["generator_objective"].inject["terms"]),
                ("ri_loss", "magnitude_loss", "time_loss", "gan_loss"),
            )
        semamba = documents["semamba"].components["generator_objective"]
        self.assertEqual(semamba.kwargs["weights"], {item[0]: item[1] for item in SEMAMBA_GENERATOR})
        self.assertEqual(documents["semamba"].components["phase_loss"].kwargs["mode"], "matrix")
        self.assertEqual(documents["semamba"].components["matching_loss"].kwargs["scale"], 2.0)
        self.assertEqual(documents["semamba"].components["matching_loss"].kwargs["reduction"], "sum")
        deepfilter = documents["deepfilternet3"]
        self.assertEqual(deepfilter.components["spectral_loss"].kwargs["factor"], 500.0)
        self.assertEqual(deepfilter.components["spectral_loss"].kwargs["fft_sizes"], [256, 512, 1024, 2048])
        self.assertEqual(deepfilter.components["local_snr_loss"].kwargs["milliseconds"], 20)
        self.assertEqual(deepfilter.components["objective"].kwargs["weights"]["local_snr"], 0.001)
        self.assertEqual(deepfilter.components["objective"].kwargs["weights"]["local_snr"], 1e-3)
        for name in ("tf_gridnet", "tf_locoformer", "gridnet_production"):
            config = documents[name]
            self.assertEqual(config.components["loss"].kwargs["windows"], [256, 512, 768, 1024])
            self.assertEqual(config.components["loss"].kwargs["time_domain_weight"], 0.5)
            self.assertIs(config.components["loss"].kwargs["normalize_variance"], False)
            self.assertEqual(config.components["objective"].kwargs["weights"], {"mr_l1": 1.0})
            self.assertEqual(config.components["trainer"].inject["objective"].ref, "objective")

    def test_yaml_build_matches_the_mp_senet_oracle_and_locks_the_claim(self) -> None:
        built = _build_losses(load_experiment(_FIXTURES["mp_senet"]))
        require_mp_senet(
            built["signal_objective"],
            built["metric_objective"],
            built["discriminator_objective"],
            "official_faithful",
        )
        require_mp_senet(
            built["signal_objective"],
            built["metric_objective"],
            built["discriminator_objective"],
            "leakage_safe",
        )
        clean_mag = torch.randn(2, 4, 6)
        generated_mag = _leaf((2, 4, 6))
        clean_phase = torch.randn(2, 4, 6)
        generated_phase = _leaf((2, 4, 6))
        clean_complex = torch.randn(2, 4, 6, 2)
        generated_complex = _leaf((2, 4, 6, 2))
        recomputed = generated_complex.detach() + 0.1
        recomputed.requires_grad_(True)
        clean_audio = torch.randn(2, 32)
        generated_audio = _leaf((2, 32))
        context = {
            "clean_audio": clean_audio,
            "clean_complex": clean_complex,
            "clean_mag": clean_mag,
            "clean_phase": clean_phase,
            "generated_audio": generated_audio,
            "generated_complex": generated_complex,
            "generated_mag": generated_mag,
            "generated_phase": generated_phase,
            "recomputed_complex": recomputed,
        }
        result = built["signal_objective"](context)
        oracle = generator_signal_losses(
            clean_mag,
            generated_mag,
            clean_phase,
            generated_phase,
            clean_complex,
            generated_complex,
            recomputed,
            clean_audio,
            generated_audio,
        )
        self.assertTrue(torch.equal(result.total, oracle["signal"]))
        for name in ("magnitude", "phase", "complex", "consistency", "time"):
            self.assertTrue(torch.equal(result.terms[name], oracle[name]), name)
        result.total.backward(retain_graph=True)
        left = [generated_mag.grad.clone(), generated_phase.grad.clone(), generated_complex.grad.clone(), generated_audio.grad.clone()]
        for tensor in (generated_mag, generated_phase, generated_complex, generated_audio, recomputed):
            tensor.grad = None
        oracle["signal"].backward()
        right = [generated_mag.grad, generated_phase.grad, generated_complex.grad, generated_audio.grad]
        for actual, expected in zip(left, right, strict=True):
            self.assertTrue(torch.equal(actual, expected))
        wrong = mp_senet_signal_objective()
        wrong.weights["magnitude"] = 0.4
        with self.assertRaises(BuildError):
            require_mp_senet(wrong, built["metric_objective"], built["discriminator_objective"], "official_faithful")
        require_mp_senet(wrong, built["metric_objective"], built["discriminator_objective"], "research")

    def test_a_different_coefficient_changes_config_sha256(self) -> None:
        config = load_experiment(_FIXTURES["mp_senet"])
        spec = config.components["signal_objective"]
        weights = dict(spec.kwargs["weights"])
        weights["magnitude"] = 0.4
        changed_spec = replace(spec, kwargs={**spec.kwargs, "weights": weights})
        changed = replace(config, components={**config.components, "signal_objective": changed_spec})
        self.assertNotEqual(config_sha256(config), config_sha256(changed))

    def test_yaml_rejects_an_unknown_key_and_a_non_finite_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            unknown = directory / "unknown.yaml"
            unknown.write_text(
                """schema: audio-db-experiment/v1
checkpoint:
  selector:
    class: tests.experiments.support.StepArchiveSelector
    kwargs: {interval: 1}
seed: 0
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs: {device: cpu, precision: fp32}
output:
  directory: unused
components:
  term:
    role: loss
    class: src.experiments.losses.ContextTerm
    kwargs:
      name: term
    not_a_field: 1
  objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights: {term: 1.0}
    inject:
      terms:
        - {ref: term}
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs: {max_logical_steps: 1}
""",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_experiment(unknown)
            non_finite = directory / "non-finite.yaml"
            non_finite.write_text(
                unknown.read_text(encoding="utf-8").replace("    not_a_field: 1\n", "").replace(
                    "weights: {term: 1.0}", "weights: {term: .nan}"
                ),
                encoding="utf-8",
            )
            loaded = load_experiment(non_finite)
            with self.assertRaises(BuildError):
                _build_losses(loaded)

    def test_mossformer_yaml_is_locked_for_both_official_claims(self) -> None:
        for name in ("mossformergan_se_16k.yaml", "mossformergan_se_16k_leakage_safe.yaml"):
            built = _build_losses(load_experiment(FIX / name))
            require_mossformer(built["generator_objective"], built["discriminator_objective"], "official_faithful")
            require_mossformer(built["generator_objective"], built["discriminator_objective"], "leakage_safe")

    def test_deepfilternet_yaml_matches_the_uninjected_oracle(self) -> None:
        torch.manual_seed(0)
        model = DeepFilterNet3Model(seed=7, architecture=dict(_ARCH))
        model.train()
        clean = torch.randn(1, 4800)
        noisy = torch.randn(1, 4800)
        stream = _df_stream(model.generator.params)
        oracle = component_losses(stream, model, clean, noisy)
        stream = _df_stream(model.generator.params)
        built = _build_losses(load_experiment(_FIXTURES["deepfilternet3"]))
        actual = component_losses(stream, model, clean, noisy, built["objective"])
        for key in ("spectral", "local_snr", "total"):
            self.assertTrue(torch.equal(actual[key], oracle[key]), key)
        helper = official_deepfilternet_objective()
        stream = _df_stream(model.generator.params)
        via_helper = component_losses(stream, model, clean, noisy, helper)
        self.assertTrue(torch.equal(via_helper["total"], oracle["total"]))

    def test_spectral_yaml_weight_one_matches_the_module(self) -> None:
        for name in ("tf_gridnet", "tf_locoformer"):
            built = _build_losses(load_experiment(_FIXTURES[name]))
            estimate = _leaf((1, 128))
            target = torch.randn(1, 128)
            direct = built["loss"](estimate, target)
            wrapped = built["objective"]({"estimate": estimate, "target": target})
            self.assertIs(wrapped.total, wrapped.terms["mr_l1"])
            self.assertTrue(torch.equal(wrapped.total, direct))


class ResearchExtensionTests(unittest.TestCase):
    def test_tf_gridnet_trainer_accepts_a_yaml_waveform_term(self) -> None:
        trainer_source = (ROOT / "src/experiments/official/tf_gridnet/trainer.py").read_text(encoding="utf-8")
        self.assertNotIn("WaveformL1", trainer_source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            official_text = _scripted_text(root / "official-out", steps=1, updates=1, max_epochs=1)
            research_base = _scripted_text(root / "research-out", steps=1, updates=1, max_epochs=1)
            zero_base = _scripted_text(root / "zero-out", steps=1, updates=1, max_epochs=1)
            research_text = research_base.replace(
                "  objective:\n",
                "  waveform_loss:\n"
                "    role: loss\n"
                "    class: src.experiments.losses.WaveformL1\n"
                "    kwargs:\n"
                "      name: waveform\n"
                "  objective:\n",
                1,
            )
            research_text = research_text.replace(
                "        mr_l1: 1.0\n",
                "        mr_l1: 1.0\n        waveform: 0.25\n",
                1,
            )
            research_text = research_text.replace(
                "        - {ref: loss}\n",
                "        - {ref: loss}\n        - {ref: waveform_loss}\n",
                1,
            )
            zero_text = zero_base.replace(
                "  objective:\n",
                "  waveform_loss:\n"
                "    role: loss\n"
                "    class: src.experiments.losses.WaveformL1\n"
                "    kwargs:\n"
                "      name: waveform\n"
                "  objective:\n",
                1,
            )
            zero_text = zero_text.replace(
                "        mr_l1: 1.0\n",
                "        mr_l1: 1.0\n        waveform: 0.0\n",
                1,
            )
            zero_text = zero_text.replace(
                "        - {ref: loss}\n",
                "        - {ref: loss}\n        - {ref: waveform_loss}\n",
                1,
            )
            (root / "official.yaml").write_text(official_text, encoding="utf-8")
            (root / "research.yaml").write_text(research_text, encoding="utf-8")
            (root / "zero.yaml").write_text(zero_text, encoding="utf-8")
            official = build_experiment(load_experiment(root / "official.yaml"))
            research = build_experiment(load_experiment(root / "research.yaml"))
            zero = build_experiment(load_experiment(root / "zero.yaml"))
            try:
                self.assertIsInstance(research.get("waveform_loss"), WaveformL1)
                self.assertEqual(research.get("objective").weights["waveform"], 0.25)
                official_loss, official_grad = _trainer_probe(official)
                research_loss, research_grad = _trainer_probe(research)
                zero_loss, zero_grad = _trainer_probe(zero)
                self.assertNotEqual(float(research_loss), float(official_loss))
                self.assertGreater(float((research_grad - official_grad).abs().max()), 0.0)
                self.assertEqual(float(zero_loss), float(official_loss))
                self.assertEqual(float((zero_grad - official_grad).abs().max()), 0.0)
                official_result = official.run()
                research_result = research.run()
                self.assertEqual(official_result.logical_step, 1)
                self.assertEqual(research_result.logical_step, 1)
                official_state = official.get("model").state_module.generator.state_dict()
                research_state = research.get("model").state_module.generator.state_dict()
                gap = max(float((official_state[key] - research_state[key]).abs().max()) for key in official_state)
                self.assertGreater(gap, 0.0)
            finally:
                official.close()
                research.close()
                zero.close()


def _trainer_probe(experiment) -> tuple[torch.Tensor, torch.Tensor]:
    trainer = experiment.get("trainer")
    runtime = experiment._graph.runtime
    view = experiment.get("data").train_epoch(0, start_update=0)
    try:
        batches = view.batch(0)
    finally:
        view.close()
    loss, _terms = trainer._forward_loss(batches, "trainer", runtime)
    experiment.get("model").module.zero_grad(set_to_none=True)
    loss.backward()
    gradient = next(experiment.get("model").module.parameters()).grad.detach().clone()
    experiment.get("model").module.zero_grad(set_to_none=True)
    return loss.detach(), gradient


class TermValueTests(unittest.TestCase):
    def test_waveform_l1_is_not_the_mossformer_reconstruction(self) -> None:
        estimate = torch.tensor([[1.0, -1.0]])
        target = torch.tensor([[0.0, 0.0]])
        value = WaveformL1("waveform")({"estimate": estimate, "target": target})
        self.assertTrue(torch.equal(value, F.l1_loss(estimate, target)))
