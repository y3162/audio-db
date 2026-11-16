"""CPU checks for the three research switches. No NeMo and no full training."""

import unittest
from pathlib import Path

import torch

from src.experiments.config import load_experiment
from src.experiments.data.sample import Batch
from src.experiments.losses import MagnitudeMSE, PhaseLoss, ScaledComplexMSE, WeightedObjective
from src.models.api import available_enhancers
from src.models.semambapp.models import SEMambaPPGenerator
from src.research.asr_guided_se.error_mask import align_token_ids, error_frames_from_alignment
from src.research.asr_guided_se.semambapp.v1.config import GuidanceV1
from src.research.asr_guided_se.semambapp.v1.loss import guided_generator_loss
from src.research.asr_guided_se.semambapp.v1.model import GuidedSEMambaPPGenerator
from src.research.asr_guided_se.semambapp.v1.trainer import _check_span, _require_batch_spans

_ROOT = Path(__file__).resolve().parents[2]


class _Token:
    def __init__(self, token_id, start, end):
        self.token_id = token_id
        self.start_offset = start
        self.end_offset = end


class FakeASR(torch.nn.Module):
    """Stand-in for the frozen encoder. It records the requested layer and stays frozen."""

    def __init__(self, *, encoder_dim=8, encoder_layers=16, layer_bias=0.0):
        super().__init__()
        self.model_id = "nvidia/parakeet-tdt-0.6b-v2"
        self.encoder_dim = encoder_dim
        self.encoder_layers = encoder_layers
        self.samples_per_frame = 1280
        self.blank_id = 1024
        self.source = torch.nn.Parameter(torch.ones(encoder_dim))
        self.loss_calls = 0
        self.seen_layers: list[int] = []
        self.layer_bias = layer_bias

    def train(self, mode=True):
        del mode
        return super().train(False)

    def encode_layer(self, audio, lengths, layer):
        self.seen_layers.append(int(layer))
        batch = int(audio.shape[0])
        values = self.source.view(1, -1, 1) + float(self.layer_bias) + float(layer)
        hidden = values.expand(batch, self.encoder_dim, 2).contiguous()
        encoded = torch.full((batch,), 2, device=audio.device, dtype=torch.long)
        del lengths
        return hidden, encoded

    def recognize(self, audio, lengths):
        del audio, lengths
        return (_Recognition(),)

    def loss_from_ids(self, audio, lengths, token_ids, *, deterministic_tdt=False):
        del lengths, token_ids, deterministic_tdt
        self.loss_calls += 1
        return audio.square().mean()

    def parameters(self, recurse=True):
        del recurse
        yield self.source


def _guidance(features=False, error=False, alpha=0.0, tdt=False, layer=15):
    return GuidanceV1(features, layer, error, alpha, tdt)


def _narrow(**overrides):
    values = dict(
        seed=1234,
        hid_feature=8,
        num_tfmamba=1,
        d_state=4,
        d_conv=2,
        expand=2,
        norm_epsilon=1e-5,
        input_channel=2,
        output_channel=1,
        n_fft=400,
        scan_backend="reference",
        hop=100,
        win=400,
        sampling_rate=16000,
    )
    values.update(overrides)
    return values


class _Recognition:
    text = "alpha"
    tokens = (_Token(7, 0, 2),)


class AlignmentTest(unittest.TestCase):
    def test_tie_break_is_correction_then_substitution_then_deletion_then_insertion(self) -> None:
        operations = [item[0] for item in align_token_ids((1, 2), (2, 1))]
        self.assertEqual(operations, ["S", "S"])

    def test_insertions_are_absent_from_the_clean_side_mask(self) -> None:
        alignment = align_token_ids((1,), (1, 9))
        self.assertEqual([item[0] for item in alignment], ["C", "I"])
        frames = error_frames_from_alignment((_Token(1, 0, 1),), alignment)
        self.assertEqual(frames, [])


class GuidanceConfigTest(unittest.TestCase):
    def test_flags_reject_integers_and_alpha_rejects_a_boolean(self) -> None:
        with self.assertRaises(Exception):
            GuidanceV1(1, 15, False, 0.0, False)
        with self.assertRaises(Exception):
            GuidanceV1(False, 15, False, True, False)
        with self.assertRaises(TypeError):
            GuidanceV1(False, 15, False, 0.0, False, extra=True)

    def test_eight_documents_share_the_training_condition(self) -> None:
        root = _ROOT / "configs/research/asr_guided_se/semambapp/v1"
        expected = {
            "all_off.yaml": (False, False, False),
            "features.yaml": (True, False, False),
            "error.yaml": (False, True, False),
            "tdt.yaml": (False, False, True),
            "features_error.yaml": (True, True, False),
            "features_tdt.yaml": (True, False, True),
            "error_tdt.yaml": (False, True, True),
            "all_on.yaml": (True, True, True),
        }
        base = load_experiment(root / "all_off.yaml")
        directories = set()
        for name, flags in expected.items():
            spec = load_experiment(root / name)
            guidance = spec.components["guidance"].kwargs
            self.assertEqual(
                (
                    guidance["asr_features_enabled"],
                    guidance["error_weight_enabled"],
                    guidance["tdt_loss_enabled"],
                ),
                flags,
            )
            self.assertEqual(guidance["encoder_layer"], 15)
            self.assertEqual(spec.components["trainer"].kwargs["experiment_claim"], "research")
            self.assertEqual(spec.components["data"].kwargs, base.components["data"].kwargs)
            self.assertEqual(
                spec.components["generator"].kwargs["num_tfmamba"],
                base.components["generator"].kwargs["num_tfmamba"],
            )
            self.assertEqual(
                spec.components["generator"].kwargs["hid_feature"],
                base.components["generator"].kwargs["hid_feature"],
            )
            directories.add(str(spec.output_directory))
        self.assertEqual(len(directories), 8)


class IdentityAndFeaturesTest(unittest.TestCase):
    def test_all_off_matches_the_official_generator(self) -> None:
        kwargs = _narrow()
        official = SEMambaPPGenerator(**kwargs)
        research = GuidedSEMambaPPGenerator(guidance=_guidance(), **kwargs)
        left = official.state_dict()
        right = research.state_dict()
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)
        torch.manual_seed(0)
        clean = torch.randn(1, 2048)
        noisy = torch.randn(1, 2048)
        official_audio = official(clean, noisy)["audio"]
        research_audio = research(clean, noisy)["audio"]
        self.assertEqual(official_audio.shape, research_audio.shape)
        self.assertEqual(float((official_audio - research_audio).abs().max()), 0.0)

    def test_selected_layer_is_detached_and_reaches_only_the_fusion(self) -> None:
        asr = FakeASR(layer_bias=0.25)
        asr.requires_grad_(False)
        model = GuidedSEMambaPPGenerator(guidance=_guidance(features=True, layer=3), asr=asr, **_narrow())
        clean = torch.randn(1, 2048)
        noisy = torch.randn(1, 2048)
        audio = model(clean, noisy)["audio"]
        audio.sum().backward()
        self.assertEqual(asr.seen_layers, [3])
        self.assertIsNone(asr.source.grad)
        self.assertFalse(asr.source.requires_grad)
        self.assertIsNotNone(model.generator.asr_proj.weight.grad)
        self.assertGreater(float(model.generator.asr_proj.weight.grad.abs().sum()), 0.0)

    def test_two_layers_change_the_waveform(self) -> None:
        clean = torch.randn(1, 2048)
        noisy = torch.randn(1, 2048)
        left = GuidedSEMambaPPGenerator(
            guidance=_guidance(features=True, layer=1),
            asr=FakeASR(),
            **_narrow(),
        )
        right = GuidedSEMambaPPGenerator(
            guidance=_guidance(features=True, layer=2),
            asr=FakeASR(layer_bias=3.0),
            **_narrow(),
        )
        right.generator.asr_proj.load_state_dict(left.generator.asr_proj.state_dict())
        right.generator.asr_fuse.load_state_dict(left.generator.asr_fuse.state_dict())
        left_audio = left(clean, noisy)["audio"]
        right_audio = right(clean, noisy)["audio"]
        self.assertGreater(float((left_audio - right_audio).abs().max()), 0.0)

    def test_a_missing_span_fails_before_a_step(self) -> None:
        batch = Batch(
            inputs=torch.zeros(1, 8),
            targets=torch.zeros(1, 8),
            sample_id="one",
            realization_id="one",
        )
        with self.assertRaisesRegex(Exception, "correspondence"):
            _require_batch_spans(batch)
        _check_span((0, 2048, 2048, 2048), 2048)
        with self.assertRaisesRegex(Exception, "window"):
            _check_span((0, 2048, 1000, 2048), 2048)


class WeightedLossTest(unittest.TestCase):
    def test_alpha_zero_matches_a_zero_mask_and_the_official_mean_without_padding(self) -> None:
        torch.manual_seed(2)
        estimate = torch.rand(1, 201, 21)
        target = torch.rand(1, 201, 21)
        context = {"generated_mag": estimate, "clean_mag": target}
        objective = WeightedObjective(
            [MagnitudeMSE("magnitude", "generated_mag", "clean_mag")],
            {"magnitude": 1.0},
        )
        official = objective(context).total
        alpha_zero = guided_generator_loss(
            objective,
            context,
            guidance=_guidance(error=True, alpha=0.0),
            sample_error=torch.ones(1, 2048, dtype=torch.bool),
            content_length=torch.tensor([2048]),
            tdt=None,
            stft_hop=100,
        ).total
        zero_mask = guided_generator_loss(
            objective,
            context,
            guidance=_guidance(error=True, alpha=1.0),
            sample_error=torch.zeros(1, 2048, dtype=torch.bool),
            content_length=torch.tensor([2048]),
            tdt=None,
            stft_hop=100,
        ).total
        self.assertEqual(float((alpha_zero - zero_mask).abs()), 0.0)
        self.assertLess(float((alpha_zero - official).abs()), 1e-6)

        phase_context = {
            "generated_mag": estimate,
            "clean_mag": target,
            "generated_phase": torch.rand(1, 201, 21),
            "clean_phase": torch.rand(1, 201, 21),
        }
        phase = WeightedObjective([PhaseLoss("phase", "matrix")], {"phase": 1.0})
        weighted = guided_generator_loss(
            phase,
            phase_context,
            guidance=_guidance(error=True, alpha=1.0),
            sample_error=torch.zeros(1, 2048, dtype=torch.bool),
            content_length=torch.tensor([2048]),
            tdt=torch.tensor(0.25),
            stft_hop=100,
        ).total
        plain = phase(phase_context).total
        self.assertLess(abs(float(weighted) - float(plain) - 0.25), 1e-5)

        complex_context = {
            "generated_mag": estimate,
            "clean_mag": target,
            "generated_complex": torch.rand(1, 201, 21, 2),
            "clean_complex": torch.rand(1, 201, 21, 2),
            "recomputed_complex": torch.rand(1, 201, 21, 2),
        }
        complex_objective = WeightedObjective(
            [ScaledComplexMSE("complex", 2.0, "generated_complex", "clean_complex")],
            {"complex": 1.0},
        )
        observed = guided_generator_loss(
            complex_objective,
            complex_context,
            guidance=_guidance(error=True, alpha=1.0),
            sample_error=torch.ones(1, 2048, dtype=torch.bool),
            content_length=torch.tensor([2048]),
            tdt=None,
            stft_hop=100,
        ).total
        self.assertTrue(torch.isfinite(observed))


class ImportBoundaryTest(unittest.TestCase):
    def test_the_research_enhancer_name_is_registered(self) -> None:
        self.assertIn("asr_guided_semambapp_v1", available_enhancers())
        self.assertIn("semambapp", available_enhancers())


if __name__ == "__main__":
    unittest.main()
