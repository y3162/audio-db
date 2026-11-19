"""Ablation configs, the VoiceBank speaker split, and shared frame aggregation."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.experiments.calc.batch import exact_accumulation
from src.experiments.config import load_experiment
from src.experiments.losses import MagnitudeMSE, WeightedObjective
from src.experiments.stopping import MetricEvent, PatienceStopPolicy
from src.research.asr_guided_se.semambapp.v1.config import MODEL_ID, GuidanceV1
from src.research.asr_guided_se.semambapp.v1.loss import frame_weight_and_valid, guided_generator_loss
from src.research.asr_guided_se.semambapp.v1.voicebank import ablation_holdout

_ROOT = Path(__file__).resolve().parents[2]
_ABLATION = _ROOT / "configs/research/asr_guided_se/semambapp/v1/ablation"
_FLAGS = {
    "A0": (False, False, False),
    "A1": (True, False, False),
    "A2": (False, False, True),
    "A3": (False, True, False),
    "A4": (True, False, True),
    "A5": (True, True, False),
    "A6": (False, True, True),
    "A7": (True, True, True),
}
_TRAIN_SNR = {0, 5, 10, 15}
_TEST_SNR = {2.5, 7.5, 12.5, 17.5}


def _guidance(error: bool, alpha: float) -> GuidanceV1:
    return GuidanceV1(False, 15, error, alpha, False)


_OLD_IDS = (
    "error_aware_se_mamba_pp",
    "error_aware_semambapp",
    "error-aware-se-mamba-pp",
    "error-aware-semambapp",
    "Error-aware SE-Mamba++",
    "semamabapp",
)
_SCAN_ROOTS = ("src", "configs", "tests", "scripts", "docs", ".agents", "AGENTS.md", "README.md")


class AblationConfigTest(unittest.TestCase):
    def test_eight_configs_share_the_run_and_differ_by_switches(self) -> None:
        base = load_experiment(_ABLATION / "A0.yaml")
        directories = set()
        for name, flags in _FLAGS.items():
            spec = load_experiment(_ABLATION / f"{name}.yaml")
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
            self.assertEqual(guidance["error_weight_alpha"], 1.0)
            self.assertEqual(spec.components["trainer"].kwargs["global_batch_size"], 8)
            self.assertEqual(spec.components["trainer"].kwargs["micro_batch_size"], 1)
            self.assertEqual(spec.components["trainer"].kwargs["model_id"], MODEL_ID)
            self.assertEqual(spec.components["trainer"].kwargs["experiment_id"], "voicebank_m15db_ablation")
            self.assertEqual(spec.components["trainer"].kwargs["condition_id"], name)
            self.assertIs(spec.components["trainer"].kwargs["valid_frame_aggregation"], True)
            self.assertTrue(
                str(spec.output_directory).endswith(
                    f"data/results/research/asr_guided_se/semambapp/v1/voicebank_m15db_ablation/{name}"
                )
            )
            self.assertEqual(spec.components["trainer"].kwargs["max_epochs"], 100)
            self.assertEqual(spec.components["trainer"].kwargs["experiment_claim"], "research")
            self.assertEqual(exact_accumulation(8, 1, 1), 8)
            self.assertEqual(spec.components["stopping"].kwargs["patience"], 10)
            self.assertEqual(spec.components["stopping"].kwargs["min_delta"], 0.0)
            self.assertEqual(spec.components["stopping"].kwargs["metric"], "validation.wer")
            self.assertEqual(spec.components["selection"].kwargs["metric"], "validation.wer")
            self.assertEqual(spec.components["source"].kwargs["split_seed"], 42)
            self.assertEqual(spec.components["source"].kwargs["valid_count"], 3)
            self.assertEqual(
                spec.components["protocol"].class_path,
                "src.research.asr_guided_se.semambapp.v1.voicebank.VoicebankAblationProtocol",
            )
            self.assertEqual(spec.components["protocol"].kwargs, base.components["protocol"].kwargs)
            self.assertEqual(
                spec.components["source"].class_path,
                base.components["source"].class_path,
            )
            self.assertEqual(spec.components["asr"].kwargs["tdt_loss"], flags[2])
            self.assertEqual(spec.seed, base.seed)
            self.assertEqual(spec.components["generator"].kwargs, base.components["generator"].kwargs)
            self.assertEqual(spec.components["batcher"].kwargs, base.components["batcher"].kwargs)
            trainer_inject = spec.components["trainer"].inject
            self.assertIn("evaluators", trainer_inject)
            self.assertIn("model_selection", trainer_inject)
            if flags[0] or flags[1] or flags[2]:
                self.assertIn("asr", trainer_inject)
            else:
                self.assertNotIn("asr", trainer_inject)
            if flags[0]:
                self.assertIn("asr", spec.components["generator"].inject)
            else:
                self.assertNotIn("asr", spec.components["generator"].inject)
            directories.add(str(spec.output_directory))
        self.assertEqual(len(directories), 8)

    def test_old_model_ids_are_absent_from_the_source_tree(self) -> None:
        found: list[str] = []
        for relative in _SCAN_ROOTS:
            path = _ROOT / relative
            files = [path] if path.is_file() else path.rglob("*") if path.exists() else []
            for child in files:
                if not child.is_file() or child.suffix in {".pyc", ".pt"} or "__pycache__" in child.parts:
                    continue
                if child.name == "test_ablation_voicebank.py":
                    continue
                text = child.read_text(encoding="utf-8", errors="ignore")
                for token in _OLD_IDS:
                    if token in text:
                        found.append(f"{child.relative_to(_ROOT)}: {token}")
        self.assertEqual(found, [])

    def test_patience_stops_after_ten_non_improvements(self) -> None:
        policy = PatienceStopPolicy("validation.wer", "min", 10, 0.0)
        self.assertFalse(policy.observe(MetricEvent("validation", 0, {"wer": 0.5})).stop)
        for step in range(1, 10):
            self.assertFalse(policy.observe(MetricEvent("validation", step, {"wer": 0.5})).stop)
        self.assertTrue(policy.observe(MetricEvent("validation", 10, {"wer": 0.5})).stop)
        restored = PatienceStopPolicy("validation.wer", "min", 10, 0.0)
        restored.load_state_dict(policy.state_dict())
        self.assertTrue(restored.stopped)
        self.assertEqual(restored.bad_observations, 10)


class FrameAggregationTest(unittest.TestCase):
    def test_alpha_one_doubles_error_frames_and_padding_matches_when_weighting_is_off(self) -> None:
        weight, valid = frame_weight_and_valid(
            torch.tensor([[True, False]]),
            torch.tensor([2]),
            4,
            1,
            1.0,
            torch.float32,
        )
        self.assertEqual(float(weight[0, 0]), 2.0)
        self.assertEqual(float(weight[0, 1]), 1.0)
        self.assertFalse(bool(valid[0, 3]))

        torch.manual_seed(4)
        estimate = torch.rand(1, 8, 6)
        target = torch.rand(1, 8, 6)
        estimate = torch.nn.functional.pad(estimate, (0, 4))
        target = torch.nn.functional.pad(target, (0, 4))
        context = {"generated_mag": estimate, "clean_mag": target}
        objective = WeightedObjective(
            [MagnitudeMSE("magnitude", "generated_mag", "clean_mag")],
            {"magnitude": 1.0},
        )
        content = torch.tensor([480])
        off = guided_generator_loss(
            objective,
            context,
            guidance=_guidance(False, 1.0),
            sample_error=None,
            content_length=content,
            tdt=None,
            stft_hop=100,
        ).total
        on = guided_generator_loss(
            objective,
            context,
            guidance=_guidance(True, 1.0),
            sample_error=torch.zeros(1, 800, dtype=torch.bool),
            content_length=content,
            tdt=None,
            stft_hop=100,
        ).total
        self.assertLess(float((off - on).abs()), 1e-6)
        changed = target.clone()
        changed[:, :, 6:] += 3
        shifted = guided_generator_loss(
            objective,
            {"generated_mag": estimate, "clean_mag": changed},
            guidance=_guidance(False, 1.0),
            sample_error=None,
            content_length=content,
            tdt=None,
            stft_hop=100,
        ).total
        self.assertLess(float((shifted - off).abs()), 1e-6)


class VoicebankSplitTest(unittest.TestCase):
    def test_seed_42_holds_out_three_speakers_without_overlap(self) -> None:
        speakers = [f"p{index:03d}" for index in range(226, 254)]
        train, valid = ablation_holdout(speakers, seed=42, valid_count=3)
        self.assertEqual(len(train), 25)
        self.assertEqual(len(valid), 3)
        self.assertFalse(set(train) & set(valid))
        again, held = ablation_holdout(list(reversed(speakers)), seed=42, valid_count=3)
        self.assertEqual(train, again)
        self.assertEqual(valid, held)

    def test_snapshot_split_labels_and_noise_scale(self) -> None:
        snapshot = _ROOT / "data/db/snapshots"
        if not snapshot.is_dir():
            self.skipTest("voicebank snapshot root is not on disk")
        from src.experiments.data.mix.voicebank_snr import noise_scale, render_case
        from src.research.asr_guided_se.semambapp.v1.voicebank import (
            VoicebankAblationProtocol,
            VoicebankAblationSource,
        )

        source = VoicebankAblationSource(
            "voicebank.snr_shift.minus_15db@1",
            snapshot,
            "train",
            split_seed=42,
            valid_count=3,
        )
        try:
            source.open()
        except Exception as exc:
            if "matched 0" in str(exc) or "missing" in str(exc).lower():
                self.skipTest(f"voicebank snapshot is not available: {exc}")
            raise
        record = source.ablation_record
        self.assertIsNotNone(record)
        assert record is not None
        train_speakers = set(record["train_speakers"])
        valid_speakers = set(record["valid_speakers"])
        test_speakers = set(record["test_speakers"])
        self.assertEqual(len(train_speakers), 25)
        self.assertEqual(len(valid_speakers), 3)
        self.assertEqual(len(test_speakers), 2)
        self.assertFalse(train_speakers & valid_speakers)
        self.assertFalse(train_speakers & test_speakers)
        self.assertFalse(valid_speakers & test_speakers)
        store = source.store
        labels = {"train": set(), "valid": set(), "test": set()}
        shortest = None
        for slot, name in (("train.cases", "train"), ("valid.cases", "valid"), ("test.cases", "test")):
            for case_id in store.case_ids(slot):
                document = store.docs[case_id]
                labels[name].add(float(document["draws"]["official_snr_db"]))
                length = int(document["output"]["length"])
                if name == "train" and (shortest is None or length < shortest[0]):
                    shortest = (length, case_id)
        self.assertEqual(labels["train"], _TRAIN_SNR)
        self.assertEqual(labels["valid"], _TRAIN_SNR)
        self.assertEqual(labels["test"], _TEST_SNR)
        self.assertIsNotNone(shortest)
        assert shortest is not None
        case_id = shortest[1]
        clean, noisy = render_case(store, case_id, snr_shift_db=0.0)
        shifted_clean, shifted = render_case(store, case_id, snr_shift_db=-15.0)
        self.assertEqual(clean.shape, noisy.shape)
        self.assertEqual(clean.shape, shifted.shape)
        self.assertTrue(torch.equal(torch.from_numpy(clean), torch.from_numpy(shifted_clean)))
        base = noisy - clean
        moved = shifted - shifted_clean
        usable = abs(base) > 1e-8
        self.assertGreater(int(usable.sum()), 0)
        ratio = float((moved[usable] / base[usable]).mean())
        self.assertAlmostEqual(ratio, noise_scale(-15.0), places=6)
        protocol = VoicebankAblationProtocol(source)
        protocol.prepare()
        protocol.seed = 42
        text = protocol.reference_transcript(case_id)
        self.assertIsInstance(text, str)
        assert text is not None
        self.assertTrue(text.strip())
        length = protocol.case_length(case_id)
        crop = protocol.locate(case_id, length, 0)
        self.assertLessEqual(crop.length, 48000)
        self.assertEqual(crop.pad_to, 48000)
        self.assertEqual(protocol.normalize, "noisy_peak_v1")
        from src.experiments.data.sampling.planner import fixed_pair_pad_name
        from src.research.asr_guided_se.semambapp.v1.voicebank import _resample

        self.assertEqual(fixed_pair_pad_name("noisy_peak_v1"), "tail_zero")
        noisy_out, clean_out = protocol._peak_pair(case_id)
        self.assertEqual(noisy_out.shape, (length,))
        self.assertEqual(clean_out.shape, (length,))
        clean_16 = _resample(shifted_clean)
        noisy_16 = _resample(shifted)
        scale = np.max(np.abs(noisy_16)) + 1e-9
        np.testing.assert_allclose(noisy_out, np.float32(noisy_16 / scale), atol=0, rtol=0)
        np.testing.assert_allclose(clean_out, np.float32(clean_16 / scale), atol=0, rtol=0)
        self.assertGreater(float(np.max(np.abs(noisy_out))), 0.999)
        self.assertLessEqual(float(np.max(np.abs(noisy_out))), 1.0)


class SharedNoisyPeakTest(unittest.TestCase):
    def test_one_noisy_scale_keeps_relative_level_and_silent_input(self) -> None:
        from src.research.asr_guided_se.semambapp.v1.voicebank import (
            VoicebankAblationProtocol,
            _shared_noisy_peak,
        )

        clean = np.array([2.0, -0.5, 0.25, 0.0], dtype=np.float32)
        noisy = np.array([0.5, -0.25, 0.0, 0.1], dtype=np.float32)
        clean_peak, noisy_peak = _shared_noisy_peak(clean, noisy)
        scale = np.max(np.abs(noisy)) + 1e-9
        np.testing.assert_allclose(clean_peak, clean / scale)
        np.testing.assert_allclose(noisy_peak, noisy / scale)
        usable = np.abs(noisy) > 0
        np.testing.assert_allclose(clean_peak[usable] / noisy_peak[usable], clean[usable] / noisy[usable])
        self.assertGreater(float(np.max(np.abs(clean_peak))), 1.0)
        self.assertAlmostEqual(float(np.max(np.abs(noisy_peak))), 1.0, places=5)
        speech = clean
        noise = noisy - clean
        before = float(np.mean(noise**2) / np.mean(speech**2))
        after_noise = noisy_peak - clean_peak
        after = float(np.mean(after_noise**2) / np.mean(clean_peak**2))
        self.assertAlmostEqual(before, after, places=5)
        silent_clean, silent_noisy = _shared_noisy_peak(np.zeros(4, np.float32), np.zeros(4, np.float32))
        self.assertTrue(np.isfinite(silent_clean).all())
        self.assertTrue(np.isfinite(silent_noisy).all())
        self.assertTrue(np.all(silent_clean == 0))
        self.assertTrue(np.all(silent_noisy == 0))

        protocol = VoicebankAblationProtocol.__new__(VoicebankAblationProtocol)
        protocol._full = {}
        protocol._shift = -15.0
        protocol.verify_on_read = False
        protocol.source = type("Source", (), {"store": object()})()
        protocol.case_length = lambda _case_id: int(clean.shape[0])
        with (
            patch(
                "src.research.asr_guided_se.semambapp.v1.voicebank.render_case",
                return_value=(clean, noisy),
            ),
            patch(
                "src.research.asr_guided_se.semambapp.v1.voicebank._resample",
                side_effect=lambda audio: np.ascontiguousarray(audio, dtype=np.float32),
            ),
        ):
            rendered_noisy, rendered_clean = protocol._peak_pair("case-shared")
        np.testing.assert_allclose(rendered_noisy, np.asarray(noisy_peak, dtype=np.float32))
        np.testing.assert_allclose(rendered_clean, np.asarray(clean_peak, dtype=np.float32))
        self.assertEqual(rendered_noisy.shape, clean.shape)
        self.assertEqual(rendered_clean.shape, clean.shape)
        from src.experiments.data.sampling.planner import fixed_pair_pad_name

        self.assertEqual(fixed_pair_pad_name("noisy_peak_v1"), fixed_pair_pad_name("independent_peak_v1"))
        self.assertEqual(fixed_pair_pad_name("noisy_peak_v1"), "tail_zero")


if __name__ == "__main__":
    unittest.main()
