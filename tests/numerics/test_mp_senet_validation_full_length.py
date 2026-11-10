"""Validation feeds the full waveform into MP-SENet. Training still rejects a length mismatch."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.se import build_model
from src.experiments.calc.rng import set_determinism
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.mp_senet import (
    MPSENetTask,
    align_validation_length,
    official_center_istft_length,
)
from src.experiments.calc.stft import mag_pha_stft
from tests.numerics.support import tiny_config

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "src/se/official/mp-senet"


def _task(loss="reconstruction", **model):
    config = tiny_config(Path("unused"), loss=loss)
    config["model"].update(
        {
            "n_fft": 400,
            "hop_size": 100,
            "win_size": 400,
            "compress_factor": 0.3,
            "sampling_rate": 16000,
        }
    )
    config["model"].update(model)
    task = MPSENetTask(config, torch.device("cpu"))
    task.eval_mode()
    return task


def _batch(length, rows=1, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    clean = torch.randn(rows, length, generator=generator)
    noisy = torch.randn(rows, length, generator=generator)
    return {"clean": clean, "noisy": noisy}


class FullLengthValidationTests(unittest.TestCase):
    def test_hop_multiple_length_is_unchanged(self):
        task = _task()
        report = task.validation_report(_batch(1600, seed=1))
        row = report["audio_lengths"][0]
        self.assertEqual(row["model_input"], 1600)
        self.assertEqual(row["clean"], 1600)
        self.assertEqual(row["audio"], 1600)
        self.assertEqual(row["pesq_reference"], 1600)
        self.assertEqual(row["pesq_estimate"], 1600)
        self.assertEqual(row["time_overlap"], 1600)
        self.assertEqual(align_validation_length(1600, 100, 400), 1600)
        self.assertEqual(report["losses"]["time"]["status"], "ok")

    def test_non_multiple_keeps_1650_as_the_model_input(self):
        task = _task()
        seen = []
        real = mag_pha_stft

        def spy(waveform, *args, **kwargs):
            seen.append(int(waveform.shape[-1]))
            return real(waveform, *args, **kwargs)

        with patch("src.experiments.calc.mp_senet.mag_pha_stft", spy):
            with patch(
                "src.experiments.calc.mp_senet.align_validation_length",
                side_effect=AssertionError("validation must not truncate with align_validation_length"),
            ):
                report = task.validation_report(_batch(1650, seed=2))
        row = report["audio_lengths"][0]
        self.assertEqual(align_validation_length(1650, 100, 400), 1600)
        self.assertEqual(row["model_input"], 1650)
        self.assertNotEqual(row["model_input"], 1600)
        self.assertEqual(row["audio"], 1600)
        self.assertEqual(row["pesq_reference"], 1650)
        self.assertEqual(row["pesq_estimate"], 1600)
        self.assertEqual(row["time_overlap"], 1600)
        self.assertEqual(seen[:2], [1650, 1650])
        self.assertEqual(seen[2], 1600)
        self.assertEqual(official_center_istft_length(1650, 100), 1600)

    def test_short_input_that_old_alignment_rejects_still_reaches_the_generator(self):
        with self.assertRaises(TrainingError):
            align_validation_length(250, 100, 400)
        self.assertEqual(official_center_istft_length(250, 100), 200)
        task = _task()
        with patch(
            "src.experiments.calc.mp_senet.align_validation_length",
            side_effect=AssertionError("validation must not truncate with align_validation_length"),
        ):
            report = task.validation_report(_batch(250, seed=3))
        row = report["audio_lengths"][0]
        self.assertEqual(row["model_input"], 250)
        self.assertEqual(row["audio"], 200)
        self.assertEqual(row["pesq_estimate"], 200)
        self.assertEqual(row["time_overlap"], 200)
        self.assertEqual(report["losses"]["magnitude"]["status"], "ok")
        self.assertEqual(report["losses"]["time"]["status"], "ok")

        tiny = _task(n_fft=32, hop_size=8, win_size=32)
        with self.assertRaises(TrainingError):
            align_validation_length(20, 8, 32)
        short = tiny.validation_report(_batch(20, seed=4))
        short_row = short["audio_lengths"][0]
        self.assertEqual(short_row["model_input"], 20)
        self.assertEqual(short_row["audio"], 16)
        self.assertEqual(short_row["pesq_estimate"], 16)

    def test_several_lengths_use_the_full_waveform(self):
        task = _task()
        for length in (800, 850, 1600, 1650, 301, 12345):
            with self.subTest(length=length):
                row = task.validation_report(_batch(length, seed=length))["audio_lengths"][0]
                produced = official_center_istft_length(length, 100)
                self.assertEqual(row["model_input"], length)
                self.assertEqual(row["audio"], produced)
                self.assertEqual(row["pesq_reference"], length)
                self.assertEqual(row["pesq_estimate"], produced)
                self.assertEqual(row["time_overlap"], produced)

    def test_update_path_still_rejects_a_non_multiple(self):
        task = _task()
        clean = torch.randn(1, 1650)
        noisy = torch.randn(1, 1650)
        with self.assertRaises(TrainingError) as caught:
            task._generate(clean, noisy, generator=task.generator)
        self.assertIn("1600", str(caught.exception))
        self.assertIn("1650", str(caught.exception))
        with self.assertRaises(TrainingError):
            task.train_update([{"clean": clean, "noisy": noisy}], 1)
        matched = torch.randn(1, 1600)
        parts = task._generate(matched, matched, generator=task.generator)
        self.assertEqual(int(parts["audio"].shape[-1]), 1600)

    def test_truncated_input_changes_generator_magnitude(self):
        official_stft, official_net = _load_official_generator()
        deterministic = torch.are_deterministic_algorithms_enabled()
        set_determinism(False, cuda=False)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            spec = {
                "dense_channel": 64,
                "num_tsconformers": 4,
                "n_fft": 400,
                "hop_size": 100,
                "win_size": 400,
                "compress_factor": 0.3,
                "beta": 2.0,
                "sampling_rate": 16000,
            }
            torch.manual_seed(1234)
            official = official_net(
                SimpleNamespace(dense_channel=64, n_fft=400, beta=2.0, num_tsconformers=4),
                num_tsblocks=4,
            ).eval()
            current = build_model("mp_senet", spec).eval()
            self.assertEqual(list(official.state_dict()), list(current.state_dict()))
            current.load_state_dict(official.state_dict())
            torch.manual_seed(7)
            clean = torch.randn(2, 1600)
            noisy = torch.randn(2, 1600)
            current_mag, current_phase, _current_complex = mag_pha_stft(noisy, 400, 100, 400, 0.3)
            official_mag, official_phase, _official_complex = official_stft(noisy, 400, 100, 400, 0.3)
            with torch.no_grad():
                from_official = official(official_mag, official_phase)[0]
                from_current = current(current_mag, current_phase)[0]
            cross_1600 = float((from_official - from_current).abs().max())
            full = torch.randn(1, 1650)
            trimmed = full[:, : align_validation_length(1650, 100, 400)]
            full_mag, full_phase, _full_complex = mag_pha_stft(full, 400, 100, 400, 0.3)
            trim_mag, trim_phase, _trim_complex = mag_pha_stft(trimmed, 400, 100, 400, 0.3)
            official_full_mag, official_full_phase, _official_full_complex = official_stft(full, 400, 100, 400, 0.3)
            with torch.no_grad():
                full_out = current(full_mag, full_phase)[0]
                trim_out = current(trim_mag, trim_phase)[0]
                official_full_out = official(official_full_mag, official_full_phase)[0]
            cross_1650 = float((official_full_out - full_out).abs().max())
            compared = full_out[..., : trim_out.shape[-1]]
            truncated_vs_full = float((compared - trim_out).abs().max())
        finally:
            torch.set_num_threads(threads)
            set_determinism(deterministic, cuda=False)
        print(
            f"official_vs_audiodb_1600 {cross_1600!r} "
            f"official_vs_audiodb_1650 {cross_1650!r} "
            f"truncated_vs_full_mag_max_abs {truncated_vs_full!r}"
        )
        self.assertEqual(tuple(full_mag.shape), tuple(trim_mag.shape))
        self.assertEqual(tuple(full_mag.shape), (1, 201, 17))
        self.assertEqual(cross_1600, 0.0)
        self.assertEqual(cross_1650, 0.0)
        # Remeasured value is 1.5627447366714478. The audit printed {:.3e}, which is 1.563e+00.
        self.assertGreater(truncated_vs_full, 1.0)
        self.assertEqual(int(clean.shape[-1]), 1600)


def _load_official_generator():
    if not (OFFICIAL_ROOT / "dataset.py").is_file() or not (OFFICIAL_ROOT / "models/model.py").is_file():
        raise unittest.SkipTest("official MP-SENet tree is not present")
    inserted = str(OFFICIAL_ROOT)
    sys.path.insert(0, inserted)
    stubbed = False
    try:
        from dataset import mag_pha_stft as official_stft

        if "matplotlib" not in sys.modules:
            matplotlib_stub = types.ModuleType("matplotlib")
            matplotlib_stub.use = lambda *args, **kwargs: None
            sys.modules["matplotlib"] = matplotlib_stub
            sys.modules["matplotlib.pylab"] = types.ModuleType("matplotlib.pylab")
            stubbed = True
        from models.model import MPNet as official_net
    except Exception as exc:
        raise unittest.SkipTest(f"official MP-SENet could not be imported: {exc}") from exc
    finally:
        while inserted in sys.path:
            sys.path.remove(inserted)
        if stubbed:
            sys.modules.pop("matplotlib.pylab", None)
            sys.modules.pop("matplotlib", None)
    return official_stft, official_net


if __name__ == "__main__":
    unittest.main()
