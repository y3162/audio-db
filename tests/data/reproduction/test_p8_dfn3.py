"""CPU checks for DeepFilterNet3 online mixing.

Distribution windows below were fixed before the first run. Do not edit them
to make a failing seed pass.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.experiments.data.ids import DFN3_ONLINE_V1
from src.experiments.data.mix.online_dfn3 import plan_online_epoch, schedule_update_count
from src.experiments.data.fixed_pair_stream import open_training_data
from src.experiments.data.protocols.online_dfn3 import (
    NUM_SAMPLES,
    SAMPLE_RATE,
    DFN3OnlineSpec,
    _combine_noises,
    _gate,
    mix_signals,
    render_mixture,
    reverb_pair,
    sample_seed,
    synthesize_noise,
)
from src.experiments.calc.batch_schedule import scheduled_batch_size
from tests.data.reproduction.dfn3_distribution import render_distribution, slices
from tests.suite_resources import distribution_workers

DISTRIBUTION_SEED = 20261106
DISTRIBUTION_N = 4000
# 6-sigma style windows around the shipped discrete distributions. n = 4000.
NOISE_COUNT_WINDOW = (800, 1200)  # uniform {2,3,4,5}, expected 1000
SNR_WINDOW = (421, 721)  # 7 bins, expected 4000/7
GAIN_WINDOW = (1133, 1533)  # 3 bins, expected 4000/3
REVERB_WINDOW = (280, 520)  # Bernoulli 0.1, expected 400
SYNTHETIC_RATE_WINDOW = (0.038, 0.062)  # per noise piece, expected 0.05


class _Clip:
    def __init__(self, data, rate: int = SAMPLE_RATE) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.sample_rate = int(rate)
        self.frames = int(self.data.shape[0])

    def read(self, start: int, stop: int) -> np.ndarray:
        return self.data[int(start) : int(stop)]


def _quiet_spec(**overrides) -> DFN3OnlineSpec:
    base = dict(
        p_reverb=0.0,
        p_remove_dc=0.0,
        p_lfilt=0.0,
        p_resample=0.0,
        p_synthetic_noise=0.0,
        p_noise_clip=0.0,
    )
    base.update(overrides)
    return DFN3OnlineSpec(**base)


class MixerTests(unittest.TestCase):
    def test_gate_zero_draws_nothing(self):
        left = np.random.Generator(np.random.PCG64(7))
        right = np.random.Generator(np.random.PCG64(7))
        self.assertFalse(_gate(left, 0.0))
        self.assertEqual(float(left.random()), float(right.random()))
        self.assertTrue(_gate(np.random.Generator(np.random.PCG64(1)), 1.0))

    def test_combine_does_not_divide_by_count(self):
        ones = np.ones(8, dtype=np.float64)
        mixed = _combine_noises(
            [ones, ones],
            [0, 0],
            8,
            np.random.Generator(np.random.PCG64(1)),
        )
        self.assertTrue(np.allclose(mixed, 2.0))

    def test_speech_is_concatenated_not_zero_padded(self):
        spec = _quiet_spec(snrs=(0,), gains=(0,), noise_count_low=2, noise_count_high_exclusive=3)
        speech = [_Clip(np.full(1000, 0.1)), _Clip(np.full(1000, 0.2))]
        noise = [_Clip(np.full(NUM_SAMPLES, 0.01))]
        mixture = render_mixture(
            spec,
            speech,
            noise,
            [],
            speech_index=0,
            split="train",
            rng=np.random.Generator(np.random.PCG64(3)),
        )
        self.assertEqual(mixture.noisy.shape, (NUM_SAMPLES,))
        self.assertEqual(mixture.sample_rate, SAMPLE_RATE)
        values = {round(float(item), 5) for item in np.unique(mixture.clean)}
        self.assertEqual(values, {0.1, 0.2})

    def test_snr_clip_guard_and_noise_only(self):
        clean = np.full(1000, 0.5)
        distorted = clean.copy()
        noise = np.full(1000, 0.25)
        out_clean, out_noise, mixture = mix_signals(clean, distorted, noise, 5.0, 0.0)
        measured = 10.0 * np.log10((np.sum(out_clean**2) + 1e-10) / (np.sum(out_noise**2) + 1e-10))
        self.assertLess(abs(measured - 5.0), 0.05)
        self.assertLessEqual(max(np.max(np.abs(out_clean)), np.max(np.abs(out_noise)), np.max(np.abs(mixture))), 1.0)
        loud_clean, loud_noise, loud_mix = mix_signals(np.full(100, 4.0), np.full(100, 4.0), np.full(100, 3.0), 0.0, 6.0)
        peak = max(np.max(np.abs(loud_clean)), np.max(np.abs(loud_noise)), np.max(np.abs(loud_mix)))
        self.assertLessEqual(peak, 1.0)
        spec = _quiet_spec(snrs=(-100,), gains=(0,), noise_count_low=2, noise_count_high_exclusive=3)
        mixture = render_mixture(
            spec,
            [_Clip(np.full(NUM_SAMPLES, 0.2))],
            [_Clip(np.full(NUM_SAMPLES, 0.05))],
            [],
            speech_index=0,
            split="valid",
            rng=np.random.Generator(np.random.PCG64(9)),
        )
        self.assertTrue(mixture.noise_only)
        self.assertIsNone(mixture.measured_snr_db)
        self.assertTrue(np.all(mixture.clean == 0))
        self.assertEqual(mixture.n_noises, 2)

    def test_missing_rir_is_an_error(self):
        spec = DFN3OnlineSpec.shipped()
        with self.assertRaises(Exception):
            render_mixture(
                spec,
                [_Clip(np.full(NUM_SAMPLES, 0.1))],
                [_Clip(np.full(NUM_SAMPLES, 0.1))],
                [],
                speech_index=0,
                split="train",
                rng=np.random.Generator(np.random.PCG64(1)),
            )

    def test_reverb_teacher_matches_independent_formula(self):
        dry = np.linspace(-0.2, 0.2, 64)
        noise = np.full(64, 0.05)
        impulse = np.array([0.0, 0.1, 1.0, 0.4, 0.2, 0.05], dtype=np.float64)
        target, reverberant, _ = reverb_pair(
            dry,
            noise,
            impulse,
            SAMPLE_RATE,
            drr=0.3,
            rt60=0.5,
            offset_samples=4,
            apply_speech=True,
            apply_noise=False,
            decay_rt60=None,
        )
        expected_target, expected_reverb = _independent_reverb(dry, impulse, drr=0.3, rt60=0.5, offset_samples=4)
        self.assertTrue(np.allclose(target, expected_target))
        self.assertTrue(np.allclose(reverberant, expected_reverb))
        negative = np.array([-1.0, 0.2, 0.05], dtype=np.float64)
        finite, _, _ = reverb_pair(
            dry,
            noise,
            negative,
            SAMPLE_RATE,
            drr=0.3,
            rt60=0.5,
            offset_samples=2,
            apply_speech=True,
            apply_noise=True,
        )
        self.assertTrue(np.isfinite(finite).all())

    def test_synthetic_decay_tilts_the_spectrum(self):
        red = synthesize_noise(np.random.Generator(np.random.PCG64(4)), 2.0, SAMPLE_RATE, SAMPLE_RATE)
        blue = synthesize_noise(np.random.Generator(np.random.PCG64(4)), -2.0, SAMPLE_RATE, SAMPLE_RATE)
        self.assertLess(_high_low(red), _high_low(blue))

    def test_same_seed_repeats_and_epochs_follow_the_split_rule(self):
        spec = _quiet_spec(snrs=(0, 5), gains=(0,))
        speech = [_Clip(np.linspace(-0.2, 0.2, NUM_SAMPLES + 10))]
        noise = [_Clip(np.linspace(-0.05, 0.05, NUM_SAMPLES + 10))]

        def once(epoch: int, split: str) -> np.ndarray:
            rng = np.random.Generator(np.random.PCG64(sample_seed("proto", "snap", 43, split, epoch, 2)))
            return render_mixture(spec, speech, noise, [], speech_index=0, split=split, rng=rng).noisy

        self.assertTrue(np.array_equal(once(0, "train"), once(0, "train")))
        self.assertFalse(np.array_equal(once(0, "train"), once(1, "train")))
        self.assertTrue(np.array_equal(once(0, "valid"), once(4, "valid")))
        self.assertEqual(sample_seed("p", "s", 1, "valid", 5, 3), sample_seed("p", "s", 1, "valid", 9, 3))
        self.assertNotEqual(sample_seed("p", "s", 1, "valid", 0, 3), sample_seed("p", "s", 1, "test", 0, 3))
        self.assertNotEqual(sample_seed("p", "s", 1, "train", 0, 3), sample_seed("p", "s", 1, "train", 1, 3))

    def test_plan_keeps_remainder_and_pads_ranks(self):
        count = 10
        micro = 4
        for world in (1, 2, 4):
            first = _plan(count, micro, world, epoch=0)
            second = _plan(count, micro, world, epoch=0)
            self.assertEqual(first.plan_hash, second.plan_hash)
            self.assertNotEqual(first.plan_hash, _plan(count, micro, world, epoch=1).plan_hash)
            lengths = [len(_indices(first, rank)) for rank in range(world)]
            self.assertEqual(len(set(lengths)), 1)
            self.assertEqual(first.updates, schedule_update_count(lengths[0], micro))
            self.assertTrue(all(len(group) == first.updates for group in first.rank_groups))
            flat = [index for rank in range(world) for index in _indices(first, rank)]
            self.assertEqual(set(flat), set(range(count)))
            self.assertEqual(len(flat) - count, int(first.stats["padded_samples"]))
            self.assertEqual(first.stats["omitted_samples"], 0)
        single = _plan(count, micro, 1, epoch=0)
        self.assertEqual(single.updates, 3)
        self.assertEqual(len(_indices(single, 0)[-2:]), 2)
        self.assertEqual(schedule_update_count(count, micro), 3)

    def test_shipped_batch_schedule_caps_at_64(self):
        schedule = [tuple(item) for item in DFN3_ONLINE_V1["batch_schedule"]]
        self.assertEqual(scheduled_batch_size(0, 64, schedule), 16)
        self.assertEqual(scheduled_batch_size(2, 64, schedule), 24)
        self.assertEqual(scheduled_batch_size(5, 64, schedule), 32)
        self.assertEqual(scheduled_batch_size(10, 64, schedule), 64)
        self.assertEqual(scheduled_batch_size(20, 64, schedule), 64)
        self.assertEqual(scheduled_batch_size(40, 64, schedule), 64)


class SnapshotTests(unittest.TestCase):
    def test_smoke_snapshot_is_repeatable_and_official_stays_blocked(self):
        import duckdb

        connection = duckdb.connect("data/db/reproduction-p8.duckdb", read_only=True)
        try:
            status = connection.execute(
                "select declared_status, claim_label from repro.datasets where name = 'dfn3.online.official'"
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(status, ("blocked", "official_reproduction"))
        data = open_training_data(
            dataset="dfn3.online.smoke@1",
            allow_test_selection=False,
            seed=43,
            global_batch=2,
            micro_batch=2,
            accumulation=1,
            world_size=1,
            segment_size=NUM_SAMPLES,
            model_sample_rate=SAMPLE_RATE,
        )
        other = open_training_data(
            dataset="dfn3.online.smoke@1",
            allow_test_selection=False,
            seed=43,
            global_batch=4,
            micro_batch=2,
            accumulation=1,
            world_size=2,
            segment_size=NUM_SAMPLES,
            model_sample_rate=SAMPLE_RATE,
        )
        try:
            self.assertEqual(data.store.dataset["rule_json"]["online"], DFN3_ONLINE_V1)
            self.assertNotIn("global_batch", data.identity())
            first = data.render_index("train", 0, 0)
            again = other.render_index("train", 0, 0)
            self.assertTrue(np.array_equal(first.noisy, again.noisy))
            self.assertEqual(first.noisy.shape, (NUM_SAMPLES,))
            self.assertEqual(first.sample_rate, SAMPLE_RATE)
            self.assertTrue(np.isfinite(first.noisy).all() and np.isfinite(first.clean).all())
            self.assertLessEqual(float(np.max(np.abs(first.noisy))), 1.0 + 1e-4)
            self.assertLessEqual(float(np.max(np.abs(first.clean))), 1.0 + 1e-4)
            valid_early = data.render_index("valid", 0, 0)
            valid_late = data.render_index("valid", 3, 0)
            self.assertTrue(np.array_equal(valid_early.noisy, valid_late.noisy))
            self.assertFalse(np.array_equal(first.noisy, data.render_index("train", 1, 0).noisy))
        finally:
            data.close()
            other.close()


class TrainerSmokeTests(unittest.TestCase):
    def test_one_deepfilternet_step_is_finite(self):
        from src.experiments.calc.rng import set_determinism
        from src.experiments.official.deepfilternet3.reference import DeepFilterNetTask
        from tests.numerics.support import accelerator_device, prepare_accelerator_numerics
        from tests.numerics.test_model_tasks import _base

        data = _smoke(1, 1)
        try:
            rendered = data.render_index("train", 0, 1)
        finally:
            data.close()
        width = 8192
        clean = torch.from_numpy(rendered.clean[:width]).view(1, width)
        noisy = torch.from_numpy(rendered.noisy[:width]).view(1, width)
        with tempfile.TemporaryDirectory() as tmp:
            config = _base(
                Path(tmp),
                "deepfilternet3",
                grad_clip=1.0,
                scheduler={"name": "exponential", "gamma": 0.99},
            )
            device = accelerator_device()
            previous = torch.are_deterministic_algorithms_enabled()
            prepare_accelerator_numerics(device)
            try:
                task = DeepFilterNetTask(config, device)
                metrics = task.train_update([{"clean": clean, "noisy": noisy, "case_id": ["x"]}], 1)
            finally:
                set_determinism(previous, cuda=False)
        for value in metrics.values():
            self.assertTrue(np.isfinite(value))


class DistributionTests(unittest.TestCase):
    def test_distribution_slices_cover_each_index_once(self):
        covered = [index for start, stop in slices(DISTRIBUTION_N, 32) for index in range(start, stop)]
        self.assertEqual(covered, list(range(DISTRIBUTION_N)))
        self.assertEqual(slices(0, 4), [])

    def test_shipped_draw_windows(self):
        # Seeds, windows, and per-draw checks match the previous in-process loop.
        # Workers only split the index range.
        summary = render_distribution(
            DISTRIBUTION_N,
            DISTRIBUTION_SEED,
            workers=distribution_workers(),
        )
        self.assertEqual(summary["failures"], [])
        for count in summary["noises"].values():
            self.assertTrue(NOISE_COUNT_WINDOW[0] <= count <= NOISE_COUNT_WINDOW[1], summary["noises"])
        for count in summary["snr"].values():
            self.assertTrue(SNR_WINDOW[0] <= count <= SNR_WINDOW[1], summary["snr"])
        for count in summary["gain"].values():
            self.assertTrue(GAIN_WINDOW[0] <= count <= GAIN_WINDOW[1], summary["gain"])
        self.assertTrue(REVERB_WINDOW[0] <= summary["reverb"] <= REVERB_WINDOW[1], summary["reverb"])
        rate = summary["synthetic"] / summary["noise_total"]
        self.assertTrue(SYNTHETIC_RATE_WINDOW[0] <= rate <= SYNTHETIC_RATE_WINDOW[1], rate)
        self.assertGreater(summary["snr_error_count"], 0)
        self.assertLess(summary["snr_error_max"], 0.05)


def _independent_reverb(dry, impulse, *, drr: float, rt60: float, offset_samples: int):
    import scipy.signal

    dry = np.asarray(dry, dtype=np.float64)
    impulse = np.asarray(impulse, dtype=np.float64)
    impulse = impulse / np.sqrt(np.sum(impulse**2))
    full = scipy.signal.fftconvolve(dry, impulse, mode="full")[: dry.size]
    max_idx = int(np.argmax(np.abs(impulse)))
    late = max_idx + int(offset_samples)
    little = impulse.copy()
    if late < little.size:
        level = 10.0 ** (-60.0 / 20.0)
        tau = -float(rt60) / np.log10(level)
        steps = np.arange(little.size - late, dtype=np.float64)
        little[late:] *= 10.0 ** (-steps / SAMPLE_RATE / tau)
    little = little / np.sqrt(np.sum(little**2))
    little_wave = scipy.signal.fftconvolve(dry, little, mode="full")[: dry.size]
    target = float(drr) * dry + (1.0 - float(drr)) * little_wave
    target = target * (np.sqrt(np.mean(dry**2)) / (np.sqrt(np.mean(target**2)) + 1e-10))
    return target, full


def _high_low(wave: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(wave))
    mid = spec.shape[0] // 2
    return float(np.mean(spec[mid:]) / (np.mean(spec[1:mid]) + 1e-12))


def _plan(count: int, micro: int, world: int, epoch: int):
    return plan_online_epoch(
        split="train",
        epoch=epoch,
        count=count,
        seed=43,
        world_size=world,
        micro_batch=micro,
        protocol_id="proto",
        snapshot_hash="snap",
    )


def _indices(plan, rank: int) -> list[int]:
    return [
        sample.index
        for group in plan.rank_groups[rank]
        for micro in group.micro_batches
        for sample in micro.samples
    ]


def _smoke(global_batch: int, world_size: int):
    return open_training_data(
        dataset="dfn3.online.smoke@1",
        allow_test_selection=False,
        seed=43,
        global_batch=global_batch,
        micro_batch=global_batch // world_size,
        accumulation=1,
        world_size=world_size,
        segment_size=NUM_SAMPLES,
        model_sample_rate=SAMPLE_RATE,
    )


def _trainer_config() -> dict:
    return {
        "seed": 43,
        "loss": "official",
        "model": {"name": "deepfilternet3"},
        "batch_schedule": [[0, 2], [1, 4]],
        "batch_size_limit": 4,
        "micro_batch_size_per_gpu": 4,
        "logical_steps_per_epoch": None,
        "data": {"load_workers": 1, "prefetch": False},
        "best_checkpoint_start_epoch": 40,
    }


class _Stub:
    def __init__(self) -> None:
        self.steps = 0
        self.validated = 0
        self.positions = []

    def train_update(self, batches, accumulation):
        del accumulation
        for batch in batches:
            if batch["noisy"].shape[-1] != NUM_SAMPLES or not torch.isfinite(batch["noisy"]).all():
                raise AssertionError("online batch is not a finite 3 second window")
        self.steps += 1
        return {"generator": 0.25}

    def validation_report(self, batch):
        self.validated += 1
        if batch["clean"].shape != batch["noisy"].shape:
            raise AssertionError("validation batch shapes differ")
        metric = {"status": "ok", "value": 0.5, "reason": None, "count": 1}
        empty = {"status": "na", "value": None, "reason": "not computed", "count": 0}
        return {"losses": {"generator": metric}, "pesq": empty, "stoi": empty}

    def retarget_deepfilter_schedule(self, steps_per_epoch: int) -> None:
        self.steps_per_epoch = int(steps_per_epoch)

    def note_schedule_position(self, index: int) -> None:
        self.positions.append(int(index))

    def step_schedulers(self, validation) -> None:
        del validation

    def learning_rates(self):
        return {"generator": 0.001}

    def distribute(self, process) -> None:
        del process

    def train_mode(self) -> None:
        return None

    def eval_mode(self) -> None:
        return None

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state) -> None:
        self.steps = int(state["steps"])

    def export_for_inference(self):
        return None


if __name__ == "__main__":
    unittest.main()
