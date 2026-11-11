"""CPU checks for SEMamba++ online degradation.

Distribution windows below were fixed before the first counted run. Do not
edit them to make a failing seed pass. Kaiser resampling uses resampy from
the existing asr environment when the se environment does not install it.

The datasets are registered into a directory of this test module. The
production database, staging, and snapshots are read by other tests and are
not written here. ``make reproduction-register-p9`` writes the production copy.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

_ASR_PACKAGES = Path(__file__).resolve().parents[3] / ".venvs/asr/lib/python3.12/site-packages"
if _ASR_PACKAGES.is_dir():
    sys.path.append(str(_ASR_PACKAGES))

from src.experiments.data.ids import SEMAMBA_GSR_V1
from src.experiments.data.mix.online_semamba import plan_semamba_epoch
from src.experiments.data.sampling.order import rank_order
from src.experiments.data.fixed_pair_stream import open_training_data
from src.experiments.data.prepare.semamba_online import register_p9
from src.experiments.data.protocols.online_semamba import (
    SAMPLE_RATE,
    SEGMENT_SIZE,
    add_reverberation,
    bandwidth_limitation,
    clipping,
    detect_non_silence,
    draw_once,
    draw_used,
    independent_peak,
    mix_noise,
    peak_and_crop,
    render_example,
    vad_power,
)
from src.se.models.semambapp.network import SelectiveSSM

DISTRIBUTION_SEED = 20261107
DISTRIBUTION_N = 4000
# 4-sigma windows. n = 4000. Expected values are exact for the HEAD config.
REVERB_WINDOW = (1872, 2128)  # Bernoulli 0.5, expected 2000
CLIP_WINDOW = (1872, 2128)
BANDWIDTH_WINDOW = (1872, 2128)
SNR_WINDOW = (572, 761)  # 6 bins, expected 4000/6
COMBO_WINDOW = (416, 584)  # 8 combinations, expected 500
RENDER_N = 16
RENDER_LENGTH = 24000

_REGISTERED = None
_WORK: tempfile.TemporaryDirectory | None = None
STAGING = DATABASE = SNAPSHOTS = Path()


def setUpModule() -> None:
    global _WORK, STAGING, DATABASE, SNAPSHOTS
    _WORK = tempfile.TemporaryDirectory(prefix="audio-db-p9-")
    root = Path(_WORK.name)
    STAGING = root / "staging"
    DATABASE = root / "db" / "reproduction-p9.duckdb"
    SNAPSHOTS = root / "snapshots"
    DATABASE.parent.mkdir()


def tearDownModule() -> None:
    global _REGISTERED
    _REGISTERED = None
    if _WORK is not None:
        _WORK.cleanup()


def _enable_official_modules():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/se/official/semambapp"))
    sys.modules.setdefault("pyroomacoustics", types.ModuleType("pyroomacoustics"))
    if "torchaudio.io" not in sys.modules:
        try:
            import torchaudio.io  # noqa: F401
        except Exception:
            package = types.ModuleType("torchaudio")
            io = types.ModuleType("torchaudio.io")
            io.AudioEffector = type("AudioEffector", (), {})
            io.CodecConfig = type("CodecConfig", (), {})
            sys.modules["torchaudio"] = package
            sys.modules["torchaudio.io"] = io
    import simulate_degradation_utils as official

    return official


def _registered():
    global _REGISTERED
    if _REGISTERED is None:
        _REGISTERED = _register()
    return _REGISTERED


def _register():
    return register_p9(staging=STAGING, database=DATABASE, snapshots=SNAPSHOTS)


def _open(global_batch: int, world_size: int, *, workers: int = 1, accumulation: int = 1):
    micro = global_batch // (world_size * accumulation)
    return open_training_data(
        dataset="semambapp.gsr.smoke@1",
        allow_test_selection=False,
        seed=7,
        global_batch=global_batch,
        micro_batch=micro,
        accumulation=accumulation,
        world_size=world_size,
        segment_size=SEGMENT_SIZE,
        model_sample_rate=SAMPLE_RATE,
        load_workers=workers,
        snapshot_root=SNAPSHOTS,
    )


class _ArrayClip:
    def __init__(self, data) -> None:
        self.data = np.asarray(data, dtype=np.float64)

    def read(self):
        return self.data.copy()


class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = _enable_official_modules()

    def test_vad_masks_match_official(self):
        source = np.random.default_rng(0)
        speech = source.normal(0.0, 0.1, (1, 8000))
        speech[0, :3000] = 0.0
        short = np.concatenate([np.zeros(100), np.full(200, 0.5), np.zeros(723)])[None, :]
        for name, signal in (("1023", short), ("1024", speech[:, :1024]), ("8000", speech)):
            left = self.official.detect_non_silence(signal.copy())
            right = detect_non_silence(signal.copy())
            self.assertTrue(np.array_equal(left, right), name)
            self.assertEqual(right.shape, signal.shape)

    def test_noise_lengths_and_vad_snr_match_official(self):
        source = np.random.default_rng(1)
        speech = source.normal(0.0, 0.1, (1, 8000))
        speech[0, :3000] = 0.0
        noises = {
            "short": source.normal(0.0, 0.2, (1, 1000)),
            "long": source.normal(0.0, 0.2, (1, 12000)),
            "equal": source.normal(0.0, 0.2, (1, 8000)),
            "zero_speech": None,
        }
        errors = []
        for name, noise in noises.items():
            if name == "zero_speech":
                clean = np.zeros((1, 8000))
                noise = source.normal(0.0, 0.2, (1, 8000))
            else:
                clean = speech
            left_rng = np.random.default_rng(7)
            right_rng = np.random.default_rng(7)
            official_noisy, official_scaled = self.official.mix_noise(clean.copy(), noise.copy(), 5.0, left_rng)
            noisy, scaled, measured = mix_noise(clean.copy(), noise.copy(), 5.0, right_rng)
            error = float(np.max(np.abs(noisy - official_noisy)))
            errors.append(error)
            self.assertEqual(noisy.shape, clean.shape)
            self.assertLess(error, 1e-12, name)
            self.assertLess(float(np.max(np.abs(scaled - official_scaled))), 1e-12, name)
            if name == "zero_speech":
                self.assertIsNone(measured)
                self.assertEqual(float(np.max(np.abs(noisy))), 0.0)
            else:
                self.assertLess(abs(measured - 5.0), 1e-6, name)
        self.assertLess(max(errors), 1e-12)

    def test_vad_scale_is_not_full_waveform_rms(self):
        speech = np.zeros((1, 8000))
        speech[0, 2000:4000] = 0.2
        noise = np.ones((1, 8000))
        rms_speech = float(np.mean(speech**2))
        vad_speech = vad_power(speech)
        self.assertGreater(abs(rms_speech - vad_speech) / vad_speech, 0.2)
        rms_scale = np.sqrt(rms_speech) / np.sqrt(float(np.mean(noise**2)))
        vad_scale = np.sqrt(vad_speech) / np.sqrt(vad_power(noise))
        self.assertGreater(abs(float(rms_scale - vad_scale)), 1e-3)
        noisy, scaled, measured = mix_noise(speech, noise, 0.0, np.random.default_rng(0))
        self.assertLess(abs(measured), 1e-6)
        self.assertLess(float(np.max(np.abs(scaled - vad_scale * noise))), 1e-6)
        self.assertGreater(float(np.max(np.abs(scaled - rms_scale * noise))), 1e-3)
        del noisy

    def test_reverb_keeps_input_length_without_rir_normalization(self):
        speech = np.random.default_rng(2).normal(0.0, 0.1, (1, 3000))
        impulse = np.zeros((1, 500))
        impulse[0, 10] = 2.0
        ours = add_reverberation(speech, impulse)
        official = self.official.add_reverberation(speech, impulse)
        self.assertEqual(ours.shape, speech.shape)
        self.assertLess(float(np.max(np.abs(ours - official))), 1e-12)
        self.assertNotEqual(float(np.max(np.abs(impulse))), 1.0)

    def test_clipping_uses_quantiles(self):
        signal = np.random.default_rng(3).normal(0.0, 1.5, (1, 5000))
        ours = clipping(signal, 0.05, 0.9)
        official = self.official.clipping(signal, 0.05, 0.9)
        self.assertLess(float(np.max(np.abs(ours - official))), 1e-12)
        self.assertGreater(float(np.max(np.abs(ours - np.clip(signal, -1.0, 1.0)))), 0.5)

    def test_bandwidth_matches_official_filter_and_resample(self):
        speech = np.random.default_rng(4).normal(0.0, 0.1, (1, 8000))
        cases = (
            (8000, "scipy", "butter", 2),
            (4000, "scipy", "cheby2", 4),
            (6000, "polyphase", "ellip", 2),
            (8000, "kaiser_fast", "bessel", 2),
        )
        for fs_new, res_type, lowpass, order in cases:
            official = self.official.bandwidth_limitation(speech, 16000, fs_new, res_type, lowpass, order)
            ours, padded = bandwidth_limitation(speech, 16000, fs_new, res_type, lowpass, order)
            self.assertEqual(ours.shape[-1], speech.shape[-1])
            if official.shape[-1] == speech.shape[-1]:
                self.assertFalse(padded)
                error = float(np.max(np.abs(ours - official)))
            else:
                self.assertTrue(padded)
                error = float(np.max(np.abs(ours[:, : official.shape[-1]] - official)))
            self.assertLess(error, 1e-5, (res_type, lowpass, order, padded))

    def test_short_resample_is_zero_padded(self):
        import librosa

        speech = np.ones((1, 32))
        original = librosa.resample

        def short(audio, *, orig_sr, target_sr, res_type):
            del orig_sr, target_sr, res_type
            return np.asarray(audio)[..., : audio.shape[-1] // 2]

        def long(audio, *, orig_sr, target_sr, res_type):
            del orig_sr, target_sr, res_type
            return np.pad(np.asarray(audio), [(0, 0), (0, 3)])

        try:
            librosa.resample = short
            padded_audio, padded = bandwidth_limitation(speech, 16000, 8000, "scipy", "butter", 2)
            self.assertTrue(padded)
            self.assertEqual(padded_audio.shape, speech.shape)
            self.assertEqual(float(padded_audio[0, -1]), 0.0)
            librosa.resample = long
            trimmed, was_padded = bandwidth_limitation(speech, 16000, 8000, "scipy", "butter", 2)
            self.assertFalse(was_padded)
            self.assertEqual(trimmed.shape, speech.shape)
        finally:
            librosa.resample = original

    def test_discarded_draw_is_the_second_sample(self):
        differed = False
        for seed in range(30):
            first = np.random.Generator(np.random.PCG64(seed))
            second = np.random.Generator(np.random.PCG64(seed))
            used, discarded = draw_used(first)
            self.assertEqual(discarded, draw_once(second))
            self.assertEqual(used, draw_once(second))
            self.assertNotIn("packet_loss", used.selected)
            self.assertEqual(list(used.selected), sorted(used.selected, key=lambda name: {"reverb": 1, "noise": 2, "clipping": 3, "bandwidth": 5}[name]))
            differed = differed or used.selected != discarded.selected
        self.assertTrue(differed)

    def test_independent_peak_rejects_the_wrong_normalizations(self):
        length = 10000
        clean = np.zeros(length)
        degraded = np.zeros(length)
        clean[0] = 4.0
        clean[100:1100] = 0.5
        degraded[-1] = 8.0
        degraded[100:1100] = 0.25

        class _Start:
            def integers(self, low, high):
                self.args = (low, high)
                return 100

        rng = _Start()
        cropped_clean, cropped_degraded, start, padded, clean_peak, degraded_peak = peak_and_crop(
            clean, degraded, rng, 1000
        )
        self.assertEqual((start, padded), (100, False))
        self.assertEqual(rng.args, (0, length - 1000 + 1))
        self.assertAlmostEqual(clean_peak, 4.0)
        self.assertAlmostEqual(degraded_peak, 8.0)
        clean_full, _ = independent_peak(clean)
        degraded_full, _ = independent_peak(degraded)
        self.assertTrue(np.array_equal(cropped_clean, clean_full[100:1100]))
        self.assertTrue(np.array_equal(cropped_degraded, degraded_full[100:1100]))
        self.assertLess(float(np.max(np.abs(cropped_clean))), 0.2)
        shared = (clean[100:1100] / np.float32(8.0 + 1e-9)).astype(np.float32)
        self.assertGreater(float(np.max(np.abs(cropped_clean - shared))), 1e-3)
        after_crop = independent_peak(clean[100:1100].astype(np.float32))[0]
        self.assertGreater(float(np.max(np.abs(cropped_clean - after_crop))), 0.5)
        speech = np.zeros((1, 8000))
        speech[0, 2000:4000] = 0.2
        noise = np.ones((1, 8000))
        noisy, _, measured = mix_noise(speech, noise, 0.0, np.random.default_rng(5))
        clean_scaled, _ = independent_peak(speech)
        noisy_scaled, _ = independent_peak(noisy)
        post = 10.0 * np.log10(vad_power(clean_scaled) / vad_power(noisy_scaled))
        self.assertLess(abs(measured), 1e-6)
        self.assertGreater(abs(post), 1.0)


class DistributionTests(unittest.TestCase):
    def test_precommitted_draw_windows(self):
        snr_counts = {value: 0 for value in SEMAMBA_GSR_V1["snr"]}
        reverb = clipping_count = bandwidth = 0
        combos = {}
        packet = 0
        for index in range(DISTRIBUTION_N):
            draw = draw_once(np.random.Generator(np.random.PCG64(DISTRIBUTION_SEED + index)))
            snr_counts[draw.snr_db] += 1
            reverb += int("reverb" in draw.selected)
            clipping_count += int("clipping" in draw.selected)
            bandwidth += int("bandwidth" in draw.selected)
            packet += int("packet_loss" in draw.selected)
            self.assertIn("noise", draw.selected)
            self.assertGreaterEqual(draw.clipping_min, 0.0)
            self.assertLess(draw.clipping_min, 0.1)
            self.assertGreaterEqual(draw.clipping_max, 0.9)
            self.assertLess(draw.clipping_max, 1.0)
            combos[draw.selected] = combos.get(draw.selected, 0) + 1
        self.assertEqual(packet, 0)
        self.assertEqual(sum(snr_counts.values()), DISTRIBUTION_N)
        for count in snr_counts.values():
            self.assertTrue(SNR_WINDOW[0] <= count <= SNR_WINDOW[1], snr_counts)
        self.assertTrue(REVERB_WINDOW[0] <= reverb <= REVERB_WINDOW[1], reverb)
        self.assertTrue(CLIP_WINDOW[0] <= clipping_count <= CLIP_WINDOW[1], clipping_count)
        self.assertTrue(BANDWIDTH_WINDOW[0] <= bandwidth <= BANDWIDTH_WINDOW[1], bandwidth)
        self.assertEqual(len(combos), 8, combos)
        for count in combos.values():
            self.assertTrue(COMBO_WINDOW[0] <= count <= COMBO_WINDOW[1], combos)

    def test_rendered_length_finite_and_snr(self):
        source = np.random.default_rng(DISTRIBUTION_SEED)
        speech = [_ArrayClip(source.normal(0.0, 0.1, RENDER_LENGTH)) for _ in range(4)]
        noise = [_ArrayClip(source.normal(0.0, 0.2, RENDER_LENGTH // 2)) for _ in range(3)]
        impulse = []
        for _ in range(2):
            rir = source.normal(0.0, 0.01, 300)
            rir[0] = 1.0
            impulse.append(_ArrayClip(rir))
        errors = []
        padded = 0
        for index in range(RENDER_N):
            example = render_example(
                speech,
                noise,
                impulse,
                speech_index=index % 4,
                split="train",
                rng=np.random.Generator(np.random.PCG64(DISTRIBUTION_SEED + 1000 + index)),
            )
            self.assertEqual(example.clean.shape, (SEGMENT_SIZE,))
            self.assertEqual(example.degraded.shape, (SEGMENT_SIZE,))
            self.assertEqual(example.sample_rate, SAMPLE_RATE)
            self.assertTrue(np.isfinite(example.clean).all())
            self.assertTrue(np.isfinite(example.degraded).all())
            self.assertLessEqual(float(np.max(np.abs(example.clean))), 1.0)
            self.assertLessEqual(float(np.max(np.abs(example.degraded))), 1.0)
            self.assertTrue(example.padded)
            padded += int(example.bandwidth_length_padded)
            if example.vad_snr_db is not None:
                errors.append(abs(example.vad_snr_db - example.snr_db))
        self.assertTrue(errors)
        self.assertLess(max(errors), 1e-6, max(errors))
        self.assertGreaterEqual(padded, 0)


class PlanTests(unittest.TestCase):
    def test_worlds_overlap_epoch_and_drop_last(self):
        for world in (1, 2, 4):
            plan = _plan(16, micro=2, world=world, epoch=0, accumulation=1)
            again = _plan(16, micro=2, world=world, epoch=0, accumulation=1)
            self.assertEqual(plan.plan_hash, again.plan_hash)
            covered = []
            for rank in range(world):
                indexes = _indexes(plan, rank)
                self.assertEqual(len(indexes), 16 // world)
                covered.extend(indexes)
            if world == 1:
                self.assertEqual(sorted(covered), list(range(16)))
                self.assertEqual(plan.stats["overlap"], [])
                self.assertEqual(plan.stats["missing"], [])
            else:
                self.assertTrue(set(covered) != set(range(16)) or len(covered) != len(set(covered)))
            later = _plan(16, micro=2, world=world, epoch=1, accumulation=1)
            self.assertNotEqual(_indexes(plan, 0), _indexes(later, 0))
        dropped = _plan(10, micro=4, world=1, epoch=0, accumulation=1)
        full = rank_order("per_rank_independent_v1", 10, seed=7, epoch=0, world_size=1, rank=0)
        self.assertEqual(_indexes(dropped, 0), full[:8])
        self.assertEqual(dropped.stats["dropped_samples"], [2])
        known = _plan(8, micro=2, world=2, epoch=0, accumulation=1, seed=1)
        known_indexes = _indexes(known, 0) + _indexes(known, 1)
        self.assertTrue(set(known_indexes) != set(range(8)) or len(known_indexes) != len(set(known_indexes)))
        grouped = _plan(8, micro=2, world=1, epoch=0, accumulation=2)
        self.assertEqual(grouped.updates, 2)
        self.assertEqual(len(grouped.rank_groups[0][0].micro_batches), 2)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered = _registered()

    def test_official_blocked_and_smoke_is_ready(self):
        import duckdb

        connection = duckdb.connect(str(DATABASE), read_only=True)
        try:
            rows = connection.execute(
                """
                select name, declared_status, claim_label, selection_uses_test
                from repro.datasets
                order by name
                """
            ).fetchall()
            leaks = connection.execute(
                """
                select d.name, a.status
                from repro.audits a
                join repro.datasets d on d.dataset_id = a.subject_id
                where a.audit_kind = 'leak_check'
                order by d.name
                """
            ).fetchall()
        finally:
            connection.close()
        found = {row[0]: row for row in rows}
        self.assertEqual(found["semambapp.gsr.official"][1:], ("blocked", "official_reproduction", False))
        self.assertEqual(found["semambapp.gsr.smoke"][1:], ("ready", "protocol_conformant_substitute_materials", False))
        self.assertEqual(found["semambapp.gsr.substitute"][1:], ("ready", "protocol_conformant_substitute_materials", False))
        self.assertTrue(all(status == "PASS" for _, status in leaks), leaks)
        official_id = next(row["dataset_id"] for row in self.registered["datasets"] if row["name"].endswith("official"))
        self.assertFalse((SNAPSHOTS / official_id).exists())

    def test_registration_is_idempotent_and_stores_no_wav(self):
        again = _register()
        self.assertTrue(all(count == 0 for count in again["counts"].values()), again["counts"])
        wavs = list(STAGING.rglob("*.wav")) + list(STAGING.rglob("*.npy"))
        for name, digest_value in self.registered["snapshots"].items():
            self.assertEqual(len(digest_value), 64, name)
            snapshot = SNAPSHOTS / next(
                row["dataset_id"] for row in self.registered["datasets"] if row["name"] == name
            )
            wavs.extend(snapshot.rglob("*.wav"))
            wavs.extend(snapshot.rglob("*.npy"))
        self.assertEqual(wavs, [])

    def test_snapshot_rebuilds_the_same_waveform(self):
        data = _open(2, 1, workers=1)
        other = _open(4, 2, workers=2)
        try:
            self.assertEqual(data.store.dataset["rule_json"]["online"], SEMAMBA_GSR_V1)
            self.assertNotIn("global_batch", data.identity())
            self.assertEqual(data.identity()["official_global_batch"], 8)
            self.assertEqual(data.identity()["normalize"], "independent_peak_v1")
            self.assertIsNotNone(data.batch_warning)
            first = data.render_index("train", 0, 0)
            again = other.render_index("train", 0, 0)
            repeated = data.render_index("train", 0, 0)
            self.assertTrue(np.array_equal(first.noisy, again.noisy))
            self.assertTrue(np.array_equal(first.clean, repeated.clean))
            self.assertIsNot(first.noisy, repeated.noisy)
            self.assertEqual(first.noisy.shape, (SEGMENT_SIZE,))
            self.assertEqual(first.clean.dtype, np.float32)
            self.assertTrue(np.isfinite(first.noisy).all() and np.isfinite(first.clean).all())
            self.assertLessEqual(float(np.max(np.abs(first.noisy))), 1.0)
            self.assertLessEqual(float(np.max(np.abs(first.clean))), 1.0)
            self.assertTrue(np.array_equal(data.render_index("valid", 0, 0).noisy, data.render_index("valid", 3, 0).noisy))
            self.assertFalse(np.array_equal(first.noisy, data.render_index("train", 1, 0).noisy))
            batch = data.epoch(0, rank=0, world_size=1).render(0)[0]
            self.assertNotIn("mag", batch)
            self.assertEqual(tuple(batch["clean"].shape), tuple(batch["noisy"].shape))
            self.assertEqual(batch["teacher_kind"], "dry_clean")
            reader = data._clip_list("train", "speech")[0]
            import librosa

            loaded, rate = librosa.load(reader.path, sr=SAMPLE_RATE, mono=True)
            self.assertEqual(rate, SAMPLE_RATE)
            self.assertLess(float(np.max(np.abs(reader.read() - loaded))), 1e-12)
        finally:
            data.close()
            other.close()


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _registered()

    def test_one_reduced_generator_step_is_finite(self):
        from src.experiments.calc.rng import set_determinism
        from src.experiments.official.semambapp.reference import SEMambaTask
        from tests.numerics.support import accelerator_device, prepare_accelerator_numerics
        from tests.numerics.test_model_tasks import _base

        data = _open(2, 1)
        try:
            rendered = data.render_index("train", 0, 0)
        finally:
            data.close()
        clean = torch.from_numpy(rendered.clean).view(1, SEGMENT_SIZE)
        noisy = torch.from_numpy(rendered.noisy).view(1, SEGMENT_SIZE)
        self.assertEqual(clean.shape[-1], 48000)
        with tempfile.TemporaryDirectory() as tmp:
            config = _base(
                Path(tmp),
                "semambapp",
                architecture={
                    "hid_feature": 8,
                    "num_tfmamba": 1,
                    "d_state": 4,
                    "d_conv": 4,
                    "expand": 2,
                    "n_fft": 400,
                    "scan_backend": "reference",
                },
                stft={"hop": 100, "win": 400},
                sampling_rate=16000,
                grad_clip=1.0,
            )
            device = accelerator_device()
            previous = torch.are_deterministic_algorithms_enabled()
            prepare_accelerator_numerics(device)
            try:
                task = SEMambaTask(config, device)
                metrics = task.train_update(
                    [{"clean": clean, "noisy": noisy, "case_id": ["x"]}],
                    1,
                )
            finally:
                set_determinism(previous, cuda=False)
        for value in metrics.values():
            self.assertTrue(np.isfinite(value), metrics)

    def test_dt_proj_bias_is_not_the_official_scratch_init(self):
        module = SelectiveSSM(8, d_state=4, d_conv=4, expand=2, scan_backend="reference")
        self.assertFalse(hasattr(module.dt_proj.bias, "_no_reinit"))
        soft = torch.nn.functional.softplus(module.dt_proj.bias.detach())
        self.assertTrue(bool(torch.any(soft > 0.1)))


def _plan(count: int, micro: int, world: int, epoch: int, accumulation: int, seed: int = 7):
    return plan_semamba_epoch(
        split="train",
        epoch=epoch,
        count=count,
        seed=seed,
        world_size=world,
        micro_batch=micro,
        accumulation=accumulation,
        protocol_id="proto",
        snapshot_hash="snap",
    )


def _indexes(plan, rank: int) -> list[int]:
    return [
        sample.index
        for group in plan.rank_groups[rank]
        for micro in group.micro_batches
        for sample in micro.samples
    ]


def _trainer_config() -> dict:
    return {
        "seed": 7,
        "loss": "official",
        "model": {"name": "semambapp", "sampling_rate": 16000},
        "logical_steps_per_epoch": None,
        "data": {"load_workers": 1, "prefetch": False},
        "best_checkpoint_start_epoch": 0,
    }


class _Stub:
    def __init__(self) -> None:
        self.steps = 0
        self.validated = 0
        self.micro_batches = 0
        self.seen = []

    def train_update(self, batches, accumulation):
        del accumulation
        self.steps += 1
        self.micro_batches += len(batches)
        self.seen.append([batch["noisy"].detach().clone() for batch in batches])
        for batch in batches:
            if batch["noisy"].shape[-1] != SEGMENT_SIZE or not torch.isfinite(batch["noisy"]).all():
                raise AssertionError("online batch is not a finite 48000-sample window")
            if "mag" in batch:
                raise AssertionError("the dataset returned an STFT")
        return {"generator": 0.25, "discriminator": 0.25}

    def validation_report(self, batch):
        self.validated += 1
        if batch["clean"].shape != batch["noisy"].shape:
            raise AssertionError("validation batch shapes differ")
        metric = {"status": "ok", "value": 0.5, "reason": None, "count": 1}
        empty = {"status": "na", "value": None, "reason": "not computed", "count": 0}
        return {"losses": {"generator": metric}, "pesq": empty, "stoi": empty}

    def step_schedulers(self, validation) -> None:
        del validation

    def learning_rates(self):
        return {"generator": 0.001, "discriminator": 0.001}

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
