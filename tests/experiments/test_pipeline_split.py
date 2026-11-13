"""Model pipelines connect to dataset snapshots without reading the snapshot rule.

The production path is SnapshotSource, the pair or chunk adapter, the sampler,
and SnapshotDataModule. AudioPairBridge is not used.
"""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from src.experiments.data.chunk import ChunkProtocol, ChunkSampler
from src.experiments.data.fixed_pair import (
    FixedPairProtocol,
    MicroBatcher,
    OrderSampler,
    SnapshotDataModule,
    SnapshotSource,
)
from src.experiments.data.online import Dfn3IndexSampler, Dfn3Protocol, SemambaProtocol
from src.experiments.data.pair import (
    WavePairData,
    check_publication,
    check_training_batch,
    require_audio_contract,
)
from src.experiments.data.protocols.online_dfn3 import mix_signals
from src.experiments.data.render.renderer import apply_fixed_pipeline, load_recorded_pair
from src.experiments.data.render.types import CropSpec
from src.experiments.data.sample import Batch
from src.experiments.data.sampling.planner import validation_refs
from src.experiments.errors import DataError, RuntimeFailure, StateError

ROOT = Path(__file__).resolve().parents[2]
VB_ROOT = ROOT / "data/db/snapshots-voicebank-float-v1"
SNAP_ROOT = ROOT / "data/db/snapshots"
VB = "mp_senet.vb28.ondemand.official@1"
MOSS_VB = "mossformergan.vb28.ondemand.official@1"
MS = "tf_gridnet.ms_snsd.espnet_recipe@1"
DNS = "tf_locoformer.dns.accept_recipe@1"
SEMAMBA = "semambapp.gsr.smoke@1"
DFN3 = "dfn3.online.smoke@1"

RESAMPLE = {
    "tool": "soxr",
    "quality": "HQ",
    "source_rate": 48000,
    "target_rate": 16000,
    "length": "source_frames_plus_one_div_3",
    "order": ["read_float32", "soxr_hq", "protocol_normalize", "protocol_window"],
}

_FIXED = {
    "mp_senet": {
        "normalize": "noisy_rms_unit_v1",
        "window": 32000,
        "validation_crop": "full_length",
        "order": ["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
        "validation_scope": "rank0",
    },
    "mossformergan": {
        "normalize": "independent_audio_norm_v1",
        "window": 32000,
        "validation_crop": "per_epoch_random",
        "order": "rank_mod_fixed_v1",
        "validation_scope": "per_rank",
    },
    "semambapp": {
        "normalize": "independent_peak_v1",
        "window": 48000,
        "validation_crop": "full_length",
        "order": "per_rank_independent_v1",
        "validation_scope": "rank0",
    },
}

_CHUNK = {
    "tf_gridnet": 48000,
    "tf_locoformer": 64000,
}


class _Runtime:
    device = torch.device("cpu")
    world_size = 1
    rank = 0


def _require_snapshots(test: unittest.TestCase) -> None:
    if not VB_ROOT.is_dir() or not SNAP_ROOT.is_dir():
        test.fail("real snapshots are not present; this connection is not a pass")


def _fixed_module(
    name: str,
    root: Path,
    spec: dict,
    *,
    resample: dict | None,
    seed: int = 0,
) -> SnapshotDataModule:
    source = SnapshotSource(name, root, "train")
    protocol = FixedPairProtocol(
        source,
        normalize=spec["normalize"],
        window=spec["window"],
        sample_rate=16000,
        validation_crop=spec["validation_crop"],
        resample=resample,
    )
    sampler = OrderSampler(
        source,
        protocol,
        seed=seed,
        order=spec["order"],
        validation_scope=spec["validation_scope"],
    )
    return SnapshotDataModule(source, sampler, protocol, MicroBatcher(1, 1), prefetch=False)


def _chunk_module(
    name: str,
    root: Path,
    chunk_length: int,
    *,
    resample: dict | None,
    num_samples: int,
) -> SnapshotDataModule:
    source = SnapshotSource(name, root, "train")
    protocol = ChunkProtocol(source, sample_rate=16000, resample=resample)
    sampler = ChunkSampler(
        source,
        seed=0,
        num_samples_per_epoch=num_samples,
        chunk_length=chunk_length,
        shift_ratio=0.5,
        num_cache_chunks=4,
        shuffle=False,
        global_batch=1,
        validation_scope="per_rank",
    )
    return SnapshotDataModule(source, sampler, protocol, MicroBatcher(1, 1), prefetch=False)


def _open(module: SnapshotDataModule) -> SnapshotDataModule:
    module.setup(_Runtime())
    return module


def _first_batch(module: SnapshotDataModule) -> Batch:
    view = module.train_epoch(0, start_update=0)
    try:
        if view.updates < 1:
            raise AssertionError("production epoch has no micro-batch")
        return view.batch(0)[0]
    finally:
        view.close()


def _assert_pair(test: unittest.TestCase, batch: Batch, *, rate: int, teacher: str, width: int | None = None) -> None:
    test.assertEqual(batch.sample_rate, rate)
    test.assertEqual(batch.teacher_kind, teacher)
    test.assertEqual(batch.inputs.ndim, 2)
    test.assertEqual(tuple(batch.inputs.shape), tuple(batch.targets.shape))
    test.assertGreater(int(batch.inputs.shape[-1]), 0)
    test.assertTrue(torch.isfinite(batch.inputs).all())
    test.assertTrue(torch.isfinite(batch.targets).all())
    test.assertIsNotNone(batch.lengths)
    test.assertGreaterEqual(int(batch.lengths.min()), 1)
    test.assertLessEqual(int(batch.lengths.max()), int(batch.inputs.shape[-1]))
    if width is not None:
        test.assertEqual(int(batch.inputs.shape[-1]), width)


def _crop_spec(protocol: FixedPairProtocol, sample) -> CropSpec | None:
    if sample.crop is None:
        return None
    pad = "none" if sample.crop.length == sample.crop.pad_to else protocol.pad_name
    return CropSpec(start=sample.crop.start, length=sample.crop.length, pad_to=sample.crop.pad_to, pad=pad)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class PipelineSplitTest(unittest.TestCase):
    def test_batch_contract_rejects_a_non_finite_or_wrong_rate_batch(self) -> None:
        good = torch.zeros(1, 8)
        batch = Batch(good, good.clone(), "s", "r", lengths=torch.tensor([8]), teacher_kind="dry_clean", sample_rate=16000)
        check_training_batch(batch, sample_rate=16000, teachers=("dry_clean",))
        poisoned = good.clone()
        poisoned[0, 0] = float("nan")
        with self.assertRaises(RuntimeFailure):
            check_training_batch(
                Batch(poisoned, good, "s", "r", lengths=torch.tensor([8]), teacher_kind="dry_clean", sample_rate=16000),
                sample_rate=16000,
                teachers=("dry_clean",),
            )
        with self.assertRaises(RuntimeFailure):
            check_training_batch(
                Batch(good, good, "s", "r", lengths=torch.tensor([8]), teacher_kind="dry_clean", sample_rate=48000),
                sample_rate=16000,
                teachers=("dry_clean",),
            )
        with self.assertRaises(RuntimeFailure):
            check_training_batch(
                Batch(good, good, "s", "r", lengths=torch.tensor([8]), teacher_kind="official_mixed_target", sample_rate=16000),
                sample_rate=16000,
                teachers=("dry_clean",),
            )

    def test_voicebank_refuses_implicit_resample(self) -> None:
        _require_snapshots(self)
        module = _fixed_module(VB, VB_ROOT, _FIXED["mp_senet"], resample=None)
        with self.assertRaises(DataError) as caught:
            _open(module)
        self.assertIn("resampling was not requested", str(caught.exception))
        module.close()

    def test_generated_audio_refuses_a_voicebank_resample_policy(self) -> None:
        _require_snapshots(self)
        module = _fixed_module(MS, SNAP_ROOT, _FIXED["mp_senet"], resample=RESAMPLE)
        with self.assertRaises(DataError) as caught:
            _open(module)
        self.assertIn("declared rate", str(caught.exception))
        module.close()

    def test_pipeline_normalize_overrides_the_snapshot_rule(self) -> None:
        _require_snapshots(self)
        module = _open(_fixed_module(MOSS_VB, VB_ROOT, _FIXED["mp_senet"], resample=dict(RESAMPLE)))
        try:
            rule = module.source.dataset()["rule_json"]
            self.assertEqual(rule["normalize"], "independent_audio_norm_v1")
            self.assertEqual(module.identity()["normalize"], "noisy_rms_unit_v1")
            self.assertEqual(module.identity()["validation_scope"], "rank0")
            self.assertNotEqual(module.identity()["validation_crop"], rule["validation"]["crop"])
            sample = module.sampler.sequence(0, 0, 1)[0]
            rendered = module.protocol.render(sample)
            raw_noisy, raw_clean = load_recorded_pair(
                module.source.store,
                sample.case_id,
                sample_rate=16000,
                resample=dict(RESAMPLE),
            )
            crop = _crop_spec(module.protocol, sample)
            followed, _, _ = apply_fixed_pipeline(
                raw_noisy,
                raw_clean,
                normalize="noisy_rms_unit_v1",
                crop=crop,
                case_id=sample.case_id,
            )
            other, _, _ = apply_fixed_pipeline(
                raw_noisy,
                raw_clean,
                normalize="independent_audio_norm_v1",
                crop=crop,
                case_id=sample.case_id,
            )
            self.assertEqual(int(np.max(np.abs(rendered[0] - followed))), 0)
            self.assertGreater(int(np.max(np.abs(followed - other))), 0)
            self.assertEqual(module.plan_hash, _open_plan(module))
        finally:
            module.close()

    def test_same_case_matches_before_the_model_pipeline(self) -> None:
        _require_snapshots(self)
        mp = _open(_fixed_module(VB, VB_ROOT, _FIXED["mp_senet"], resample=dict(RESAMPLE)))
        moss = _open(_fixed_module(VB, VB_ROOT, _FIXED["mossformergan"], resample=dict(RESAMPLE)))
        try:
            case_id = mp.source.case_ids("train.cases")[0]
            left = load_recorded_pair(mp.source.store, case_id, sample_rate=16000, resample=dict(RESAMPLE))
            right = load_recorded_pair(moss.source.store, case_id, sample_rate=16000, resample=dict(RESAMPLE))
            self.assertEqual(int(np.max(np.abs(left[0] - right[0]))), 0)
            self.assertEqual(int(np.max(np.abs(left[1] - right[1]))), 0)
            mp_audio = apply_fixed_pipeline(left[0], left[1], normalize="noisy_rms_unit_v1", crop=None, case_id=case_id)
            moss_audio = apply_fixed_pipeline(left[0], left[1], normalize="independent_audio_norm_v1", crop=None, case_id=case_id)
            self.assertGreater(int(np.max(np.abs(mp_audio[0] - moss_audio[0]))), 0)
        finally:
            mp.close()
            moss.close()

    def test_connection_matrix_renders_one_production_batch(self) -> None:
        _require_snapshots(self)
        datasets = (
            ("voicebank", VB, VB_ROOT, dict(RESAMPLE), 256),
            ("ms_snsd", MS, SNAP_ROOT, None, 2),
            ("dns2020", DNS, SNAP_ROOT, None, 2),
        )
        opened: list[SnapshotDataModule] = []
        try:
            for dataset_name, snapshot, root, resample, samples in datasets:
                for model, spec in _FIXED.items():
                    module = _open(_fixed_module(snapshot, root, spec, resample=None if resample is None else dict(resample)))
                    opened.append(module)
                    batch = _first_batch(module)
                    _assert_pair(self, batch, rate=16000, teacher="dry_clean", width=spec["window"])
                    self.assertEqual(module.identity()["normalize"], spec["normalize"])
                    self.assertEqual(module.identity()["dataset_id"], module.source.dataset().get("dataset_id"))
                for model, chunk_length in _CHUNK.items():
                    module = _open(
                        _chunk_module(
                            snapshot,
                            root,
                            chunk_length,
                            resample=None if resample is None else dict(resample),
                            num_samples=samples,
                        )
                    )
                    opened.append(module)
                    batch = _first_batch(module)
                    _assert_pair(self, batch, rate=16000, teacher="dry_clean", width=chunk_length)
                    self.assertIsNone(module.identity()["normalize"])
                    self.assertEqual(module.identity()["validation_scope"], "per_rank")
        finally:
            for module in opened:
                module.close()

    def test_generated_chunk_is_a_slice_of_the_unnormalized_mixture(self) -> None:
        _require_snapshots(self)
        chunk = _open(_chunk_module(MS, SNAP_ROOT, 8000, resample=None, num_samples=1))
        pair = _open(_fixed_module(MS, SNAP_ROOT, _FIXED["mp_senet"], resample=None))
        try:
            sample = chunk.sampler.sequence(0, 0, 1)[0]
            self.assertIsNotNone(sample.chunk)
            span_noisy, span_clean, valid = chunk.protocol._adapter.render_span(
                sample.case_id,
                int(sample.chunk.start),
                int(sample.chunk.stop),
            )
            full_noisy, full_clean = pair.protocol._adapter.render_full(sample.case_id)
            self.assertEqual(valid, 8000)
            self.assertEqual(int(np.max(np.abs(span_noisy - full_noisy[sample.chunk.start : sample.chunk.stop]))), 0)
            self.assertEqual(int(np.max(np.abs(span_clean - full_clean[sample.chunk.start : sample.chunk.stop]))), 0)
        finally:
            chunk.close()
            pair.close()

    def test_online_adapters_keep_rate_and_teacher(self) -> None:
        _require_snapshots(self)
        sem_source = SnapshotSource(SEMAMBA, SNAP_ROOT, "train")
        sem_protocol = SemambaProtocol(sem_source)
        sem = SnapshotDataModule(
            sem_source,
            OrderSampler(
                sem_source,
                sem_protocol,
                seed=0,
                global_batch=1,
                order="per_rank_independent_v1",
                validation_scope="rank0",
                validation_batch=1,
            ),
            sem_protocol,
            MicroBatcher(1, 1),
            prefetch=False,
        )
        dfn_source = SnapshotSource(DFN3, SNAP_ROOT, "train")
        dfn = SnapshotDataModule(
            dfn_source,
            Dfn3IndexSampler(dfn_source, seed=0, global_batch=1),
            Dfn3Protocol(dfn_source),
            MicroBatcher(1, 1),
            prefetch=False,
        )
        opened = [sem, dfn]
        try:
            _open(sem)
            _open(dfn)
            sem_batch = _first_batch(sem)
            dfn_batch = _first_batch(dfn)
            _assert_pair(self, sem_batch, rate=16000, teacher="dry_clean", width=48000)
            _assert_pair(self, dfn_batch, rate=48000, teacher="official_mixed_target", width=144000)
            with self.assertRaises(RuntimeFailure):
                require_audio_contract(dfn, sample_rate=16000, teachers=("dry_clean",))
            with self.assertRaises(RuntimeFailure):
                require_audio_contract(sem, sample_rate=16000, teachers=("official_mixed_target",))
        finally:
            for module in opened:
                module.close()

    def test_dfn3_refuses_a_dry_clean_snapshot(self) -> None:
        _require_snapshots(self)
        source = SnapshotSource(VB, VB_ROOT, "train")
        module = SnapshotDataModule(
            source,
            Dfn3IndexSampler(source, seed=0),
            Dfn3Protocol(source),
            MicroBatcher(1, 1),
        )
        with self.assertRaises(DataError) as caught:
            _open(module)
        message = str(caught.exception)
        self.assertTrue("48000" in message or "teacher" in message)
        module.close()

    def test_wave_pair_adapter_supplies_a_mixed_target_without_a_trainer_edit(self) -> None:
        rng = np.random.default_rng(0)
        speech = rng.normal(0.0, 0.1, size=4800).astype(np.float64)
        noise = rng.normal(0.0, 0.05, size=4800).astype(np.float64)
        clean, _scaled, mixture = mix_signals(speech, speech, noise, snr_db=5.0, gain_db=-6.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "noisy").mkdir()
            (root / "target").mkdir()
            _write_wav(root / "noisy" / "pair.wav", np.asarray(mixture, dtype=np.float32), 48000)
            _write_wav(root / "target" / "pair.wav", np.asarray(clean, dtype=np.float32), 48000)
            data = WavePairData(str(root), sample_rate=48000, teacher="official_mixed_target")
            require_audio_contract(data, sample_rate=48000, teachers=("official_mixed_target",))
            with self.assertRaises(RuntimeFailure):
                require_audio_contract(data, sample_rate=16000, teachers=("official_mixed_target",))
            with self.assertRaises(RuntimeFailure):
                require_audio_contract(data, sample_rate=48000, teachers=("dry_clean",))
            batch = data.train_epoch(0, start_update=0).batch(0)[0]
            _assert_pair(self, batch, rate=48000, teacher="official_mixed_target")
            self.assertNotIn("algorithm", data.identity())
            stereo = root / "noisy" / "pair.wav"
            with wave.open(str(stereo), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(48000)
                handle.writeframes(np.zeros(16, dtype=np.int16).tobytes())
            with self.assertRaises(DataError):
                WavePairData(str(root), sample_rate=48000, teacher="official_mixed_target")

    def test_resume_refuses_a_changed_pipeline_or_dataset(self) -> None:
        _require_snapshots(self)
        original = _open(_fixed_module(VB, VB_ROOT, _FIXED["mp_senet"], resample=dict(RESAMPLE), seed=3))
        same = _open(_fixed_module(VB, VB_ROOT, _FIXED["mp_senet"], resample=dict(RESAMPLE), seed=3))
        other_pipeline = _open(_fixed_module(VB, VB_ROOT, _FIXED["mossformergan"], resample=dict(RESAMPLE), seed=3))
        try:
            state = original.state_dict()
            same.load_state_dict(state)
            self.assertNotEqual(original.plan_hash, other_pipeline.plan_hash)
            with self.assertRaises(StateError):
                other_pipeline.load_state_dict(state)
            other_dataset = _open(_fixed_module(MS, SNAP_ROOT, _FIXED["mp_senet"], resample=None, seed=3))
            try:
                with self.assertRaises(StateError):
                    other_dataset.load_state_dict(state)
            finally:
                other_dataset.close()
        finally:
            original.close()
            same.close()
            other_pipeline.close()

    def test_validation_plan_follows_the_pipeline_and_replays(self) -> None:
        _require_snapshots(self)
        module = _open(_fixed_module(VB, VB_ROOT, _FIXED["mossformergan"], resample=dict(RESAMPLE)))
        try:
            store = module.source.store
            first = validation_refs(
                store,
                seed=7,
                epoch=1,
                world_size=1,
                policy="rank_mod_fixed_v1",
                rank=0,
                crop_name="per_epoch_random",
                window=32000,
                normalize="independent_audio_norm_v1",
                resampled=True,
            )
            second = validation_refs(
                store,
                seed=7,
                epoch=1,
                world_size=1,
                policy="rank_mod_fixed_v1",
                rank=0,
                crop_name="per_epoch_random",
                window=32000,
                normalize="independent_audio_norm_v1",
                resampled=True,
            )
            moved = validation_refs(
                store,
                seed=7,
                epoch=2,
                world_size=1,
                policy="rank_mod_fixed_v1",
                rank=0,
                crop_name="per_epoch_random",
                window=32000,
                normalize="independent_audio_norm_v1",
                resampled=True,
            )
            self.assertEqual(first, second)
            self.assertNotEqual(first, moved)
            self.assertEqual(module.identity()["validation_scope"], "per_rank")
            self.assertEqual(module.identity()["validation_crop"], "per_epoch_random")
        finally:
            module.close()

    def test_research_claim_does_not_inherit_the_dataset_profile(self) -> None:
        class _Trainer:
            experiment_claim = "research"
            claim_label = None
            data = _Identity(profile="official_faithful", selection_uses_test=False, claim_label="espnet_public_se_recipe_reproduction")

        check_publication(_Trainer())

        class _Official:
            experiment_claim = "official_faithful"
            claim_label = None
            data = _Trainer.data

        with self.assertRaises(RuntimeFailure):
            check_publication(_Official())


class _Identity:
    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def identity(self) -> dict[str, object]:
        return dict(self._fields)


def _open_plan(module: SnapshotDataModule) -> str:
    return module.plan_hash
