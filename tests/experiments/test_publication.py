"""Preflight acceptance and refusal for official, research, and leakage-safe claims."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.data.fixed_pair import (
    FixedPairProtocol,
    MicroBatcher,
    OrderSampler,
    SnapshotDataModule,
    SnapshotSource,
)
from src.experiments.data.pair import WavePairData, check_publication
from src.experiments.data.publication import (
    MOSS_TRAINER,
    MP_TRAINER,
    OFFICIAL_CONDITIONS,
    OFFICIAL_ORDERS,
    PUBLICATION_KEYS,
    SNAPSHOT_CLASS,
    require_declared_publication,
)
from src.experiments.data.sampling.order import resolve_order_policy
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mossformergan.trainer import MossFormerGANTrainer
from src.experiments.official.mp_senet.trainer import MPSENetGANTrainer
from tests.experiments.test_mp_senet_gan import _yaml_text

ROOT = Path(__file__).resolve().parents[2]
FLOAT_ROOT = ROOT / "data/db/snapshots-voicebank-float-v1"
MP_FIXTURE = ROOT / "tests/fixtures/configs/experiments/mp_senet_gan.yaml"
MP_ORDER = list(OFFICIAL_ORDERS[MP_TRAINER])
RESAMPLE = {
    "tool": "soxr",
    "quality": "HQ",
    "source_rate": 48000,
    "target_rate": 16000,
    "length": "source_frames_plus_one_div_3",
    "order": ["read_float32", "soxr_hq", "protocol_normalize", "protocol_window"],
}


class _Claim:
    def __init__(self, claim: str | None, data: object, label: str | None = None) -> None:
        self.experiment_claim = claim
        self.claim_label = label
        self.data = data


class _Identity:
    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def identity(self) -> dict[str, object]:
        return dict(self._fields)


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_official_configs_declare_the_snapshot_condition(self) -> None:
        paths = {
            MP_TRAINER: (
                ROOT / "configs/reproduction/experiments/official/mp_senet_gan.yaml",
                ROOT / "tests/fixtures/configs/experiments/mp_senet_gan.yaml",
            ),
            MOSS_TRAINER: (
                ROOT / "configs/reproduction/experiments/official/mossformergan_se_16k.yaml",
                ROOT / "tests/fixtures/configs/experiments/mossformergan_se_16k.yaml",
            ),
        }
        self.assertNotIn("order_policy", PUBLICATION_KEYS)
        self.assertNotIn("max_epoch_samples", PUBLICATION_KEYS)
        self.assertNotIn("max_validation_batches", PUBLICATION_KEYS)
        for trainer_path, documents in paths.items():
            condition = OFFICIAL_CONDITIONS[trainer_path]
            snapshot = FLOAT_ROOT / condition["dataset_id"]
            document = json.loads((snapshot / "dataset.json").read_text(encoding="utf-8"))
            manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(document["dataset_id"], condition["dataset_id"])
            self.assertEqual(document["protocol_id"], condition["protocol_id"])
            self.assertEqual(document["claim_label"], condition["claim_label"])
            self.assertEqual(document["profile"], condition["profile"])
            self.assertEqual(document["selection_uses_test"], condition["selection_uses_test"])
            self.assertEqual(manifest["snapshot_hash"], condition["snapshot_hash"])
            for path in documents:
                config = load_experiment(path)
                trainer = _blank(_trainer_type(trainer_path))
                require_declared_publication(trainer, config.publication)

    def test_live_official_snapshots_are_accepted(self) -> None:
        opened = (
            (
                MPSENetGANTrainer,
                "mp_senet.vb28.ondemand.official@1",
                "noisy_rms_unit_v1",
                "full_length",
                "rank0",
                ["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
                "official_faithful",
            ),
            (
                MossFormerGANTrainer,
                "mossformergan.vb28.ondemand.official@1",
                "independent_audio_norm_v1",
                "per_epoch_random",
                "per_rank",
                "rank_mod_fixed_v1",
                "official_faithful",
            ),
        )
        for trainer_type, snapshot, normalize, crop, scope, order, profile in opened:
            data = _open(snapshot, normalize=normalize, crop=crop, scope=scope, order=order, profile=profile)
            try:
                trainer = _blank(trainer_type)
                trainer.experiment_claim = "official_faithful"
                trainer.data = data
                check_publication(trainer)
            finally:
                data.close()

    def test_research_accepts_voicebank_dns_ms_snsd_and_wave_pairs(self) -> None:
        voicebank = _identity(OFFICIAL_CONDITIONS[MP_TRAINER])
        dns = {
            "class": SNAPSHOT_CLASS,
            "algorithm": "pregenerated.dns_tflocoformer",
            "dataset_id": "5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe",
            "protocol_id": "cadda632ebd08a9066e5b7a81851920762cc2a5fa5bddc21f8488bf1a1630116",
            "sample_rate": 16000,
            "teacher": "dry_clean",
            "profile": "official_faithful",
            "claim_label": "espnet_public_se_recipe_reproduction",
            "selection_uses_test": False,
        }
        ms_snsd = {
            "class": SNAPSHOT_CLASS,
            "algorithm": "pregenerated.ms_snsd_tfgridnet",
            "dataset_id": "13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6",
            "protocol_id": "9f5ffa7c14a43b23b5db0a0c931c8a80e91cb666518aa10c826813c90abf7e97",
            "sample_rate": 16000,
            "teacher": "dry_clean",
            "profile": "official_faithful",
            "claim_label": "espnet_public_se_recipe_reproduction",
            "selection_uses_test": False,
        }
        for fields in (voicebank, dns, ms_snsd):
            check_publication(_Claim("research", _Identity(**fields)))
        root = self.base / "waves"
        noisy = root / "noisy"
        target = root / "target"
        noisy.mkdir(parents=True)
        target.mkdir()
        tone = np.zeros(16, dtype=np.float32)
        sf.write(noisy / "a.wav", tone, 16000)
        sf.write(target / "a.wav", tone, 16000)
        check_publication(_Claim("research", WavePairData(str(root), sample_rate=16000, teacher="dry_clean")))

    def test_leakage_safe_snapshot_is_accepted_and_unknown_test_use_is_refused(self) -> None:
        data = _open(
            "mp_senet.vb28.ondemand.safe@1",
            normalize="noisy_rms_unit_v1",
            crop="full_length",
            scope="rank0",
            order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
            profile="leakage_safe",
        )
        try:
            trainer = _blank(MPSENetGANTrainer)
            trainer.experiment_claim = "leakage_safe"
            trainer.data = data
            check_publication(trainer)
        finally:
            data.close()
        safe = {
            "class": SNAPSHOT_CLASS,
            "profile": "leakage_safe",
            "selection_uses_test": False,
            "claim_label": "leakage_safe_comparison",
        }
        check_publication(_Claim("leakage_safe", _Identity(**safe)))
        with self.assertRaisesRegex(RuntimeFailure, "unknown"):
            check_publication(_Claim("leakage_safe", _Identity(profile="leakage_safe", selection_uses_test=None)))
        with self.assertRaisesRegex(RuntimeFailure, "test split"):
            check_publication(_Claim("leakage_safe", _Identity(profile="leakage_safe", selection_uses_test=True)))

    def test_official_faithful_refuses_incomplete_or_foreign_data(self) -> None:
        condition = OFFICIAL_CONDITIONS[MP_TRAINER]
        cases = {
            "profile": ("missing profile", {"profile": None}),
            "selection_uses_test": ("missing selection_uses_test", {"selection_uses_test": None}),
            "dataset_id": ("missing dataset_id", {"dataset_id": None}),
            "protocol_id": ("missing protocol_id", {"protocol_id": None}),
            "foreign protocol": (
                "protocol_id",
                {"protocol_id": OFFICIAL_CONDITIONS[MOSS_TRAINER]["protocol_id"]},
            ),
            "unrelated snapshot": (
                "snapshot_hash",
                {"snapshot_hash": "265f256b938a89746dcaa62cd44daa2c6bcca3bce4cc1b1719eb2170717d33d5"},
            ),
            "wave pair": (
                "WavePairData",
                {"class": "src.experiments.data.pair.WavePairData"},
            ),
            "pipeline": (
                "pipeline normalize",
                {"normalize": "independent_audio_norm_v1"},
            ),
            "claim": (
                "claim_label",
                {"claim_label": "research_checkpoint_selection"},
            ),
        }
        trainer = _blank(MPSENetGANTrainer)
        trainer.experiment_claim = "official_faithful"
        for name, (pattern, overrides) in cases.items():
            fields = _identity(condition)
            fields.update(overrides)
            trainer.data = _Identity(**fields)
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeFailure, pattern):
                    check_publication(trainer)

    def test_semamba_official_reproduction_stays_blocked(self) -> None:
        data = _Identity(claim_label="official_reproduction", profile="official_faithful", selection_uses_test=False)
        with self.assertRaisesRegex(RuntimeFailure, "semambapp.gsr.official is blocked"):
            check_publication(_Claim(None, data, label="protocol_conformant_substitute_materials"))

    def test_builder_rejects_official_faithful_without_the_declaration(self) -> None:
        directory = self.base / "undeclared"
        directory.mkdir()
        text = _yaml_text(directory, steps=1, updates=2).replace(
            "experiment_claim: research",
            "experiment_claim: official_faithful",
        )
        text = text.replace("dense_channel: 4", "dense_channel: 64").replace(
            "num_tsconformers: 1", "num_tsconformers: 4"
        )
        path = directory / "experiment.yaml"
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(BuildError, "publication"):
            build_experiment(load_experiment(path))

    def test_mp_senet_order_follows_world_size_and_rejects_a_narrower_order(self) -> None:
        wide = _open(
            "mp_senet.vb28.ondemand.official@1",
            normalize="noisy_rms_unit_v1",
            crop="full_length",
            scope="rank0",
            order=list(MP_ORDER),
            profile="official_faithful",
            world_size=2,
        )
        try:
            trainer = _official(MPSENetGANTrainer, wide)
            check_publication(trainer)
            self.assertEqual(wide.sampler.policy(2), "torch_distributed_sampler_v1")
            self.assertEqual(wide.identity()["order_policy"], "torch_distributed_sampler_v1")
            self.assertIsNone(wide.sampler.max_epoch_samples)
            self.assertIsNone(wide.max_validation_batches)
        finally:
            wide.close()

        data = _open(
            "mp_senet.vb28.ondemand.official@1",
            normalize="noisy_rms_unit_v1",
            crop="full_length",
            scope="rank0",
            order=list(MP_ORDER),
            profile="official_faithful",
        )
        try:
            trainer = _official(MPSENetGANTrainer, data)
            check_publication(trainer)
            self.assertEqual(data.sampler.policy(1), "fixed_single_shuffle_v1")
            self.assertEqual(data.identity()["order_policy"], "fixed_single_shuffle_v1")
            data.sampler.order = list(reversed(MP_ORDER))
            check_publication(trainer)
            self.assertEqual(data.identity()["order_policy"], "fixed_single_shuffle_v1")
            data.sampler.order = list(MP_ORDER)
            data.world_size = 4
            check_publication(trainer)
            self.assertEqual(data.identity()["order_policy"], "torch_distributed_sampler_v1")
            refusals = (
                (1, "fixed_single_shuffle_v1"),
                (1, "torch_distributed_sampler_v1"),
                (2, "torch_distributed_sampler_v1"),
                (2, "fixed_single_shuffle_v1"),
                (1, "rank_mod_fixed_v1"),
            )
            for world_size, order in refusals:
                data.world_size = world_size
                data.sampler.order = order
                with self.subTest(world_size=world_size, order=order):
                    with self.assertRaisesRegex(RuntimeFailure, "sampler order"):
                        check_publication(trainer)
        finally:
            data.close()

    def test_mossformergan_order_is_rank_mod_and_a_different_policy_is_refused(self) -> None:
        data = _open(
            "mossformergan.vb28.ondemand.official@1",
            normalize="independent_audio_norm_v1",
            crop="per_epoch_random",
            scope="per_rank",
            order="rank_mod_fixed_v1",
            profile="official_faithful",
        )
        try:
            trainer = _official(MossFormerGANTrainer, data)
            check_publication(trainer)
            self.assertEqual(data.sampler.policy(1), "rank_mod_fixed_v1")
            self.assertEqual(data.identity()["order_policy"], "rank_mod_fixed_v1")
            data.world_size = 2
            check_publication(trainer)
            self.assertEqual(data.identity()["order_policy"], "rank_mod_fixed_v1")
            data.sampler.order = ["rank_mod_fixed_v1"]
            check_publication(trainer)
            for order in ("fixed_single_shuffle_v1", MP_ORDER):
                data.sampler.order = order
                with self.subTest(order=order):
                    with self.assertRaisesRegex(RuntimeFailure, "sampler order"):
                        check_publication(trainer)
        finally:
            data.close()

    def test_official_sample_caps_are_refused_and_research_keeps_them(self) -> None:
        data = _open(
            "mp_senet.vb28.ondemand.official@1",
            normalize="noisy_rms_unit_v1",
            crop="full_length",
            scope="rank0",
            order=list(MP_ORDER),
            profile="official_faithful",
        )
        try:
            trainer = _official(MPSENetGANTrainer, data)
            check_publication(trainer)
            data.sampler.max_epoch_samples = 1
            with self.assertRaisesRegex(RuntimeFailure, "max_epoch_samples"):
                check_publication(trainer)
            data.sampler.max_epoch_samples = None
            data.max_validation_batches = 1
            with self.assertRaisesRegex(RuntimeFailure, "max_validation_batches"):
                check_publication(trainer)
            trainer.experiment_claim = "research"
            for samples, batches in ((1, None), (None, 1)):
                data.sampler.max_epoch_samples = samples
                data.max_validation_batches = batches
                with self.subTest(max_epoch_samples=samples, max_validation_batches=batches):
                    check_publication(trainer)
        finally:
            data.close()

    def test_resolved_policy_and_identity_are_checked_when_the_order_list_matches(self) -> None:
        official = list(MP_ORDER)
        lying = _Policy(official, "rank_mod_fixed_v1")
        trainer = _official(
            MPSENetGANTrainer,
            _sampling_data(lying, order_policy="fixed_single_shuffle_v1"),
        )
        with self.assertRaisesRegex(RuntimeFailure, "order_policy 'rank_mod_fixed_v1'"):
            check_publication(trainer)
        honest = _Policy(official, None)
        trainer.data = _sampling_data(honest, order_policy="rank_mod_fixed_v1")
        with self.assertRaisesRegex(RuntimeFailure, "order_policy 'rank_mod_fixed_v1'"):
            check_publication(trainer)

    def test_builder_accepts_the_official_mp_senet_fixture_before_training(self) -> None:
        path = self._stage_official("accept-mp", MP_FIXTURE)
        experiment = build_experiment(load_experiment(path))
        try:
            self.assertEqual(experiment.config.schema, "audio-db-experiment/v1")
            self.assertFalse(experiment._ran)
            data = experiment.get("data")
            self.assertEqual(data.identity()["order_policy"], "fixed_single_shuffle_v1")
            self.assertEqual(data.sampler.policy(data.world_size), data.identity()["order_policy"])
            self.assertIsNone(data.sampler.max_epoch_samples)
            self.assertIsNone(data.max_validation_batches)
        finally:
            experiment.close()

    def test_builder_rejects_a_narrowed_official_order_before_training(self) -> None:
        path = self._stage_official(
            "narrow-order",
            MP_FIXTURE,
            (
                (
                    "      order:\n        - fixed_single_shuffle_v1\n        - torch_distributed_sampler_v1\n",
                    "      order: fixed_single_shuffle_v1\n",
                ),
            ),
        )
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(path))
        self.assertRegex(str(caught.exception), "sampler order")
        self.assertIsInstance(caught.exception.__cause__, RuntimeFailure)

    def test_builder_rejects_official_sample_caps_before_training(self) -> None:
        cases = {
            "max_epoch_samples": (
                "      validation_scope: rank0\n",
                "      validation_scope: rank0\n      max_epoch_samples: 1\n",
            ),
            "max_validation_batches": (
                "      require_profile: official_faithful\n      prefetch: false\n",
                "      require_profile: official_faithful\n      prefetch: false\n      max_validation_batches: 1\n",
            ),
        }
        for name, (old, new) in cases.items():
            with self.subTest(name=name):
                path = self._stage_official(name, MP_FIXTURE, ((old, new),))
                with self.assertRaises(BuildError) as caught:
                    build_experiment(load_experiment(path))
                self.assertRegex(str(caught.exception), name)
                self.assertIsInstance(caught.exception.__cause__, RuntimeFailure)

    def _stage_official(
        self,
        name: str,
        source: Path,
        replacements: tuple[tuple[str, str], ...] = (),
    ) -> Path:
        directory = self.base / name
        directory.mkdir()
        text = source.read_text(encoding="utf-8")
        rewritten = text
        for label in ("runs/mp-senet-gan", "runs/mossformergan-se-16k"):
            rewritten = rewritten.replace(f"directory: {label}", f"directory: {directory.as_posix()}", 1)
        if rewritten == text or "directory: runs/" in rewritten:
            raise AssertionError("output directory was not retargeted")
        text = rewritten
        for old, new in replacements:
            count = text.count(old)
            if count != 1:
                raise AssertionError(f"{old!r} occurred {count} times")
            text = text.replace(old, new, 1)
        path = directory / "experiment.yaml"
        path.write_text(text, encoding="utf-8")
        return path


def _trainer_type(path: str) -> type:
    if path == MP_TRAINER:
        return MPSENetGANTrainer
    if path == MOSS_TRAINER:
        return MossFormerGANTrainer
    raise AssertionError(path)


def _blank(kind: type) -> object:
    trainer = kind.__new__(kind)
    trainer.experiment_claim = None
    trainer.claim_label = None
    trainer.data = None
    return trainer


def _official(kind: type, data: object) -> object:
    trainer = _blank(kind)
    trainer.experiment_claim = "official_faithful"
    trainer.data = data
    return trainer


class _Policy:
    def __init__(self, order: list[str], resolved: str | None) -> None:
        self.order = order
        self.max_epoch_samples = None
        self._resolved = resolved

    def policy(self, world_size: int) -> str:
        if self._resolved is not None:
            return self._resolved
        return resolve_order_policy(self.order, world_size)


def _sampling_data(sampler: _Policy, *, order_policy: str, world_size: int = 1) -> _Identity:
    fields = _identity(OFFICIAL_CONDITIONS[MP_TRAINER])
    fields["order_policy"] = order_policy
    data = _Identity(**fields)
    data.sampler = sampler
    data.world_size = world_size
    data.max_validation_batches = None
    return data


def _identity(condition: dict[str, object]) -> dict[str, object]:
    fields = copy.deepcopy(condition)
    fields["class"] = SNAPSHOT_CLASS
    return fields


def _open(
    snapshot: str,
    *,
    normalize: str,
    crop: str,
    scope: str,
    order: str | list[str],
    profile: str,
    world_size: int = 1,
) -> SnapshotDataModule:
    source = SnapshotSource(
        snapshot=snapshot,
        snapshot_root=FLOAT_ROOT,
        split="train",
    )
    protocol = FixedPairProtocol(
        source,
        normalize=normalize,
        window=32000,
        sample_rate=16000,
        validation_crop=crop,
        resample=dict(RESAMPLE),
    )
    sampler = OrderSampler(source, protocol, seed=1, validation_scope=scope, order=order)
    data = SnapshotDataModule(
        source,
        sampler,
        protocol,
        MicroBatcher(micro_batch=1, group_size=1),
        require_profile=profile,
        prefetch=False,
    )

    class _Runtime:
        def __init__(self) -> None:
            self.world_size = world_size
            self.rank = 0
            self.device = "cpu"

    data.setup(_Runtime())
    return data


if __name__ == "__main__":
    unittest.main()
