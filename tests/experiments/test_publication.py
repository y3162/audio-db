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
    SNAPSHOT_CLASS,
    require_declared_publication,
)
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mossformergan.trainer import MossFormerGANTrainer
from src.experiments.official.mp_senet.trainer import MPSENetGANTrainer
from tests.experiments.test_mp_senet_gan import _yaml_text

ROOT = Path(__file__).resolve().parents[2]
FLOAT_ROOT = ROOT / "data/db/snapshots-voicebank-float-v1"
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
        path = directory / "experiment.yaml"
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(BuildError, "publication"):
            build_experiment(load_experiment(path))


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
        world_size = 1
        rank = 0
        device = "cpu"

    data.setup(_Runtime())
    return data


if __name__ == "__main__":
    unittest.main()
