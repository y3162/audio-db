"""YAML architecture changes reach the built modules.

Official settings keep the published state_dict. Research settings change
the module that is actually constructed. These tests do not claim a paper score.
"""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.config import config_sha256, load_experiment
from src.experiments.errors import BuildError
from src.experiments.official.deepfilternet3.model import DeepFilterNet3Model
from src.experiments.official.mossformergan.models import (
    OFFICIAL_ARCHITECTURE as MOSS_OFFICIAL,
    MossFormerGANGenerator,
    require_official_architecture as require_moss,
)
from src.experiments.official.mp_senet.models import (
    OFFICIAL_ARCHITECTURE as MP_OFFICIAL,
    MPSENetGANGenerator,
    MPSENetMetricDiscriminator,
    architecture_config as mp_architecture,
    require_official_architecture as require_mp,
)
from src.experiments.official.semambapp.models import SEMambaPPGenerator, SEMambaPPMRD
from src.experiments.official.tf_gridnet.model import TFGridNetModel
from src.experiments.official.tf_locoformer.model import TFLocoformerModel
from src.se import build_model

ROOT = Path(__file__).resolve().parents[2]


def _instantiate(component) -> torch.nn.Module:
    module_name, _, class_name = component.class_path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**component.kwargs)


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _mp(**overrides: object) -> dict[str, object]:
    values = dict(MP_OFFICIAL)
    values.update(overrides)
    return values


class ModelArchitectureTests(unittest.TestCase):
    def test_mp_senet_official_keys_match_the_published_structure(self) -> None:
        plain = MPSENetGANGenerator(seed=3, **{key: MP_OFFICIAL[key] for key in (
            "dense_channel", "num_tsconformers", "n_fft", "hop_size", "win_size",
            "compress_factor", "beta", "sampling_rate",
        )})
        explicit = MPSENetGANGenerator(seed=3, **MP_OFFICIAL)
        self.assertEqual(plain.architecture, MP_OFFICIAL)
        self.assertEqual(explicit.architecture, MP_OFFICIAL)
        self.assertEqual(plain.generator.state_dict().keys(), explicit.generator.state_dict().keys())
        for key, left in plain.state_dict().items():
            self.assertTrue(torch.equal(left, explicit.state_dict()[key]), key)
        require_mp(explicit.architecture, "official_faithful")
        with self.assertRaises(BuildError):
            require_mp(mp_architecture(**_mp(num_tsconformers=2)), "official_faithful")
        require_mp(mp_architecture(**_mp(num_tsconformers=2)), "research")

    def test_mp_senet_research_depth_and_discriminator_change_the_modules(self) -> None:
        common = _mp(
            dense_channel=8,
            num_tsconformers=1,
            n_fft=32,
            hop_size=8,
            win_size=32,
            n_heads=2,
            dense_depth=2,
            disc_dim=8,
        )
        shallow = MPSENetGANGenerator(seed=1, **common)
        deep = MPSENetGANGenerator(seed=1, **dict(common, dense_depth=3))
        self.assertNotEqual(_count(shallow), _count(deep))
        self.assertEqual(shallow.generator.dense_encoder.dense_block.depth, 2)
        self.assertEqual(deep.generator.dense_encoder.dense_block.depth, 3)
        waveform = torch.randn(1, 64)
        generated = shallow(waveform, waveform)
        generated[-1].square().mean().backward()
        wide = MPSENetMetricDiscriminator(seed=1, **common)
        narrow = MPSENetMetricDiscriminator(seed=1, **dict(common, disc_dim=4))
        self.assertNotEqual(wide.architecture, narrow.architecture)
        self.assertNotEqual(_count(wide), _count(narrow))
        with self.assertRaises(BuildError):
            MPSENetGANGenerator(seed=1, **dict(common, n_heads=3))
        with self.assertRaises(TypeError):
            MPSENetGANGenerator(seed=1, not_a_parameter=1, **common)
        ignored = build_model("mp_senet", {"dense_channel": 8, "num_tsconformers": 1, "not_a_parameter": 1})
        self.assertEqual(ignored.dense_encoder.dense_conv_1[0].out_channels, 8)

    def test_mossformer_research_layers_change_the_generator(self) -> None:
        base = dict(MOSS_OFFICIAL)
        small = dict(base, num_channel=32, n_layers=1, n_head=4, approx_qk_dim=128, dense_depth=1, query_key_dim=32, expansion_factor=2.0, dropout=0.0)
        one = MossFormerGANGenerator(seed=2, **small)
        two = MossFormerGANGenerator(seed=2, **dict(small, n_layers=2))
        self.assertEqual(len(one.generator.blocks), 1)
        self.assertEqual(len(two.generator.blocks), 2)
        self.assertNotEqual(_count(one), _count(two))
        spectrum = torch.randn(1, 2, 4, 201)
        real, imag = one(spectrum)
        (real.square().mean() + imag.square().mean()).backward()
        require_moss(MossFormerGANGenerator(seed=2).architecture, "official_faithful")
        with self.assertRaises(BuildError):
            require_moss(one.architecture, "official_faithful")
        with self.assertRaises(BuildError):
            MossFormerGANGenerator(seed=2, **dict(small, n_head=3))
        with self.assertRaises(ValueError):
            build_model("mossformergan", {"num_channel": 8, "not_a_parameter": 1}, variant="se_16k")

    def test_semamba_discriminator_resolution_is_applied(self) -> None:
        common = dict(
            seed=4,
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
        official = SEMambaPPMRD(**common)
        changed = SEMambaPPMRD(**common, mrd_channels=8, mrd_resolutions=((128, 32, 64),))
        self.assertEqual(len(official.discriminator.discriminators), 3)
        self.assertEqual(len(changed.discriminator.discriminators), 1)
        self.assertEqual(changed.discriminator.discriminators[0].resolution, (128, 32, 64))
        self.assertNotEqual(_count(official.discriminator), _count(changed.discriminator))
        generator = SEMambaPPGenerator(**common)
        clean = torch.randn(1, 800)
        outputs = generator(clean, clean)
        outputs[-1].square().mean().backward()
        with self.assertRaises(BuildError):
            SEMambaPPGenerator(**common, cqt_hops=(32,), cqt_octaves=(2, 2))

    def test_deepfilternet_explicit_defaults_match_the_empty_mapping(self) -> None:
        blank = DeepFilterNet3Model(seed=5, architecture={})
        explicit = DeepFilterNet3Model(seed=5, architecture={
            "sr": 48000,
            "fft_size": 960,
            "hop_size": 480,
            "nb_erb": 32,
            "nb_df": 96,
            "norm_tau": 1.0,
            "lsnr_max": 35,
            "lsnr_min": -15,
            "min_nb_freqs": 2,
            "df_order": 5,
            "df_lookahead": 2,
            "conv_lookahead": 2,
            "conv_ch": 64,
            "emb_hidden_dim": 256,
            "emb_num_layers": 3,
            "enc_lin_groups": 32,
            "lin_groups": 16,
            "df_hidden_dim": 256,
            "df_num_layers": 2,
            "df_gru_skip": "groupedlinear",
            "df_pathway_kernel_size_t": 5,
            "enc_concat": False,
            "conv_kernel": [1, 3],
            "convt_kernel": [1, 3],
            "conv_kernel_inp": [3, 3],
            "df_n_iter": 1,
            "mask_pf": False,
            "pf_beta": 0.02,
        })
        self.assertEqual(blank.architecture["sr"], 48000)
        self.assertEqual(blank.architecture["conv_ch"], 64)
        for key, left in blank.state_dict().items():
            self.assertTrue(torch.equal(left, explicit.state_dict()[key]), key)
        research = DeepFilterNet3Model(seed=5, architecture={"conv_ch": 32})
        self.assertEqual(research.generator.params.conv_ch, 32)
        self.assertEqual(research.generator.params.sr, 48000)
        with self.assertRaises(RuntimeError):
            research.generator.load_state_dict(blank.generator.state_dict(), strict=True)
        with self.assertRaises(ValueError):
            build_model("deepfilternet3", {"conv_ch": 32, "not_a_parameter": 1})
        with self.assertRaises(ValueError):
            build_model("deepfilternet3", {"lsnr_dropout": True})
        with self.assertRaises(ValueError):
            build_model("deepfilternet3", {"emb_num_layers": 1})

    def test_gridnet_and_locoformer_research_widths(self) -> None:
        grid = TFGridNetModel(seed=0, architecture={"n_layers": 1, "emb_dim": 8, "attn_n_head": 2, "lstm_hidden_units": 8, "n_fft": 32, "stride": 16, "emb_ks": 2, "emb_hs": 2, "attn_approx_qk_dim": 16})
        wider = TFGridNetModel(seed=0, architecture={"n_layers": 2, "emb_dim": 8, "attn_n_head": 2, "lstm_hidden_units": 8, "n_fft": 32, "stride": 16, "emb_ks": 2, "emb_hs": 2, "attn_approx_qk_dim": 16})
        self.assertEqual(len(grid.generator.blocks), 1)
        self.assertEqual(len(wider.generator.blocks), 2)
        self.assertEqual(grid.architecture["activation"], "prelu")
        wave = torch.randn(1, 64)
        grid(wave).square().mean().backward()
        with self.assertRaises(BuildError):
            TFGridNetModel(seed=0, architecture={"emb_dim": 8, "attn_n_head": 3})
        with self.assertRaises(BuildError):
            TFGridNetModel(seed=0, architecture={"n_srcs": 1})
        loco = TFLocoformerModel(seed=0, architecture={
            "n_layers": 1,
            "emb_dim": 8,
            "num_groups": 4,
            "n_heads": 2,
            "attention_dim": 8,
            "ffn_hidden_dim": [8, 8],
            "normalize_variance": False,
            "n_fft": 32,
            "hop_length": 8,
            "pos_enc": "nope",
        })
        loco_more = TFLocoformerModel(seed=0, architecture={
            "n_layers": 2,
            "emb_dim": 8,
            "num_groups": 4,
            "n_heads": 2,
            "attention_dim": 8,
            "ffn_hidden_dim": [8, 8],
            "normalize_variance": False,
            "n_fft": 32,
            "hop_length": 8,
            "pos_enc": "nope",
        })
        self.assertEqual(len(loco.generator.separator.blocks), 1)
        self.assertEqual(len(loco_more.generator.separator.blocks), 2)
        loco(wave).square().mean().backward()
        with self.assertRaises(BuildError):
            TFLocoformerModel(seed=0, architecture={"attention_dim": 8, "n_heads": 3})
        with self.assertRaises(BuildError):
            TFLocoformerModel(seed=0, architecture={"t_ksize": 5})

    def test_fixture_yaml_builds_the_declared_structure(self) -> None:
        fixtures = ROOT / "tests/fixtures/configs/experiments"
        expected = {
            "mp_senet_gan.yaml": ("generator", "n_heads", 4),
            "mossformergan_se_16k.yaml": ("generator", "n_layers", 6),
            "semambapp_gsr.yaml": ("mrd", "mrd_channels", 32),
            "deepfilternet3.yaml": ("model", "conv_ch", 64),
            "tf_gridnet_ms_snsd.yaml": ("model", "n_layers", 4),
            "tf_locoformer_dns.yaml": ("model", "n_layers", 6),
        }
        for name, (component_id, key, value) in expected.items():
            loaded = load_experiment(fixtures / name)
            module = _instantiate(loaded.components[component_id])
            self.assertEqual(module.architecture[key], value, name)
            if name == "mp_senet_gan.yaml":
                discriminator = _instantiate(loaded.components["metric_discriminator"])
                self.assertEqual(module.architecture, discriminator.architecture)
            if name == "semambapp_gsr.yaml":
                self.assertEqual(len(module.discriminator.discriminators), 3)

    def test_structure_change_changes_config_identity(self) -> None:
        source = ROOT / "tests/fixtures/configs/experiments/mp_senet_gan.yaml"
        original = load_experiment(source)
        text = source.read_text(encoding="utf-8").replace("n_heads: 4", "n_heads: 8", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(text, encoding="utf-8")
            changed = load_experiment(path)
        self.assertNotEqual(config_sha256(original), config_sha256(changed))
        self.assertEqual(config_sha256(original), config_sha256(load_experiment(source)))
