"""Call ClearerVoice ``_run_one_epoch_mossformergan_se_16k`` and ``MossFormerGANTrainer.run``.

The official method is the released accumulation loop. PESQ stays the official
``pesq_loss``. On CPU the released ``batch_pesq`` hardcodes ``.to('cuda')``, so
that placement is redirected to the comparison device. The score is unchanged.

Outputs go to ``AUDIO_DB_NUMERICS_OUT`` (default ``/tmp/audio-db-official-final``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-final"))
SE = ROOT / "src/se/official/clearervoice-studio/train/speech_enhancement"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SE))

from models.mossformer_gan.discriminator import Discriminator as OfficialDiscriminator  # noqa: E402
from models.mossformer_gan.generator import SyncANet  # noqa: E402
from losses import loss as official_loss  # noqa: E402
from solver import Solver  # noqa: E402
from utils.misc import pesq_loss  # noqa: E402
from src.experiments.data import Batch  # noqa: E402
from tests.numerics.reference.common import NoCheckpoints  # noqa: E402
from src.experiments.official.mossformergan.models import (  # noqa: E402
    MossFormerGANDiscriminator,
    MossFormerGANGenerator,
)
from src.experiments.official.mossformergan.reference import MossFormerGANTask  # noqa: E402
from src.experiments.official.mossformergan.trainer import MossFormerGANTrainer  # noqa: E402
from src.experiments.runtime import PreparePolicy, ProcessRuntime  # noqa: E402
from tests.experiments.test_mossformergan import _as_batch, _task_config  # noqa: E402


def _gap(left: torch.nn.Module, right: torch.nn.Module) -> dict[str, float | str]:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, value in left.named_parameters():
        delta = float((value.detach() - named[name].detach()).abs().max())
        if delta > worst:
            worst = delta
            worst_name = name
    buffers = dict(right.named_buffers())
    buffer_worst = 0.0
    buffer_name = ""
    missing = []
    for name, value in left.named_buffers():
        other = buffers.get(name)
        if other is None:
            missing.append(name)
            continue
        delta = float((value.detach() - other.detach()).abs().max())
        if delta > buffer_worst:
            buffer_worst = delta
            buffer_name = name
    return {
        "parameter_max_abs": worst,
        "parameter_worst": worst_name,
        "buffer_max_abs": buffer_worst,
        "buffer_worst": buffer_name,
        "buffer_missing_on_right": missing,
    }


def _optim_gap(left: torch.optim.Optimizer, right: torch.optim.Optimizer) -> float:
    worst = 0.0
    left_params = [parameter for group in left.param_groups for parameter in group["params"]]
    right_params = [parameter for group in right.param_groups for parameter in group["params"]]
    for lp, rp in zip(left_params, right_params, strict=True):
        for key, value in left.state.get(lp, {}).items():
            other = right.state.get(rp, {}).get(key)
            if key == "step" or not torch.is_tensor(value) or not torch.is_tensor(other):
                continue
            worst = max(worst, float((value.detach() - other.detach()).abs().max()))
    return worst


def _tone_pair(index: int, length: int = 8000) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(length, dtype=torch.float32) / 16000
    clean = 0.2 * torch.sin(2 * torch.pi * (140 + 15 * index) * time)
    noisy = 0.15 * torch.sin(2 * torch.pi * (140 + 15 * index) * time + 0.4) + 0.05 * torch.sin(
        2 * torch.pi * (420 + 20 * index) * time
    )
    return noisy.unsqueeze(0), clean.unsqueeze(0)


def _cpu_batch_pesq(clean_rows, enhanced_rows):
    values = [pesq_loss(clean, enhanced) for clean, enhanced in zip(clean_rows, enhanced_rows, strict=True)]
    if -1 in values:
        return None
    return torch.tensor([(value - 1) / 3.5 for value in values], dtype=torch.float32)


class _Pairs:
    def __init__(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)


class _Epoch:
    def __init__(self, batches: list[Batch]) -> None:
        # Longer than the consumed window so run() stops on max_logical_steps
        # before the epoch boundary tries to checkpoint.
        self.updates = len(batches) + 4
        self._batches = batches

    def batch(self, index: int) -> list[Batch]:
        return [self._batches[index]]


class _Data:
    group_size = 2
    epoch = 0
    update_index = 0

    def __init__(self, batches: list[Batch]) -> None:
        self._batches = batches

    def train_epoch(self, epoch: int, *, start_update: int):
        del epoch, start_update
        return _Epoch(self._batches)

    def validation_epoch(self, epoch: int):
        del epoch
        return iter(())

    def identity(self) -> dict[str, object]:
        return {
            "algorithm": "fixed_pair.mossformergan_se_16k",
            "profile": "official_faithful",
            "selection_uses_test": True,
        }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and os.environ.get("AUDIO_DB_MOSS_DEVICE") != "cpu" else "cpu")
    torch.manual_seed(7)
    fresh = MossFormerGANTask(_task_config(), device)
    held_g = {key: value.detach().cpu().clone() for key, value in fresh.generator.state_dict().items()}
    held_d = {key: value.detach().cpu().clone() for key, value in fresh.discriminator.state_dict().items()}
    official_g = SyncANet(num_channel=64, num_features=201).to(device)
    official_d = OfficialDiscriminator(16).to(device)
    official_g.load_state_dict(held_g, strict=True)
    official_d.load_state_dict(held_d, strict=True)
    opt_g = torch.optim.AdamW(official_g.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01)
    opt_d = torch.optim.AdamW(official_d.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    solver = Solver.__new__(Solver)
    solver.device = device
    solver.model = official_g
    solver.discriminator = official_d
    solver.optimizer = opt_g
    solver.optimizer_disc = opt_d
    solver.epoch = 0
    solver.step = 0
    solver.accu_count = 0
    solver.args = SimpleNamespace(
        accu_grad=1,
        effec_batch_size=2,
        batch_size=1,
        clip_grad_norm=10.0,
        print_freq=10**9,
        checkpoint_save_freq=10**9,
        max_epoch=1,
        win_type="hamming",
        win_len=400,
        win_inc=100,
        fft_len=400,
        device=device,
    )
    pairs = [_tone_pair(index) for index in range(6)]
    batches = [{"noisy": noisy, "clean": clean} for noisy, clean in pairs]
    official_loss.batch_pesq = lambda clean_rows, enhanced_rows: _cpu_batch_pesq(clean_rows, enhanced_rows).to(device)
    official_g.train()
    official_d.train()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def _seed() -> None:
        torch.manual_seed(19)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(19)

    _seed()
    official_mean = float(Solver._run_one_epoch_mossformergan_se_16k(solver, _Pairs(pairs), state="train"))
    _seed()
    task_losses = [fresh.train_update(batches[index : index + 2], 2) for index in (0, 2, 4)]
    policy = PreparePolicy(trainable=True, distribute="none")
    runtime_device = "cuda" if device.type == "cuda" else "cpu"
    runtime = ProcessRuntime(device=runtime_device, precision="fp32")
    generator = runtime.prepare_model(MossFormerGANGenerator(seed=7), policy)
    discriminator = runtime.prepare_model(MossFormerGANDiscriminator(seed=7), policy)
    generator.state_module.generator.load_state_dict(held_g, strict=True)
    discriminator.state_module.discriminator.load_state_dict(held_d, strict=True)
    optim_g = torch.optim.AdamW(generator.parameters, lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01)
    optim_d = torch.optim.AdamW(discriminator.parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    optim_g._experiment_component_id = "optim_g"  # type: ignore[attr-defined]
    optim_d._experiment_component_id = "optim_d"  # type: ignore[attr-defined]
    data = _Data([_as_batch({"noisy": noisy, "clean": clean}, index) for index, (noisy, clean) in enumerate(pairs)])
    trainer = MossFormerGANTrainer(
        generator,
        discriminator,
        data,
        optim_g,
        optim_d,
        max_logical_steps=3,
        accumulation=2,
        grad_clip=10.0,
        validate_on_epoch_end=False,
    )
    checkpoint = NoCheckpoints(OUT / "mossformer-trainer")
    checkpoint.directory.mkdir(parents=True, exist_ok=True)
    _seed()
    result = trainer.run(runtime, checkpoint)
    report = {
        "device": str(device),
        "official_function": "solver.Solver._run_one_epoch_mossformergan_se_16k",
        "official_commit": "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61",
        "width": "SyncANet num_channel=64 num_features=201, Discriminator ndf=16",
        "micro_batches": 6,
        "logical_steps": 3,
        "accumulation": 2,
        "pesq_adapter": "official pesq_loss; batch_pesq device redirected off the hardcoded cuda string",
        "official_epoch_mean_generator_loss": official_mean,
        "task_updates": task_losses,
        "trainer_run": {
            "class": "MossFormerGANTrainer.run",
            "stop_reason": result.stop_reason,
            "logical_step": result.logical_step,
            "optimizer_updates": result.optimizer_updates,
        },
        "official_versus_task": {
            "generator": _gap(official_g, fresh.generator),
            "discriminator": _gap(official_d, fresh.discriminator),
            "optim_g": _optim_gap(opt_g, fresh.optim_g),
            "optim_d": _optim_gap(opt_d, fresh.optim_d),
        },
        "trainer_versus_task": {
            "generator": _gap(generator.state_module.generator, fresh.generator),
            "discriminator": _gap(discriminator.state_module.discriminator, fresh.discriminator),
            "optim_g": _optim_gap(optim_g, fresh.optim_g),
            "optim_d": _optim_gap(optim_d, fresh.optim_d),
        },
        "zero_grad_default_set_to_none": torch.optim.Optimizer.zero_grad.__defaults__,
    }
    path = OUT / "mossformer_solver.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(report["official_versus_task"], indent=2))
    print(json.dumps(report["trainer_versus_task"], indent=2))
    runtime.shutdown()


if __name__ == "__main__":
    main()
