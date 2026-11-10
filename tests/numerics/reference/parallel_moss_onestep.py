"""One MossFormerGAN logical step on CUDA, native STFT versus shared index STFT.

Two micro-batches, accumulation 2. Captures generator gradients at the
optimizer step and the weight gap after that step. This does not repeat the
finished 6-micro-batch CPU run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-continue"))
SE = ROOT / "src/se/official/clearervoice-studio/train/speech_enhancement"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SE))

from models.mossformer_gan.discriminator import Discriminator as OfficialDiscriminator  # noqa: E402
from models.mossformer_gan.generator import SyncANet  # noqa: E402
from losses import loss as official_loss  # noqa: E402
import solver as solver_mod  # noqa: E402
import utils.misc as misc  # noqa: E402
from solver import Solver  # noqa: E402
from utils.misc import pesq_loss  # noqa: E402
from src.experiments.calc.stft import centered_stft  # noqa: E402
from src.experiments.official.mossformergan.reference import MossFormerGANTask  # noqa: E402
from tests.experiments.test_mossformergan import _task_config  # noqa: E402
from tests.numerics.reference.final_mossformer_solver import _Pairs, _tone_pair  # noqa: E402

FLOOR = 1e-6


def _cpu_batch_pesq(clean_rows, enhanced_rows):
    values = [pesq_loss(clean, enhanced) for clean, enhanced in zip(clean_rows, enhanced_rows, strict=True)]
    if -1 in values:
        return None
    return torch.tensor([(value - 1) / 3.5 for value in values], dtype=torch.float32)


def _masked(left: torch.nn.Module, right: torch.nn.Module) -> dict:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    large = 0.0
    large_name = ""
    grad_worst = 0.0
    grad_name = ""
    for name, value in left.named_parameters():
        other = named[name]
        delta = (value.detach() - other.detach()).abs()
        found = float(delta.max())
        if found > worst:
            worst = found
            worst_name = name
        if value.grad is None or other.grad is None:
            continue
        gdelta = float((value.grad.detach() - other.grad.detach()).abs().max())
        if gdelta > grad_worst:
            grad_worst = gdelta
            grad_name = name
        mask = value.grad.detach().abs() >= FLOOR
        if bool(mask.any()):
            large_found = float(delta[mask].max())
            if large_found > large:
                large = large_found
                large_name = name
    element = {}
    if worst_name:
        left_p = dict(left.named_parameters())[worst_name]
        right_p = named[worst_name]
        delta = (left_p.detach() - right_p.detach()).abs()
        index = int(delta.view(-1).argmax())
        element = {
            "name": worst_name,
            "index": index,
            "weight_abs": float(delta.view(-1)[index]),
            "grad_official": None if left_p.grad is None else float(left_p.grad.detach().view(-1)[index]),
            "grad_task": None if right_p.grad is None else float(right_p.grad.detach().view(-1)[index]),
        }
    return {
        "parameter_max_abs": worst,
        "parameter_worst": worst_name,
        "large_grad_weight_max_abs": large,
        "large_grad_weight_worst": large_name,
        "grad_max_abs": grad_worst,
        "grad_worst": grad_name,
        "element": element,
    }


def _install_index_stft() -> None:
    def stft(x, args, center=False):
        del center
        window = torch.hamming_window(args.win_len, periodic=False, dtype=x.dtype).to(x.device)
        return torch.view_as_real(centered_stft(x, args.fft_len, args.win_inc, window))

    misc.stft = stft
    solver_mod.stft = stft
    official_loss.stft = stft


def _restore_native_stft() -> None:
    misc.stft = misc_stft
    solver_mod.stft = misc_stft
    official_loss.stft = misc_stft


misc_stft = misc.stft


def _one(device: torch.device, shared: bool) -> dict:
    if shared:
        _install_index_stft()
    else:
        _restore_native_stft()
    torch.manual_seed(7)
    task = MossFormerGANTask(_task_config(), device)
    held_g = {key: value.detach().cpu().clone() for key, value in task.generator.state_dict().items()}
    held_d = {key: value.detach().cpu().clone() for key, value in task.discriminator.state_dict().items()}
    official_g = SyncANet(num_channel=64, num_features=201).to(device)
    official_d = OfficialDiscriminator(16).to(device)
    official_g.load_state_dict(held_g, strict=True)
    official_d.load_state_dict(held_d, strict=True)
    opt_g = torch.optim.AdamW(official_g.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01)
    opt_d = torch.optim.AdamW(official_d.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    from types import SimpleNamespace

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
    pairs = [_tone_pair(index) for index in range(2)]
    batches = [{"noisy": noisy, "clean": clean} for noisy, clean in pairs]
    official_loss.batch_pesq = lambda clean_rows, enhanced_rows: _cpu_batch_pesq(clean_rows, enhanced_rows).to(device)
    official_g.train()
    official_d.train()
    captured: dict[str, torch.nn.Module] = {}

    def wrap(optim, module, label):
        original = optim.step

        def step(*args, **kwargs):
            captured[label] = module
            return original(*args, **kwargs)

        optim.step = step

    wrap(opt_g, official_g, "official")
    wrap(task.optim_g, task.generator, "task")
    torch.manual_seed(19)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(19)
    official_mean = float(Solver._run_one_epoch_mossformergan_se_16k(solver, _Pairs(pairs), state="train"))
    torch.manual_seed(19)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(19)
    # Gradients on the task are read inside its optimizer step, after the
    # official model has already stepped. Clone official grads at its step.
    official_grads = {}

    def keep_official(*args, **kwargs):
        official_grads.update(
            {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in official_g.named_parameters()
            }
        )
        return torch.optim.AdamW.step(opt_g, *args, **kwargs)

    opt_g.step = keep_official
    # The wrap above already replaced step, and the official epoch already
    # ran. Rebuild a fresh pair instead of replaying a spent step.
    del captured
    return _fresh_pair(device, shared, held_g, held_d)


def _fresh_pair(device, shared, held_g, held_d) -> dict:
    if shared:
        _install_index_stft()
    else:
        _restore_native_stft()
    task = MossFormerGANTask(_task_config(), device)
    task.generator.load_state_dict(held_g, strict=True)
    task.discriminator.load_state_dict(held_d, strict=True)
    task.optim_g = torch.optim.AdamW(task.generator.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01)
    task.optim_d = torch.optim.AdamW(task.discriminator.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    official_g = SyncANet(num_channel=64, num_features=201).to(device)
    official_d = OfficialDiscriminator(16).to(device)
    official_g.load_state_dict(held_g, strict=True)
    official_d.load_state_dict(held_d, strict=True)
    opt_g = torch.optim.AdamW(official_g.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01)
    opt_d = torch.optim.AdamW(official_d.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    from types import SimpleNamespace

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
    pairs = [_tone_pair(index) for index in range(2)]
    batches = [{"noisy": noisy, "clean": clean} for noisy, clean in pairs]
    official_loss.batch_pesq = lambda clean_rows, enhanced_rows: _cpu_batch_pesq(clean_rows, enhanced_rows).to(device)
    official_g.train()
    official_d.train()
    official_grads: dict[str, torch.Tensor | None] = {}
    task_grads: dict[str, torch.Tensor | None] = {}

    def bind_official(optim):
        original = optim.step

        def step(*args, **kwargs):
            official_grads.update(
                {name: None if p.grad is None else p.grad.detach().clone() for name, p in official_g.named_parameters()}
            )
            return original(*args, **kwargs)

        optim.step = step

    def bind_task(optim):
        original = optim.step

        def step(*args, **kwargs):
            task_grads.update(
                {name: None if p.grad is None else p.grad.detach().clone() for name, p in task.generator.named_parameters()}
            )
            return original(*args, **kwargs)

        optim.step = step

    bind_official(opt_g)
    bind_task(task.optim_g)

    def seed() -> None:
        torch.manual_seed(19)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(19)

    seed()
    official_mean = float(Solver._run_one_epoch_mossformergan_se_16k(solver, _Pairs(pairs), state="train"))
    seed()
    task_metrics = task.train_update(batches, 2)
    grad_gap = 0.0
    grad_name = ""
    large_weight = 0.0
    large_name = ""
    for name, left in official_grads.items():
        right = task_grads.get(name)
        if left is None or right is None:
            continue
        value = float((left - right).abs().max())
        if value > grad_gap:
            grad_gap = value
            grad_name = name
        weight_delta = (
            dict(official_g.named_parameters())[name].detach() - dict(task.generator.named_parameters())[name].detach()
        ).abs()
        mask = left.abs() >= FLOOR
        if bool(mask.any()):
            found = float(weight_delta[mask].max())
            if found > large_weight:
                large_weight = found
                large_name = name
    weight = _masked(official_g, task.generator)
    weight["large_grad_weight_max_abs"] = large_weight
    weight["large_grad_weight_worst"] = large_name
    weight["grad_max_abs"] = grad_gap
    weight["grad_worst"] = grad_name
    return {
        "shared_index_stft": shared,
        "official_mean_generator_loss": official_mean,
        "task": task_metrics,
        "weight_after_step": weight,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(7)
    probe = MossFormerGANTask(_task_config(), device)
    held_g = {key: value.detach().cpu().clone() for key, value in probe.generator.state_dict().items()}
    held_d = {key: value.detach().cpu().clone() for key, value in probe.discriminator.state_dict().items()}
    del probe
    report = {
        "device": str(device),
        "official_commit": "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61",
        "micro_batches": 2,
        "logical_steps": 1,
        "accumulation": 2,
        "native": _fresh_pair(device, False, held_g, held_d),
        "shared_index_stft": _fresh_pair(device, True, held_g, held_d),
    }
    path = OUT / "mossformer_onestep.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    print(json.dumps({"native": report["native"]["weight_after_step"], "shared": report["shared_index_stft"]["weight_after_step"]}, indent=2))


if __name__ == "__main__":
    main()
