"""First MossFormerGAN divergence on one CPU micro-batch.

Eval mode removes dropout so a leaf mismatch is the operator graph. Train mode
is the forward that feeds the optimizer. Gradients are taken before any step.
The GAN term uses a private copy of the task discriminator on each side so
BatchNorm updates do not cross the comparison. Outputs go to
``AUDIO_DB_NUMERICS_OUT``.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-continue"))
SE = ROOT / "src/se/official/clearervoice-studio/train/speech_enhancement"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SE))

from models.mossformer_gan.generator import SyncANet  # noqa: E402
from utils.misc import power_compress, stft  # noqa: E402
from src.experiments.official.mossformergan.reference import (  # noqa: E402
    MossFormerGANTask,
    _GAN,
    _MAG,
    _RI,
    _TIME,
    _compress,
    _istft,
    _stft,
    _uncompress,
)
from tests.experiments.test_mossformergan import _task_config  # noqa: E402
from tests.numerics.reference.final_mossformer_solver import _tone_pair  # noqa: E402

LR = 5e-4
WD = 0.01
ADAM_EPS = 1e-8


def _args() -> SimpleNamespace:
    return SimpleNamespace(win_type="hamming", win_len=400, win_inc=100, fft_len=400, batch_size=1, device=torch.device("cpu"))


def _official_spec(noisy: torch.Tensor, args: SimpleNamespace) -> torch.Tensor:
    scale = torch.sqrt(noisy.shape[-1] / (noisy.square().sum(dim=-1) + 1e-6)).unsqueeze(-1)
    spectrum = stft(noisy * scale, args, center=True).to(torch.float32)
    return power_compress(spectrum).permute(0, 1, 3, 2)


def _task_spec(task: MossFormerGANTask, noisy: torch.Tensor) -> torch.Tensor:
    scale = torch.sqrt(noisy.shape[-1] / (noisy.square().sum(dim=-1) + 1e-6)).unsqueeze(-1)
    return _compress(_stft(noisy * scale, task.n_fft, task.hop, task.win)).permute(0, 1, 3, 2)


def _max_pair(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.detach() - right.detach()).abs().max())


def _leaf_outputs(module: torch.nn.Module, spectrum: torch.Tensor, seed: int) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    torch.manual_seed(seed)
    captured: list[tuple[str, torch.Tensor]] = []
    handles = []

    def hook(name: str):
        def record(_module, _inputs, output):
            if torch.is_tensor(output):
                captured.append((name, output.detach()))
        return record

    for name, child in module.named_modules():
        if any(child.children()):
            continue
        handles.append(child.register_forward_hook(hook(name)))
    try:
        output = module(spectrum)
    finally:
        for handle in handles:
            handle.remove()
    return output[0], captured


def _first_leaf(official, task) -> dict:
    mismatch = None
    compared = 0
    for (left_name, left), (right_name, right) in zip(official, task, strict=False):
        compared += 1
        gap = _max_pair(left, right)
        if gap != 0.0:
            mismatch = {
                "official": left_name,
                "task": right_name,
                "max_abs": gap,
                "shape": list(left.shape) if left.shape == right.shape else [list(left.shape), list(right.shape)],
            }
            break
    return {
        "leaves_compared_before_stop": compared,
        "first_mismatch": mismatch,
        "official_leaves": len(official),
        "task_leaves": len(task),
    }


def _adam_delta(param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return param * (LR * WD) + LR * grad / (grad.abs() + ADAM_EPS)


def _grad_report(official: torch.nn.Module, task: torch.nn.Module) -> dict:
    named = dict(task.named_parameters())
    rows = []
    meaningful = 0.0
    meaningful_name = ""
    for name, left in official.named_parameters():
        right = named[name]
        if left.grad is None or right.grad is None:
            continue
        delta = (left.grad.detach() - right.grad.detach()).abs()
        index = int(delta.view(-1).argmax())
        grad_left = left.grad.detach().view(-1)[index]
        grad_right = right.grad.detach().view(-1)[index]
        param = left.detach().view(-1)[index]
        predicted = _adam_delta(param, grad_left) - _adam_delta(named[name].detach().view(-1)[index], grad_right)
        magnitude = max(float(left.grad.detach().abs().max()), float(right.grad.detach().abs().max()))
        entry = {
            "name": name,
            "index": index,
            "grad_abs": float(delta.view(-1)[index]),
            "grad_official": float(grad_left),
            "grad_task": float(grad_right),
            "param": float(param),
            "predicted_weight_abs": float(predicted.abs()),
            "grad_magnitude": magnitude,
        }
        rows.append(entry)
        if magnitude >= 1e-4 and entry["grad_abs"] > meaningful:
            meaningful = entry["grad_abs"]
            meaningful_name = name
    by_grad = sorted(rows, key=lambda row: row["grad_abs"], reverse=True)
    by_weight = sorted(rows, key=lambda row: row["predicted_weight_abs"], reverse=True)
    named_rows = {row["name"]: row for row in rows}
    return {
        "worst_grad": by_grad[:8],
        "worst_predicted_weight": by_weight[:8],
        "conv1_bias": named_rows.get("dense_encoder.dilated_dense.conv1.bias"),
        "conv_1_bias": named_rows.get("dense_encoder.conv_1.0.bias"),
        "meaningful_grad_floor": 1e-4,
        "meaningful_worst": meaningful,
        "meaningful_name": meaningful_name,
        "parameters": len(rows),
    }


def _generator_loss(module, discriminator, spectrum, waveform, task: MossFormerGANTask) -> tuple[torch.Tensor, dict[str, float]]:
    torch.manual_seed(19)
    output = module(spectrum)
    pred_real = output[0].permute(0, 1, 3, 2)
    pred_imag = output[1].permute(0, 1, 3, 2)
    magnitude = torch.sqrt(pred_real ** 2 + pred_imag ** 2)
    label_spec = _compress(_stft(waveform, task.n_fft, task.hop, task.win))
    label_real = label_spec[:, 0, :, :].unsqueeze(1)
    label_imag = label_spec[:, 1, :, :].unsqueeze(1)
    label_mag = torch.sqrt(label_real ** 2 + label_imag ** 2)
    gan = F.mse_loss(
        discriminator(label_mag, magnitude).flatten(),
        torch.ones(waveform.shape[0], device=waveform.device),
    )
    loss_mag = F.mse_loss(magnitude, label_mag)
    loss_ri = F.mse_loss(pred_real, label_real) + F.mse_loss(pred_imag, label_imag)
    audio = _istft(_uncompress(label_real, pred_imag).squeeze(1), task.n_fft, task.hop, task.win, waveform.shape[-1])
    length = min(int(audio.shape[-1]), int(waveform.shape[-1]))
    time_loss = (audio[..., :length] - waveform[..., :length]).abs().mean()
    total = _RI * loss_ri + _MAG * loss_mag + _TIME * time_loss + _GAN * gan
    parts = {
        "ri": float(loss_ri.detach()),
        "magnitude": float(loss_mag.detach()),
        "time": float(time_loss.detach()),
        "gan": float(gan.detach()),
        "total": float(total.detach()),
    }
    return total, parts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(7)
    task = MossFormerGANTask(_task_config(), device)
    held_g = {key: value.detach().clone() for key, value in task.generator.state_dict().items()}
    held_d = {key: value.detach().clone() for key, value in task.discriminator.state_dict().items()}
    official = SyncANet(num_channel=64, num_features=201).to(device)
    official.load_state_dict(held_g, strict=True)
    noisy, _clean = _tone_pair(0)
    official_spec = _official_spec(noisy, _args())
    task_spec = _task_spec(task, noisy)
    scale = torch.sqrt(noisy.shape[-1] / (noisy.square().sum(dim=-1) + 1e-6)).unsqueeze(-1)
    waveform = noisy * scale
    report: dict = {
        "device": "cpu",
        "samples": int(noisy.shape[-1]),
        "batch_size": 1,
        "official_function": "solver preprocessing plus loss_mossformergan_se_16k coefficients 0.1/0.9/0.2/0.05",
        "spectrum_max_abs": _max_pair(official_spec, task_spec),
        "spectrum_shape": list(official_spec.shape),
    }
    for mode in ("eval", "train"):
        official.load_state_dict(held_g, strict=False)
        task.generator.load_state_dict(held_g, strict=True)
        official.train(mode == "train")
        task.generator.train(mode == "train")
        official_real, official_leaves = _leaf_outputs(official, official_spec, 19)
        task_real, task_leaves = _leaf_outputs(task.generator, official_spec, 19)
        report[mode] = {
            "same_official_spectrum_output_max_abs": _max_pair(official_real, task_real),
            "leaves": _first_leaf(official_leaves, task_leaves),
        }
    official.load_state_dict(held_g, strict=False)
    task.generator.load_state_dict(held_g, strict=True)
    official.train()
    task.generator.train()
    disc_official = copy.deepcopy(task.discriminator).train()
    disc_task = copy.deepcopy(task.discriminator).train()
    disc_official.load_state_dict(held_d, strict=True)
    disc_task.load_state_dict(held_d, strict=True)
    official_loss, official_parts = _generator_loss(official, disc_official, official_spec, waveform, task)
    task_loss, task_parts = _generator_loss(task.generator, disc_task, task_spec, waveform, task)
    official_loss.backward()
    task_loss.backward()
    report["gradients_before_step"] = _grad_report(official, task.generator)
    opt_official = torch.optim.AdamW(official.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=WD)
    opt_task = torch.optim.AdamW(task.generator.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=WD)
    opt_official.step()
    opt_task.step()
    named = dict(task.generator.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, left in official.named_parameters():
        delta = float((left.detach() - named[name].detach()).abs().max())
        if delta > worst:
            worst = delta
            worst_name = name
    report["weight_after_one_generator_step"] = {"max_abs": worst, "worst": worst_name}
    report["generator_loss"] = {
        "official": official_parts,
        "task": task_parts,
        "total_gap": abs(official_parts["total"] - task_parts["total"]),
        "preprocessing": "each side uses its own spectrum; the label STFT is centered_stft on both so a gap is the noisy analysis or the generator",
    }
    path = OUT / "moss_first_grad.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    summary = {
        "spectrum": report["spectrum_max_abs"],
        "eval_output": report["eval"]["same_official_spectrum_output_max_abs"],
        "eval_first": report["eval"]["leaves"]["first_mismatch"],
        "train_output": report["train"]["same_official_spectrum_output_max_abs"],
        "train_first": report["train"]["leaves"]["first_mismatch"],
        "loss_gap": report["generator_loss"]["total_gap"],
        "loss": {"official": official_parts, "task": task_parts},
        "grad_worst": report["gradients_before_step"]["worst_predicted_weight"][:3],
        "weight_after_step": report["weight_after_one_generator_step"],
        "meaningful": {
            "name": report["gradients_before_step"]["meaningful_name"],
            "grad_abs": report["gradients_before_step"]["meaningful_worst"],
        },
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
