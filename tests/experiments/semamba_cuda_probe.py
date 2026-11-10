"""One CUDA SEMamba++ GAN step against the current task.

This file is not named ``test_*.py``, so the CPU regression does not collect
it. ``tests/experiments/test_cuda_amp.py`` launches it when a GPU is free.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from src.experiments.api import build_experiment, load_experiment
from src.experiments.data.sample import Batch
from src.experiments.official.semambapp.reference import SEMambaTask
from tests.experiments.test_semambapp import _ARCH, _adam_gap, _state_gap, _tensor_gap, _yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "fused", "both"), required=True)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--state", type=int, default=4)
    parser.add_argument("--conv", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible")
    torch.cuda.set_device(0)
    reports = []
    saved = []
    own_context = False
    if args.mode in {"reference", "both"}:
        report, state = _match("reference", args, allow_busy=own_context)
        own_context = True
        reports.append(report)
        saved.append(state)
    if args.mode in {"fused", "both"}:
        report, state = _match("fused", args, allow_busy=own_context)
        reports.append(report)
        saved.append(state)
    if args.mode == "both" and len(saved) == 2:
        reports.append(_cross(reports[0], reports[1], saved[0], saved[1]))
    args.out.write_text(json.dumps({"runs": reports}, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))


def _match(backend: str, args: argparse.Namespace, *, allow_busy: bool) -> dict[str, object]:
    directory = args.directory / backend
    config = _yaml(
        directory,
        steps=1,
        count=1,
        length=args.length,
        scan_backend=backend,
        device="cuda",
        validation_interval=0,
        deterministic=args.deterministic,
    )
    # The narrow architecture is the probe. The declaration YAML stays official width.
    text = config.read_text(encoding="utf-8")
    text = text.replace("hid_feature: 8", f"hid_feature: {args.width}")
    text = text.replace("num_tfmamba: 1", f"num_tfmamba: {args.layers}")
    text = text.replace("d_state: 4", f"d_state: {args.state}")
    text = text.replace("d_conv: 2", f"d_conv: {args.conv}")
    if allow_busy:
        text = text.replace(
            "    precision: fp32\n",
            "    precision: fp32\n    allow_busy_gpu: true\n",
            1,
        )
    config.write_text(text, encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    experiment = build_experiment(load_experiment(config))
    task = None
    try:
        task = _task(backend, args)
        _require_same_init(task, experiment)
        generator = torch.Generator().manual_seed(19)
        noisy = torch.randn(1, args.length, generator=generator)
        clean = torch.randn(1, args.length, generator=generator)
        task.capture_update = True
        task.capture_gradients = True
        step_started = time.perf_counter()
        old = task.train_update([{"clean": clean.clone(), "noisy": noisy.clone()}], 1)
        trainer = experiment.get("trainer")
        trainer.capture_gradients = True
        new = trainer.official_step(
            [Batch(inputs=noisy.clone(), targets=clean.clone(), sample_id="s", realization_id="r")],
            experiment._graph.runtime,
        )
        step_seconds = time.perf_counter() - step_started
        grad_gap = max(
            _tensor_gap(value, trainer.last_generator_grads[key])
            for key, value in task.last_capture["generator_grad"].items()
        )
        disc_grad_gap = max(
            _tensor_gap(value, trainer.last_discriminator_grads[key])
            for key, value in task.last_capture["discriminator_grad"].items()
        )
        state = {
            "cqt": {key: value.detach().cpu() for key, value in experiment.get("cqt").state_module.discriminator.state_dict().items()},
            "generator": {
                key: value.detach().cpu()
                for key, value in experiment.get("generator").state_module.generator.state_dict().items()
            },
            "mrd": {key: value.detach().cpu() for key, value in experiment.get("mrd").state_module.discriminator.state_dict().items()},
        }
        return {
            "backend": backend,
            "build_seconds": time.perf_counter() - started - step_seconds,
            "device": torch.cuda.get_device_name(0),
            "discriminator_grad_max_abs": disc_grad_gap,
            "discriminator_state_max_abs": max(
                _state_gap(task.discriminator_mrd, experiment.get("mrd").state_module.discriminator),
                _state_gap(task.discriminator_band, experiment.get("cqt").state_module.discriminator),
            ),
            "finite": all(torch.isfinite(torch.tensor(float(new[key]))) for key in ("generator", "discriminator")),
            "generator_forwards": new["generator_forwards"],
            "generator_grad_max_abs": grad_gap,
            "generator_state_max_abs": _state_gap(task.generator, experiment.get("generator").state_module.generator),
            "layers": args.layers,
            "length": args.length,
            "loss_max_abs": max(abs(float(old[key]) - float(new[key])) for key in old if isinstance(old[key], float)),
            "optim_g_max_abs": _adam_gap(
                task.optim_g,
                task.generator.parameters(),
                experiment.get("optim_g"),
                experiment.get("generator").state_module.generator.parameters(),
            ),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "scan_configured": new["scan_configured"],
            "scan_executed": new["scan_executed"],
            "step_seconds": step_seconds,
            "width": args.width,
        }, state
    finally:
        experiment.close()
        del task
        del experiment
        gc.collect()
        torch.cuda.empty_cache()


def _task(backend: str, args: argparse.Namespace) -> SEMambaTask:
    architecture = dict(_ARCH)
    architecture.update(
        {
            "d_conv": args.conv,
            "d_state": args.state,
            "hid_feature": args.width,
            "num_tfmamba": args.layers,
            "scan_backend": backend,
        }
    )
    return SEMambaTask(
        {
            "loss": "official",
            "model": {
                "architecture": architecture,
                "name": "semambapp",
                "sampling_rate": 16000,
                "stft": {"hop": 100, "win": 400},
                "variant": "semambapp",
            },
            "optimizer": {
                "betas": [0.8, 0.99],
                "lr": 0.0002,
                "name": "adamw",
                "weight_decay": 0.01,
            },
            "scheduler": {"gamma": 0.99, "name": "exponential"},
            "seed": 1234,
        },
        torch.device("cuda"),
    )


def _require_same_init(task: SEMambaTask, experiment) -> None:
    if _state_gap(task.generator, experiment.get("generator").state_module.generator) != 0.0:
        raise RuntimeError("CUDA initial generator state does not match the task")


def _cross(
    reference: dict[str, object],
    fused: dict[str, object],
    reference_state: dict[str, dict[str, torch.Tensor]],
    fused_state: dict[str, dict[str, torch.Tensor]],
) -> dict[str, object]:
    gaps = {
        name: max(_tensor_gap(reference_state[name][key], fused_state[name][key]) for key in reference_state[name])
        for name in reference_state
    }
    return {
        "backend": "reference_versus_fused",
        "cqt_state_max_abs": gaps["cqt"],
        "fused_executed": fused["scan_executed"],
        "fused_peak_allocated_bytes": fused["peak_allocated_bytes"],
        "fused_step_seconds": fused["step_seconds"],
        "generator_state_max_abs": gaps["generator"],
        "mrd_state_max_abs": gaps["mrd"],
        "note": "same seed and input; bit equality is not required",
        "reference_executed": reference["scan_executed"],
        "reference_peak_allocated_bytes": reference["peak_allocated_bytes"],
        "reference_step_seconds": reference["step_seconds"],
    }


if __name__ == "__main__":
    main()
