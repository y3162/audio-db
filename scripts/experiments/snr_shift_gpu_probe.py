"""Small single-GPU checks for the four SNR-shift trainers.

Temporary YAML and outputs stay under data/results/investigations/. The
published training YAML is not modified. This does not start the full run.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/results/investigations/snr-shift-20261114/gpu"
CONFIGS = {
    "mp_senet": ROOT / "configs/reproduction/experiments/snr_shift/mp_senet.yaml",
    "semambapp": ROOT / "configs/reproduction/experiments/snr_shift/semambapp.yaml",
    "tf_gridnet": ROOT / "configs/reproduction/experiments/snr_shift/tf_gridnet.yaml",
    "tf_locoformer": ROOT / "configs/reproduction/experiments/snr_shift/tf_locoformer.yaml",
}


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(model: str, document: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _prepare(model: str, steps: int, directory: Path, *, accumulation: int | None = None, micro_batch: int | None = None) -> Path:
    document = _load(CONFIGS[model])
    document["output"] = {"directory": str(directory / "run")}
    document["checkpoint"]["selector"]["kwargs"]["latest_every"] = 1
    trainer = document["components"]["trainer"]["kwargs"]
    trainer["stop_after_logical_steps"] = steps
    trainer["max_logical_steps"] = max(int(trainer["max_logical_steps"]), steps)
    if accumulation is not None:
        trainer["accumulation"] = accumulation
        document["components"]["batcher"]["kwargs"]["group_size"] = accumulation
    if micro_batch is not None:
        document["components"]["batcher"]["kwargs"]["micro_batch"] = micro_batch
    if "global_batch" in document["components"].get("sampler", {}).get("kwargs", {}):
        micro = document["components"]["batcher"]["kwargs"]["micro_batch"]
        group = document["components"]["batcher"]["kwargs"]["group_size"]
        document["components"]["sampler"]["kwargs"]["global_batch"] = micro * group
    return _write(model, document, directory)


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _gap(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    shared = sorted(set(left) & set(right))
    if not shared:
        return float("nan")
    return max(float((left[key] - right[key]).abs().max()) for key in shared)


def _timed_forward(model: torch.nn.Module, batch: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = model(batch)
    torch.cuda.synchronize()
    forward = time.perf_counter() - started
    loss = torch.mean((prediction - batch) ** 2)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward = time.perf_counter() - started
    return forward, backward, loss.detach()


def _validate(experiment: Any) -> dict[str, Any]:
    trainer = experiment.get("trainer")
    runtime = experiment._graph.runtime
    data = experiment.get("data")
    cursor = (int(data.epoch), int(data.update_index))
    started = time.perf_counter()
    summary = trainer._validate(runtime, 0, "trainer")
    compact = {}
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, (int, float, str)) or value is None:
                compact[key] = value
            elif isinstance(value, dict):
                compact[key] = {
                    item: item_value
                    for item, item_value in value.items()
                    if isinstance(item_value, (int, float, str)) or item_value is None
                }
    moved = (int(data.epoch), int(data.update_index))
    if moved != cursor:
        raise RuntimeError(f"validation moved the cursor {cursor} -> {moved}")
    planned_chunks = None
    unique = None
    if type(data.sampler).__name__ == "ChunkSampler":
        ranks = data.sampler.validation_ranks(0, 1)[0]
        planned_chunks = len(ranks)
        unique = len({sample.case_id for sample in ranks})
    return {
        "summary_keys": sorted(str(key) for key in summary) if isinstance(summary, dict) else None,
        "summary": compact,
        "loss": summary.get("loss") if isinstance(summary, dict) else None,
        "weight": summary.get("weight") if isinstance(summary, dict) else None,
        "planned_chunks": planned_chunks,
        "unique_utterances": unique,
        "seconds": time.perf_counter() - started,
        "cursor_unchanged": True,
    }


def _one_run(model: str, directory: Path, steps: int) -> dict[str, Any]:
    from src.experiments.api import build_experiment, load_experiment

    config = _prepare(model, steps, directory)
    experiment = build_experiment(load_experiment(config))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = experiment.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "stop_reason": result.stop_reason,
        "logical_steps": int(experiment.get("trainer").logical_step),
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(steps, 1),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
        "latest": (directory / "run" / "latest.pt").is_file(),
    }
    experiment.close()
    return payload


def _tree_gap(left: object, right: object) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape:
        return float((left.detach().float() - right.detach().float()).abs().max())
    if isinstance(left, dict) and isinstance(right, dict):
        gaps = [_tree_gap(left[key], right[key]) for key in left.keys() & right.keys()]
        return max(gaps) if gaps else 0.0
    return 0.0


def _resume(model: str, directory: Path) -> dict[str, Any]:
    from src.experiments.api import build_experiment, load_experiment

    resumed = directory / "resume"
    _one_run(model, resumed, 1)
    config = _prepare(model, 2, resumed)
    experiment = build_experiment(load_experiment(config))
    result = experiment.run(resume=resumed / "run")
    payload = {
        "stop_reason": result.stop_reason,
        "logical_steps": int(experiment.get("trainer").logical_step),
        "latest": (resumed / "run" / "latest.pt").is_file(),
    }
    experiment.close()
    straight = directory / "straight"
    _one_run(model, straight, 2)
    left = torch.load(resumed / "run" / "latest.pt", map_location="cpu", weights_only=False)
    right = torch.load(straight / "run" / "latest.pt", map_location="cpu", weights_only=False)
    payload["checkpoint_logical_step_continued"] = left.get("logical_step")
    payload["checkpoint_logical_step_straight"] = right.get("logical_step")
    payload["same_logical_step"] = left.get("logical_step") == right.get("logical_step")
    payload["state_abs_max"] = _tree_gap(left, right)
    return payload


def _cat_batch(batches: list[Any]) -> Any:
    from src.experiments.data.sample import Batch

    return Batch(
        inputs=torch.cat([batch.inputs for batch in batches], 0),
        targets=torch.cat([batch.targets for batch in batches], 0),
        sample_id=",".join(batch.sample_id for batch in batches),
        realization_id=",".join(batch.realization_id for batch in batches),
        lengths=None if batches[0].lengths is None else torch.cat([batch.lengths for batch in batches], 0),
        sample_ids=tuple(item for batch in batches for item in batch.sample_ids),
        realization_ids=tuple(item for batch in batches for item in batch.realization_ids),
        teacher_kind=batches[0].teacher_kind,
    )


def _module_blobs(experiment: Any) -> dict[str, dict[str, torch.Tensor]]:
    blobs = {}
    components = experiment._graph.components
    names = tuple(getattr(components, "_objects", {}))
    for name in names:
        component = components.get(name) if hasattr(components, "get") else components[name]
        module = getattr(component, "state_module", None)
        optimizer = component if hasattr(component, "state_dict") and component.__class__.__module__.startswith("torch.optim") else None
        if isinstance(module, torch.nn.Module):
            blobs[name] = copy.deepcopy(module.state_dict())
        elif optimizer is not None and hasattr(component, "param_groups"):
            blobs[name] = copy.deepcopy(component.state_dict())
    return blobs


def _restore_blobs(experiment: Any, blobs: dict[str, dict[str, torch.Tensor]]) -> None:
    for name, state in blobs.items():
        component = experiment.get(name)
        module = getattr(component, "state_module", None)
        if isinstance(module, torch.nn.Module):
            module.load_state_dict(state)
        else:
            component.load_state_dict(state)


def _official_accumulation(experiment: Any, batches: list[Any]) -> dict[str, Any]:
    trainer = experiment.get("trainer")
    runtime = experiment._graph.runtime
    original = trainer.accumulation
    baseline = _module_blobs(experiment)
    try:
        trainer.accumulation = len(batches)
        split = trainer.official_step(batches, runtime)
        split_state = _module_blobs(experiment)
        _restore_blobs(experiment, baseline)
        trainer.accumulation = 1
        whole = trainer.official_step([_cat_batch(batches)], runtime)
        whole_state = _module_blobs(experiment)
    finally:
        trainer.accumulation = original
        _restore_blobs(experiment, baseline)
    keys = sorted(set(split) & set(whole))
    numeric = {
        key: abs(float(split[key]) - float(whole[key]))
        for key in keys
        if isinstance(split[key], (int, float)) and isinstance(whole[key], (int, float))
    }
    return {
        "split": {key: split[key] for key in keys if isinstance(split[key], (int, float))},
        "whole": {key: whole[key] for key in keys if isinstance(whole[key], (int, float))},
        "abs": numeric,
        "state_abs_max": max(_tree_gap(split_state[name], whole_state[name]) for name in split_state.keys() & whole_state.keys()),
        "note": "accumulation=1 on one CUDA process uses the overlapped GAN step; accumulation>1 updates the discriminator across micro-batches first",
    }


def _accumulation(model: str, directory: Path) -> dict[str, Any]:
    from src.experiments.api import build_experiment, load_experiment

    document = _load(CONFIGS[model])
    micro = int(document["components"]["batcher"]["kwargs"]["micro_batch"])
    group = int(document["components"]["batcher"]["kwargs"]["group_size"])
    report: dict[str, Any] = {"micro_batch": micro, "accumulation": group, "effective": micro * group}
    if directory.exists():
        shutil.rmtree(directory)
    config = _prepare(model, 1, directory)
    experiment = build_experiment(load_experiment(config))
    try:
        data = experiment.get("data")
        runtime = experiment._graph.runtime
        view = data.train_epoch(0, start_update=0)
        batches = [view.batch(index)[0] for index in range(group)]
        view.close()
        module = experiment.get("model").module if "model" in experiment._graph.components else experiment.get("generator").module
        loss_fn = experiment.get("loss") if "loss" in experiment._graph.components else None
        inputs = [batch.inputs.to(runtime.device) for batch in batches]
        targets = [batch.targets.to(runtime.device) for batch in batches]
        module.train()
        baseline = {key: value.detach().clone() for key, value in module.state_dict().items()}

        def predict(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
            try:
                prediction = module(noisy)
            except TypeError:
                prediction = module(noisy, clean)
            if isinstance(prediction, tuple):
                prediction = next((item for item in prediction if torch.is_tensor(item) and item.shape == clean.shape), None)
            if not torch.is_tensor(prediction) or prediction.shape != clean.shape:
                raise TypeError("forward did not return a waveform")
            if loss_fn is not None:
                return loss_fn(prediction, clean)
            return torch.mean((prediction - clean).abs())

        def grads(chunks_in: list[torch.Tensor], chunks_out: list[torch.Tensor]) -> tuple[float, dict[str, torch.Tensor]]:
            module.load_state_dict(baseline)
            module.zero_grad(set_to_none=True)
            total = None
            for noisy, clean in zip(chunks_in, chunks_out, strict=True):
                value = predict(noisy, clean)
                (value / len(chunks_in)).backward()
                total = value.detach() if total is None else total + value.detach()
            assert total is not None
            named = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in module.named_parameters()
                if parameter.grad is not None
            }
            return float(total.cpu()) / len(chunks_in), named

        try:
            split_loss, split_grad = grads(inputs, targets)
            whole_loss, whole_grad = grads([torch.cat(inputs, 0)], [torch.cat(targets, 0)])
        except (TypeError, RuntimeError) as exc:
            report["waveform_comparison"] = f"skipped: {type(exc).__name__}: {exc}"
        else:
            report["split_loss"] = split_loss
            report["whole_loss"] = whole_loss
            report["loss_abs"] = abs(split_loss - whole_loss)
            report["grad_abs_max"] = _gap(split_grad, whole_grad)
            report["loss_kind"] = "official_loss" if loss_fn is not None else "waveform_l1"
        try:
            forward, backward, _loss = _timed_forward(module, inputs[0])
            report["forward_seconds"] = forward
            report["backward_seconds"] = backward
        except TypeError as exc:
            report["timing_error"] = str(exc)
        trainer = experiment.get("trainer")
        if hasattr(trainer, "official_step"):
            report["official_step"] = _official_accumulation(experiment, batches)
    finally:
        experiment.close()
    wide = directory / "accum1"
    if wide.exists():
        shutil.rmtree(wide)
    try:
        config = _prepare(model, 1, wide, accumulation=1, micro_batch=micro * group)
        experiment = build_experiment(load_experiment(config))
        torch.cuda.reset_peak_memory_stats()
        result = experiment.run()
        report["accumulation_1"] = {
            "stop_reason": result.stop_reason,
            "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
            "oom": False,
        }
        experiment.close()
    except torch.cuda.OutOfMemoryError as exc:
        report["accumulation_1"] = {"oom": True, "error": str(exc)}
    return report


def main() -> None:
    print(f"executable {sys.executable}", flush=True)
    model = os.environ["SNR_SHIFT_PROBE_MODEL"]
    part = os.environ.get("SNR_SHIFT_PROBE_PART", "all")
    OUT.mkdir(parents=True, exist_ok=True)
    directory = OUT / model
    if part == "all" and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    existing = json.loads(report_path.read_text(encoding="utf-8")) if part != "all" and report_path.is_file() else {}
    report: dict[str, Any] = {
        "model": model,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    try:
        if part in {"all", "train"}:
            report["train"] = _one_run(model, directory / "train", 2)
            report["resume"] = _resume(model, directory)
        if part in {"all", "accum"}:
            report["accumulation"] = _accumulation(model, directory / "grad")
        if part in {"all", "valid"}:
            from src.experiments.api import build_experiment, load_experiment

            config = _prepare(model, 1, directory / "valid")
            experiment = build_experiment(load_experiment(config))
            try:
                report["validation"] = _validate(experiment)
            finally:
                experiment.close()
        report["ok"] = True
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["trace"] = traceback.format_exc()
    if report.get("ok"):
        existing.pop("error", None)
        existing.pop("trace", None)
    existing.update(report)
    report = existing
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "trace"}, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(report["error"])


if __name__ == "__main__":
    main()
