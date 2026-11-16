"""One real batch before training, then the same state the run started with.

The probe runs forward, loss, and backward. It does not call the optimizer.
RNG, parameters, buffers, gradients, optimizer state, scaler bookkeeping, and
data caches created by the probe are put back so step 1 matches a run that
skipped the probe.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.errors import BuildError
from src.experiments.execution.runtime import PreparedModel


def run_batch_probe(experiment: object) -> dict[str, Any]:
    """Probe when the runtime asks for it. Restore the graph either way."""

    graph = getattr(experiment, "_graph", None)
    if graph is None:
        raise BuildError("experiment is not built")
    runtime = graph.runtime
    if not getattr(runtime, "preflight", True):
        return {"enabled": False, "status": "disabled"}
    trainer = graph.trainer
    probe = getattr(trainer, "preflight_probe", None)
    if not callable(probe):
        return {
            "enabled": True,
            "status": "unsupported",
            "trainer": type(trainer).__name__,
        }
    _attach_probe_runtime(trainer, runtime)
    snapshot = _snapshot(experiment, runtime)
    try:
        report = probe()
    finally:
        _restore(snapshot)
    if not isinstance(report, dict):
        raise BuildError("preflight probe did not return a report")
    report["enabled"] = True
    return report


def _attach_probe_runtime(trainer: object, runtime: object) -> None:
    """Give the probe the process runtime.

    A wrapper trainer stores ``runtime`` on itself and forwards unknown
    attributes to an inner trainer. The probe is the inner method, so the
    inner object needs the same runtime before it runs.
    """

    if getattr(trainer, "runtime", None) is None:
        setattr(trainer, "runtime", runtime)
    inner = getattr(trainer, "_trainer", None)
    if inner is None or inner is trainer:
        return
    if getattr(inner, "runtime", None) is None:
        setattr(inner, "runtime", getattr(trainer, "runtime", runtime))


def _snapshot(experiment: object, runtime: object) -> dict[str, Any]:
    graph = experiment._graph  # type: ignore[attr-defined]
    config = experiment.config  # type: ignore[attr-defined]
    modules: list[dict[str, Any]] = []
    seen: set[int] = set()
    for _component_id, component in graph.components.items():
        if not isinstance(component, PreparedModel):
            continue
        module = component.state_module
        if id(module) in seen:
            continue
        seen.add(id(module))
        modules.append(
            {
                "module": module,
                "execution": component.module,
                "training": bool(component.module.training),
                "state": _clone_tree(module.state_dict()),
                "grads": [
                    None if parameter.grad is None else parameter.grad.detach().clone()
                    for parameter in module.parameters()
                ],
            }
        )
    optimizers = [
        component
        for _component_id, component in graph.components.items()
        if isinstance(component, torch.optim.Optimizer)
    ]
    data_modules = []
    for component_id, spec in config.components.items():
        if spec.role != "data" or spec.ref is not None:
            continue
        data = graph.components.get(component_id)
        plans = getattr(data, "_plans", None)
        cache = getattr(data, "_cache", None)
        data_modules.append(
            {
                "data": data,
                "epoch": getattr(data, "epoch", None),
                "update_index": getattr(data, "update_index", None),
                "plan_keys": set(plans) if isinstance(plans, dict) else None,
                "cache_keys": set(cache) if isinstance(cache, dict) else None,
            }
        )
    trainer = graph.trainer
    scaler = getattr(runtime, "_scaler", None)
    finite = getattr(runtime, "_loss_finite", None)
    return {
        "modules": modules,
        "optimizers": [(item, _clone_tree(item.state_dict())) for item in optimizers],
        "data": data_modules,
        "trainer": trainer,
        "logical_step": getattr(trainer, "logical_step", None),
        "optimizer_updates": _clone_tree(getattr(trainer, "optimizer_updates", None)),
        "losses": list(getattr(trainer, "losses", [])),
        "stop_reason": getattr(trainer, "stop_reason", None),
        "runtime": runtime,
        "loss_finite": None if finite is None else finite.detach().clone(),
        "clipped": set(getattr(runtime, "_clipped", ())),
        "unscaled": set(getattr(runtime, "_unscaled", ())),
        "stepped": set(getattr(runtime, "_stepped", ())),
        "scaler": None if scaler is None else _clone_tree(scaler.state_dict()),
        "rng": capture_rank_rng(cuda=getattr(runtime, "device", torch.device("cpu")).type == "cuda"),
        "cuda": getattr(runtime, "device", torch.device("cpu")).type == "cuda",
    }


def _restore(snapshot: dict[str, Any]) -> None:
    for item in snapshot["modules"]:
        module = item["module"]
        module.load_state_dict(_clone_tree(item["state"]))
        item["execution"].train(item["training"])
        for parameter, grad in zip(module.parameters(), item["grads"], strict=True):
            parameter.grad = None if grad is None else grad.detach().clone()
    for optimizer, state in snapshot["optimizers"]:
        optimizer.load_state_dict(_clone_tree(state))
    for item in snapshot["data"]:
        data = item["data"]
        if item["epoch"] is not None and hasattr(data, "epoch"):
            data.epoch = item["epoch"]
        if item["update_index"] is not None and hasattr(data, "update_index"):
            data.update_index = item["update_index"]
        _drop_new_keys(getattr(data, "_plans", None), item["plan_keys"])
        _drop_new_keys(getattr(data, "_cache", None), item["cache_keys"])
    trainer = snapshot["trainer"]
    if snapshot["logical_step"] is not None and hasattr(trainer, "logical_step"):
        trainer.logical_step = snapshot["logical_step"]
    if snapshot["optimizer_updates"] is not None and hasattr(trainer, "optimizer_updates"):
        trainer.optimizer_updates = _clone_tree(snapshot["optimizer_updates"])
    if hasattr(trainer, "losses"):
        trainer.losses = list(snapshot["losses"])
    if snapshot["stop_reason"] is not None and hasattr(trainer, "stop_reason"):
        trainer.stop_reason = snapshot["stop_reason"]
    runtime = snapshot["runtime"]
    runtime._loss_finite = snapshot["loss_finite"]
    runtime._clipped = set(snapshot["clipped"])
    runtime._unscaled = set(snapshot["unscaled"])
    runtime._stepped = set(snapshot["stepped"])
    scaler = getattr(runtime, "_scaler", None)
    if scaler is not None and snapshot["scaler"] is not None:
        scaler.load_state_dict(_clone_tree(snapshot["scaler"]))
    restore_rank_rng(snapshot["rng"], cuda=snapshot["cuda"])


def _drop_new_keys(mapping: object, keys: set[Any] | None) -> None:
    if not isinstance(mapping, dict) or keys is None:
        return
    for key in list(mapping):
        if key not in keys:
            del mapping[key]


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value
