"""CPU / Gloo entry points for the MossFormerGAN tests.

``torchrun --nproc_per_node=2`` starts this module. The directory and the
mode are arguments. Each rank writes its own JSON before exiting.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from src.experiments.api import build_experiment, load_experiment
from src.experiments.errors import RuntimeFailure
from src.experiments.runtime import PreparePolicy, ProcessRuntime


_SEQUENCE_CALLS = 0


def _pesq_sequence(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    """Success, failure, success, repeating. Both ranks share the call pattern."""

    global _SEQUENCE_CALLS
    del enhanced, sample_rate
    _SEQUENCE_CALLS += 1
    if _SEQUENCE_CALLS % 3 == 2:
        return None
    return torch.full((clean.shape[0],), 0.4)


def _pesq_split(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    del enhanced, sample_rate
    if os.environ.get("RANK", "0") == "0":
        return None
    return torch.full((clean.shape[0],), 0.25)


def _pesq_constant(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor:
    del enhanced, sample_rate
    return torch.full((clean.shape[0],), 0.4)


def _install_pesq(mode: str) -> None:
    import src.experiments.calc.mossformergan as mossformer

    if mode == "skip":
        mossformer.metric_targets = _pesq_split
    elif mode == "sequence":
        mossformer.metric_targets = _pesq_sequence
    elif mode == "run":
        mossformer.metric_targets = _pesq_constant
    else:
        raise RuntimeFailure(f"unknown train mode {mode}")


def _digest(module: torch.nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for tensor in list(module.parameters()) + list(module.buffers()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train(directory: Path, mode: str, resume: Path | None) -> None:
    _install_pesq(mode)
    experiment = build_experiment(load_experiment(directory / "experiment.yaml"))
    phase = os.environ.get("AUDIO_DB_MOSS_FAIL_PHASE")
    if phase:
        experiment.get("trainer").fail_phase = phase
    result = experiment.run(resume=resume)
    trainer = experiment.get("trainer")
    runtime = experiment._graph.runtime
    generator = experiment.get("generator").state_module
    discriminator = experiment.get("discriminator").state_module
    payload = {
        "accu_count": int(trainer._accu_count),
        "average_calls": int(trainer._average_calls),
        "counters": dict(trainer.counters),
        "data_epoch": int(trainer.data.epoch),
        "data_update_index": int(trainer.data.update_index),
        "device": str(runtime.device),
        "discriminator_sha256": _digest(discriminator),
        "generator_sha256": _digest(generator),
        "halt_reason": trainer.halt_reason,
        "logical_step": int(result.logical_step),
        "optimizer_updates": dict(result.optimizer_updates),
        "rank": int(runtime.rank),
        "stop_reason": result.stop_reason,
        "stop_training": bool(trainer.stop_training),
        "sync_batchnorm": sum(isinstance(module, torch.nn.SyncBatchNorm) for module in discriminator.modules()),
        "torch": torch.__version__,
    }
    path = directory / f"worker-rank{payload['rank']}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _increment(directory: Path) -> None:
    """Show per-micro-batch clipping is not the same as one clip at the end."""

    runtime = ProcessRuntime(device="cpu", precision="fp32", timeout_seconds=60)
    try:
        torch.manual_seed(0)
        each = torch.nn.Linear(4, 1, bias=False)
        once = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            weight = torch.tensor([[0.4, -0.2, 0.7, 0.1]])
            each.weight.copy_(weight)
            once.weight.copy_(weight)
        policy = PreparePolicy(
            trainable=True,
            distribute="ddp",
            find_unused_parameters=True,
            broadcast_buffers=True,
        )
        each_prepared = runtime.prepare_model(each, policy)
        once_prepared = runtime.prepare_model(once, policy)
        patterns = (
            torch.tensor([8.0, -1.0, 0.5, 2.0]),
            torch.tensor([-1.0, 9.0, -4.0, 0.2]),
        )
        scale = 1.0 if runtime.rank == 0 else -0.3
        each_grad = _accumulate(runtime, each_prepared, patterns, scale, clip_each=True)
        once_grad = _accumulate(runtime, once_prepared, patterns, scale, clip_each=False)
        payload = {
            "differ": not torch.equal(each_grad, once_grad),
            "each": each_grad.tolist(),
            "once": once_grad.tolist(),
            "rank": runtime.rank,
        }
        gathered = runtime.gather_objects(payload)
        if runtime.rank == 0:
            payload["ranks_agree"] = gathered[0]["each"] == gathered[1]["each"]
        else:
            payload["ranks_agree"] = gathered[0]["each"] == gathered[1]["each"]
        path = directory / f"increment-rank{runtime.rank}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    finally:
        runtime.shutdown()


def _accumulate(runtime, prepared, patterns, scale: float, *, clip_each: bool) -> torch.Tensor:
    for parameter in prepared.parameters:
        parameter.grad = None
    for pattern in patterns:
        with runtime.sync_context([prepared], sync=False):
            loss = prepared.module(pattern * scale).sum() / 2
            baselines = {
                id(parameter): None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in prepared.parameters
            }
            runtime.backward(loss)
        runtime.average_new_gradients(prepared.parameters, baselines)
        if clip_each:
            runtime.clip_repeated(prepared.parameters, 0.5)
    if not clip_each:
        runtime.clip_repeated(prepared.parameters, 0.5)
    gradient = prepared.parameters[0].grad
    if gradient is None:
        raise RuntimeFailure("increment test produced no gradient")
    return gradient.detach().cpu().clone()


def main() -> None:
    directory = Path(sys.argv[1])
    mode = sys.argv[2]
    resume = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    if mode == "increment":
        _increment(directory)
        return
    _train(directory, mode, resume)


if __name__ == "__main__":
    main()
