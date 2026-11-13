"""CPU Gloo worker for Phase 3-E data acceptance.

``torchrun --nproc_per_node=2`` starts this module. It is not a new DataModule.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.data.chunk import ChunkProtocol, ChunkSampler
from src.experiments.data.fixed_pair import (
    FixedPairProtocol,
    MicroBatcher,
    OrderSampler,
    SnapshotDataModule,
    SnapshotSource,
)
from src.experiments.data.online import Dfn3IndexSampler, Dfn3Protocol, SemambaProtocol
from src.experiments.runtime import ProcessRuntime
from src.experiments.data.render.chunking import plan_chunk_epoch
from src.experiments.data.mix.online_dfn3 import plan_online_epoch
from src.experiments.data.mix.online_semamba import plan_semamba_epoch
from src.experiments.data.sampling.order import rank_order, resolve_order_policy
from src.experiments.data.sampling.planner import chunk_case_length, validation_refs
from src.experiments.data.protocols.online_dfn3 import speech_key_positions


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: distributed_data_worker.py SPEC.json", file=sys.stderr)
        return 2
    spec = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    runtime = ProcessRuntime(
        device="cpu",
        precision="fp32",
        world_size=2,
        timeout_seconds=float(spec.get("timeout_seconds", 30)),
    )
    error: str | None = None
    report: dict[str, Any] = {}
    try:
        mode = str(spec["mode"])
        if mode == "probe":
            report = {"pipelines": [_probe_pipeline(runtime, item) for item in spec["pipelines"]]}
        elif mode == "resume-waves":
            report = {"waves": [_resume_wave(item) for item in spec["checkpoints"]]}
        else:
            raise RuntimeError(f"unknown worker mode {mode}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    out = Path(spec["report_dir"]) / f"rank{runtime.rank}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"error": error, "report": report}, sort_keys=True), encoding="utf-8")
    try:
        if runtime.world_size > 1:
            errors = runtime.gather_objects(error)
            if error is not None:
                print(error, file=sys.stderr)
                return 1
            remote = next((item for item in errors if item), None)
            if remote:
                print(str(remote), file=sys.stderr)
                return 1
            runtime.barrier()
    finally:
        runtime.shutdown()
    return 0


def _probe_pipeline(runtime: ProcessRuntime, item: dict[str, Any]) -> dict[str, Any]:
    primary = _module(item, runtime, load_workers=0)
    try:
        epoch = 0
        view = primary.train_epoch(epoch, start_update=0)
        groups = _groups(primary, epoch)
        if view.updates != len(groups):
            raise RuntimeError(f"{item['name']} update count {view.updates} != {len(groups)}")
        if view.updates < 1:
            raise RuntimeError(f"{item['name']} produced no micro-batches")
        sample_ids = [sample.sample_id for group in groups for sample in group]
        realization_ids = [sample.realization_id for group in groups for sample in group]
        batch_sizes = [len(group) for group in groups]
        _assert_plan(primary, item, groups)
        waves = _worker_waves(item, runtime, epoch)
        cursor = primary.update_index
        first = view.batch(0)[0]
        if primary.update_index != cursor:
            raise RuntimeError(f"{item['name']} prefetch advanced the data cursor")
        validation = _validation(primary, item)
        plan_hash = primary.plan_hash
        hashes = runtime.gather_objects(plan_hash)
        if any(value != plan_hash for value in hashes):
            raise RuntimeError(f"{item['name']} plan hash differs across ranks")
        counts = runtime.gather_objects(view.updates)
        if any(value != view.updates for value in counts):
            raise RuntimeError(f"{item['name']} micro-batch counts differ across ranks")
        identities = runtime.gather_objects(primary.identity()["dataset_id"])
        if any(value != identities[0] for value in identities):
            raise RuntimeError(f"{item['name']} dataset identity differs across ranks")
        worlds = runtime.gather_objects(primary.world_size)
        if any(value != 2 for value in worlds):
            raise RuntimeError(f"{item['name']} world size is not 2")
        return {
            "batch_sizes": batch_sizes,
            "dataset_id": identities[0],
            "name": item["name"],
            "plan_hash": plan_hash,
            "policy": primary.sampler.policy(primary.world_size),
            "rank": runtime.rank,
            "realization_ids": realization_ids,
            "sample_ids": sample_ids,
            "updates": view.updates,
            "validation": validation,
            "wave_digest": _digest(first.inputs),
            "worker_max_abs": waves,
            "world_size": primary.world_size,
        }
    finally:
        primary.close()


def _worker_waves(item: dict[str, Any], runtime: ProcessRuntime, epoch: int) -> float:
    if not item.get("render", True):
        return 0.0
    modules = [_module(item, runtime, load_workers=count) for count in (0, 1, 2)]
    try:
        batches = []
        hashes = []
        for module in modules:
            if module.plan_hash != modules[0].plan_hash:
                raise RuntimeError(f"{item['name']} plan hash changed with load_workers")
            batch = module.train_epoch(epoch, start_update=0).batch(0)[0]
            if module.update_index != 0:
                raise RuntimeError(f"{item['name']} worker render advanced the cursor")
            batches.append(batch)
            hashes.append(module.plan_hash)
        worst = 0.0
        for batch in batches[1:]:
            worst = max(worst, _pair_max_abs(batches[0], batch, item["name"]))
        if worst != 0.0:
            raise RuntimeError(f"{item['name']} worker waveforms differ by {worst}")
        if len(set(hashes)) != 1:
            raise RuntimeError(f"{item['name']} plan hash depends on load_workers")
        return worst
    finally:
        for module in modules:
            module.close()


def _validation(module: SnapshotDataModule, item: dict[str, Any]) -> dict[str, Any]:
    rows = []
    realizations = []
    sizes = []
    for batch in module.validation_epoch(0):
        rows.append(list(batch.sample_ids))
        realizations.append(list(batch.realization_ids))
        sizes.append(int(batch.inputs.shape[0]))
    second_rows = []
    for batch in module.validation_epoch(1):
        second_rows.append(list(batch.sample_ids))
    _assert_validation(module, item, rows, realizations)
    wave_delta = None
    if item.get("render", True) and item["kind"] in {"dfn3", "semamba"} and module.rank == 0:
        wave_delta = _same_index_wave(module, 0)
        if wave_delta != 0.0:
            raise RuntimeError(f"{item['name']} validation waveform changed with epoch")
    elif rows and item.get("render", True) and (item["kind"] == "chunk" or item.get("validation_wave", False)):
        first = next(iter(module.validation_epoch(0)))
        later = next(iter(module.validation_epoch(1)))
        wave_delta = _pair_max_abs(
            first,
            later,
            item["name"],
            require_realization=item["kind"] != "fixed",
        )
        if wave_delta != 0.0:
            raise RuntimeError(f"{item['name']} validation waveform changed with epoch")
    return {
        "batch_sizes": sizes,
        "epoch0": rows,
        "epoch1": second_rows,
        "wave_max_abs": wave_delta,
    }


def _same_index_wave(module: SnapshotDataModule, index: int) -> float:
    left = _validation_sample(module, 0, index)
    right = _validation_sample(module, 1, index)
    left_wave = module.protocol.render(left)
    right_wave = module.protocol.render(right)
    worst = 0.0
    for lhs, rhs in zip(left_wave[:2], right_wave[:2]):
        worst = max(worst, float(np.max(np.abs(np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)))))
    return worst


def _validation_sample(module: SnapshotDataModule, epoch: int, index: int) -> Any:
    ranks = module.sampler.validation_ranks(epoch, module.world_size)  # type: ignore[attr-defined]
    for sample in ranks[module.rank]:
        if int(sample.index) == index:
            return sample
    raise RuntimeError(f"validation epoch {epoch} has no sample index {index}")


def _assert_validation(
    module: SnapshotDataModule,
    item: dict[str, Any],
    rows: list[list[str]],
    realizations: list[list[str]],
) -> None:
    found = [sample_id for batch in rows for sample_id in batch]
    found_realizations = [value for batch in realizations for value in batch]
    if item["kind"] == "fixed":
        from src.experiments.data.fixed_pair import validation_scope

        policy = module.sampler.policy(module.world_size)
        if validation_scope(policy) == "rank0" and module.rank != 0:
            if found:
                raise RuntimeError("rank-0 validation leaked onto a non-zero rank")
            return
        raw = validation_refs(
            module.source.store,
            seed=int(item["seed"]),
            epoch=0,
            world_size=module.world_size,
            policy=policy,
            rank=None if validation_scope(policy) == "rank0" else module.rank,
        )
        expected = [str(sample.case_id) for sample in raw]
        if found != expected:
            raise RuntimeError(f"{item['name']} validation ids do not match validation_refs")
        return
    if item["kind"] == "chunk":
        case_ids = module.source.case_ids("valid.cases")
        samples = [(case_id, chunk_case_length(module.source.store, case_id)) for case_id in case_ids]
        plan = plan_chunk_epoch(
            samples,
            espnet_epoch=1,
            seed=int(item["seed"]),
            world_size=module.world_size,
            num_samples=None,
            chunk_length=int(item["chunk_length"]),
            global_batch=int(item["micro_batch"]) * module.world_size,
            num_cache_chunks=int(item["num_cache_chunks"]),
            shuffle=False,
            shift_ratio=0.5,
        )
        expected = [str(chunk.case_id) for batch in plan.rank_batches[module.rank] for chunk in batch]
        expected_realizations = [
            f"{chunk.case_id}:{int(chunk.start)}:{int(chunk.stop)}"
            for batch in plan.rank_batches[module.rank]
            for chunk in batch
        ]
        if found != expected or found_realizations != expected_realizations:
            raise RuntimeError(f"{item['name']} validation spans do not match plan_chunk_epoch")
        return
    if item["kind"] in {"dfn3", "semamba"} and module.rank != 0 and found:
        raise RuntimeError(f"{item['name']} validation leaked onto a non-zero rank")


def _assert_plan(module: SnapshotDataModule, item: dict[str, Any], groups: tuple[tuple[Any, ...], ...]) -> None:
    kind = item["kind"]
    if kind == "fixed":
        case_ids = module.source.case_ids("train.cases")
        policy = resolve_order_policy(module.source.dataset()["rule_json"]["order"], module.world_size)
        expected = rank_order(
            policy,
            len(case_ids),
            seed=int(item["seed"]),
            epoch=0,
            world_size=module.world_size,
            rank=module.rank,
        )
        sequence = module.sampler.sequence(0, module.rank, module.world_size)
        found = [sample.case_id for sample in sequence]
        wanted = [case_ids[position] for position in expected]
        if found != wanted:
            raise RuntimeError(f"{item['name']} rank order does not match rank_order")
        return
    if kind == "chunk":
        case_ids = module.source.case_ids("train.cases")
        samples = [(case_id, chunk_case_length(module.source.store, case_id)) for case_id in case_ids]
        plan = plan_chunk_epoch(
            samples,
            espnet_epoch=1,
            seed=int(item["seed"]),
            world_size=module.world_size,
            num_samples=int(item["num_samples_per_epoch"]),
            chunk_length=int(item["chunk_length"]),
            global_batch=int(item["micro_batch"]) * module.world_size,
            num_cache_chunks=int(item["num_cache_chunks"]),
            shuffle=True,
            shift_ratio=0.5,
        )
        expected = [
            (str(chunk.case_id), int(chunk.start), int(chunk.stop))
            for batch in plan.rank_batches[module.rank]
            for chunk in batch
        ]
        found = [
            (str(sample.case_id), int(sample.chunk.start), int(sample.chunk.stop))
            for group in groups
            for sample in group
            if sample.chunk is not None
        ]
        if found != expected:
            raise RuntimeError(f"{item['name']} chunk spans do not match plan_chunk_epoch")
        if plan.stats["iterator_stop"] != "min_rank_batches":
            raise RuntimeError(f"{item['name']} iterator_stop changed")
        return
    if kind == "dfn3":
        speech = module.source.material_ids("train.speech")
        count = len(
            speech_key_positions(
                len(speech),
                float(item["factor"]),
                protocol_id=str(module.source.dataset()["protocol_id"]),
                snapshot_hash=module.source.snapshot_hash,
                seed=int(item["seed"]),
                split="train",
                epoch=0,
            )
        )
        plan = plan_online_epoch(
            split="train",
            epoch=0,
            count=count,
            seed=int(item["seed"]),
            world_size=module.world_size,
            micro_batch=int(item["micro_batch"]),
            protocol_id=str(module.source.dataset()["protocol_id"]),
            snapshot_hash=module.source.snapshot_hash,
            case_ids=None,
        )
        expected = [
            int(sample.index)
            for group in plan.rank_groups[module.rank]
            for micro in group.micro_batches
            for sample in micro.samples
        ]
        found = [int(sample.index) for group in groups for sample in group]
        if found != expected:
            raise RuntimeError(f"{item['name']} indexes do not match plan_online_epoch")
        return
    if kind == "semamba":
        count = len(module.source.material_ids("train.speech"))
        plan = plan_semamba_epoch(
            split="train",
            epoch=0,
            count=count,
            seed=int(item["seed"]),
            world_size=module.world_size,
            micro_batch=int(item["micro_batch"]),
            accumulation=int(item["group_size"]),
            protocol_id=str(module.source.dataset()["protocol_id"]),
            snapshot_hash=module.source.snapshot_hash,
        )
        expected = [
            int(sample.index)
            for group in plan.rank_groups[module.rank]
            for micro in group.micro_batches
            for sample in micro.samples
        ]
        found = [int(sample.index) for group in groups for sample in group]
        if found != expected:
            raise RuntimeError(f"{item['name']} indexes do not match plan_semamba_epoch")
        return
    raise RuntimeError(f"unknown pipeline kind {kind}")


def _groups(module: SnapshotDataModule, epoch: int) -> tuple[tuple[Any, ...], ...]:
    module._bind_sampler()
    return module.batcher.groups(module.sampler.sequence(epoch, module.rank, module.world_size))


def _module(item: dict[str, Any], runtime: ProcessRuntime, *, load_workers: int) -> SnapshotDataModule:
    source = SnapshotSource(
        str(item["selector"]),
        item["root"],
        "train",
        derived=item["derived"],
        corpora=item["corpora"],
        generated=item["generated"],
    )
    kind = item["kind"]
    seed = int(item["seed"])
    if kind == "fixed":
        from tests.experiments.pipeline_args import fixed_protocol, order_sampler

        protocol: object = fixed_protocol(source)
        sampler: object = order_sampler(source, protocol, seed=seed)
    elif kind == "chunk":
        protocol = ChunkProtocol(source)
        sampler = ChunkSampler(
            source,
            seed=seed,
            num_samples_per_epoch=int(item["num_samples_per_epoch"]),
            chunk_length=int(item["chunk_length"]),
            num_cache_chunks=int(item["num_cache_chunks"]),
            shuffle=True,
        )
    elif kind == "dfn3":
        protocol = Dfn3Protocol(source)
        sampler = Dfn3IndexSampler(source, seed=seed)
    elif kind == "semamba":
        from tests.experiments.pipeline_args import order_sampler

        protocol = SemambaProtocol(source)
        sampler = order_sampler(source, protocol, seed=seed)
    else:
        raise RuntimeError(f"unknown pipeline kind {kind}")
    module = SnapshotDataModule(
        source,
        sampler,  # type: ignore[arg-type]
        protocol,  # type: ignore[arg-type]
        MicroBatcher(int(item["micro_batch"]), int(item["group_size"])),
        require_profile=item.get("require_profile"),
        prefetch=load_workers > 0,
        load_workers=load_workers,
    )
    module.setup(runtime)
    return module


def _resume_wave(item: dict[str, Any]) -> dict[str, Any]:
    from src.experiments.api import Experiment
    from src.experiments.config import load_experiment

    experiment = Experiment(load_experiment(item["config"]))
    experiment.build()
    try:
        experiment._checkpoint.restore(Path(item["checkpoint"]), mode="resume")  # type: ignore[union-attr]
        data = experiment.get("data")
        stored_epoch = int(data.epoch)
        stored_index = int(data.update_index)
        view = data.train_epoch(stored_epoch, start_update=stored_index)
        epoch = stored_epoch
        index = stored_index
        if index >= view.updates:
            epoch += 1
            index = 0
            view = data.train_epoch(epoch, start_update=0)
        batch = view.batch(index)[0]
        if int(data.update_index) != stored_index or int(data.epoch) != stored_epoch:
            raise RuntimeError("resume render advanced the stored cursor")
        return {
            "epoch": epoch,
            "inputs": _digest(batch.inputs),
            "name": item["name"],
            "plan_hash": data.plan_hash,
            "realization_id": batch.realization_id,
            "sample_id": batch.sample_id,
            "stored_epoch": stored_epoch,
            "stored_index": stored_index,
            "targets": _digest(batch.targets),
            "update_index": index,
        }
    finally:
        data_close = getattr(experiment.get("data"), "close", None)
        if callable(data_close):
            data_close()
        # Leave the process group up. The outer worker owns it.
        experiment._closed = True


def _pair_max_abs(left: Any, right: Any, label: str, *, require_realization: bool = True) -> float:
    if left.sample_id != right.sample_id or (require_realization and left.realization_id != right.realization_id):
        raise RuntimeError(f"{label} changed sample identity")
    worst = 0.0
    for name in ("inputs", "targets"):
        lhs = getattr(left, name).detach().cpu().numpy()
        rhs = getattr(right, name).detach().cpu().numpy()
        if lhs.shape != rhs.shape:
            raise RuntimeError(f"{name} shape changed across workers")
        worst = max(worst, float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64)))))
    return worst


def _digest(tensor: Any) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
