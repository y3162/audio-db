"""Phase 3-E CPU Gloo acceptance for the six snapshot data pipelines."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.experiments.data.fixed_pair import MicroBatcher, OrderSampler, SnapshotDataModule, SnapshotSource
from src.experiments.data.fixed_pair import FixedPairProtocol
from src.reproduction.ids import SEMAMBA_GSR_V1
from src.reproduction.pipeline.order import distributed_sampler_coverage, rank_order
from src.reproduction.protocols.online_semamba import bandwidth_limitation
from tests.experiments.test_chunk_data import _write_snapshot as _write_chunk
from tests.experiments.test_fixed_pair_data import _write_snapshot as _write_fixed
from tests.experiments.test_online_data import _write_online

ROOT = Path(__file__).resolve().parents[2]


class PrefetchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.spec = _write_fixed(
            self.base / "snapshots",
            derived=self.base / "derived",
            name="toy.prefetch",
            algorithm="fixed_pair.mp_senet",
            normalize="noisy_rms_unit_v1",
            order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
            profile="leakage_safe",
            selection_uses_test=False,
            claim_label="leakage_safe_comparison",
            validation_crop="full_length",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_worker_count_keeps_order_identity_and_waveforms(self) -> None:
        runtime = _Runtime()
        modules = [_fixed_module(self.spec, runtime, workers) for workers in (0, 1, 2)]
        try:
            hashes = [module.plan_hash for module in modules]
            self.assertEqual(len(set(hashes)), 1)
            views = [module.train_epoch(0, start_update=0) for module in modules]
            first = [view.batch(0)[0] for view in views]
            second = [view.batch(1)[0] for view in views]
            again = views[2].batch(0)[0]
            self.assertEqual(_max_abs(first[0].inputs, first[1].inputs), 0.0)
            self.assertEqual(_max_abs(first[0].inputs, first[2].inputs), 0.0)
            self.assertEqual(_max_abs(first[0].targets, first[2].targets), 0.0)
            self.assertEqual(_max_abs(second[0].inputs, second[2].inputs), 0.0)
            self.assertEqual(first[0].sample_id, first[2].sample_id)
            self.assertEqual(first[0].realization_id, first[2].realization_id)
            self.assertEqual(_max_abs(first[2].inputs, again.inputs), 0.0)
            for module in modules:
                self.assertEqual(module.update_index, 0)
            for index in range(views[1].updates):
                views[1].batch(index)
            self.assertLessEqual(len(modules[1]._cache), modules[1].load_workers + 1)
        finally:
            for module in modules:
                module.close()

    def test_prefetch_error_propagates_and_close_reports_a_stuck_worker(self) -> None:
        runtime = _Runtime()
        module = _fixed_module(self.spec, runtime, 1)
        try:
            original = module.protocol.render
            def fail(sample: object) -> tuple[np.ndarray, np.ndarray, int]:
                if int(getattr(sample, "index")) != 0:
                    raise RuntimeError("injected prefetch failure")
                return original(sample)  # type: ignore[arg-type, no-any-return]
            module.protocol.render = fail  # type: ignore[method-assign]
            module.train_epoch(0, start_update=0).batch(0)
            with self.assertRaises(RuntimeError):
                module.train_epoch(0, start_update=0).batch(1)
        finally:
            module.protocol.render = original  # type: ignore[method-assign]
            module.close()
        stuck = _fixed_module(self.spec, runtime, 1)
        held = threading.Event()
        thread = threading.Thread(target=held.wait, name="audio-db-prefetch-stuck")
        thread.start()
        stuck._workers.append(thread)
        stuck.worker_join_timeout = 0.2
        try:
            with self.assertRaises(Exception) as caught:
                stuck.close()
            self.assertIn("did not exit", str(caught.exception))
        finally:
            held.set()
            thread.join(timeout=2)
            stuck._workers = [worker for worker in stuck._workers if worker.is_alive()]
            if stuck._workers:
                stuck.close()


class BandwidthPathTests(unittest.TestCase):
    def test_each_resampler_and_lowpass_runs(self) -> None:
        speech = np.linspace(-0.2, 0.2, 1600, dtype=np.float64)[None, :]
        ran: list[str] = []
        for res_type in ("scipy", "polyphase", "kaiser_best", "kaiser_fast"):
            limited, _padded = bandwidth_limitation(speech, 16000, 8000, res_type, "butter", 4)
            self.assertEqual(limited.shape, speech.shape)
            self.assertTrue(np.isfinite(limited).all())
            ran.append(res_type)
        for lowpass in ("butter", "cheby1", "cheby2", "bessel", "ellip"):
            limited, _padded = bandwidth_limitation(speech, 16000, 8000, "kaiser_best", lowpass, 4)
            self.assertEqual(limited.shape, speech.shape)
            self.assertTrue(np.isfinite(limited).all())
            ran.append(lowpass)
        self.assertEqual(list(SEMAMBA_GSR_V1["bandwidth_type"]), ["kaiser_best", "kaiser_fast", "scipy", "polyphase"])
        self.assertEqual(ran, ["scipy", "polyphase", "kaiser_best", "kaiser_fast", "butter", "cheby1", "cheby2", "bessel", "ellip"])


class RankRuleTests(unittest.TestCase):
    def test_divisible_and_indivisible_fixed_pair_rules(self) -> None:
        even = [rank_order("torch_distributed_sampler_v1", 8, seed=7, epoch=0, world_size=2, rank=rank) for rank in (0, 1)]
        odd = [rank_order("torch_distributed_sampler_v1", 7, seed=7, epoch=0, world_size=2, rank=rank) for rank in (0, 1)]
        self.assertEqual(len(even[0]), 4)
        self.assertEqual(len(odd[0]), 3)
        self.assertEqual(set(even[0]).intersection(even[1]), set())
        self.assertEqual(len(set(odd[0] + odd[1])), 6)
        self.assertEqual(len(set(range(7)) - set(odd[0] + odd[1])), 1)
        covered = distributed_sampler_coverage(7, 2)
        self.assertEqual(covered["dropped_tail"], 1)
        self.assertEqual(covered["padded_duplicates"], 2)
        moss = [rank_order("rank_mod_fixed_v1", 7, seed=7, epoch=0, world_size=2, rank=rank) for rank in (0, 1)]
        self.assertEqual(len(moss[0]), len(moss[1]))
        self.assertEqual(len(moss[0]), 4)
        self.assertTrue(set(moss[0]).intersection(moss[1]) or len(set(moss[0] + moss[1])) < 7)


class DistributedSupplyTests(unittest.TestCase):
    def test_six_pipelines_match_on_two_processes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            spec = _supply_spec(base)
            completed = _torchrun_worker(spec, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            ranks = [_read_report(base / "probe" / f"rank{rank}.json") for rank in (0, 1)]
            self.assertIsNone(ranks[0]["error"])
            self.assertIsNone(ranks[1]["error"])
            left = {item["name"]: item for item in ranks[0]["report"]["pipelines"]}
            right = {item["name"]: item for item in ranks[1]["report"]["pipelines"]}
            self.assertEqual(set(left), set(right))
            for name, item in left.items():
                other = right[name]
                self.assertEqual(item["plan_hash"], other["plan_hash"])
                self.assertEqual(item["updates"], other["updates"])
                self.assertEqual(item["dataset_id"], other["dataset_id"])
                self.assertEqual(item["world_size"], 2)
                self.assertEqual(item["worker_max_abs"], 0.0)
                self.assertGreater(item["updates"], 0)
            self.assertNotEqual(left["mp_senet"]["validation"]["epoch0"], [])
            self.assertEqual(right["mp_senet"]["validation"]["epoch0"], [])
            self.assertNotEqual(left["dfn3"]["validation"]["epoch0"], [])
            self.assertEqual(right["dfn3"]["validation"]["epoch0"], [])
            self.assertNotEqual(left["semamba"]["validation"]["epoch0"], [])
            self.assertEqual(right["semamba"]["validation"]["epoch0"], [])
            self.assertNotEqual(left["mossformer"]["validation"]["epoch0"], [])
            self.assertNotEqual(right["mossformer"]["validation"]["epoch0"], [])
            self.assertEqual(left["mp_odd"]["updates"], right["mp_odd"]["updates"])
            self.assertTrue(any(size < left["dfn3"]["batch_sizes"][0] for size in left["dfn3"]["batch_sizes"]))
            self.assertEqual(left["dfn3"]["batch_sizes"], right["dfn3"]["batch_sizes"])
            semamba_ids = set(left["semamba"]["sample_ids"])
            other_ids = set(right["semamba"]["sample_ids"])
            self.assertTrue(semamba_ids and other_ids)


class DistributedResumeTests(unittest.TestCase):
    def test_continuous_and_resumed_runs_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            pipelines = _training_pipelines(base)
            compared: list[str] = []
            waves = []
            for pipeline in pipelines:
                continuous = base / pipeline["name"] / "continuous"
                interrupted = base / pipeline["name"] / "interrupted"
                _train(pipeline, continuous, steps=4, stop_after=None)
                _train(pipeline, interrupted, steps=4, stop_after=2)
                saved = interrupted / "run" / "step_00000002.pt"
                self.assertTrue(saved.is_file(), pipeline["name"])
                before = saved.read_bytes()
                _train(pipeline, interrupted, steps=4, stop_after=None, resume=saved)
                self.assertEqual(saved.read_bytes(), before)
                left = torch.load(continuous / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
                right = torch.load(interrupted / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
                self.assertEqual(left["states"]["data"]["epoch"], right["states"]["data"]["epoch"])
                self.assertEqual(left["states"]["data"]["update_index"], right["states"]["data"]["update_index"])
                self.assertEqual(left["states"]["data"]["plan_hash"], right["states"]["data"]["plan_hash"])
                self.assertEqual(left["states"]["data"]["world_size"], 2)
                self.assertEqual(_sample_ids(continuous / "run"), _sample_ids(interrupted / "run"))
                if pipeline["name"] == "mp_senet":
                    self.assertEqual(_max_tree(left["states"]["model"], right["states"]["model"]), 0.0)
                    self.assertEqual(_max_tree(left["states"]["optimizer"], right["states"]["optimizer"]), 0.0)
                    self.assertEqual(_max_tree(left["states"]["scheduler"], right["states"]["scheduler"]), 0.0)
                    self.assertEqual(left["states"]["trainer"], right["states"]["trainer"])
                    self.assertTrue(_same_tree(left["rng_by_rank"], right["rng_by_rank"]))
                    self.assertEqual(_losses(continuous / "run"), _losses(interrupted / "run"))
                waves.append(
                    {
                        "checkpoint": str(saved),
                        "config": str(interrupted / "resume.yaml"),
                        "name": pipeline["name"],
                    }
                )
                compared.append(pipeline["name"])
            spec = {"checkpoints": waves, "mode": "resume-waves", "report_dir": str(base / "waves"), "timeout_seconds": 60}
            (base / "waves-spec.json").write_text(json.dumps(spec), encoding="utf-8")
            completed = _torchrun_worker(base / "waves-spec.json", timeout=240)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            reports = [_read_report(base / "waves" / f"rank{rank}.json") for rank in (0, 1)]
            self.assertEqual(
                [item["name"] for item in reports[0]["report"]["waves"]],
                [item["name"] for item in reports[1]["report"]["waves"]],
            )
            for left, right in zip(reports[0]["report"]["waves"], reports[1]["report"]["waves"]):
                self.assertEqual(left["stored_epoch"], right["stored_epoch"])
                self.assertEqual(left["stored_index"], right["stored_index"])
                self.assertEqual(left["plan_hash"], right["plan_hash"])
                self.assertNotEqual(left["sample_id"], "")
                self.assertTrue(left["inputs"])
                self.assertTrue(left["realization_id"])
            self.assertEqual(compared, [item["name"] for item in pipelines])


class DistributedFailureTests(unittest.TestCase):
    def test_rank_disagreement_does_not_replace_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            pipeline = _training_pipelines(base)[0]
            directory = base / "skew"
            cursor = _train(
                pipeline,
                directory,
                steps=1,
                stop_after=None,
                trainer="tests.experiments.support.CursorSkewTrainer",
                expect_ok=False,
            )
            self.assertNotEqual(cursor.returncode, 0)
            self.assertIn("update_index", cursor.stderr + cursor.stdout)
            self.assertFalse((directory / "run" / "step_00000001.pt").is_file())
            self.assertEqual(_status(directory / "run"), "FAILED")
            plan = _train(
                pipeline,
                base / "plan",
                steps=1,
                stop_after=None,
                trainer="tests.experiments.support.PlanSkewTrainer",
                expect_ok=False,
            )
            self.assertNotEqual(plan.returncode, 0)
            self.assertIn("plan_hash", plan.stderr + plan.stdout)
            self.assertFalse((base / "plan" / "run" / "step_00000001.pt").is_file())

    def test_tampered_data_state_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            pipeline = _training_pipelines(base)[0]
            directory = base / "tamper"
            _train(pipeline, directory, steps=2, stop_after=1)
            saved = directory / "run" / "step_00000001.pt"
            original = saved.read_bytes()
            payload = torch.load(saved, map_location="cpu", weights_only=False)
            payload["states"]["data"]["plan_hash"] = "0" * 64
            tampered = directory / "run" / "tampered.pt"
            torch.save(payload, tampered)
            refused = _train(pipeline, directory, steps=2, stop_after=None, resume=tampered, expect_ok=False)
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(saved.read_bytes(), original)
            self.assertIn("plan_hash", refused.stderr + refused.stdout)

    def test_build_and_render_failures_finish_within_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            pipeline = _training_pipelines(base)[0]
            build = _train(
                pipeline,
                base / "build",
                steps=1,
                stop_after=None,
                data_class="tests.experiments.support.RankBuildFailModule",
                timeout_seconds=8,
                expect_ok=False,
                extra_env={"AUDIO_DB_FAIL_BUILD_RANK": "0"},
                process_timeout=25,
            )
            self.assertNotEqual(build.returncode, 0)
            self.assertIn("injected build failure", build.stderr + build.stdout)
            render = _train(
                pipeline,
                base / "render",
                steps=1,
                stop_after=None,
                data_class="tests.experiments.support.RankRenderFailModule",
                timeout_seconds=8,
                expect_ok=False,
                extra_env={"AUDIO_DB_FAIL_RENDER_RANK": "1"},
                process_timeout=25,
            )
            self.assertNotEqual(render.returncode, 0)
            self.assertIn("injected render failure", render.stderr + render.stdout)

    def test_load_workers_changes_config_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            pipeline = _training_pipelines(base)[0]
            directory = base / "workers"
            _train(pipeline, directory, steps=1, stop_after=None, load_workers=1)
            saved = directory / "run" / "step_00000001.pt"
            original = saved.read_bytes()
            refused = _train(
                pipeline,
                directory,
                steps=2,
                stop_after=None,
                resume=saved,
                load_workers=2,
                expect_ok=False,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("config", refused.stderr + refused.stdout)
            self.assertEqual(saved.read_bytes(), original)


def _fixed_module(spec: dict[str, object], runtime: object, workers: int) -> SnapshotDataModule:
    source = SnapshotSource(
        str(spec["selector"]),
        spec["root"],  # type: ignore[arg-type]
        "train",
        derived=spec["derived"],  # type: ignore[arg-type]
        corpora=spec["corpora"],  # type: ignore[arg-type]
        generated=spec["generated"],  # type: ignore[arg-type]
    )
    protocol = FixedPairProtocol(source)
    module = SnapshotDataModule(
        source,
        OrderSampler(source, protocol, seed=7),
        protocol,
        MicroBatcher(1, 1),
        require_profile="leakage_safe",
        prefetch=workers > 0,
        load_workers=workers,
    )
    module.setup(runtime)
    return module


def _supply_spec(base: Path) -> Path:
    derived = base / "derived"
    snapshots = base / "snapshots"
    mp = _write_fixed(
        snapshots,
        derived=derived,
        name="toy.mp",
        algorithm="fixed_pair.mp_senet",
        normalize="noisy_rms_unit_v1",
        order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
        profile="leakage_safe",
        selection_uses_test=False,
        claim_label="leakage_safe_comparison",
        validation_crop="full_length",
    )
    odd = _write_fixed(
        snapshots,
        derived=derived,
        name="toy.mpodd",
        algorithm="fixed_pair.mp_senet",
        normalize="noisy_rms_unit_v1",
        order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
        profile="leakage_safe",
        selection_uses_test=False,
        claim_label="leakage_safe_comparison",
        validation_crop="full_length",
        train_count=7,
        valid_count=2,
        test_count=2,
    )
    moss = _write_fixed(
        snapshots,
        derived=derived,
        name="toy.moss",
        algorithm="fixed_pair.mossformergan_se_16k",
        normalize="independent_audio_norm_v1",
        order="rank_mod_fixed_v1",
        profile="leakage_safe",
        selection_uses_test=False,
        claim_label="leakage_safe_comparison",
        validation_crop="per_epoch_random",
    )
    grid = _write_chunk(snapshots, derived=derived, kind="ms", name="toy.grid")
    loco = _write_chunk(snapshots, derived=derived, kind="dns", name="toy.loco")
    dfn = _write_online(
        snapshots,
        corpora=base / "corpora",
        derived=derived,
        kind="dfn3",
        name="toy.dfn",
        speech=5,
        factor=1.0,
        global_batch=2,
        valid_batch=2,
    )
    semamba = _write_online(
        snapshots,
        corpora=base / "corpora",
        derived=derived,
        kind="semamba",
        name="toy.semamba",
        speech=7,
        factor=1.0,
        global_batch=2,
        valid_batch=2,
    )
    common = {"num_cache_chunks": 4, "num_samples_per_epoch": 2, "seed": 7}
    pipelines = [
        _item("mp_senet", "fixed", mp, micro_batch=2, group_size=1, render=True, validation_wave=True, require_profile="leakage_safe"),
        _item("mp_odd", "fixed", odd, micro_batch=2, group_size=1, render=True, require_profile="leakage_safe"),
        _item("mossformer", "fixed", moss, micro_batch=2, group_size=1, render=True, require_profile="leakage_safe"),
        _item("tf_gridnet", "chunk", grid, micro_batch=1, group_size=1, render=True, chunk_length=32, require_profile="leakage_safe", **common),
        _item("tf_locoformer", "chunk", loco, micro_batch=1, group_size=1, render=True, chunk_length=32, require_profile="leakage_safe", **common),
        _item("dfn3", "dfn3", dfn, micro_batch=2, group_size=1, render=True, factor=1.0),
        _item("semamba", "semamba", semamba, micro_batch=1, group_size=1, render=True),
        _item("semamba_group2", "semamba", semamba, micro_batch=1, group_size=2, render=True),
    ]
    path = base / "probe-spec.json"
    payload = {"mode": "probe", "pipelines": pipelines, "report_dir": str(base / "probe"), "timeout_seconds": 60}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _training_pipelines(base: Path) -> list[dict[str, Any]]:
    derived = base / "derived"
    snapshots = base / "snapshots"
    specs = {
        "mp_senet": ("fixed", _write_fixed(
            snapshots, derived=derived, name="train.mp", algorithm="fixed_pair.mp_senet",
            normalize="noisy_rms_unit_v1", order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
            profile="leakage_safe", selection_uses_test=False, claim_label="leakage_safe_comparison",
            validation_crop="full_length",
        ), {"scheduler": True}),
        "mossformer": ("fixed", _write_fixed(
            snapshots, derived=derived, name="train.moss", algorithm="fixed_pair.mossformergan_se_16k",
            normalize="independent_audio_norm_v1", order="rank_mod_fixed_v1", profile="leakage_safe",
            selection_uses_test=False, claim_label="leakage_safe_comparison", validation_crop="per_epoch_random",
        ), {}),
        "tf_gridnet": ("chunk", _write_chunk(snapshots, derived=derived, kind="ms", name="train.grid"), {}),
        "tf_locoformer": ("chunk", _write_chunk(snapshots, derived=derived, kind="dns", name="train.loco"), {}),
        "dfn3": ("dfn3", _write_online(
            snapshots, corpora=base / "corpora", derived=derived, kind="dfn3", name="train.dfn",
            speech=5, factor=1.0, global_batch=2, valid_batch=2,
        ), {}),
        "semamba": ("semamba", _write_online(
            snapshots, corpora=base / "corpora", derived=derived, kind="semamba", name="train.semamba",
            speech=7, factor=1.0, global_batch=2, valid_batch=2,
        ), {}),
    }
    pipelines = []
    for name, (kind, spec, extra) in specs.items():
        pipelines.append({"extra": extra, "kind": kind, "name": name, "spec": spec})
    return pipelines


def _item(name: str, kind: str, spec: dict[str, object], **kwargs: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "corpora": str(spec["corpora"]),
        "derived": str(spec["derived"]),
        "generated": str(spec["generated"]),
        "group_size": 1,
        "kind": kind,
        "micro_batch": 1,
        "name": name,
        "render": True,
        "require_profile": None,
        "root": str(spec["root"]),
        "seed": 7,
        "selector": spec["selector"],
    }
    payload.update(kwargs)
    return payload


def _train(
    pipeline: dict[str, Any],
    directory: Path,
    *,
    steps: int,
    stop_after: int | None,
    resume: Path | None = None,
    trainer: str = "src.experiments.trainers.StandardTrainer",
    data_class: str = "src.experiments.data.SnapshotDataModule",
    load_workers: int = 0,
    timeout_seconds: int = 30,
    expect_ok: bool = True,
    extra_env: dict[str, str] | None = None,
    process_timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    directory.mkdir(parents=True, exist_ok=True)
    config = _write_training_yaml(
        directory,
        pipeline,
        steps=steps,
        stop_after=stop_after,
        trainer=trainer,
        data_class=data_class,
        load_workers=load_workers,
        timeout_seconds=timeout_seconds,
        filename="resume.yaml" if resume is not None else "experiment.yaml",
    )
    extra = () if resume is None else ("--resume", str(resume))
    completed = _torchrun_cli(config, *extra, timeout=process_timeout, extra_env=extra_env)
    if expect_ok:
        if completed.returncode != 0:
            raise AssertionError(completed.stderr + completed.stdout)
    return completed


def _write_training_yaml(
    directory: Path,
    pipeline: dict[str, Any],
    *,
    steps: int,
    stop_after: int | None,
    trainer: str,
    data_class: str,
    load_workers: int,
    timeout_seconds: int,
    filename: str,
) -> Path:
    spec = pipeline["spec"]
    kind = pipeline["kind"]
    if kind == "fixed":
        sampler = f"""    class: src.experiments.data.OrderSampler
    kwargs:
      seed: 7
    inject:
      source: {{ref: source}}
      protocol: {{ref: protocol}}"""
        protocol = "src.experiments.data.FixedPairProtocol"
        micro = 1
        group = 1
        chunk = ""
    elif kind == "chunk":
        sampler = """    class: src.experiments.data.ChunkSampler
    kwargs:
      seed: 7
      num_samples_per_epoch: 2
      chunk_length: 32
      num_cache_chunks: 4
      shuffle: true
    inject:
      source: {ref: source}"""
        protocol = "src.experiments.data.ChunkProtocol"
        micro = 1
        group = 1
        chunk = ""
    elif kind == "dfn3":
        sampler = """    class: src.experiments.data.Dfn3IndexSampler
    kwargs:
      seed: 7
    inject:
      source: {ref: source}"""
        protocol = "src.experiments.data.Dfn3Protocol"
        micro = 1
        group = 1
        chunk = ""
    else:
        sampler = """    class: src.experiments.data.OrderSampler
    kwargs:
      seed: 7
    inject:
      source: {ref: source}
      protocol: {ref: protocol}"""
        protocol = "src.experiments.data.SemambaProtocol"
        micro = 1
        group = 1
        chunk = ""
    scheduler = ""
    scheduler_inject = ""
    if pipeline["extra"].get("scheduler"):
        scheduler = """
  scheduler:
    role: scheduler
    class: torch.optim.lr_scheduler.StepLR
    kwargs: {step_size: 1, gamma: 0.5}
    inject:
      optimizer: {ref: optimizer}"""
        scheduler_inject = "      scheduler: {ref: scheduler}\n"
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    profile = '      require_profile: "leakage_safe"\n' if kind in {"fixed", "chunk"} else ""
    text = f"""schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
    timeout_seconds: {timeout_seconds}
output:
  directory: {directory / "run"}
components:
  model:
    role: module
    class: tests.experiments.support.GainWave
    prepare: {{trainable: true, distribute: ddp}}
  loss:
    role: loss
    class: torch.nn.L1Loss
  source:
    role: other
    class: src.experiments.data.SnapshotSource
    kwargs:
      snapshot: "{spec["selector"]}"
      snapshot_root: "{spec["root"]}"
      split: train
      derived: "{spec["derived"]}"
      corpora: "{spec["corpora"]}"
      generated: "{spec["generated"]}"
  protocol:
    role: other
    class: {protocol}
    kwargs: {{}}
    inject:
      source: {{ref: source}}
  sampler:
    role: other
{sampler}
  batcher:
    role: other
    class: src.experiments.data.MicroBatcher
    kwargs:
      micro_batch: {micro}
      group_size: {group}
  data:
    role: data
    class: {data_class}
    kwargs:
      prefetch: {str(load_workers > 0).lower()}
      load_workers: {load_workers}
{profile}{chunk}    inject:
      source: {{ref: source}}
      sampler: {{ref: sampler}}
      protocol: {{ref: protocol}}
      batcher: {{ref: batcher}}
  optimizer:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: model
    kwargs: {{lr: 0.01}}{scheduler}
  trainer:
    role: trainer
    class: {trainer}
    kwargs:
      max_logical_steps: {steps}
      accumulation: {group}
      checkpoint_interval: 1
{stop}    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
{scheduler_inject}"""
    path = directory / filename
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(text)
    return path


def _torchrun_worker(spec: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    return _launch(
        [sys.executable, "-m", "tests.experiments.distributed_data_worker", str(spec)],
        timeout=timeout,
    )


def _torchrun_cli(
    config: Path,
    *extra: str,
    timeout: int,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return _launch(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        timeout=timeout,
        extra_env=extra_env,
    )


def _launch(
    arguments: list[str],
    *,
    timeout: int,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    if extra_env:
        env.update(extra_env)
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "LOCAL_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        env.pop(name, None)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        "--master_port",
        str(_free_port()),
        *arguments[1:],
    ]
    # arguments[0] is the interpreter. torchrun uses the same interpreter via -m above
    # only when arguments begin with the interpreter and a module flag.
    if arguments[1:3] != ["-m", arguments[2] if len(arguments) > 2 else ""]:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "2",
            "--master_port",
            str(_free_port()),
            *arguments,
        ]
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_ids(directory: Path) -> list[list[str]]:
    rows = []
    for rank in (0, 1):
        values = []
        for line in (directory / f"samples-rank{rank}.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                values.append(str(json.loads(line)["sample_id"]))
        rows.append(values)
    return rows


def _losses(directory: Path) -> list[float]:
    values = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            values.append(float(json.loads(line)["loss"]))
    return values


def _status(directory: Path) -> str:
    return str(json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"])


def _same_tree(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_tree(lhs, rhs) for lhs, rhs in zip(left, right))
    return left == right


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.cpu() - right.cpu())).item())


def _max_tree(left: object, right: object) -> float:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return _max_abs(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return max(_max_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return max(_max_tree(lhs, rhs) for lhs, rhs in zip(left, right))
    if left != right:
        raise AssertionError(f"{left!r} != {right!r}")
    return 0.0


class _Runtime:
    def __init__(self, rank: int = 0, world_size: int = 1) -> None:
        self.rank = rank
        self.world_size = world_size


