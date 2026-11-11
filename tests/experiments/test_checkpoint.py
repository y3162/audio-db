"""Checkpoint save, restore, resume, and initialization on CPU.

Latest is one file that is replaced only by a complete file. Archives are
what the selector keeps. Both hold the same full state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from src.experiments.api import build_experiment
from src.experiments.checkpoint import FORMAT, CheckpointManager
from src.experiments.config import config_generations, config_sha256, load_experiment
from src.experiments.data import TensorPairData
from src.experiments.errors import BuildError, CheckpointError, ConfigError, StateError
from src.experiments.selectors import (
    DEFAULT_LATEST_EVERY,
    Decision,
    PeriodicArchiveSelector,
    SavePoint,
    Selector,
    StandardSelector,
    TopKSelector,
    build_selector,
    scalar_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
STEP_ARCHIVE = ("tests.experiments.support.StepArchiveSelector", {"interval": 1})
STANDARD = "src.experiments.selectors.StandardSelector"
PERIODIC = "src.experiments.selectors.PeriodicArchiveSelector"
TOP_K = "src.experiments.selectors.TopKSelector"


class _Base(unittest.TestCase):
    def _directory(self) -> Path:
        return Path(self.enterContext(tempfile.TemporaryDirectory()))


class ResumeTests(_Base):
    def test_interrupted_resume_matches_a_continuous_run_in_another_process(self) -> None:
        continuous_dir = self._directory()
        interrupted_dir = self._directory()
        continuous_config = _write_yaml(continuous_dir, "continuous", steps=4, stop_after=None, selector=STEP_ARCHIVE)
        interrupted_config = _write_yaml(
            interrupted_dir, "interrupted", "run", steps=4, stop_after=2, selector=STEP_ARCHIVE
        )
        resume_config = _write_yaml(interrupted_dir, "resume", "run", steps=4, stop_after=None, selector=STEP_ARCHIVE)
        self.assertEqual(config_sha256(load_experiment(interrupted_config)), config_sha256(load_experiment(resume_config)))
        self.assertEqual(config_sha256(load_experiment(continuous_config)), config_sha256(load_experiment(resume_config)))

        continuous = _cli(continuous_config)
        self.assertEqual(continuous.returncode, 0, continuous.stderr)
        interrupted = _cli(interrupted_config)
        self.assertEqual(interrupted.returncode, 0, interrupted.stderr)
        interrupted_out = interrupted_dir / "run"
        self.assertEqual(_status(interrupted_out), "INTERRUPTED")
        self.assertFalse((interrupted_out / "complete.json").is_file())
        saved = interrupted_out / "archive" / "step_00000002.pt"
        self.assertTrue(saved.is_file())
        self.assertEqual(saved.read_bytes(), (interrupted_out / "latest.pt").read_bytes())
        data = TensorPairData(features=8, examples=32, batch_size=4)
        prefetched = data.train_epoch(0, start_update=0).batch(2)[0].sample_id
        self.assertEqual(data.update_index, 0)
        metrics_path = interrupted_out / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "applied": True,
                        "logical_step": 3,
                        "loss": 123.0,
                        "optimizer_updates": {"optimizer": 3},
                        "sample_id": "not-consumed",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        resumed = _cli(resume_config, "--resume", str(saved))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)

        continuous_out = continuous_dir / "continuous"
        self.assertEqual(_status(continuous_out), "COMPLETED")
        self.assertEqual(_status(interrupted_out), "COMPLETED")
        self.assertTrue((continuous_out / "complete.json").is_file())
        self.assertTrue((interrupted_out / "complete.json").is_file())
        left = _load(continuous_out / "archive" / "step_00000004.pt")
        right = _load(interrupted_out / "archive" / "step_00000004.pt")
        self.assertEqual(left["logical_step"], 4)
        self.assertEqual(right["logical_step"], 4)
        self.assertEqual(left["config_sha256"], right["config_sha256"])
        self.assertEqual(left["optimizer_updates"], right["optimizer_updates"])
        _assert_equal(left["states"], right["states"])
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        _assert_equal(left["selector"], right["selector"])
        left_metrics = _metrics(continuous_out)
        right_metrics = _metrics(interrupted_out)
        self.assertEqual([row["logical_step"] for row in left_metrics], [1, 2, 3, 4])
        self.assertEqual(left_metrics, right_metrics)
        self.assertNotIn("not-consumed", [row["sample_id"] for row in right_metrics])
        self.assertEqual(right_metrics[2]["sample_id"], prefetched)
        self.assertEqual(right["states"]["data"]["update_index"], 4)
        optimizer_step = int(next(iter(right["states"]["optimizer"]["state"].values()))["step"])
        self.assertEqual(optimizer_step, 4)
        delta = _max_abs(left["states"]["model"], right["states"]["model"])
        self.assertEqual(delta, 0.0)
        self.assertEqual(right["states"]["scheduler"]["last_epoch"], left["states"]["scheduler"]["last_epoch"])
        # Latest and the Archive of the same point are the same file content.
        for out in (continuous_out, interrupted_out):
            self.assertEqual((out / "latest.pt").read_bytes(), (out / "archive" / "step_00000004.pt").read_bytes())

    def test_the_standard_selector_keeps_one_latest_and_the_directory_resumes(self) -> None:
        continuous_dir = self._directory()
        interrupted_dir = self._directory()
        continuous_config = _write_yaml(continuous_dir, "continuous", steps=4, stop_after=None)
        interrupted_config = _write_yaml(interrupted_dir, "interrupted", "run", steps=4, stop_after=2)
        resume_config = _write_yaml(interrupted_dir, "resume", "run", steps=4, stop_after=None)
        _run(continuous_config)
        _run(interrupted_config)
        interrupted_out = interrupted_dir / "run"
        self.assertEqual(_status(interrupted_out), "INTERRUPTED")
        self.assertEqual(_load(interrupted_out / "latest.pt")["logical_step"], 2)
        # ``--resume`` given the experiment directory means its Latest.
        resumed = _cli(resume_config, "--resume", str(interrupted_out))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        continuous_out = continuous_dir / "continuous"
        for out in (continuous_out, interrupted_out):
            self.assertEqual(_status(out), "COMPLETED")
            self.assertEqual(
                sorted(path.name for path in out.iterdir()),
                [
                    ".lock",
                    "complete.json",
                    "data_provenance.json",
                    "latest.data_provenance.json",
                    "latest.pt",
                    "metrics.jsonl",
                    "status.json",
                ],
            )
        left = _load(continuous_out / "latest.pt")
        right = _load(interrupted_out / "latest.pt")
        self.assertEqual(right["logical_step"], 4)
        _assert_equal(left["states"], right["states"])
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(_metrics(continuous_out), _metrics(interrupted_out))

    def test_resume_refuses_completion_a_copied_directory_and_a_rewind(self) -> None:
        directory = self._directory()
        config = _write_yaml(directory, "out", steps=1, stop_after=None)
        _run(config)
        finished = directory / "out" / "latest.pt"
        with self.assertRaises(CheckpointError) as caught:
            build_experiment(load_experiment(config)).run(resume=finished)
        self.assertIn("complete.json", str(caught.exception))
        copied_dir = self._directory()
        copied = copied_dir / "out"
        copied.mkdir()
        target = copied / finished.name
        shutil.copyfile(finished, target)
        copied_config = _write_yaml(copied_dir, "out", steps=1, stop_after=None)
        with self.assertRaises(CheckpointError) as caught:
            build_experiment(load_experiment(copied_config)).run(resume=target)
        self.assertIn("different directory", str(caught.exception))

        rewind_dir = self._directory()
        stopped = _write_yaml(rewind_dir, "stopped", "run", steps=5, stop_after=3, selector=STEP_ARCHIVE)
        whole = _write_yaml(rewind_dir, "whole", "run", steps=5, stop_after=None, selector=STEP_ARCHIVE)
        _run(stopped)
        out = rewind_dir / "run"
        before = _tree(out)
        self.assertEqual(_load(out / "latest.pt")["logical_step"], 3)
        with self.assertRaises(CheckpointError) as caught:
            build_experiment(load_experiment(whole)).run(resume=out / "archive" / "step_00000001.pt")
        self.assertIn("behind latest.pt", str(caught.exception))
        self.assertIn("initialize", str(caught.exception))
        self.assertEqual(_tree(out), before)
        # The Archive of the newest state is the same state as Latest and resumes.
        result = build_experiment(load_experiment(whole)).run(resume=out / "archive" / "step_00000003.pt")
        self.assertEqual((result.stop_reason, result.logical_step), ("completed", 5))

    def test_resume_requires_the_same_selector_class_but_not_the_same_settings(self) -> None:
        directory = self._directory()
        periodic = (PERIODIC, {"archive_every": 1, "keep_last": 2})
        _run(_write_yaml(directory, "stopped", "run", steps=4, stop_after=2, selector=periodic))
        out = directory / "run"
        before = _tree(out)
        other = _write_yaml(directory, "other", "run", steps=4, stop_after=None, selector=(TOP_K, {"metric": "loss"}))
        with self.assertRaises(CheckpointError) as caught:
            build_experiment(load_experiment(other)).run(resume=out)
        self.assertIn("selector", str(caught.exception))
        self.assertEqual(_tree(out), before)
        # Settings of the same class may change: they are not part of the experiment identity.
        wider = _write_yaml(
            directory, "wider", "run", steps=4, stop_after=None, selector=(PERIODIC, {"archive_every": 1, "keep_last": 3})
        )
        result = build_experiment(load_experiment(wider)).run(resume=out)
        self.assertEqual(result.logical_step, 4)
        self.assertEqual(_archives(out), ["step_00000002.pt", "step_00000003.pt", "step_00000004.pt"])

    def test_state_identity_and_foreign_formats_are_refused(self) -> None:
        directory = self._directory()
        config = _write_yaml(directory, "out", steps=1, stop_after=None)
        _run(config)
        source = directory / "out" / "latest.pt"
        payload = _load(source)
        experiment = build_experiment(load_experiment(config))
        cases = {
            "missing": _without_state(payload, "scheduler"),
            "extra": _with_extra_state(payload),
            "config": _replace(payload, "config_sha256", "0" * 64),
            "plan": _replace_plan(payload),
            "version": _replace(payload, "schema_version", 2),
            "unknown": _replace(payload, "unexpected", 1),
            "world": _replace(payload, "world_size", 2),
            "selector": _replace(payload, "selector", {"class": "x.Y", "state": {}}),
            "selector_state": _replace(payload, "selector", {"class": payload["selector"]["class"], "state": None}),
        }
        expected = {
            "missing": StateError,
            "extra": StateError,
            "config": CheckpointError,
            "plan": CheckpointError,
            "version": CheckpointError,
            "unknown": CheckpointError,
            "world": CheckpointError,
            "selector": CheckpointError,
            "selector_state": CheckpointError,
        }
        for name, mutated in cases.items():
            path = directory / "out" / f"{name}.pt"
            torch.save(mutated, path)
            with self.subTest(name=name):
                with self.assertRaises(expected[name]):
                    experiment._checkpoint.restore(path, mode="resume")
        foreign = directory / "out" / "legacy.pt"
        torch.save({"format": "audio-db-training-v1", "global_step": 1}, foreign)
        with self.assertRaises(CheckpointError) as raised:
            experiment._checkpoint.restore(foreign, mode="resume")
        self.assertIn("audio-db-training-v1", str(raised.exception))
        self.assertEqual(payload["format"], FORMAT)


class LatestTests(_Base):
    def test_a_failed_save_keeps_the_previous_latest(self) -> None:
        directory = self._directory()
        config = _write_yaml(directory, "out", steps=1, stop_after=None, selector=(STANDARD, {"latest_every": 1}))
        experiment = build_experiment(load_experiment(config))
        result = experiment.run()
        self.assertEqual(result.logical_step, 1)
        out = directory / "out"
        latest = out / "latest.pt"
        original = latest.read_bytes()
        temporary = out / ".latest.pt.tmp"

        def partial_write(payload: object, stream: object) -> None:
            del payload
            stream.write(b"half a checkpoint")  # type: ignore[attr-defined]
            raise OSError("disk full")

        failures = {
            "replace": patch("src.experiments.execution.checkpoint.os.replace", side_effect=OSError("injected")),
            "write": patch("src.experiments.execution.checkpoint.torch.save", side_effect=partial_write),
            "verify": patch(
                "src.experiments.execution.checkpoint._verify_written",
                side_effect=CheckpointError("checkpoint verification failed: injected"),
            ),
            "interrupt": patch("src.experiments.execution.checkpoint._verify_written", side_effect=KeyboardInterrupt),
        }
        for name, failure in failures.items():
            with self.subTest(name=name):
                with failure:
                    with self.assertRaises(KeyboardInterrupt if name == "interrupt" else CheckpointError):
                        experiment._checkpoint.offer("step", logical_step=1)
                self.assertEqual(latest.read_bytes(), original)
                self.assertFalse(temporary.exists())
                self.assertEqual(sorted(path.name for path in out.glob("*.pt")), ["latest.pt"])
                self.assertFalse((out / "archive").exists())
        # A temporary file and a truncated file are never a resume source.
        shutil.copyfile(latest, temporary)
        with self.assertRaises(CheckpointError) as caught:
            experiment._checkpoint.restore(temporary, mode="resume")
        self.assertIn("temporary", str(caught.exception))
        temporary.unlink()
        truncated = out / "truncated.pt"
        truncated.write_bytes(original[: len(original) // 2])
        with self.assertRaises(CheckpointError):
            experiment._checkpoint.restore(truncated, mode="resume")
        truncated.unlink()
        # The next successful save replaces the one Latest.
        report = experiment._checkpoint.offer("step", logical_step=1)
        self.assertEqual(report.latest, latest)
        self.assertEqual(sorted(path.name for path in out.glob("*.pt")), ["latest.pt"])
        self.assertEqual(_load(latest)["logical_step"], 1)

    def test_a_failed_archive_keeps_latest_and_the_other_archives(self) -> None:
        directory = self._directory()
        config = _write_yaml(directory, "out", steps=2, stop_after=None, selector=STEP_ARCHIVE)
        experiment = build_experiment(load_experiment(config))
        experiment.run()
        out = directory / "out"
        before = _tree(out)
        with patch("src.experiments.execution.checkpoint._link_or_copy", side_effect=CheckpointError("injected")):
            with self.assertRaises(CheckpointError):
                experiment._checkpoint.offer("step", logical_step=2)
        self.assertEqual(_tree(out), before)
        self.assertFalse(list(out.rglob(".*.tmp")))

    def test_stop_and_final_do_not_write_a_current_latest_again(self) -> None:
        cases = [
            # steps, latest_every, stop_after, expected writes, expected status
            (3, 1, None, [1, 2, 3], "COMPLETED"),
            (3, 2, None, [2, 3], "COMPLETED"),
            (4, 2, 3, [2, 3], "INTERRUPTED"),
            (4, 2, 2, [2], "INTERRUPTED"),
            (2, 5000, None, [2], "COMPLETED"),
            (3, None, None, [3], "COMPLETED"),
            (4, None, 2, [2], "INTERRUPTED"),
        ]
        original = CheckpointManager._write
        for steps, every, stop_after, expected, status in cases:
            with self.subTest(steps=steps, every=every, stop_after=stop_after):
                directory = self._directory()
                selector = None if every is None else (STANDARD, {"latest_every": every})
                config = _write_yaml(
                    directory, "out", steps=steps, stop_after=stop_after, selector=selector
                )
                written: list[int] = []

                def counted(manager: CheckpointManager, payload: dict, provenance: bytes, decision: Decision) -> None:
                    written.append(int(payload["logical_step"]))
                    original(manager, payload, provenance, decision)

                with patch.object(CheckpointManager, "_write", counted):
                    _run(config)
                out = directory / "out"
                self.assertEqual(written, expected)
                self.assertEqual(_status(out), status)
                self.assertEqual(_load(out / "latest.pt")["logical_step"], expected[-1])
                self.assertFalse((out / "archive").exists())

    def test_sigkill_during_a_save_keeps_the_previous_latest(self) -> None:
        directory = self._directory()
        killed_dir = directory / "killed"
        whole_dir = directory / "whole"
        killed_dir.mkdir()
        whole_dir.mkdir()
        marker = killed_dir / "killed.marker"
        spent = whole_dir / "killed.marker"
        spent.write_text("never\n", encoding="utf-8")
        killed_config = _kill_yaml(killed_dir, marker)
        whole_config = _kill_yaml(whole_dir, spent)
        self.assertEqual(config_sha256(load_experiment(killed_config)).__len__(), 64)

        first = _cli(killed_config)
        self.assertEqual(first.returncode, -signal.SIGKILL, first.stderr)
        self.assertTrue(marker.is_file())
        out = killed_dir / "out"
        # The process died while step 3 was being written. Latest is still step 2.
        self.assertEqual(_load(out / "latest.pt")["logical_step"], 2)
        self.assertFalse((out / "complete.json").exists())
        self.assertEqual(_status(out), "RUNNING")
        leftover = out / ".latest.pt.tmp"
        self.assertTrue(leftover.is_file())
        experiment = build_experiment(load_experiment(killed_config))
        try:
            with self.assertRaises(CheckpointError):
                experiment._checkpoint.restore(leftover, mode="resume")
        finally:
            experiment.close()
        resumed = _cli(killed_config, "--resume", str(out))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(_status(out), "COMPLETED")
        self.assertFalse(leftover.exists())
        whole = _cli(whole_config)
        self.assertEqual(whole.returncode, 0, whole.stderr)
        left = _load(whole_dir / "out" / "latest.pt")
        right = _load(out / "latest.pt")
        self.assertEqual(right["logical_step"], 4)
        _assert_equal(left["states"], right["states"])
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(_metrics(whole_dir / "out"), _metrics(out))


class SelectorTests(_Base):
    def test_standard_selector_decisions(self) -> None:
        selector = StandardSelector(latest_every=3)
        self.assertEqual(selector.decide(SavePoint("step", 2)), Decision())
        self.assertEqual(selector.decide(SavePoint("step", 3)), Decision(latest=True))
        self.assertEqual(selector.decide(SavePoint("partial", 3)), Decision())
        self.assertEqual(selector.decide(SavePoint("boundary_pending", 4)), Decision(latest=True))
        self.assertEqual(selector.decide(SavePoint("boundary", 4)), Decision(latest=True))
        self.assertEqual(selector.decide(SavePoint("stop", 5)), Decision(latest=True))
        self.assertEqual(selector.decide(SavePoint("stop", 5, latest_is_current=True)), Decision())
        self.assertEqual(selector.decide(SavePoint("final", 5, latest_is_current=True)), Decision())
        self.assertEqual(selector.decide(SavePoint("final", 5)), Decision(latest=True))
        self.assertEqual(StandardSelector(partial=True).decide(SavePoint("partial", 0)), Decision(latest=True))
        standard = StandardSelector()
        self.assertEqual(standard.latest_every, DEFAULT_LATEST_EVERY)
        self.assertEqual(standard.decide(SavePoint("step", 1)), Decision())
        self.assertEqual(standard.decide(SavePoint("step", DEFAULT_LATEST_EVERY)), Decision(latest=True))
        self.assertEqual(StandardSelector(latest_every=1).decide(SavePoint("step", 1)), Decision(latest=True))
        self.assertEqual(PeriodicArchiveSelector().latest_every, 1)
        self.assertEqual(TopKSelector(metric="loss").latest_every, 1)
        self.assertEqual(selector.state_dict(), {})
        with self.assertRaises(CheckpointError):
            selector.decide(SavePoint("unknown", 1))
        with self.assertRaises(CheckpointError):
            selector.load_state_dict({"kept": []})
        for bad in (0, -1, True, 1.5, "2"):
            with self.subTest(bad=bad):
                with self.assertRaises(BuildError):
                    StandardSelector(latest_every=bad)  # type: ignore[arg-type]

    def test_periodic_archive_selector_decisions_and_state(self) -> None:
        selector = PeriodicArchiveSelector(latest_every=1, archive_every=2, keep_last=2)
        self.assertEqual(selector.decide(SavePoint("step", 1)), Decision(latest=True))
        self.assertEqual(selector.decide(SavePoint("step", 2)), Decision(latest=True, archive=("step_00000002.pt",)))
        self.assertEqual(selector.decide(SavePoint("step", 4)), Decision(latest=True, archive=("step_00000004.pt",)))
        # A boundary at the same step replaces that Archive and drops nothing.
        self.assertEqual(
            selector.decide(SavePoint("boundary", 4)), Decision(latest=True, archive=("step_00000004.pt",))
        )
        self.assertEqual(
            selector.decide(SavePoint("step", 6)),
            Decision(latest=True, archive=("step_00000006.pt",), delete=("step_00000002.pt",)),
        )
        # A stop is the state of the previous offer and adds no Archive.
        self.assertEqual(selector.decide(SavePoint("stop", 6, latest_is_current=True)), Decision())
        state = selector.state_dict()
        self.assertEqual(state, {"kept": ["step_00000004.pt", "step_00000006.pt"]})
        twin = PeriodicArchiveSelector(latest_every=1, archive_every=2, keep_last=2)
        twin.load_state_dict(json.loads(json.dumps(state)))
        self.assertEqual(
            twin.decide(SavePoint("step", 8)),
            Decision(latest=True, archive=("step_00000008.pt",), delete=("step_00000004.pt",)),
        )
        unbounded = PeriodicArchiveSelector(archive_every=1, name="every_{step}.pt")
        for step in range(1, 6):
            self.assertEqual(unbounded.decide(SavePoint("step", step)).delete, ())
        self.assertEqual(len(unbounded.state_dict()["kept"]), 5)
        with self.assertRaises(CheckpointError):
            twin.load_state_dict({"kept": [1]})
        with self.assertRaises(BuildError):
            PeriodicArchiveSelector(name="fixed.pt")
        with self.assertRaises(BuildError):
            PeriodicArchiveSelector(keep_last=0)

    def test_top_k_selector_decisions_and_state(self) -> None:
        selector = TopKSelector(metric="pesq", mode="max", k=2)
        self.assertEqual(selector.decide(SavePoint("step", 1, {"pesq": 9.0})), Decision(latest=True))
        self.assertEqual(
            selector.decide(SavePoint("boundary", 10, {"pesq": 2.0})),
            Decision(latest=True, archive=("best_00000010.pt",)),
        )
        self.assertEqual(
            selector.decide(SavePoint("boundary", 20, {"pesq": 3.0})),
            Decision(latest=True, archive=("best_00000020.pt",)),
        )
        # Worse than both kept boundaries: Latest only.
        self.assertEqual(selector.decide(SavePoint("boundary", 30, {"pesq": 1.0})), Decision(latest=True))
        # A tie does not displace the earlier step.
        self.assertEqual(selector.decide(SavePoint("boundary", 40, {"pesq": 2.0})), Decision(latest=True))
        # A missing or non-finite metric is not ranked and is not a zero.
        self.assertEqual(selector.decide(SavePoint("boundary", 50, {})), Decision(latest=True))
        self.assertEqual(selector.decide(SavePoint("boundary", 60, {"pesq": float("nan")})), Decision(latest=True))
        self.assertEqual(
            selector.decide(SavePoint("boundary", 70, {"pesq": 2.5})),
            Decision(latest=True, archive=("best_00000070.pt",), delete=("best_00000010.pt",)),
        )
        state = selector.state_dict()
        self.assertEqual([entry["name"] for entry in state["entries"]], ["best_00000020.pt", "best_00000070.pt"])
        twin = TopKSelector(metric="pesq", mode="max", k=2)
        twin.load_state_dict(json.loads(json.dumps(state)))
        self.assertEqual(
            twin.decide(SavePoint("boundary", 80, {"pesq": 4.0})),
            Decision(latest=True, archive=("best_00000080.pt",), delete=("best_00000070.pt",)),
        )
        lowest = TopKSelector(metric="loss", mode="min", k=1)
        self.assertEqual(lowest.decide(SavePoint("boundary", 1, {"loss": 0.5})).archive, ("best_00000001.pt",))
        self.assertEqual(lowest.decide(SavePoint("boundary", 2, {"loss": 0.7})).archive, ())
        self.assertEqual(
            lowest.decide(SavePoint("boundary", 3, {"loss": 0.1})),
            Decision(latest=True, archive=("best_00000003.pt",), delete=("best_00000001.pt",)),
        )
        with self.assertRaises(CheckpointError):
            twin.load_state_dict({"entries": [{"name": "x.pt", "step": True, "value": 1.0}]})
        with self.assertRaises(BuildError):
            TopKSelector(metric="loss", mode="best")
        with self.assertRaises(BuildError):
            TopKSelector(metric="")

    def test_scalar_metrics_leave_out_missing_values(self) -> None:
        summary = {
            "loss": 0.25,
            "pesq": {"status": "ok", "value": 2.5},
            "stoi": {"status": "na", "value": None},
            "wer": {"status": "unavailable", "value": 0.0},
            "count": 4,
            "flag": True,
            "nan": float("nan"),
            "name": "valid",
        }
        self.assertEqual(scalar_metrics(summary), {"loss": 0.25, "pesq": 2.5, "count": 4.0})
        self.assertEqual(scalar_metrics(None), {})

    def test_the_yaml_names_the_selector_and_its_settings(self) -> None:
        directory = self._directory()
        default = load_experiment(_write_yaml(directory, "default", steps=1, stop_after=None))
        self.assertEqual(default.selector_class, STANDARD)
        self.assertEqual(default.selector_kwargs, {})
        top = load_experiment(
            _write_yaml(directory, "top", steps=1, stop_after=None, selector=(TOP_K, {"metric": "loss", "k": 3}))
        )
        self.assertEqual(top.selector_class, TOP_K)
        selector = build_selector(top.selector_class, top.selector_kwargs)
        self.assertIsInstance(selector, TopKSelector)
        self.assertEqual(selector.k, 3)
        # The save policy is not part of the experiment identity.
        self.assertEqual(config_sha256(default), config_sha256(top))
        with self.assertRaises(BuildError):
            build_selector(STANDARD, {"latest_every": 0})
        with self.assertRaises(BuildError):
            build_selector(STANDARD, {"best_metric": "loss"})
        with self.assertRaises(BuildError):
            build_selector("tests.experiments.support.Marker", {})
        text = (directory / "default.yaml").read_text(encoding="utf-8")
        for name, section in {
            "unknown": {"checkpoint": {"interval": 1}},
            "missing": {"checkpoint": {}},
            "class": {"checkpoint": {"selector": {"class": "Standard"}}},
            "extra": {"checkpoint": {"selector": {"class": STANDARD, "mode": "max"}}},
        }.items():
            with self.subTest(name=name):
                path = directory / f"{name}.yaml"
                path.write_text(text + yaml.safe_dump(section), encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_experiment(path)
        # The trainer no longer owns a save period.
        old = directory / "old.yaml"
        old.write_text(text.replace("      accumulation: 1\n", "      accumulation: 1\n      checkpoint_interval: 1\n"), encoding="utf-8")
        with self.assertRaises(BuildError):
            build_experiment(load_experiment(old))

    def test_periodic_archives_follow_the_selector_and_survive_a_resume(self) -> None:
        selector = (PERIODIC, {"latest_every": 3, "archive_every": 2, "keep_last": 2})
        continuous_dir = self._directory()
        interrupted_dir = self._directory()
        _run(_write_yaml(continuous_dir, "run", steps=7, stop_after=None, selector=selector))
        _run(_write_yaml(interrupted_dir, "stopped", "run", steps=7, stop_after=5, selector=selector))
        stopped = interrupted_dir / "run"
        self.assertEqual(_archives(stopped), ["step_00000002.pt", "step_00000004.pt"])
        self.assertEqual(_load(stopped / "latest.pt")["logical_step"], 5)
        self.assertEqual(
            _load(stopped / "latest.pt")["selector"],
            {
                "class": "src.experiments.execution.selector.PeriodicArchiveSelector",
                "state": {"kept": ["step_00000002.pt", "step_00000004.pt"]},
            },
        )
        _run(_write_yaml(interrupted_dir, "resume", "run", steps=7, stop_after=None, selector=selector), resume=stopped)
        for out in (continuous_dir / "run", stopped):
            self.assertEqual(_archives(out), ["step_00000004.pt", "step_00000006.pt"])
            self.assertEqual(
                sorted(path.name for path in (out / "archive").iterdir()),
                [
                    "step_00000004.data_provenance.json",
                    "step_00000004.pt",
                    "step_00000006.data_provenance.json",
                    "step_00000006.pt",
                ],
            )
            self.assertEqual(sorted(path.name for path in out.glob("*.pt")), ["latest.pt"])
            for step in (4, 6):
                self.assertEqual(_load(out / "archive" / f"step_{step:08d}.pt")["logical_step"], step)
        left = _load(continuous_dir / "run" / "latest.pt")
        right = _load(stopped / "latest.pt")
        self.assertEqual(right["logical_step"], 7)
        _assert_equal(left["states"], right["states"])
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        _assert_equal(left["selector"], right["selector"])
        for step in (4, 6):
            _assert_equal(
                _load(continuous_dir / "run" / "archive" / f"step_{step:08d}.pt")["states"],
                _load(stopped / "archive" / f"step_{step:08d}.pt")["states"],
            )

    def test_top_k_archives_are_written_and_deleted_as_the_selector_says(self) -> None:
        directory = self._directory()
        config = _write_yaml(
            directory, "out", steps=9, stop_after=None, selector=(TOP_K, {"metric": "loss", "mode": "min", "k": 2})
        )
        experiment = build_experiment(load_experiment(config))
        self.addCleanup(experiment.close)
        trainer = experiment.get("trainer")
        manager = experiment._checkpoint
        out = directory / "out"
        expected = [
            (1, 0.5, ["best_00000001.pt"]),
            (2, 0.7, ["best_00000001.pt", "best_00000002.pt"]),
            (3, 0.3, ["best_00000001.pt", "best_00000003.pt"]),
            (4, 0.9, ["best_00000001.pt", "best_00000003.pt"]),
            (5, 0.1, ["best_00000003.pt", "best_00000005.pt"]),
        ]
        for step, loss, kept in expected:
            trainer.logical_step = step
            report = manager.offer("boundary", logical_step=step, metrics={"loss": loss})
            self.assertEqual(report.latest, out / "latest.pt")
            self.assertEqual(_archives(out), kept)
            self.assertEqual(_load(out / "latest.pt")["logical_step"], step)
        # Each Archive is a full checkpoint of the boundary it was selected at.
        best = _load(out / "archive" / "best_00000005.pt")
        self.assertEqual(best["logical_step"], 5)
        self.assertEqual(set(best["states"]), {"data", "model", "optimizer", "runtime", "scheduler", "trainer"})
        self.assertEqual(
            [entry["name"] for entry in best["selector"]["state"]["entries"]],
            ["best_00000005.pt", "best_00000003.pt"],
        )
        self.assertEqual(
            sorted(path.name for path in (out / "archive").iterdir()),
            [
                "best_00000003.data_provenance.json",
                "best_00000003.pt",
                "best_00000005.data_provenance.json",
                "best_00000005.pt",
            ],
        )
        with self.assertRaises(CheckpointError):
            manager.offer("boundary", logical_step=5, metrics={"loss": "low"})  # type: ignore[dict-item]

    def test_a_selector_cannot_reach_outside_the_archive_directory(self) -> None:
        directory = self._directory()
        config = _write_yaml(directory, "out", steps=1, stop_after=None)
        experiment = build_experiment(load_experiment(config))
        experiment.run()
        out = directory / "out"
        outside = directory / "outside.pt"
        outside.write_bytes(b"keep")
        before = _tree(directory)
        manager = experiment._checkpoint
        hostile = [
            Decision(delete=("../latest.pt",)),
            Decision(delete=("../../outside.pt",)),
            Decision(delete=(str(outside),)),
            Decision(delete=("sub/old.pt",)),
            Decision(delete=(".latest.pt",)),
            Decision(delete=("latest.data_provenance.json",)),
            Decision(archive=("../latest.pt",)),
            Decision(archive=("../outside.pt",)),
            Decision(archive=(str(outside),)),
            Decision(archive=("notes.txt",)),
            Decision(archive=("",)),
            Decision(archive=(5,)),  # type: ignore[arg-type]
            Decision(archive=("a.pt", "a.pt")),
            Decision(archive=("a.pt",), delete=("a.pt",)),
            Decision(latest="yes"),  # type: ignore[arg-type]
            "latest",
        ]
        for decision in hostile:
            with self.subTest(decision=decision):
                manager._selector = _Scripted(decision)
                with self.assertRaises(CheckpointError):
                    manager.offer("step", logical_step=1)
                self.assertEqual(_tree(directory), before)
        # ``latest.pt`` as an Archive name is a file inside archive/, not the Latest.
        manager._selector = _Scripted(Decision(archive=("latest.pt",)))
        report = manager.offer("step", logical_step=1)
        self.assertIsNone(report.latest)
        self.assertEqual(report.archives, (out / "archive" / "latest.pt",))
        manager._selector = _Scripted(Decision(delete=("latest.pt", "missing.pt")))
        manager.offer("step", logical_step=1)
        self.assertEqual(_tree(directory), before)


class InitializeTests(_Base):
    def setUp(self) -> None:
        self.source_dir = self._directory()
        _run(_write_yaml(self.source_dir, "out", steps=2, stop_after=None, selector=STEP_ARCHIVE))
        self.source = self.source_dir / "out" / "archive" / "step_00000002.pt"
        self.payload = _load(self.source)
        self.before = _tree(self.source_dir)

    def _new(self, states: list[str], *, steps: int = 1, stop_after: int | None = None, name: str = "out") -> Path:
        directory = self._directory()
        return _write_yaml(
            directory,
            name,
            steps=steps,
            stop_after=stop_after,
            seed=99,
            initialize={"checkpoint": str(self.source), "states": states},
        )

    def test_weights_only_start_a_new_experiment(self) -> None:
        config = self._new(["model"])
        loaded = load_experiment(config)
        fresh = build_experiment(loaded)
        report = fresh._checkpoint.restore(self.source, mode="initialize", states=["model"])
        self.assertEqual(report.initialized_states, ("model",))
        self.assertTrue(_tensors_equal(_module_tensors(self.payload), _live_tensors(fresh)))
        self.assertEqual(len(fresh.get("optimizer").state), 0)
        self.assertEqual(fresh.get("trainer").logical_step, 0)
        self.assertEqual(fresh.get("data").update_index, 0)
        # The RNG of the source is not restored: it is the new experiment's seed.
        restored_rng = torch.random.get_rng_state().clone()
        self.assertFalse(torch.equal(restored_rng, self.payload["rng_by_rank"][0]["torch"].cpu()))
        twin = build_experiment(load_experiment(self._new(["model"])))
        self.assertTrue(torch.equal(restored_rng, torch.random.get_rng_state()))
        twin.close()
        result = fresh.run()
        self.assertEqual(result.logical_step, 1)
        parameter = next(fresh.get("model").state_module.parameters())
        self.assertEqual(int(fresh.get("optimizer").state[parameter]["step"]), 1)

        out = config.parent / "out"
        record = json.loads((out / "initialization.json").read_text(encoding="utf-8"))
        self.assertEqual(record["format"], "audio-db-initialization-v1")
        self.assertEqual(record["checkpoint"], str(self.source.resolve()))
        self.assertEqual(record["checkpoint_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(record["states"], ["model"])
        self.assertEqual(record["logical_step"], 2)
        self.assertEqual(record["config_sha256"], self.payload["config_sha256"])
        provenance = json.loads((out / "data_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["initialized_from"]["checkpoint_sha256"], record["checkpoint_sha256"])
        self.assertEqual(provenance["initialized_from"]["states"], ["model"])
        saved = _load(out / "latest.pt")
        self.assertEqual(saved["logical_step"], 1)
        self.assertIsNone(saved["initialize"])
        self.assertEqual(saved["config_sha256"], config_sha256(loaded))
        # Another initial state is another experiment.
        plain = load_experiment(_write_yaml(self._directory(), "out", steps=1, stop_after=None, seed=99))
        self.assertNotEqual(config_sha256(plain), config_sha256(loaded))
        # The source experiment is read and never written.
        self.assertEqual(_tree(self.source_dir), self.before)

    def test_optimizer_and_scheduler_state_are_selectable(self) -> None:
        with_optimizer = build_experiment(load_experiment(self._new(["model", "optimizer"])))
        report = with_optimizer._checkpoint.restore(self.source, mode="initialize", states=["optimizer", "model"])
        self.assertEqual(report.initialized_states, ("model", "optimizer"))
        parameter = next(with_optimizer.get("model").state_module.parameters())
        self.assertEqual(int(with_optimizer.get("optimizer").state[parameter]["step"]), 2)
        self.assertEqual(with_optimizer.get("scheduler").last_epoch, 0)
        self.assertEqual(with_optimizer.run().logical_step, 1)
        self.assertEqual(int(with_optimizer.get("optimizer").state[parameter]["step"]), 3)

        everything = build_experiment(load_experiment(self._new(["model", "optimizer", "scheduler"])))
        everything._checkpoint.restore(self.source, mode="initialize", states=["model", "optimizer", "scheduler"])
        self.assertEqual(everything.get("scheduler").last_epoch, self.payload["states"]["scheduler"]["last_epoch"])
        self.assertEqual(
            everything.get("optimizer").param_groups[0]["lr"],
            self.payload["states"]["optimizer"]["param_groups"][0]["lr"],
        )
        self.assertEqual(everything.get("trainer").logical_step, 0)
        everything.close()
        self.assertEqual(_tree(self.source_dir), self.before)

    def test_incompatible_selections_are_refused(self) -> None:
        cases = {
            "optimizer without its module": (["optimizer"], CheckpointError, "without model"),
            "scheduler without its optimizer": (["model", "scheduler"], CheckpointError, "without optimizer"),
            "trainer state": (["model", "trainer"], CheckpointError, "resume"),
            "data position": (["model", "data"], CheckpointError, "resume"),
            "unknown state": (["model", "ghost"], StateError, "ghost"),
        }
        for name, (states, error, message) in cases.items():
            with self.subTest(name=name):
                config = self._new(states)
                experiment = build_experiment(load_experiment(config))
                weights = _live_tensors(experiment)
                with self.assertRaises(error) as caught:
                    experiment.run()
                self.assertIn(message, str(caught.exception))
                out = config.parent / "out"
                self.assertFalse((out / "latest.pt").exists())
                self.assertFalse((out / "initialization.json").exists())
                self.assertEqual(_status(out), "FAILED")
                del weights
        for name, section in {
            "empty": {"checkpoint": str(self.source), "states": []},
            "repeated": {"checkpoint": str(self.source), "states": ["model", "model"]},
            "no checkpoint": {"states": ["model"]},
            "unknown key": {"checkpoint": str(self.source), "states": ["model"], "allow": ["model"]},
        }.items():
            with self.subTest(name=name):
                with self.assertRaises(ConfigError):
                    load_experiment(_write_yaml(self._directory(), "out", steps=1, stop_after=None, initialize=section))
        # A shape that does not fit is refused by the module itself, before any training.
        wide = _write_yaml(
            self._directory(),
            "out",
            steps=1,
            stop_after=None,
            initialize={"checkpoint": str(self.source), "states": ["model"]},
        )
        wide.write_text(
            wide.read_text(encoding="utf-8").replace("out_features: 8", "out_features: 4"), encoding="utf-8"
        )
        with self.assertRaises(Exception) as caught:
            build_experiment(load_experiment(wide)).run()
        self.assertIn("size mismatch", str(caught.exception))
        missing = self._new(["model"])
        missing.write_text(
            missing.read_text(encoding="utf-8").replace(str(self.source), str(self.source) + ".missing"),
            encoding="utf-8",
        )
        with self.assertRaises(CheckpointError):
            build_experiment(load_experiment(missing)).run()
        self.assertEqual(_tree(self.source_dir), self.before)

    def test_the_source_directory_cannot_be_started_over(self) -> None:
        # Pointing a new run at the directory of the source experiment would rewind it.
        config = _write_yaml(
            self.source_dir,
            "again",
            "out",
            steps=3,
            stop_after=None,
            selector=STEP_ARCHIVE,
            initialize={"checkpoint": str(self.source_dir / "out" / "archive" / "step_00000001.pt"), "states": ["model"]},
        )
        before = _tree(self.source_dir)
        with self.assertRaises(CheckpointError) as caught:
            build_experiment(load_experiment(config)).run()
        self.assertIn("already holds a run", str(caught.exception))
        self.assertEqual(_tree(self.source_dir), before)

    def test_an_initialized_experiment_resumes_as_itself(self) -> None:
        states = ["model", "optimizer"]
        whole = self._new(states, steps=4)
        _run(whole)
        stopped = self._new(states, steps=4, stop_after=2, name="stopped")
        _run(stopped)
        out = stopped.parent / "stopped"
        self.assertEqual(_status(out), "INTERRUPTED")
        # Resume reads this experiment's own files. The source may be gone.
        moved = self.source_dir / "moved"
        (self.source_dir / "out").rename(moved)
        try:
            text = stopped.read_text(encoding="utf-8").replace("      stop_after_logical_steps: 2\n", "")
            stopped.write_text(text, encoding="utf-8")
            result = build_experiment(load_experiment(stopped)).run(resume=out)
        finally:
            moved.rename(self.source_dir / "out")
        self.assertEqual((result.stop_reason, result.logical_step), ("completed", 4))
        left = _load(whole.parent / "out" / "latest.pt")
        right = _load(out / "latest.pt")
        _assert_equal(left["states"], right["states"])
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(left["config_sha256"], right["config_sha256"])
        self.assertEqual(
            (whole.parent / "out" / "data_provenance.json").read_bytes(),
            (out / "data_provenance.json").read_bytes(),
        )
        self.assertEqual(_tree(self.source_dir), self.before)

    def test_the_cli_takes_the_source_from_the_yaml(self) -> None:
        config = self._new(["model"])
        completed = _cli(config)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        out = config.parent / "out"
        self.assertEqual(_status(out), "COMPLETED")
        self.assertTrue((out / "initialization.json").is_file())
        removed = _cli(config, "--initialize-from", str(self.source))
        self.assertEqual(removed.returncode, 2)
        self.assertEqual(_tree(self.source_dir), self.before)


class LegacyTests(_Base):
    def test_a_checkpoint_written_before_selectors_resumes(self) -> None:
        continuous_dir = self._directory()
        legacy_dir = self._directory()
        _run(_write_yaml(continuous_dir, "run", steps=4, stop_after=None))
        _run(_write_yaml(legacy_dir, "stopped", "run", steps=4, stop_after=2))
        out = legacy_dir / "run"
        # The former layout: ``step_XXXXXXXX.pt`` in the directory itself, no selector entry.
        payload = _load(out / "latest.pt")
        del payload["selector"]
        legacy = out / "step_00000002.pt"
        torch.save(payload, legacy)
        (out / "latest.data_provenance.json").rename(out / "step_00000002.data_provenance.json")
        (out / "latest.pt").unlink()
        original = legacy.read_bytes()
        whole = _write_yaml(legacy_dir, "whole", "run", steps=4, stop_after=None)
        with self.assertRaises(CheckpointError) as caught:
            _run(whole)
        self.assertIn("already holds a run (step_00000002.pt)", str(caught.exception))
        _run(whole, resume=legacy)
        self.assertEqual(_status(out), "COMPLETED")
        self.assertEqual(legacy.read_bytes(), original)
        left = _load(continuous_dir / "run" / "latest.pt")
        right = _load(out / "latest.pt")
        self.assertEqual(right["logical_step"], 4)
        _assert_equal(left["states"], right["states"])
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(_metrics(continuous_dir / "run"), _metrics(out))

    def test_the_identity_of_a_document_before_selectors_is_accepted(self) -> None:
        directory = self._directory()
        modern = load_experiment(
            _write_yaml(directory, "modern", steps=4, stop_after=None, selector=(STANDARD, {"latest_every": 3}))
        )
        text = _write_yaml(directory, "old", steps=4, stop_after=None).read_text(encoding="utf-8")
        accepted = {digest for _, digest in config_generations(modern)}
        self.assertEqual(config_generations(modern)[0][1], config_sha256(modern))
        for interval, expected in ((3, True), (None, True), (5, False)):
            with self.subTest(interval=interval):
                line = "" if interval is None else f"      checkpoint_interval: {interval}\n"
                path = directory / "old.yaml"
                path.write_text(text.replace("      accumulation: 1\n", "      accumulation: 1\n" + line), encoding="utf-8")
                self.assertEqual(config_sha256(load_experiment(path)) in accepted, expected)

    def test_the_former_partial_argument_is_part_of_the_accepted_identity(self) -> None:
        directory = self._directory()
        text = _write_yaml(directory, "old", steps=4, stop_after=None).read_text(encoding="utf-8")

        def old_identity(lines: str) -> str:
            path = directory / "old.yaml"
            path.write_text(text.replace("      accumulation: 1\n", "      accumulation: 1\n" + lines), encoding="utf-8")
            return config_sha256(load_experiment(path))

        for partial in (False, True):
            modern = load_experiment(
                _write_yaml(
                    directory,
                    f"modern-{partial}",
                    steps=4,
                    stop_after=None,
                    selector=(STANDARD, {"latest_every": 3, "partial": partial}),
                )
            )
            accepted = {digest for _, digest in config_generations(modern)}
            same = str(partial).lower()
            other = str(not partial).lower()
            with self.subTest(partial=partial):
                self.assertIn(old_identity(f"      checkpoint_interval: 3\n      checkpoint_partial: {same}\n"), accepted)
                self.assertIn(old_identity(f"      checkpoint_partial: {same}\n"), accepted)
                self.assertNotIn(old_identity(f"      checkpoint_interval: 3\n      checkpoint_partial: {other}\n"), accepted)

    def test_an_omitted_interval_matches_the_default_not_every_step(self) -> None:
        directory = self._directory()
        modern = load_experiment(_write_yaml(directory, "modern", steps=4, stop_after=None))
        self.assertEqual(modern.selector_kwargs, {})
        text = _write_yaml(directory, "old", steps=4, stop_after=None).read_text(encoding="utf-8")
        accepted = {digest for _, digest in config_generations(modern)}

        def old_identity(line: str) -> str:
            path = directory / "old.yaml"
            path.write_text(
                text.replace("      accumulation: 1\n", "      accumulation: 1\n" + line),
                encoding="utf-8",
            )
            return config_sha256(load_experiment(path))

        self.assertIn(old_identity(""), accepted)
        self.assertIn(old_identity(f"      checkpoint_interval: {DEFAULT_LATEST_EVERY}\n"), accepted)
        self.assertNotIn(old_identity("      checkpoint_interval: 1\n"), accepted)


class _Scripted(Selector):
    """Answers every offer with one fixed decision."""

    def __init__(self, decision: object) -> None:
        self._decision = decision

    def decide(self, point: SavePoint) -> Decision:
        del point
        return self._decision  # type: ignore[return-value]


def _write_yaml(
    directory: Path,
    folder: str,
    output: str | None = None,
    *,
    steps: int,
    stop_after: int | None,
    seed: int = 17,
    selector: tuple[str, dict[str, object]] | None = None,
    initialize: dict[str, object] | None = None,
    trainer: str = "src.experiments.trainers.StandardTrainer",
    trainer_kwargs: dict[str, object] | None = None,
) -> Path:
    stop = ""
    if stop_after is not None:
        stop = f"      stop_after_logical_steps: {stop_after}\n"
    extra = "".join(f"      {key}: {json.dumps(value)}\n" for key, value in (trainer_kwargs or {}).items())
    output_dir = directory / (output or folder)
    text = f"""schema: audio-db-experiment/v1
seed: {seed}
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
output:
  directory: "{output_dir}"
components:
  model:
    role: module
    class: tests.experiments.support.DropoutLinear
    kwargs: {{in_features: 8, out_features: 8, p: 0.5}}
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: 8
      examples: 32
      batch_size: 4
  optimizer:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: model
    kwargs: {{lr: 0.001}}
  scheduler:
    role: scheduler
    class: torch.optim.lr_scheduler.StepLR
    kwargs: {{step_size: 1, gamma: 0.5}}
    inject:
      optimizer: {{ref: optimizer}}
  trainer:
    role: trainer
    class: {trainer}
    kwargs:
      max_logical_steps: {steps}
      accumulation: 1
{extra}{stop}    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
      scheduler: {{ref: scheduler}}
"""
    if selector is not None:
        text += yaml.safe_dump({"checkpoint": {"selector": {"class": selector[0], "kwargs": selector[1]}}})
    if initialize is not None:
        text += yaml.safe_dump({"initialize": initialize})
    path = directory / f"{folder}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _kill_yaml(directory: Path, marker: Path) -> Path:
    return _write_yaml(
        directory,
        "out",
        steps=4,
        stop_after=None,
        trainer="tests.experiments.support.KillDuringSaveTrainer",
        trainer_kwargs={"kill_step": 3, "marker": str(marker)},
        selector=(STANDARD, {"latest_every": 1}),
    )


def _run(config: Path, *, resume: Path | None = None) -> None:
    experiment = build_experiment(load_experiment(config))
    try:
        experiment.run(resume=resume)
    finally:
        experiment.close()


def _cli(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _load(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _status(directory: Path) -> str:
    return json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"]


def _archives(directory: Path) -> list[str]:
    return sorted(path.name for path in (directory / "archive").glob("*.pt"))


def _tree(directory: Path) -> dict[str, str]:
    """Content hash of every file under ``directory``. The lock file has no content."""

    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != ".lock"
    }


def _metrics(directory: Path) -> list[dict[str, object]]:
    rows = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _replace(payload: dict[str, object], key: str, value: object) -> dict[str, object]:
    copied = dict(payload)
    copied[key] = value
    return copied


def _without_state(payload: dict[str, object], state_id: str) -> dict[str, object]:
    copied = dict(payload)
    states = dict(copied["states"])
    states.pop(state_id)
    copied["states"] = states
    copied["state_ids"] = [item for item in copied["state_ids"] if item != state_id]
    return copied


def _with_extra_state(payload: dict[str, object]) -> dict[str, object]:
    copied = dict(payload)
    states = dict(copied["states"])
    states["ghost"] = {"value": 1}
    copied["states"] = states
    copied["state_ids"] = [*list(copied["state_ids"]), "ghost"]
    return copied


def _replace_plan(payload: dict[str, object]) -> dict[str, object]:
    copied = dict(payload)
    states = dict(copied["states"])
    data = dict(states["data"])
    data["plan_hash"] = "0" * 64
    states["data"] = data
    copied["states"] = states
    return copied


def _module_tensors(payload: dict[str, object]) -> list[torch.Tensor]:
    tensors = []
    for value in payload["states"]["model"].values():
        if isinstance(value, torch.Tensor):
            tensors.append(value.detach().cpu().clone())
    return tensors


def _live_tensors(experiment: object) -> list[torch.Tensor]:
    module = experiment.get("model").state_module  # type: ignore[attr-defined]
    return [tensor.detach().cpu().clone() for tensor in module.state_dict().values() if isinstance(tensor, torch.Tensor)]


def _tensors_equal(left: list[torch.Tensor], right: list[torch.Tensor]) -> bool:
    return len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right))


def _max_abs(left: dict[str, object], right: dict[str, object]) -> float:
    values: list[float] = []

    def walk(lhs: object, rhs: object) -> None:
        if isinstance(lhs, torch.Tensor):
            values.append(float((lhs.detach().cpu() - rhs.detach().cpu()).abs().max()))  # type: ignore[union-attr]
            return
        if isinstance(lhs, dict):
            for key, value in lhs.items():
                walk(value, rhs[key])  # type: ignore[index]

    walk(left, right)
    return max(values)


def _assert_equal(left: object, right: object, path: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(path)
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(path)
        for key, value in left.items():
            _assert_equal(value, right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise AssertionError(path)
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            _assert_equal(lhs, rhs, f"{path}[{index}]")
        return
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise AssertionError(path)
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


if __name__ == "__main__":
    unittest.main()
