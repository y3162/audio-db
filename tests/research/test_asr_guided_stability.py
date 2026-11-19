"""Narrow multi-step resume and NCCL checks for ASR-guided SEMamba++ v1.

Hidden CUDA is NOT_RUN. The launches go through the existing launcher and
write only under a temporary directory. all_off is expected to resume
bitwise. ASR-on runs are measured even when TDT is not bitwise.

Set GPU_IDS to the comma-separated numbers of free GPUs:

    CUDA_VISIBLE_DEVICES="$GPU_IDS" scripts/setup/with_env.sh \\
      .venvs/se/bin/python -m tests.suite_guard \\
      .venvs/se/bin/python -m unittest tests.research.test_asr_guided_stability -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_NEMO_MARKERS = ("preprocessor", "joint.", "decoder.prediction", "fastconformer")
_PYTHON = _ROOT / ".venvs/se/bin/python"


def _visible() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return []
    if raw.strip() in {"", "-1"}:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _reason(needed: int) -> str | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        return "NOT_RUN: CUDA_VISIBLE_DEVICES is unset"
    devices = _visible()
    if not devices:
        return "NOT_RUN: CUDA is hidden"
    if len(devices) < needed:
        return f"NOT_RUN: need {needed} visible GPUs, saw {devices}"
    if not _PYTHON.is_file():
        return "integrated environment is absent"
    return None


def _launch(config: Path, *, nproc: int, resume: Path | None = None, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(_visible()[:nproc])
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(_ROOT)
    env["AUDIO_DB_GPU_MEMORY_INTERVAL"] = "0"
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    command = [
        str(_ROOT / "scripts/setup/with_env.sh"),
        str(_PYTHON),
        "-m",
        "src.experiments.launcher",
        "--config",
        str(config),
        "--nproc",
        str(nproc),
        "--gpu-memory-interval",
        "0",
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return subprocess.run(
        command,
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _yaml(directory: Path, *, steps: int, count: int, world: int, preflight: bool, stop_after: int | None, **switches) -> Path:
    from tests.research.test_asr_guided_loop import _research_yaml

    if switches.get("real_asr"):
        switches.setdefault("layer", 15)
    return _research_yaml(
        directory,
        steps=steps,
        count=count,
        world=world,
        validation_interval=0,
        device="cuda",
        length=16000,
        preflight=preflight,
        stop_after=stop_after,
        **switches,
    )


def _status(directory: Path) -> str:
    path = directory / "status.json"
    if not path.is_file():
        return "MISSING"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("status") or payload.get("state") or "MISSING")


def _metrics(directory: Path) -> list[dict]:
    path = directory / "metrics.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _gap(left, right) -> tuple[float, str]:
    import numpy as np
    import torch

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return 1.0, "array-type"
        if left.shape != right.shape or left.dtype != right.dtype:
            return 1.0, "array-shape"
        if not np.array_equal(left, right):
            if np.issubdtype(left.dtype, np.floating):
                return float(np.max(np.abs(left - right))), "array"
            return 1.0, "array"
        return 0.0, ""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return 1.0, "tensor-type"
        if left.shape != right.shape or left.dtype != right.dtype:
            return 1.0, "tensor-shape"
        if not left.dtype.is_floating_point:
            return (0.0, "") if torch.equal(left.cpu(), right.cpu()) else (1.0, "tensor-value")
        worst = float((left.detach().cpu() - right.detach().cpu()).abs().max())
        return worst, ""
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 1.0, "dict-type"
        if set(left) != set(right):
            return 1.0, "keys " + ",".join(sorted(set(left) ^ set(right))[:6])
        worst = 0.0
        where = ""
        for key in left:
            gap, place = _gap(left[key], right[key])
            if gap > worst:
                worst = gap
                where = f"{key}.{place}" if place else str(key)
        return worst, where
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
            return 1.0, "list"
        worst = 0.0
        where = ""
        for index, (item, other) in enumerate(zip(left, right, strict=True)):
            gap, place = _gap(item, other)
            if gap > worst:
                worst = gap
                where = f"[{index}].{place}" if place else f"[{index}]"
        return worst, where
    if left != right:
        return 1.0, "value"
    return 0.0, ""


def _checkpoint_report(left: Path, right: Path) -> dict:
    import torch

    lhs = torch.load(left, map_location="cpu", weights_only=False)
    rhs = torch.load(right, map_location="cpu", weights_only=False)
    report = {}
    for key in ("logical_step", "optimizer_updates", "label", "world_size", "parameter_ranks_agree"):
        report[key] = {"left": lhs.get(key), "right": rhs.get(key), "equal": lhs.get(key) == rhs.get(key)}
    gap, where = _gap(lhs.get("states"), rhs.get("states"))
    rng_gap, rng_where = _gap(lhs.get("rng_by_rank", lhs.get("rng")), rhs.get("rng_by_rank", rhs.get("rng")))
    keys = []
    for name, blob in lhs.get("states", {}).items():
        if isinstance(blob, dict):
            keys.extend(f"{name}.{key}" for key in blob)
    report["state_max_abs"] = gap
    report["state_where"] = where
    report["rng_max_abs"] = rng_gap
    report["rng_where"] = rng_where
    report["nemo_keys"] = [key for key in keys if any(marker in key for marker in _NEMO_MARKERS)]
    report["bitwise"] = gap == 0.0 and rng_gap == 0.0 and all(item["equal"] for item in report.values() if isinstance(item, dict))
    return report


def _metric_report(left: list[dict], right: list[dict]) -> dict:
    numeric = (
        "generator",
        "discriminator",
        "tdt",
        "error_mask_fraction",
        "adversarial",
        "matching",
        "mel",
        "magnitude",
        "phase",
        "complex",
        "consistency",
    )
    rows = []
    for index, (first, second) in enumerate(zip(left, right, strict=False)):
        gaps = {}
        for key in numeric:
            if key not in first and key not in second:
                continue
            if key not in first or key not in second:
                gaps[key] = "missing"
                continue
            if isinstance(first[key], (int, float)) and isinstance(second[key], (int, float)):
                gaps[key] = abs(float(first[key]) - float(second[key]))
        rows.append(
            {
                "step": index + 1,
                "sample_id_equal": first.get("sample_id") == second.get("sample_id"),
                "sample_id": [first.get("sample_id"), second.get("sample_id")],
                "gaps": gaps,
            }
        )
    return {"count": [len(left), len(right)], "rows": rows}


def _tail(completed: subprocess.CompletedProcess[str]) -> str:
    return (completed.stderr or "")[-1500:] + (completed.stdout or "")[-500:]


def _resume(name: str, directory: Path, *, world: int, steps: int, **switches) -> dict:
    continuous = directory / "continuous"
    interrupted = directory / "interrupted"
    _yaml(continuous, steps=steps, count=steps, world=world, preflight=False, stop_after=None, **switches)
    _yaml(interrupted, steps=steps, count=steps, world=world, preflight=False, stop_after=max(steps // 2, 1), **switches)
    first = _launch(continuous / "experiment.yaml", nproc=world)
    mid = _launch(interrupted / "experiment.yaml", nproc=world)
    mid_status = _status(interrupted)
    _yaml(interrupted, steps=steps, count=steps, world=world, preflight=False, stop_after=None, **switches)
    second = _launch(interrupted / "experiment.yaml", nproc=world, resume=interrupted)
    report = {
        "name": name,
        "world": world,
        "steps": steps,
        "switches": switches,
        "continuous_code": first.returncode,
        "interrupted_code": mid.returncode,
        "resume_code": second.returncode,
        "continuous_status": _status(continuous),
        "mid_status": mid_status,
        "resume_status": _status(interrupted),
    }
    if first.returncode != 0:
        report["continuous_tail"] = _tail(first)
    if mid.returncode not in {0, 3}:
        report["interrupted_tail"] = _tail(mid)
    if second.returncode != 0:
        report["resume_tail"] = _tail(second)
    left_metrics = _metrics(continuous)
    right_metrics = _metrics(interrupted)
    report["metrics"] = _metric_report(left_metrics, right_metrics)
    left_step = continuous / "archive" / f"step_{steps:08d}.pt"
    right_step = interrupted / "archive" / f"step_{steps:08d}.pt"
    if left_step.is_file() and right_step.is_file():
        report["checkpoint"] = _checkpoint_report(left_step, right_step)
    else:
        report["checkpoint"] = {"missing": [left_step.is_file(), right_step.is_file()]}
    hypotheses = []
    for folder in (continuous, interrupted):
        files = sorted(folder.glob("asr_hypotheses.rank*.jsonl"))
        hypotheses.append([path.read_text(encoding="utf-8") for path in files])
    report["hypotheses_equal"] = hypotheses[0] == hypotheses[1]
    report["hypothesis_files"] = [len(item) for item in hypotheses]
    print("STABILITY " + json.dumps(report, default=str), flush=True)
    return report


def _assert_ran(test: unittest.TestCase, report: dict, *, bitwise: bool) -> None:
    test.assertEqual(report["continuous_code"], 0, report.get("continuous_tail"))
    test.assertEqual(report["interrupted_code"], 3, report.get("interrupted_tail"))
    test.assertEqual(report["resume_code"], 0, report.get("resume_tail"))
    test.assertEqual(report["continuous_status"], "COMPLETED")
    test.assertEqual(report["mid_status"], "INTERRUPTED")
    test.assertEqual(report["resume_status"], "COMPLETED")
    checkpoint = report["checkpoint"]
    test.assertNotIn("missing", checkpoint, checkpoint)
    test.assertEqual(checkpoint.get("nemo_keys"), [])
    if report["world"] > 1:
        test.assertTrue(checkpoint["parameter_ranks_agree"]["left"])
        test.assertTrue(checkpoint["parameter_ranks_agree"]["right"])
    if bitwise:
        test.assertTrue(checkpoint["bitwise"], checkpoint)
        test.assertEqual(report["metrics"]["count"][0], report["metrics"]["count"][1])
        for row in report["metrics"]["rows"]:
            test.assertTrue(row["sample_id_equal"], row)
            for key, gap in row["gaps"].items():
                test.assertEqual(gap, 0.0, f"{key} {row}")


class NarrowResumeTest(unittest.TestCase):
    def _run(self, name: str, *, bitwise: bool, **switches) -> None:
        reason = _reason(1)
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory(prefix=f"asr-stability-{name}-") as directory:
            report = _resume(name, Path(directory), world=1, steps=4, **switches)
        _assert_ran(self, report, bitwise=bitwise)

    def test_all_off(self) -> None:
        self._run("all_off", bitwise=True)

    def test_features(self) -> None:
        self._run("features", bitwise=False, features=True, real_asr=True)

    def test_error(self) -> None:
        self._run("error", bitwise=False, error=True, real_asr=True)

    def test_tdt(self) -> None:
        self._run("tdt", bitwise=False, tdt=True, real_asr=True)

    def test_all_on(self) -> None:
        self._run("all_on", bitwise=False, features=True, error=True, tdt=True, real_asr=True)

    def test_preflight_on_and_off(self) -> None:
        reason = _reason(1)
        if reason:
            self.skipTest(reason)
        from tests.research.test_asr_guided_loop import _research_yaml

        reports = []
        for name, switches in (("tdt", dict(tdt=True)), ("all_on", dict(features=True, error=True, tdt=True))):
            with tempfile.TemporaryDirectory(prefix=f"asr-preflight-{name}-") as raw:
                root = Path(raw)
                enabled = root / "on"
                disabled = root / "off"
                _research_yaml(
                    enabled,
                    steps=2,
                    count=2,
                    world=1,
                    validation_interval=0,
                    device="cuda",
                    length=16000,
                    preflight=True,
                    real_asr=True,
                    layer=15,
                    **switches,
                )
                _research_yaml(
                    disabled,
                    steps=2,
                    count=2,
                    world=1,
                    validation_interval=0,
                    device="cuda",
                    length=16000,
                    preflight=False,
                    real_asr=True,
                    layer=15,
                    **switches,
                )
                first = _launch(enabled / "experiment.yaml", nproc=1)
                second = _launch(disabled / "experiment.yaml", nproc=1)
                report = {
                    "name": name,
                    "on_code": first.returncode,
                    "off_code": second.returncode,
                    "on_status": _status(enabled),
                    "off_status": _status(disabled),
                    "metrics": _metric_report(_metrics(enabled), _metrics(disabled)),
                }
                if first.returncode != 0:
                    report["on_tail"] = _tail(first)
                if second.returncode != 0:
                    report["off_tail"] = _tail(second)
                print("STABILITY_PREFLIGHT " + json.dumps(report, default=str), flush=True)
                reports.append(report)
        for report in reports:
            self.assertEqual(report["on_code"], 0, report.get("on_tail"))
            self.assertEqual(report["off_code"], 0, report.get("off_tail"))
            self.assertEqual(report["metrics"]["count"], [2, 2])


class NCCLResumeTest(unittest.TestCase):
    def test_training_and_resume(self) -> None:
        reason = _reason(2)
        if reason:
            self.skipTest(reason)
        reports = []
        with tempfile.TemporaryDirectory(prefix="asr-nccl-") as raw:
            root = Path(raw)
            for name, bitwise, switches in (
                ("all_off", True, {}),
                ("tdt", False, dict(tdt=True, real_asr=True)),
                ("all_on", False, dict(features=True, error=True, tdt=True, real_asr=True)),
            ):
                report = _resume(f"nccl-{name}", root / name, world=2, steps=2, **switches)
                reports.append((bitwise, report))
        for bitwise, report in reports:
            _assert_ran(self, report, bitwise=bitwise)


class OfficialWidthTest(unittest.TestCase):
    def test_all_off_matches_the_official_trainer(self) -> None:
        self._compare("reference")

    def test_fused_all_off_matches_when_it_runs(self) -> None:
        self._compare("fused")

    def _compare(self, backend: str) -> None:
        reason = _reason(1)
        if reason:
            self.skipTest(reason)
        source = _ROOT / "configs/research/asr_guided_se/semambapp/v1/all_off.yaml"
        if not source.is_file():
            self.skipTest("all_off.yaml is absent")
        base = source.read_text(encoding="utf-8")
        if backend == "fused":
            base = base.replace("scan_backend: reference", "scan_backend: fused")
        with tempfile.TemporaryDirectory(prefix="asr-official-width-") as raw:
            root = Path(raw)
            guided_dir = root / "guided"
            official_dir = root / "official"
            guided_dir.mkdir()
            official_dir.mkdir()
            guided = base.replace("device: cpu", "device: cuda", 1).replace(
                "directory: runs/asr-guided-se/semambapp-v1/all-off",
                f"directory: {guided_dir}",
                1,
            )
            official = guided.replace(
                "class: src.research.asr_guided_se.semambapp.v1.model.GuidedSEMambaPPGenerator",
                "class: src.models.semambapp.models.SEMambaPPGenerator",
                1,
            )
            official = official.replace(
                "class: src.research.asr_guided_se.semambapp.v1.trainer.GuidedSEMambaPPTrainer",
                "class: src.models.semambapp.trainer.SEMambaPPTrainer",
                1,
            )
            official = official.replace("      experiment_claim: research\n", "", 1)
            official = official.replace(
                """    prepare:
      trainable: true
      distribute: ddp
    inject:
      guidance: {ref: guidance}
  mrd:""",
                """    prepare:
      trainable: true
      distribute: ddp
  mrd:""",
                1,
            )
            official = official.replace("      guidance: {ref: guidance}\n", "", 1)
            guidance = """  guidance:
    role: other
    class: src.research.asr_guided_se.semambapp.v1.config.GuidanceV1
    kwargs:
      asr_features_enabled: false
      encoder_layer: 15
      error_weight_enabled: false
      error_weight_alpha: 0.0
      tdt_loss_enabled: false
"""
            official = official.replace(guidance, "", 1)
            official = official.replace(
                f"directory: {guided_dir}",
                f"directory: {official_dir}",
                1,
            )
            guided_yaml = guided_dir / "experiment.yaml"
            official_yaml = official_dir / "experiment.yaml"
            guided_yaml.write_text(guided, encoding="utf-8")
            official_yaml.write_text(official, encoding="utf-8")
            first = _launch(guided_yaml, nproc=1, timeout=2400)
            second = _launch(official_yaml, nproc=1, timeout=2400)
            report = {
                "guided_code": first.returncode,
                "official_code": second.returncode,
                "guided_status": _status(guided_dir),
                "official_status": _status(official_dir),
                "metrics": _metric_report(_metrics(guided_dir), _metrics(official_dir)),
                "width": "hid_48_tfmamba_4_window_48000",
                "scan_backend": backend,
                "guided_scan": {
                    "configured": (_metrics(guided_dir) or [{}])[0].get("scan_configured"),
                    "executed": (_metrics(guided_dir) or [{}])[0].get("scan_executed"),
                },
                "official_scan": {
                    "configured": (_metrics(official_dir) or [{}])[0].get("scan_configured"),
                    "executed": (_metrics(official_dir) or [{}])[0].get("scan_executed"),
                },
                "device": "cuda",
            }
            if first.returncode != 0:
                report["guided_tail"] = _tail(first)
            if second.returncode != 0:
                report["official_tail"] = _tail(second)
            left = guided_dir / "latest.pt"
            right = official_dir / "latest.pt"
            if left.is_file() and right.is_file():
                report["checkpoint"] = _checkpoint_report(left, right)
            else:
                report["checkpoint"] = {"missing": [left.is_file(), right.is_file()]}
            print("OFFICIAL_WIDTH " + json.dumps(report, default=str), flush=True)
        self.assertEqual(report["guided_code"], 0, report.get("guided_tail"))
        self.assertEqual(report["official_code"], 0, report.get("official_tail"))
        self.assertEqual(report["guided_status"], "COMPLETED")
        self.assertEqual(report["official_status"], "COMPLETED")
        checkpoint = report["checkpoint"]
        self.assertNotIn("missing", checkpoint, checkpoint)
        self.assertEqual(report["guided_scan"]["executed"], backend)
        self.assertEqual(report["official_scan"]["executed"], backend)
        if backend == "reference":
            self.assertTrue(checkpoint["bitwise"], checkpoint)
        self.assertEqual(report["metrics"]["count"], [1, 1])
        for row in report["metrics"]["rows"]:
            for key, gap in row["gaps"].items():
                self.assertEqual(gap, 0.0, f"{key} {row}")


if __name__ == "__main__":
    unittest.main()
