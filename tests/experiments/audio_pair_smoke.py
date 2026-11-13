"""One common-contract step for each official trainer.

The models are the reduced architectures already used by the trainer tests.
The data object is AudioPairBridge, so the run declares a sample rate and a
teacher and does not name a dataset algorithm. Writes stay outside data/.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import torch
import yaml

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("AUDIO_PAIR_SMOKE_DIR", "/tmp/audio-db-pair-smoke"))
DEVICE = os.environ.get("AUDIO_PAIR_SMOKE_DEVICE", "cuda")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"device": DEVICE, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "models": []}
    jobs = (
        ("mp_senet", _mp_senet, 16000, "dry_clean"),
        ("mossformergan", _mossformer, 16000, "dry_clean"),
        ("semambapp", _semamba, 16000, "dry_clean"),
        ("tf_gridnet", _grid, 16000, "dry_clean"),
        ("tf_locoformer", _loco, 16000, "dry_clean"),
        ("deepfilternet3", _dfn, 48000, "official_mixed_target"),
    )
    only = os.environ.get("AUDIO_PAIR_SMOKE_ONLY")
    failed = False
    for name, writer, rate, teacher in jobs:
        if only and name != only:
            continue
        row = _run_pair(name, writer, sample_rate=rate, teacher=teacher, origin="synthetic")
        report["models"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["status"] != "PASS":
            failed = True
    if not failed and (only is None or only == "tf_gridnet"):
        for origin in ("voicebank", "ms_snsd"):
            row = _run_pair("tf_gridnet", _grid, sample_rate=16000, teacher="dry_clean", origin=origin, length=2048)
            row["role"] = "independence"
            report["models"].append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["status"] != "PASS":
                failed = True
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report {path}", flush=True)
    return 1 if failed else 0


def _run_pair(name: str, writer, *, sample_rate: int, teacher: str, origin: str, length: int | None = None) -> dict[str, object]:
    label = name if origin == "synthetic" else f"{name}-{origin}"
    directory = OUT / label
    row: dict[str, object] = {
        "model": name,
        "origin": origin,
        "sample_rate": sample_rate,
        "teacher": teacher,
        "status": "FAIL",
        "directory": str(directory),
    }
    try:
        if directory.exists():
            raise RuntimeError(f"{directory} already exists")
        directory.mkdir(parents=True)
        config = writer(directory / "first")
        _install_bridge(config, sample_rate=sample_rate, teacher=teacher, origin=origin, length=length)
        first = build_experiment(load_experiment(config))
        try:
            stopped = first.run()
        finally:
            first.close()
        row["interrupted"] = {"stop_reason": stopped.stop_reason, "logical_step": stopped.logical_step, "optimizer_updates": stopped.optimizer_updates}
        checkpoint = directory / "first" / "archive" / "step_00000001.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"missing checkpoint {checkpoint}")
        resumed_config = writer(directory / "resume-config")
        _install_bridge(resumed_config, sample_rate=sample_rate, teacher=teacher, origin=origin, length=length, stop_after=None)
        _retarget(resumed_config, directory / "first")
        second = build_experiment(load_experiment(resumed_config))
        try:
            resumed = second.run(resume=checkpoint)
        finally:
            second.close()
        row["resumed"] = {"stop_reason": resumed.stop_reason, "logical_step": resumed.logical_step, "optimizer_updates": resumed.optimizer_updates}
        evidence = _validation_evidence(directory / "first")
        row["validation"] = evidence
        if stopped.stop_reason != "interrupted" or stopped.logical_step != 1:
            raise RuntimeError(f"first run was {stopped.stop_reason} at {stopped.logical_step}")
        if resumed.stop_reason != "completed" or resumed.logical_step != 2:
            raise RuntimeError(f"resume was {resumed.stop_reason} at {resumed.logical_step}")
        if not evidence["ran"]:
            raise RuntimeError(f"validation did not run: {evidence}")
        updates = resumed.optimizer_updates or {}
        if not updates or any(int(value) < 1 for value in updates.values()):
            raise RuntimeError(f"optimizer did not step: {updates}")
        row["status"] = "PASS"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["trace"] = traceback.format_exc(limit=20)
    finally:
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def _install_bridge(path: Path, *, sample_rate: int, teacher: str, origin: str, length: int | None, stop_after: int | None = 1) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    runtime = document["runtime"]["kwargs"]
    runtime["device"] = DEVICE
    runtime["timeout_seconds"] = 600
    document["deterministic"] = False
    current = document["components"]["data"]["kwargs"]
    chosen = length if length is not None else int(current.get("length", 4800 if sample_rate == 48000 else 2048))
    document["components"]["data"] = {
        "role": "data",
        "class": "tests.experiments.audio_pair_bridge.AudioPairBridge",
        "kwargs": {
            "sample_rate": sample_rate,
            "teacher": teacher,
            "length": chosen,
            "count": 2,
            "seed": 17,
            "origin": origin,
        },
    }
    trainer = document["components"]["trainer"]["kwargs"]
    if stop_after is None:
        trainer.pop("stop_after_logical_steps", None)
    else:
        trainer["stop_after_logical_steps"] = stop_after
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _retarget(path: Path, output: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["output"]["directory"] = str(output)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _validation_evidence(directory: Path) -> dict[str, object]:
    payload = torch.load(directory / "archive" / "step_00000002.pt", map_location="cpu", weights_only=False)
    trainer = payload["states"]["trainer"]
    history = trainer.get("history") or []
    boundaries = trainer.get("completed_boundaries") or []
    boundary = trainer.get("boundary") if isinstance(trainer.get("boundary"), dict) else {}
    ran = bool(history or boundaries)
    if boundary.get("validation_done") or boundary.get("validation_summary") is not None:
        ran = True
    if boundary.get("pesq_status") not in (None,):
        ran = True
    losses = trainer.get("losses") or []
    return {
        "ran": ran,
        "history": len(history),
        "completed_boundaries": len(boundaries),
        "losses": [float(item) for item in losses],
        "logical_step": trainer.get("logical_step"),
    }


def _mp_senet(directory: Path) -> Path:
    from tests.experiments.test_mp_senet_gan import _yaml

    directory.mkdir(parents=True, exist_ok=True)
    return _yaml(directory, steps=2, updates=2, validation_interval=2, best_start=0, stop_after=1, accumulation=1, timeout=600)


def _mossformer(directory: Path) -> Path:
    from tests.experiments.test_mossformergan import _yaml

    directory.mkdir(parents=True, exist_ok=True)
    # PESQ refuses buffers shorter than 0.25 s. 4000 samples at 16 kHz clears that.
    return _yaml(
        directory,
        steps=2,
        updates=2,
        accumulation=1,
        length=4000,
        validate=True,
        stop_after=1,
        timeout=600,
        device=DEVICE,
        deterministic=False,
    )


def _semamba(directory: Path) -> Path:
    from tests.experiments.test_semambapp import _yaml

    directory.mkdir(parents=True, exist_ok=True)
    return _yaml(directory, steps=2, count=2, accumulation=1, validation_interval=2, best_start=0, stop_after=1, device=DEVICE, deterministic=False)


def _grid(directory: Path) -> Path:
    from tests.experiments.test_tf_gridnet_training import _scripted_yaml

    directory.mkdir(parents=True, exist_ok=True)
    return _scripted_yaml(directory, steps=2, updates=2, max_epochs=1, stop_after=1)


def _loco(directory: Path) -> Path:
    from tests.experiments.test_tf_locoformer_training import _scripted_yaml

    directory.mkdir(parents=True, exist_ok=True)
    return _scripted_yaml(directory, steps=2, updates=2, max_epochs=1, stop_after=1)


def _dfn(directory: Path) -> Path:
    from tests.experiments.test_deepfilternet3 import _tiny_yaml

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(_tiny_yaml(directory, stop_after=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
