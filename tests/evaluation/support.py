"""Shared builders for evaluation tests."""

from __future__ import annotations

from src.evaluation.config import config_from_dict


def experiment_config(root, run_id="run", **overrides):
    raw = {
        "experiment": {"name": "unit", "seed": 0, "limit": 2},
        "dataset": {
            "source": "synthetic",
            "count": 2,
            "sample_rate": 16000,
            "seconds": 1.0,
            "seed": 0,
        },
        "mixing": {"enabled": True, "snr_db": [0.0], "prevent_clip": True, "hard_clip": False},
        "models": [{"name": "mock"}],
        "asr": {
            "enabled": True,
            "name": "mock",
            "default_text": "sample 0",
            "conditions": ["noisy", "enhanced"],
        },
        "metrics": {"wer": True, "pesq": True, "stoi": True, "estoi": False, "si_sdr": False},
        "output": {"root": str(root), "run_id": run_id, "save_audio": False, "resume": False},
        "runtime": {"device": "cpu"},
    }
    for key, value in overrides.items():
        raw[key] = value
    return config_from_dict(raw)
