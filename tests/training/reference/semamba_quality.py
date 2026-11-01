"""Quality-metric helpers for the SEMamba++ official comparison.

PESQ and STOI are the libraries in ``.venvs/lab``. This module does not import
the reference tree and does not install a dummy ``pesq`` module.

Tolerances below were chosen before any score in
``data/results/training/semamba-quality-20261101/`` was computed. They are an
equivalence band for a short run, not a claim about a fully trained model.

    .venvs/lab/bin/python tests/training/reference/semamba_quality.py prepare \\
        --out data/results/training/semamba-quality-20261101/phase-a
    .venvs/lab/bin/python tests/training/reference/semamba_quality.py score \\
        --run data/results/training/semamba-quality-20261101/phase-a
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

# Fixed before measurement. Do not retune after reading scores.
TOLERANCES = {
    "pesq_abs": 0.10,
    "stoi_abs": 0.02,
    "si_sdr_abs_db": 1.0,
}
JUDGED_UPDATES = 20
METRIC_KEYS = ("pesq", "stoi", "si_sdr")
MIN_VALID_SECONDS = 3.0
MAX_VALID_SECONDS = 10.0
SAMPLE_RATE = 16000


def choose_max_seconds(source_seconds: float) -> float:
    """Length stored in a validation recipe.

    Sources shorter than 3 seconds are rejected. Longer files are capped at
    10 seconds. The value is rounded to 6 decimal places, matching recipe
    canonicalization.
    """
    if source_seconds < MIN_VALID_SECONDS:
        raise ValueError(f"validation source is {source_seconds:.3f}s, shorter than {MIN_VALID_SECONDS:.0f}s")
    chosen = min(float(source_seconds), MAX_VALID_SECONDS)
    return float(f"{chosen:.6f}")


def energy_normalize(noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Match ``src.training.data.dataset._normalize`` and return the scale.

    The scale is ``sqrt(N / energy(noisy))`` applied to both waveforms. It is
    the training input level, not a peak normalize and not a clip.
    """
    noisy = noisy.detach().to(dtype=torch.float32).reshape(-1).contiguous()
    clean = clean.detach().to(dtype=torch.float32).reshape(-1).contiguous()
    if noisy.shape != clean.shape:
        raise ValueError(f"noisy {tuple(noisy.shape)} and clean {tuple(clean.shape)} differ")
    energy = torch.sum(noisy * noisy)
    if not torch.isfinite(energy) or float(energy) <= 0.0:
        raise ValueError("noisy waveform has no energy")
    scale = torch.sqrt(noisy.new_tensor(float(noisy.numel())) / energy)
    value = float(scale)
    return noisy * scale, clean * scale, value


def tensor_sha256(pairs: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in pairs:
        array = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def module_sha256(module: torch.nn.Module) -> str:
    items = sorted(module.state_dict().items(), key=lambda item: item[0])
    return tensor_sha256(items)


def waveform_gap(reference: torch.Tensor, other: torch.Tensor) -> dict[str, Any]:
    """Cosine and max-abs on the overlapping prefix. Non-finite samples are not zero."""
    left = reference.detach().to(dtype=torch.float32).reshape(-1).cpu()
    right = other.detach().to(dtype=torch.float32).reshape(-1).cpu()
    trimmed = abs(int(left.numel() - right.numel()))
    length = min(int(left.numel()), int(right.numel()))
    if length <= 0:
        return {"status": "na", "reason": "empty audio", "max_abs": None, "cosine": None, "trimmed_samples": trimmed}
    left = left[:length]
    right = right[:length]
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        return {
            "status": "na",
            "reason": "non_finite",
            "max_abs": None,
            "cosine": None,
            "trimmed_samples": trimmed,
        }
    delta = (left - right).abs()
    cosine = 1.0 if left.numel() < 2 else float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    return {
        "status": "ok",
        "reason": None,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "cosine": cosine,
        "trimmed_samples": trimmed,
    }


def _metric_value(result: dict[str, Any]) -> float | None:
    if result.get("status") != "ok":
        return None
    value = result.get("value")
    if value is None:
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """``left - right``. A missing score stays missing. It is not replaced by 0."""
    a = _metric_value(left)
    b = _metric_value(right)
    if a is None or b is None:
        reason = left.get("reason") or right.get("reason") or left.get("status") or right.get("status")
        return {"status": "na", "reason": f"missing_score:{reason}", "value": None}
    return {"status": "ok", "reason": None, "value": a - b}


def score_waveform(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    """PESQ, STOI, and SI-SDR. Importing this uses the real metric modules."""
    from src.evaluation.metrics.pesq import score_pesq
    from src.evaluation.metrics.sisdr import score_si_sdr
    from src.evaluation.metrics.stoi import score_stoi

    if not torch.isfinite(reference).all() or not torch.isfinite(estimate).all():
        missing = {"status": "na", "reason": "non_finite", "value": None}
        return {"pesq": dict(missing), "stoi": dict(missing), "si_sdr": dict(missing)}
    return {
        "pesq": score_pesq(reference, estimate, sample_rate),
        "stoi": score_stoi(reference, estimate, sample_rate, extended=False),
        "si_sdr": score_si_sdr(reference, estimate),
    }


def score_conditions(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    official: torch.Tensor,
    independent: torch.Tensor,
    sample_rate: int,
) -> dict[str, Any]:
    conditions = {
        "noisy": score_waveform(clean, noisy, sample_rate),
        "official": score_waveform(clean, official, sample_rate),
        "independent": score_waveform(clean, independent, sample_rate),
    }
    delta = {key: _difference(conditions["official"][key], conditions["independent"][key]) for key in METRIC_KEYS}
    improvement = {
        "official": {key: _difference(conditions["official"][key], conditions["noisy"][key]) for key in METRIC_KEYS},
        "independent": {
            key: _difference(conditions["independent"][key], conditions["noisy"][key]) for key in METRIC_KEYS
        },
    }
    return {
        "conditions": conditions,
        "official_minus_independent": delta,
        "improvement_over_noisy": improvement,
        "wave": waveform_gap(official, independent),
        "alignment": {
            "samples": {name: int(signal.numel()) for name, signal in {
                "clean": clean, "noisy": noisy, "official": official, "independent": independent
            }.items()},
            "same_length": len({int(signal.numel()) for signal in (clean, noisy, official, independent)}) == 1,
        },
    }


def _mean(values: list[float | None]) -> dict[str, Any]:
    finite = [float(item) for item in values if item is not None]
    if not finite:
        return {"status": "na", "reason": "no_finite_scores", "value": None, "n": 0, "n_missing": len(values)}
    return {
        "status": "ok" if len(finite) == len(values) else "partial",
        "reason": None if len(finite) == len(values) else "missing_scores_omitted",
        "value": sum(finite) / len(finite),
        "n": len(finite),
        "n_missing": len(values) - len(finite),
    }


def summarize_step(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-metric means. Missing scores are omitted from the mean and counted."""
    summary: dict[str, Any] = {"n": len(utterances), "metrics": {}}
    for key, limit_name in zip(METRIC_KEYS, ("pesq_abs", "stoi_abs", "si_sdr_abs_db"), strict=True):
        official = [_metric_value(item["conditions"]["official"][key]) for item in utterances]
        independent = [_metric_value(item["conditions"]["independent"][key]) for item in utterances]
        noisy = [_metric_value(item["conditions"]["noisy"][key]) for item in utterances]
        delta = [_metric_value(item["official_minus_independent"][key]) for item in utterances]
        abs_delta = [None if item is None else abs(item) for item in delta]
        gain_o = [_metric_value(item["improvement_over_noisy"]["official"][key]) for item in utterances]
        gain_i = [_metric_value(item["improvement_over_noisy"]["independent"][key]) for item in utterances]
        summary["metrics"][key] = {
            "noisy": _mean(noisy),
            "official": _mean(official),
            "independent": _mean(independent),
            "official_minus_independent": _mean(delta),
            "abs_official_minus_independent": _mean(abs_delta),
            "official_improvement": _mean(gain_o),
            "independent_improvement": _mean(gain_i),
            "tolerance": TOLERANCES[limit_name],
        }
    cosines = [item["wave"]["cosine"] if item["wave"]["status"] == "ok" else None for item in utterances]
    max_abs = [item["wave"]["max_abs"] if item["wave"]["status"] == "ok" else None for item in utterances]
    summary["wave"] = {"cosine": _mean(cosines), "max_abs": _mean(max_abs)}
    return summary


def tolerance_result(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass only when every utterance is finite and inside the pre-set band.

    A missing score fails the check. It is not treated as a zero gap. The mean
    is reported separately and is not a substitute for the per-utterance check.
    """
    failures: list[str] = []
    if not utterances:
        failures.append("no_utterances")
    limits = {"pesq": "pesq_abs", "stoi": "stoi_abs", "si_sdr": "si_sdr_abs_db"}
    for item in utterances:
        case_id = str(item.get("case_id", "?"))
        if item.get("alignment", {}).get("same_length") is False:
            failures.append(f"{case_id}:alignment:length_mismatch")
        for key, limit_name in limits.items():
            diff = item["official_minus_independent"][key]
            value = _metric_value(diff)
            if value is None:
                failures.append(f"{case_id}:{key}:na")
                continue
            if abs(value) > TOLERANCES[limit_name]:
                failures.append(f"{case_id}:{key}:{abs(value):.6g}>{TOLERANCES[limit_name]}")
    return {
        "passed": not failures,
        "failures": failures,
        "n": len(utterances),
        "tolerances": dict(TOLERANCES),
        "judged_updates": JUDGED_UPDATES,
    }


def assert_real_metric_libraries() -> dict[str, str]:
    """Refuse the dummy ``pesq`` module installed by the training comparison."""
    import importlib.metadata as metadata

    import pesq
    import pystoi

    pesq_file = getattr(pesq, "__file__", None)
    stoi_file = getattr(pystoi, "__file__", None)
    if not pesq_file or "site-packages" not in pesq_file.replace("\\", "/"):
        raise RuntimeError("pesq is not the installed library; refusing a dummy module")
    if not stoi_file or "site-packages" not in stoi_file.replace("\\", "/"):
        raise RuntimeError("pystoi is not the installed library")
    return {"pesq": metadata.version("pesq"), "pystoi": metadata.version("pystoi")}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _walk_tensors(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    found: list[tuple[str, torch.Tensor]] = []
    if torch.is_tensor(value):
        found.append((prefix, value))
    elif isinstance(value, dict):
        for key in sorted(value):
            found.extend(_walk_tensors(value[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_walk_tensors(item, f"{prefix}[{index}]"))
    return found


def assert_tensor_roundtrip(path: Path, payload: dict[str, Any]) -> str:
    """Write ``payload`` and require every tensor to reload unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    original = _walk_tensors(payload)
    again = _walk_tensors(loaded)
    if [name for name, _ in original] != [name for name, _ in again]:
        raise RuntimeError(f"tensor names changed on reload: {path}")
    for (name, left), (_, right) in zip(original, again, strict=True):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left.cpu(), right.cpu()):
            raise RuntimeError(f"tensor {name} changed on reload: {path}")
    return tensor_sha256(original)


def prepare_validation(out: Path, *, database: Path, project: Path) -> dict[str, Any]:
    """Build 3–10 s validation recipes and render them. Does not write DuckDB."""
    import duckdb

    from src.data.recipe import canonical_recipe, case_id_for
    from src.evaluation.audio.render import prepare_noise_start, render_recipe

    out.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        valid_rows = connection.execute(
            """
            SELECT i.position, c.case_id, c.recipe_json, c.metadata
            FROM eval.set_items i
            JOIN eval.cases c ON c.case_id = i.case_id
            WHERE i.set_name = 'libri_demand_smoke_valid'
            ORDER BY i.position
            """
        ).fetchall()
        train_rows = connection.execute(
            """
            SELECT c.recipe_json
            FROM eval.set_items i
            JOIN eval.cases c ON c.case_id = i.case_id
            WHERE i.set_name = 'libri_demand_smoke_train'
            """
        ).fetchall()
    finally:
        connection.close()
    if len(valid_rows) < 2:
        raise RuntimeError(f"expected multiple validation cases, found {len(valid_rows)}")
    root = database.parent
    train_speech = set()
    train_noise = set()
    for (recipe_json,) in train_rows:
        recipe = json.loads(recipe_json)
        train_speech.add(recipe["sources"]["clean"]["path"])
        train_noise.add(recipe["sources"]["noise"]["path"])

    utterances = []
    recipes = []
    for position, source_case_id, recipe_json, metadata_json in valid_rows:
        recipe = json.loads(recipe_json)
        metadata = json.loads(metadata_json)
        speech = recipe["sources"]["clean"]["path"]
        noise = recipe["sources"]["noise"]["path"]
        if speech in train_speech or noise in train_noise:
            raise RuntimeError(f"validation source overlaps training: {speech}")
        clean_path = (root / speech).resolve()
        noise_path = (root / noise).resolve()
        info = _audio_seconds(clean_path)
        max_seconds = choose_max_seconds(info)
        operation = recipe["operations"][0]
        seed = int(operation["seed"])
        noise_start = int(prepare_noise_start(clean_path, noise_path, SAMPLE_RATE, seed, max_seconds))
        extended = json.loads(recipe_json)
        extended["output"]["max_seconds"] = max_seconds
        extended["operations"][0]["noise_start"] = noise_start
        canonical = canonical_recipe(extended)
        case_id = case_id_for(canonical)
        if case_id == source_case_id:
            raise RuntimeError("extended recipe hashed to the 2 second case id")
        first = render_recipe(canonical, root)
        second = render_recipe(canonical, root)
        if first.sample_rate != SAMPLE_RATE or first.reference is None:
            raise RuntimeError(f"{case_id} did not render a 16 kHz clean reference")
        if not torch.equal(first.observed, second.observed) or not torch.equal(first.reference, second.reference):
            raise RuntimeError(f"{case_id} render is not repeatable")
        seconds = float(first.observed.numel()) / SAMPLE_RATE
        if seconds < MIN_VALID_SECONDS or seconds > MAX_VALID_SECONDS + 1e-3:
            raise RuntimeError(f"{case_id} rendered {seconds:.3f}s")
        noisy, clean, scale = energy_normalize(first.observed, first.reference)
        record = {
            "position": int(position),
            "case_id": case_id,
            "source_case_id": source_case_id,
            "seconds": seconds,
            "samples": int(clean.numel()),
            "snr_db": float(operation["snr_db"]),
            "noise_start": noise_start,
            "seed": seed,
            "scale": scale,
            "speech": speech,
            "noise": noise,
            "noise_scene": metadata.get("noise_scene"),
            "clean": clean,
            "noisy": noisy,
            "recipe": canonical,
        }
        utterances.append(record)
        recipes.append(
            {
                "case_id": case_id,
                "source_case_id": source_case_id,
                "recipe": canonical,
                "seconds": seconds,
                "samples": int(clean.numel()),
                "snr_db": float(operation["snr_db"]),
                "scale": scale,
                "speech": speech,
                "noise": noise,
            }
        )
    payload = {
        "sample_rate": SAMPLE_RATE,
        "normalization": "training_energy_v1",
        "clipping": None,
        "resampled": False,
        "role": "validation",
        "utterances": utterances,
    }
    digest = assert_tensor_roundtrip(out / "validation.pt", payload)
    save_json(out / "recipes.json", {"recipes": recipes, "validation_sha256": digest})
    config = {
        "sample_rate": SAMPLE_RATE,
        "normalization": "training_energy_v1",
        "clipping": None,
        "resampled": False,
        "resampler": None,
        "length_alignment": "stft_roundtrip_trimmed_to_clean",
        "tolerances": dict(TOLERANCES),
        "judged_updates": JUDGED_UPDATES,
        "validation_utterances": len(utterances),
        "validation_seconds": [item["seconds"] for item in utterances],
        "validation_sha256": digest,
        "database": str(database),
        "database_writes": False,
        "quality_source": "extended libri_demand_smoke_valid recipes, not the 0.5 s training crop",
    }
    save_json(out / "config.json", config)
    return config


def _audio_seconds(path: Path) -> float:
    import torchaudio

    info = torchaudio.info(str(path))
    return float(info.num_frames) / float(info.sample_rate)


def _load_wave_rows(path: Path) -> list[dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for utterance in saved["utterances"]:
        scored = score_conditions(
            utterance["clean"],
            utterance["noisy"],
            utterance["official"],
            utterance["independent"],
            int(saved["sample_rate"]),
        )
        scored["case_id"] = utterance["case_id"]
        scored["samples"] = int(utterance["clean"].numel())
        scored["seconds"] = scored["samples"] / int(saved["sample_rate"])
        rows.append(scored)
    return rows


def score_run(run: Path) -> dict[str, Any]:
    libraries = assert_real_metric_libraries()
    wave_dir = run / "waves"
    paths = sorted(wave_dir.glob("step_*.pt"))
    if not paths:
        raise RuntimeError(f"no waveform files in {wave_dir}")
    steps = []
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        reloaded = torch.load(path, map_location="cpu", weights_only=False)
        if _walk_tensors(saved)[0][0] != _walk_tensors(reloaded)[0][0]:
            raise RuntimeError(f"reload structure changed: {path}")
        for (name, left), (_, right) in zip(_walk_tensors(saved), _walk_tensors(reloaded), strict=True):
            if not torch.equal(left, right):
                raise RuntimeError(f"{name} changed on reload of {path}")
        rows = _load_wave_rows(path)
        summary = summarize_step(rows)
        steps.append(
            {
                "updates_completed": int(saved["updates_completed"]),
                "file": path.name,
                "utterances": rows,
                "summary": summary,
            }
        )
    judged = [item for item in steps if item["updates_completed"] == JUDGED_UPDATES]
    judgment = tolerance_result(judged[0]["utterances"]) if judged else {
        "passed": False,
        "failures": ["judged_step_missing"],
        "n": 0,
        "tolerances": dict(TOLERANCES),
        "judged_updates": JUDGED_UPDATES,
    }
    training = None
    train_path = run / "train_waves.pt"
    if train_path.is_file():
        saved = torch.load(train_path, map_location="cpu", weights_only=False)
        training_steps = []
        for item in saved["steps"]:
            scored = score_conditions(item["clean"], item["noisy"], item["official"], item["independent"], SAMPLE_RATE)
            training_steps.append(
                {
                    "updates_completed": int(item["updates_completed"]),
                    "case_id": item["case_id"],
                    "seconds": 8000 / SAMPLE_RATE,
                    "quality_source": False,
                    "reason": "0.5 s training crop is not the quality reference",
                    "wave": item["wave"],
                    "scores": scored,
                }
            )
        training = training_steps
    payload = {
        "libraries": libraries,
        "sample_rate": SAMPLE_RATE,
        "resampled": False,
        "clipping": None,
        "normalization": "training_energy_v1",
        "steps": steps,
        "training_segments": training,
        "tolerance": judgment,
    }
    save_json(run / "scores.json", _jsonable(payload))
    return payload


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        raise TypeError("tensors are not written into score JSON")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return value


def main() -> None:
    import argparse
    import sys

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--database", type=Path, default=root / "data" / "db" / "training.duckdb")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.out is None:
            raise SystemExit("--out is required")
        config = prepare_validation(args.out, database=args.database, project=root)
        print(json.dumps({key: config[key] for key in ("validation_utterances", "validation_seconds")}, indent=2))
    else:
        if args.run is None:
            raise SystemExit("--run is required")
        payload = score_run(args.run)
        print(json.dumps({"tolerance": payload["tolerance"], "libraries": payload["libraries"]}, indent=2))


if __name__ == "__main__":
    main()
