"""Run dataset selection, mixing, enhancement, recognition, and metrics.

Each stage is a function. A missing ASR backend or a missing transcript skips
only the stage that needs it. Speech enhancement models are loaded once per
model, and only when at least one utterance still needs inference.
"""

from __future__ import annotations

import gc
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.evaluation.audio.io import AudioNotFound, as_mono, load_waveform, save_waveform
from src.evaluation.audio.mixing import MixResult, mix_snr
from src.evaluation.config import (
    PLANNED_ASR,
    ExperimentConfig,
    validate_registered_models,
)
from src.evaluation.config import config_hash as hash_config
from src.evaluation.datasets.load import DatasetBundle, load_dataset
from src.evaluation.envinfo import capture_environment, git_metadata
from src.evaluation.errors import ConfigError, DatasetError
from src.evaluation.gpu import assert_gpu_idle
from src.evaluation.identity import (
    canonical_snr,
    checkpoint_fingerprint,
    checkpoint_status,
    derive_seed,
    make_identity,
    make_slot,
    record_id,
    sha256_json,
    tensor_sha256,
)
from src.evaluation.metrics.compute import not_computed, score_utterance
from src.evaluation.metrics.pesq import library_status as pesq_library_status
from src.evaluation.metrics.stoi import library_status as stoi_library_status
from src.evaluation.models.asr import load_recognizer, recognizer_settings
from src.evaluation.models.se import (
    enhance_utterance,
    enhancement_seed,
    load_evaluation_enhancer,
)
from src.evaluation.results.reader import read_json, read_record
from src.evaluation.results.summary import build_summary, format_summary, write_summary
from src.evaluation.results.writer import atomic_write_json, write_jsonl, write_record
from src.evaluation.sample import AudioSample

__all__ = ["execute", "format_plan", "format_summary"]


def execute(
    config: ExperimentConfig,
    *,
    dry_run: bool = False,
    enhancer_factory=None,
    recognizer_factory=None,
    bundle: DatasetBundle | None = None,
) -> dict[str, Any]:
    if enhancer_factory is None:
        validate_registered_models(config.models)
    digest = hash_config(config)
    dataset_error = None
    if bundle is None:
        bundle = DatasetBundle(samples=[])
        try:
            bundle = load_dataset(config.dataset, config.experiment.seed)
        except DatasetError as exc:
            dataset_error = str(exc)
            if not dry_run:
                raise
    samples = list(bundle.samples)
    if config.experiment.limit is not None:
        samples = samples[: config.experiment.limit]
    plan = _build_plan(config, digest, samples, bundle, dataset_error)
    if dry_run:
        return {"dry_run": True, "plan": plan}
    if dataset_error:
        raise DatasetError(dataset_error)
    if not plan["ready"]:
        joined = "; ".join(plan["problems"])
        raise ConfigError(f"experiment is not ready: {joined}")
    assert_gpu_idle(config.runtime.device, allow_busy=config.runtime.allow_busy_gpu)
    fingerprints = [checkpoint_fingerprint(model) for model in config.models]
    _reject_identical_models(config, fingerprints)
    run_id, run_dir = _prepare_run(config, digest)
    environment = capture_environment()
    git = git_metadata()
    meta = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "config_hash": digest,
        "git": git,
        "environment": environment,
        "device": config.runtime.device,
        "started_at": _now(),
        "finished_at": None,
        "cwd": str(Path.cwd()),
    }
    if config.output.resume:
        stored = read_json(run_dir / "run.json")
        if isinstance(stored, dict) and stored.get("started_at"):
            meta["started_at"] = stored["started_at"]
    else:
        atomic_write_json(run_dir / "config.json", config.to_dict())
        atomic_write_json(run_dir / "run.json", meta)
    recognizer = None
    if config.asr.enabled:
        if recognizer_factory is None:
            recognizer = load_recognizer(config.asr)
        else:
            recognizer = recognizer_factory(config.asr)
    factory = enhancer_factory or load_evaluation_enhancer
    records: list[dict[str, Any]] = []
    reused = 0
    written = 0
    try:
        for model, fingerprint in zip(config.models, fingerprints, strict=True):
            model_reused, model_written = _run_model(
                config,
                bundle,
                samples,
                model,
                fingerprint,
                digest,
                run_id,
                run_dir,
                recognizer,
                factory,
                git,
                environment,
                records,
            )
            reused += model_reused
            written += model_written
        meta["status"] = "completed"
    except Exception:
        meta["status"] = "failed"
        meta["finished_at"] = _now()
        atomic_write_json(run_dir / "run.json", meta)
        raise
    meta["finished_at"] = _now()
    meta["reused"] = reused
    meta["written"] = written
    atomic_write_json(run_dir / "run.json", meta)
    summary = _publish(run_dir, records)
    return {
        "dry_run": False,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config_hash": digest,
        "reused": reused,
        "written": written,
        "summary": summary,
        "records": records,
    }


def format_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"experiment: {plan['experiment']}",
        f"config_hash: {plan['config_hash']}",
        f"device: {plan['device']}",
        f"samples: {plan['num_samples']}",
    ]
    for sample in plan["samples_preview"]:
        missing = "" if not sample["missing"] else " missing=" + ",".join(sample["missing"])
        lines.append(f"  - {sample['utterance_id']} ({sample['dataset']}){missing}")
    if plan["num_samples"] > len(plan["samples_preview"]):
        extra = plan["num_samples"] - len(plan["samples_preview"])
        lines.append(f"  ... {extra} more")
    lines.append("models:")
    for model in plan["models"]:
        lines.append(
            f"  - {model['name']} variant={model['variant']} ready={model['ready']} checkpoints={model['checkpoints']}"
        )
    lines.append(f"snrs: {plan['snrs']}")
    lines.append(f"planned_records: {plan['planned_records']}")
    lines.append(f"asr: {plan['asr']}")
    lines.append(f"metrics: {plan['metrics']}")
    lines.append(f"output: {plan['output']}")
    lines.append(f"save_audio: {plan['save_audio']}")
    if plan["warnings"]:
        lines.append("warnings:")
        for warning in plan["warnings"]:
            lines.append(f"  - {warning}")
    if plan["problems"]:
        lines.append("problems:")
        for problem in plan["problems"]:
            lines.append(f"  - {problem}")
    lines.append(f"ready: {plan['ready']}")
    return "\n".join(lines)


def _run_model(
    config,
    bundle,
    samples,
    model,
    fingerprint,
    digest,
    run_id,
    run_dir,
    recognizer,
    factory,
    git,
    environment,
    records,
):
    enhancer = None
    reused = 0
    written = 0
    try:
        for index, sample in enumerate(samples):
            for snr in _snrs(config, sample):
                record, did_reuse, did_write, enhancer = _one_utterance(
                    config,
                    bundle,
                    sample,
                    index,
                    snr,
                    model,
                    fingerprint,
                    digest,
                    run_id,
                    run_dir,
                    recognizer,
                    factory,
                    enhancer,
                    git,
                    environment,
                )
                records.append(record)
                reused += int(did_reuse)
                written += int(did_write)
                if did_write:
                    write_record(run_dir, record)
                    _publish(run_dir, records)
    finally:
        if enhancer is not None:
            del enhancer
            gc.collect()
            if str(config.runtime.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return reused, written


def _one_utterance(
    config,
    bundle,
    sample,
    index,
    snr,
    model,
    fingerprint,
    digest,
    run_id,
    run_dir,
    recognizer,
    factory,
    enhancer,
    git,
    environment,
):
    noise_item = _paired_noise(config, bundle, sample, index)
    prepared = _prepare(config, sample, noise_item, snr)
    _assign_enhance_seed(config, sample, model, snr, prepared)
    slot = make_slot(
        config_hash=digest,
        torch_version=torch.__version__,
        utterance_id=sample.utterance_id,
        dataset=sample.dataset,
        model_name=model.name,
        variant=model.variant,
        checkpoint_sha256=fingerprint,
        snr_db=snr,
        sample_rate=prepared["sample_rate"] if prepared["sample_rate"] is not None else sample.sample_rate,
    )
    identity = make_identity(
        slot,
        input_sha256=tensor_sha256(prepared["input_wave"]),
        clean_sha256=tensor_sha256(prepared["clean_wave"]),
        enhance_seed=prepared["enhance_seed"],
        mix_seed=prepared["mix_seed"],
    )
    slot_id = record_id(slot)
    identity_hash = sha256_json(identity)
    if prepared["status"] == "ok":
        existing = read_record(run_dir, slot_id)
        if (
            config.output.resume
            and existing
            and existing.get("status") == "ok"
            and existing.get("identity_hash") == identity_hash
        ):
            return existing, True, False, enhancer
    if prepared["status"] != "ok":
        record = _finish_record(
            config,
            sample,
            model,
            fingerprint,
            run_id,
            slot_id,
            identity,
            identity_hash,
            prepared,
            git,
            environment,
            hypotheses=None,
            metrics=not_computed(prepared["reason"] or prepared["status"]),
            timing={"prepare": prepared["seconds"], "enhance": None, "asr": None, "metrics": None},
            saved_audio=None,
        )
        return record, False, True, enhancer
    if enhancer is None:
        enhancer = factory(model, config.runtime.device)
    try:
        enhanced = enhance_utterance(
            enhancer,
            prepared["input_wave"],
            prepared["sample_rate"],
            prepared["enhance_seed"],
        )
    except Exception as exc:
        prepared["status"] = "error"
        prepared["error"] = _error_payload(exc)
        prepared["reason"] = "enhancement_failed"
        record = _finish_record(
            config,
            sample,
            model,
            fingerprint,
            run_id,
            slot_id,
            identity,
            identity_hash,
            prepared,
            git,
            environment,
            hypotheses=None,
            metrics=not_computed("enhancement_failed"),
            timing={"prepare": prepared["seconds"], "enhance": None, "asr": None, "metrics": None},
            saved_audio=None,
        )
        return record, False, True, enhancer
    hypotheses, asr_seconds = _recognize(
        config,
        recognizer,
        sample,
        prepared,
        enhanced.waveform,
    )
    metric_begin = _perf()
    metrics = score_utterance(
        config.metrics,
        clean=prepared["clean_wave"],
        estimates={"noisy": prepared["input_wave"], "enhanced": enhanced.waveform},
        sample_rate=prepared["sample_rate"],
        reference=sample.transcript,
        hypotheses=hypotheses or {},
    )
    metric_seconds = _perf() - metric_begin
    saved = _save_enhanced(config, run_dir, slot_id, enhanced.waveform, prepared["sample_rate"])
    record = _finish_record(
        config,
        sample,
        model,
        fingerprint,
        run_id,
        slot_id,
        identity,
        identity_hash,
        prepared,
        git,
        environment,
        hypotheses=hypotheses,
        metrics=metrics,
        timing={
            "prepare": prepared["seconds"],
            "enhance": enhanced.seconds,
            "asr": asr_seconds,
            "metrics": metric_seconds,
        },
        saved_audio=saved,
    )
    return record, False, True, enhancer


def _prepare(config: ExperimentConfig, sample: AudioSample, noise_item: AudioSample | None, snr) -> dict[str, Any]:
    started = _perf()
    paths = {
        "audio": sample.audio_path,
        "clean": sample.clean_path,
        "noisy": sample.noisy_path,
        "noise": sample.noise_path,
    }
    payload = {
        "status": "ok",
        "reason": None,
        "error": None,
        "input_wave": None,
        "clean_wave": None,
        "sample_rate": sample.sample_rate,
        "paths": paths,
        "mix": None,
        "mix_seed": None,
        "enhance_seed": None,
        "snr_db": canonical_snr(snr),
        "seconds": 0.0,
    }
    try:
        recipe = _mix_recipe(sample)
        if recipe is not None or config.mixing.enabled:
            clean_limit = config.dataset.max_seconds
            noise_limit = config.dataset.max_seconds
            if recipe is not None:
                clean_limit = recipe.get("max_seconds")
                noise_limit = None
            clean = _load_any(sample.clean_waveform, sample.sample_rate, sample.clean_path, clean_limit)
            if clean is None:
                return _mark(payload, started, "skipped", "clean_required")
            if recipe is not None:
                noise_loaded = _load_any(
                    sample.noise_waveform, sample.sample_rate, sample.noise_path, noise_limit
                )
                noise_path = sample.noise_path
                if noise_loaded is None:
                    return _mark(payload, started, "skipped", "noise_required")
                noise_tensor, noise_rate = noise_loaded
            else:
                noise_loaded, noise_path = _load_noise(sample, noise_item, noise_limit)
                if noise_loaded is None:
                    return _mark(payload, started, "skipped", "noise_required")
                noise_tensor, noise_rate = noise_loaded
            paths["noise"] = noise_path
            clean_wave, clean_rate = clean
            if recipe is not None:
                mix_seed = int(recipe["seed"])
                protocol = str(recipe.get("protocol") or config.mixing.protocol)
                prevent_clip = bool(recipe.get("prevent_clip", False))
                hard_clip = bool(recipe.get("hard_clip", False))
                noise_start = recipe.get("noise_start")
                snr_value = float(sample.snr_db)
            else:
                mix_seed = derive_seed(
                    config.experiment.seed,
                    "mix",
                    sample.utterance_id,
                    canonical_snr(snr),
                    noise_path or "memory",
                )
                protocol = config.mixing.protocol
                prevent_clip = config.mixing.prevent_clip
                hard_clip = config.mixing.hard_clip
                noise_start = None
                snr_value = float(snr)
            mixed = mix_snr(
                clean_wave,
                noise_tensor,
                clean_rate,
                noise_rate,
                snr_value,
                mix_seed,
                protocol=protocol,
                prevent_clip=prevent_clip,
                hard_clip=hard_clip,
                noise_start=None if noise_start is None else int(noise_start),
            )
            payload["snr_db"] = canonical_snr(snr_value)
            payload["mix"] = _mix_payload(mixed)
            if recipe is not None and recipe.get("max_seconds") is not None:
                payload["mix"]["clean_max_seconds"] = float(recipe["max_seconds"])
            payload["mix_seed"] = mix_seed
            if mixed.clean_silent:
                payload["input_wave"] = mixed.noisy
                payload["clean_wave"] = mixed.clean
                payload["sample_rate"] = mixed.sample_rate
                return _mark(payload, started, "skipped", "clean_has_zero_energy")
            payload["input_wave"] = mixed.noisy
            payload["clean_wave"] = mixed.clean
            payload["sample_rate"] = mixed.sample_rate
        else:
            loaded = _load_primary(sample, config.dataset.max_seconds)
            if loaded is None:
                return _mark(payload, started, "skipped", "no_audio")
            kind, wave, rate, clean = loaded
            payload["paths"]["input_kind"] = kind
            payload["input_wave"] = wave
            payload["clean_wave"] = clean
            payload["sample_rate"] = rate
        _check_wave(payload["input_wave"], "input")
        if payload["clean_wave"] is not None:
            _check_wave(payload["clean_wave"], "clean")
    except AudioNotFound as exc:
        return _mark(payload, started, "skipped", "audio_not_found", error=_error_payload(exc))
    except Exception as exc:
        return _mark(payload, started, "error", "preparation_failed", error=_error_payload(exc))
    payload["seconds"] = _perf() - started
    return payload


def _assign_enhance_seed(config, sample, model, snr, prepared) -> None:
    prepared["enhance_seed"] = enhancement_seed(
        config.experiment.seed,
        sample.utterance_id,
        model.name,
        model.variant,
        canonical_snr(snr),
    )


def _load_primary(sample: AudioSample, max_seconds):
    if sample.noisy_waveform is not None or sample.noisy_path:
        wave, rate = _require_load(sample.noisy_waveform, sample.sample_rate, sample.noisy_path, max_seconds)
        clean = _load_any(sample.clean_waveform, sample.sample_rate, sample.clean_path, max_seconds)
        clean_wave = None if clean is None else _match_rate(clean[0], clean[1], rate)
        return "noisy", wave, rate, clean_wave
    if sample.clean_waveform is not None or sample.clean_path:
        wave, rate = _require_load(sample.clean_waveform, sample.sample_rate, sample.clean_path, max_seconds)
        return "clean", wave, rate, wave
    if sample.audio_path:
        wave, rate = load_waveform(sample.audio_path, max_seconds)
        return "audio", wave, rate, None
    return None


def _load_noise(sample: AudioSample, noise_item: AudioSample | None, max_seconds):
    if sample.noise_waveform is not None or sample.noise_path:
        loaded = _load_any(sample.noise_waveform, sample.sample_rate, sample.noise_path, max_seconds)
        return loaded, sample.noise_path
    if noise_item is None:
        return None, None
    path = noise_item.noise_path or noise_item.audio_path
    loaded = _load_any(noise_item.noise_waveform, noise_item.sample_rate, path, max_seconds)
    return loaded, path


def _load_any(waveform, sample_rate, path, max_seconds):
    if waveform is None and not path:
        return None
    return _require_load(waveform, sample_rate, path, max_seconds)


def _require_load(waveform, sample_rate, path, max_seconds):
    if waveform is not None:
        if sample_rate is None:
            raise ValueError("in-memory audio is missing sample_rate")
        return as_mono(waveform, int(sample_rate), max_seconds)
    if not path:
        raise ValueError("audio path is missing")
    return load_waveform(path, max_seconds)


def _match_rate(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    import torchaudio.functional as AF

    return AF.resample(waveform, source_rate, target_rate).to(torch.float32).contiguous()


def _check_wave(waveform: torch.Tensor, label: str) -> None:
    if waveform.numel() == 0:
        raise ValueError(f"{label} audio is empty")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{label} audio contains non-finite samples")


def _recognize(config, recognizer, sample, prepared, enhanced):
    if recognizer is None:
        return None, None
    audio = {
        "noisy": prepared["input_wave"],
        "enhanced": enhanced,
        "clean": prepared["clean_wave"],
    }
    hypotheses = {}
    elapsed = 0.0
    for condition in config.asr.conditions:
        waveform = audio.get(condition)
        if waveform is None:
            hypotheses[condition] = {"text": None, "error": "audio_unavailable", "seconds": None, "seed": None}
            continue
        seed = derive_seed(
            config.experiment.seed,
            "asr",
            sample.utterance_id,
            condition,
            config.asr.name,
        )
        try:
            if hasattr(recognizer, "bind"):
                recognizer.bind(sample.utterance_id, condition)
            torch.manual_seed(seed)
            random.seed(seed)
            started = _perf()
            result = recognizer.transcribe(waveform, prepared["sample_rate"])
            seconds = _perf() - started
            elapsed += seconds
            hypotheses[condition] = {
                "text": result.text,
                "model": result.model,
                "error": None,
                "seconds": seconds,
                "seed": seed,
            }
        except Exception as exc:
            hypotheses[condition] = {
                "text": None,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": None,
                "seed": seed,
            }
    return hypotheses, elapsed


def _finish_record(
    config,
    sample,
    model,
    fingerprint,
    run_id,
    slot_id,
    identity,
    identity_hash,
    prepared,
    git,
    environment,
    *,
    hypotheses,
    metrics,
    timing,
    saved_audio,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_id": slot_id,
        "identity": identity,
        "identity_hash": identity_hash,
        "run_id": run_id,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "status": prepared["status"],
        "skip_reason": prepared["reason"] if prepared["status"] == "skipped" else None,
        "error": prepared["error"],
        "utterance_id": sample.utterance_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "speaker_id": sample.speaker_id,
        "reference": sample.transcript,
        "paths": prepared["paths"],
        "sample_rate": prepared["sample_rate"],
        "snr_db": prepared["snr_db"],
        "max_seconds": config.dataset.max_seconds,
        "mix": prepared["mix"],
        "seeds": {
            "experiment": config.experiment.seed,
            "mix": prepared["mix_seed"],
            "enhance": prepared["enhance_seed"],
        },
        "model": {
            "name": model.name,
            "variant": model.variant,
            "checkpoint": model.checkpoint,
            "checkpoints": model.checkpoints,
            "checkpoint_sha256": fingerprint,
        },
        "asr": recognizer_settings(config.asr),
        "hypotheses": hypotheses,
        "wer": metrics["wer"],
        "pesq": metrics["pesq"],
        "stoi": metrics["stoi"],
        "estoi": metrics["estoi"],
        "si_sdr": metrics["si_sdr"],
        "timing_sec": timing,
        "saved_audio": saved_audio,
        "environment": {
            "python": environment.get("python"),
            "platform": environment.get("platform"),
            "torch": environment.get("torch"),
            "device": config.runtime.device,
        },
    }


def _save_enhanced(config, run_dir, slot_id, waveform, sample_rate):
    if not config.output.save_audio:
        return None
    relative = Path("audio") / f"{slot_id}.wav"
    save_waveform(run_dir / relative, waveform, sample_rate)
    return {"enhanced": relative.as_posix()}


def _publish(run_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = build_summary(records)
    write_jsonl(run_dir / "results.jsonl", records)
    write_summary(run_dir, summary)
    return summary


def _prepare_run(config: ExperimentConfig, digest: str) -> tuple[str, Path]:
    root = Path(config.output.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    run_id = config.output.run_id or _new_run_id(config.experiment.name, digest)
    run_dir = root / run_id
    if config.output.resume:
        if not config.output.run_id:
            raise ConfigError("resume requires output.run_id")
        meta_path = run_dir / "run.json"
        if not meta_path.is_file():
            raise ConfigError(f"cannot resume because run metadata is missing: {run_dir}")
        stored = read_json(meta_path)
        if not isinstance(stored, dict) or stored.get("config_hash") != digest:
            stored_hash = None if not isinstance(stored, dict) else stored.get("config_hash")
            raise ConfigError(
                "this run_id was created with a different configuration "
                f"(stored {stored_hash}, current {digest}). Choose a new run_id."
            )
        return run_id, run_dir
    if run_dir.exists():
        raise ConfigError(
            f"run directory already exists: {run_dir}. Use --resume or choose another run_id."
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _build_plan(config, digest, samples, bundle, dataset_error) -> dict[str, Any]:
    problems = []
    warnings = []
    if dataset_error:
        problems.append(dataset_error)
    if config.asr.enabled and config.asr.name in PLANNED_ASR:
        label = PLANNED_ASR[config.asr.name]
        problems.append(f"ASR backend {config.asr.name} ({label}) is not installed")
    if config.metrics.pesq:
        message = pesq_library_status()
        if message:
            warnings.append(message)
    if config.metrics.stoi or config.metrics.estoi:
        message = stoi_library_status()
        if message:
            warnings.append(message)
    if (config.metrics.pesq or config.metrics.stoi or config.metrics.si_sdr) and config.dataset.source == "duckdb":
        if config.dataset.role != "clean" and not config.mixing.enabled:
            warnings.append("signal metrics need a clean reference; dataset.role is not clean, so scores will be na")
    preview = []
    missing_count = 0
    for sample in samples:
        missing = _missing_paths(sample)
        missing_count += len(missing)
        if len(preview) < 20:
            described = sample.describe()
            described["missing"] = missing
            preview.append(described)
    if missing_count:
        problems.append(f"{missing_count} audio paths are missing")
    for item in bundle.noise_pool:
        missing = _missing_paths(item)
        if missing:
            problems.append(f"noise {item.utterance_id} is missing: {', '.join(missing)}")
    models = []
    for model in config.models:
        status = checkpoint_status(model)
        models.append(
            {
                "name": model.name,
                "variant": model.variant,
                "ready": status["ready"],
                "checkpoints": status["paths"],
            }
        )
        if not status["ready"]:
            problems.append(f"checkpoint for {model.name} is missing or was not provided")
    if samples:
        sequences = [_snrs(config, sample) for sample in samples]
        first = sequences[0]
        if all(item == first for item in sequences):
            snrs = first
        else:
            snrs = sorted({value for item in sequences for value in item if value is not None})
        planned_records = sum(len(item) for item in sequences) * len(config.models)
    else:
        snrs = list(config.mixing.snr_db) if config.mixing.enabled else [None]
        planned_records = 0
    output = config.output.root
    if config.output.run_id:
        output = str(Path(config.output.root) / config.output.run_id)
    if str(config.runtime.device).startswith("cuda"):
        warnings.append("CUDA is requested. Execution checks nvidia-smi and refuses a busy GPU.")
    return {
        "experiment": config.experiment.name,
        "config_hash": digest,
        "device": config.runtime.device,
        "seed": config.experiment.seed,
        "num_samples": len(samples),
        "samples_preview": preview,
        "missing_paths": missing_count,
        "models": models,
        "snrs": snrs,
        "planned_records": planned_records,
        "asr": recognizer_settings(config.asr),
        "metrics": {
            "wer": config.metrics.wer,
            "normalization": config.metrics.normalization,
            "pesq": config.metrics.pesq,
            "stoi": config.metrics.stoi,
            "estoi": config.metrics.estoi,
            "si_sdr": config.metrics.si_sdr,
        },
        "mixing": {
            "enabled": config.mixing.enabled,
            "protocol": config.mixing.protocol if config.mixing.enabled else None,
            "prevent_clip": config.mixing.prevent_clip,
            "hard_clip": config.mixing.hard_clip,
        },
        "output": output,
        "save_audio": config.output.save_audio,
        "resume": config.output.resume,
        "warnings": warnings,
        "problems": problems,
        "ready": not problems,
    }


def _missing_paths(sample: AudioSample) -> list[str]:
    missing = []
    for label in ("audio_path", "clean_path", "noisy_path", "noise_path"):
        path = getattr(sample, label)
        if path and not Path(path).is_file():
            missing.append(f"{label}={path}")
    return missing


def _paired_noise(config, bundle, sample, index):
    if sample.noise_waveform is not None or sample.noise_path:
        return sample
    if not bundle.noise_pool:
        return None
    if config.dataset.pair == "cycle":
        return bundle.noise_pool[index % len(bundle.noise_pool)]
    offset = derive_seed(config.experiment.seed, "noise-pair", sample.utterance_id) % len(bundle.noise_pool)
    return bundle.noise_pool[offset]


def _mix_recipe(sample: AudioSample) -> dict[str, Any] | None:
    """Stored mixture for a synthetic case. Experiment seed does not change it."""
    recipe = sample.metadata.get("mix")
    if not isinstance(recipe, dict):
        return None
    seed = recipe.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or sample.snr_db is None:
        return None
    return recipe


def _snrs(config: ExperimentConfig, sample: AudioSample) -> list[float | None]:
    if _mix_recipe(sample) is not None:
        return [sample.snr_db]
    if config.mixing.enabled:
        return list(config.mixing.snr_db)
    return [sample.snr_db]


def _reject_identical_models(config: ExperimentConfig, fingerprints: list[dict[str, str]]) -> None:
    seen = set()
    for model, fingerprint in zip(config.models, fingerprints, strict=True):
        key = (model.name, model.variant, tuple(sorted(fingerprint.items())))
        if key in seen:
            raise ConfigError(
                f"duplicate model {model.name!r} variant {model.variant!r} with the same checkpoint contents"
            )
        seen.add(key)


def _mix_payload(result: MixResult) -> dict[str, Any]:
    return {
        "protocol": result.protocol,
        "snr_db_requested": result.snr_db_requested,
        "snr_db_achieved": result.snr_db_achieved,
        "noise_start": result.noise_start,
        "scale": result.scale,
        "peak_scaled": result.peak_scaled,
        "hard_clipped": result.hard_clipped,
        "clean_silent": result.clean_silent,
        "definition": "10*log10(mean(clean^2)/mean(scaled_noise^2)) after alignment; silence stays in the mean",
    }


def _mark(payload, started, status, reason, error=None):
    payload["status"] = status
    payload["reason"] = reason
    payload["error"] = error
    payload["seconds"] = _perf() - started
    return payload


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _new_run_id(name: str, digest: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"
    return f"{slug}-{stamp}-{digest[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _perf() -> float:
    return time.perf_counter()
