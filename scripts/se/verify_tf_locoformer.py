"""Re-run TF-Locoformer official comparisons and resource measurements.

Lightweight unit tests do not load the published DNS checkpoint. This script
does. It writes JSON under the given output directory and does not modify
``data/db`` or ``data/checkpoints``.

Exit status is 0 when every executed comparison is PASS or an expected SKIP.
FAIL and NOT_TESTED are non-zero. A missing official checkout or checkpoint
is SKIP, not PASS.

Pre-registered waveform tolerance, fixed before the official comparison:
maximum absolute error <= 1e-5 and relative L2 <= 1e-5. The published 1-second
GPU difference was 1.11e-9. This bound leaves room for longer FFT accumulation
and fails a real frontend or weight mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OFFICIAL = ROOT / "src/se/official/tf-locoformer"
CHECKPOINT = ROOT / "data/checkpoints/tf-locoformer/dns/valid.loss.ave_5best.pth"
ABS_TOL = 1e-5
REL_TOL = 1e-5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--durations", type=float, nargs="+", default=[1.0, 5.0, 20.0])
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--train-batches", type=int, nargs="*", default=[])
    parser.add_argument("--chunk", type=int, default=64000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_arg": args.device,
        "abs_tol": ABS_TOL,
        "rel_tol": REL_TOL,
        "versions": _versions(),
        "rows": [],
    }
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        index = 0 if args.device == "cuda" else int(args.device.split(":")[1])
        free, total = torch.cuda.mem_get_info(index)
        used = total - free
        report["gpu"] = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "used_bytes": used,
            "total_bytes": total,
        }
        if used >= 1024 * 1024 * 1024:
            raise SystemExit(f"GPU {index} is using {used} bytes; refusing")
    if not args.skip_inference:
        report["rows"].extend(_inference(args))
    for batch in args.train_batches:
        report["rows"].append(_train_step(args.device, batch, args.chunk))
    path = args.output / f"report-{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    failed = [row for row in report["rows"] if row["status"] in {"FAIL", "NOT_TESTED"}]
    if failed:
        raise SystemExit(f"{len(failed)} rows failed")


def _versions() -> dict[str, str]:
    import importlib.metadata as metadata

    found = {}
    for name in ("torchaudio", "rotary-embedding-torch"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "missing"
    return found


def _inference(args) -> list[dict]:
    if not OFFICIAL.is_dir() or not CHECKPOINT.is_file():
        return [{
            "status": "SKIP",
            "reason": "official checkout or DNS checkpoint missing",
            "official": OFFICIAL.is_dir(),
            "checkpoint": CHECKPOINT.is_file(),
        }]
    import sys

    from src.se import build_model

    root = str(OFFICIAL)
    if root not in sys.path:
        sys.path.insert(0, root)
    from standalone.tflocoformer_separator import TFLocoformerSeparator

    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    separator_state = {key.removeprefix("separator."): value for key, value in state.items()}
    official = TFLocoformerSeparator(
        num_spk=1, n_layers=6, emb_dim=128, norm_type="rmsgroupnorm", num_groups=4,
        tf_order="ft", n_heads=4, flash_attention=False, attention_dim=128, pos_enc="rope",
        ffn_type=["swiglu_conv1d", "swiglu_conv1d"], ffn_hidden_dim=[384, 384],
        conv1d_kernel=4, conv1d_shift=1, dropout=0.0, eps=1e-5,
    )
    official.load_state_dict(separator_state, strict=True)
    independent = build_model("tf_locoformer", {}, variant="dns")
    independent.separator.load_state_dict(separator_state, strict=True)
    device = torch.device(args.device)
    official.to(device).eval()
    independent.to(device).eval()
    rows = []
    inputs = _inputs()
    for duration in args.durations:
        inputs.append(("synthetic", 16000, int(duration * 16000), _tone(int(duration * 16000), device)))
    import torchaudio.functional as AF

    for rate in (8000, 48000):
        raw = _tone(rate, torch.device("cpu"))
        waveform = AF.resample(raw, rate, 16000).to(device)
        inputs.append((f"resampled-from-{rate}", 16000, int(waveform.numel()), waveform))
    for kind, rate, length, waveform in inputs:
        rows.append(_compare_row(kind, rate, length, waveform, official, independent, device, digest))
    return rows


def _inputs() -> list[tuple]:
    found = []
    database = ROOT / "data/db/resource.duckdb"
    if not database.is_file():
        found.append(("real", 0, 0, None))
        return found
    from src.evaluation.audio.render import render_recipe
    from src.evaluation.datasets.evaldb import EvaluationDB

    dataset = EvaluationDB(database).load_dataset("libri_demand_snr0")
    sample = dataset[0]
    rendered = render_recipe(sample.metadata["recipe"], sample.metadata["audio_root"])
    speech = rendered.reference
    noisy = rendered.observed
    for name, waveform in (("real-speech", speech), ("real-noisy", noisy)):
        if waveform is None:
            found.append((name, 0, 0, None))
            continue
        found.append((name, int(rendered.sample_rate), int(waveform.numel()), waveform.to(dtype=torch.float32)))
    return found


def _tone(length: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261105)
    return torch.randn(length, generator=generator).to(device)


def _compare_row(kind, rate, length, waveform, official, independent, device, digest) -> dict:
    row = {
        "kind": kind,
        "sample_rate": rate,
        "length": length,
        "device": str(device),
        "checkpoint_sha256": digest,
        "seed": 20261105,
        "status": "NOT_TESTED",
    }
    if waveform is None:
        row["status"] = "SKIP"
        row["reason"] = "waveform missing"
        return row
    waveform = waveform.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        left = _frontend(official, waveform.unsqueeze(0))
        right = independent(waveform.unsqueeze(0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    elapsed = time.perf_counter() - started
    diff = (left - right).abs()
    max_abs = float(diff.max())
    denom = float(left.square().sum().sqrt().clamp_min(1e-12))
    rel = float((left - right).square().sum().sqrt() / denom)
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    row.update({
        "seconds": elapsed,
        "output_length": int(right.shape[-1]),
        "finite": finite,
        "max_abs": max_abs,
        "rel_l2": rel,
        "rtf": elapsed / max(length / 16000.0, 1e-6),
        "status": "PASS" if (
            left.shape == right.shape and finite and max_abs <= ABS_TOL and rel <= REL_TOL
        ) else "FAIL",
    })
    return row


def _frontend(separator, waveform: torch.Tensor) -> torch.Tensor:
    scale = torch.std(waveform, dim=1, keepdim=True)
    window = torch.hann_window(256, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform / scale, 256, 128, 256, window, center=True, return_complex=True
    ).transpose(1, 2)
    enhanced = separator(spectrum)[:, 0].transpose(1, 2)
    restored = torch.istft(enhanced, 256, 128, 256, window, center=True, length=waveform.shape[-1])
    return restored * scale


def _train_step(device_name: str, batch: int, chunk: int) -> dict:
    from src.se.models.tf_locoformer.model import TFLocoformer
    from src.experiments.calc.initialization import apply_espnet_initialization
    from src.experiments.calc.spectral import multi_resolution_l1

    device = torch.device(device_name)
    row = {"kind": "train-step", "batch": batch, "chunk": chunk, "device": device_name, "status": "NOT_TESTED"}
    try:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20261105)
        torch.manual_seed(20261105)
        model = TFLocoformer()
        apply_espnet_initialization(model, "xavier_uniform")
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        clean = torch.randn(batch, chunk, generator=generator).to(device)
        noisy = torch.randn(batch, chunk, generator=generator).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        estimate = model(noisy)
        loss = multi_resolution_l1(clean, estimate, time_domain_weight=0.5, normalize_variance=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        row.update({
            "seconds": time.perf_counter() - started,
            "loss": float(loss.detach()),
            "finite": bool(torch.isfinite(loss).all()),
            "status": "PASS" if torch.isfinite(loss).all() else "FAIL",
        })
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "FAIL"
        row["reason"] = f"OOM: {exc}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


if __name__ == "__main__":
    main()
