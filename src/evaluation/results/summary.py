"""Micro-averaged summaries.

WER in a group is (S + D + I) / N. Per-utterance ratios are not averaged.
PESQ, STOI, ESTOI, and SI-SDR are arithmetic means of finite utterance scores.
Missing scores are left out of the mean. They are not filled with zero.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from src.evaluation.metrics.wer import add_counts, corpus_wer, empty_counts
from src.evaluation.results.reader import read_jsonl, read_record_files
from src.evaluation.results.writer import atomic_write_json, atomic_write_text

CONDITIONS = ("clean", "noisy", "enhanced")
SIGNAL_METRICS = ("pesq", "stoi", "estoi", "si_sdr")


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        identity = record.get("identity") or {}
        for condition in CONDITIONS:
            if not _has_condition(record, condition):
                continue
            key = (
                identity.get("config_hash"),
                record.get("model", {}).get("name"),
                record.get("model", {}).get("variant"),
                record.get("snr_db"),
                condition,
            )
            grouped.setdefault(key, []).append(record)
    groups = [
        _render_group(key, items)
        for key, items in sorted(grouped.items(), key=_sort_key)
    ]
    return {
        "schema_version": 1,
        "wer_aggregation": "micro",
        "num_records": len(records),
        "num_ok": sum(record.get("status") == "ok" for record in records),
        "num_error": sum(record.get("status") == "error" for record in records),
        "num_skipped": sum(record.get("status") == "skipped" for record in records),
        "groups": groups,
    }


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "summary.json", summary)
    _write_csv(run_dir / "summary.csv", summary["groups"])


def summarize_directory(run_dir: Path) -> dict[str, Any]:
    jsonl = run_dir / "results.jsonl"
    records = read_jsonl(jsonl) if jsonl.is_file() else read_record_files(run_dir)
    summary = build_summary(records)
    write_summary(run_dir, summary)
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"records: ok={summary['num_ok']} error={summary['num_error']} skipped={summary['num_skipped']}",
        "wer_aggregation: micro",
    ]
    if not summary["groups"]:
        lines.append("groups: 0")
        return "\n".join(lines)
    for group in summary["groups"]:
        lines.append(
            " ".join(
                [
                    f"{group['condition']}",
                    f"model={group['model']}",
                    f"variant={group['variant']}",
                    f"snr={group['snr_db']}",
                    f"n={group['utterances']}",
                    f"wer={_fmt(group['wer_normalized'])}",
                    f"pesq={_fmt(group['pesq_mean'])}",
                    f"stoi={_fmt(group['stoi_mean'])}",
                    f"si_sdr={_fmt(group['si_sdr_mean'])}",
                ]
            )
        )
    return "\n".join(lines)


def _render_group(key: tuple, items: list[dict[str, Any]]) -> dict[str, Any]:
    config_hash, model, variant, snr_db, condition = key
    raw = empty_counts()
    normalized = empty_counts()
    saw_wer = False
    for record in items:
        bucket = record.get("wer", {}).get("values", {}).get(condition)
        if not bucket or bucket.get("status") != "ok":
            continue
        saw_wer = True
        add_counts(raw, bucket["raw"])
        add_counts(normalized, bucket["normalized"])
    payload: dict[str, Any] = {
        "config_hash": config_hash,
        "model": model,
        "variant": variant,
        "snr_db": snr_db,
        "condition": condition,
        "utterances": len(items),
        "wer_aggregation": "micro",
        "wer_normalized": corpus_wer(normalized) if saw_wer else None,
        "wer_raw": corpus_wer(raw) if saw_wer else None,
        "edits_normalized": normalized if saw_wer else None,
        "edits_raw": raw if saw_wer else None,
    }
    for metric in SIGNAL_METRICS:
        values = _metric_values(items, metric, condition)
        payload[f"{metric}_mean"] = None if not values else sum(values) / len(values)
        payload[f"{metric}_count"] = len(values)
    if condition == "enhanced":
        payload["enhance_seconds"] = sum(float(record.get("timing_sec", {}).get("enhance") or 0.0) for record in items)
    else:
        payload["enhance_seconds"] = None
    return payload


def _metric_values(items: list[dict[str, Any]], metric: str, condition: str) -> list[float]:
    values = []
    for record in items:
        cell = record.get(metric, {}).get("values", {}).get(condition)
        if not cell or cell.get("status") != "ok" or cell.get("value") is None:
            continue
        values.append(float(cell["value"]))
    return values


def _has_condition(record: dict[str, Any], condition: str) -> bool:
    if condition in record.get("wer", {}).get("values", {}):
        return True
    for metric in SIGNAL_METRICS:
        if condition in record.get(metric, {}).get("values", {}):
            return True
    return False


def _sort_key(item: tuple) -> tuple:
    key, _records = item
    config_hash, model, variant, snr_db, condition = key
    return (
        config_hash or "",
        model or "",
        variant or "",
        snr_db is None,
        -1 if snr_db is None else float(snr_db),
        condition,
    )


def _write_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config_hash",
        "model",
        "variant",
        "snr_db",
        "condition",
        "utterances",
        "wer_aggregation",
        "wer_normalized",
        "wer_raw",
        "normalized_S",
        "normalized_D",
        "normalized_I",
        "normalized_N",
        "pesq_mean",
        "pesq_count",
        "stoi_mean",
        "stoi_count",
        "estoi_mean",
        "estoi_count",
        "si_sdr_mean",
        "si_sdr_count",
        "enhance_seconds",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for group in groups:
        edits = group.get("edits_normalized") or {}
        writer.writerow(
            {
                "config_hash": group["config_hash"],
                "model": group["model"],
                "variant": group["variant"],
                "snr_db": group["snr_db"],
                "condition": group["condition"],
                "utterances": group["utterances"],
                "wer_aggregation": group["wer_aggregation"],
                "wer_normalized": group["wer_normalized"],
                "wer_raw": group["wer_raw"],
                "normalized_S": edits.get("S"),
                "normalized_D": edits.get("D"),
                "normalized_I": edits.get("I"),
                "normalized_N": edits.get("N"),
                "pesq_mean": group["pesq_mean"],
                "pesq_count": group["pesq_count"],
                "stoi_mean": group["stoi_mean"],
                "stoi_count": group["stoi_count"],
                "estoi_mean": group["estoi_mean"],
                "estoi_count": group["estoi_count"],
                "si_sdr_mean": group["si_sdr_mean"],
                "si_sdr_count": group["si_sdr_count"],
                "enhance_seconds": group["enhance_seconds"],
            }
        )
    atomic_write_text(path, buffer.getvalue())


def _fmt(value: Any) -> str:
    if value is None:
        return "na"
    return f"{float(value):.6g}"
