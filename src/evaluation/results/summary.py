"""Micro-averaged summaries.

WER in a group is (S + D + I) / N over the target utterances. Per-utterance
ratios are not averaged. N = 0 stays out of the denominator and is not stored
as a zero WER.

``wer_normalized`` is the official value only when every required condition
was scored for every target utterance. A missing required WER leaves the
official value unset. A provisional figure, when one can be computed from the
scores that do exist, is stored separately and is not a substitute for the
official WER. Missing scores are never filled with zero.

PESQ, STOI, ESTOI, and SI-SDR stay arithmetic means of finite utterance
scores. A missing auxiliary score does not by itself withhold the official WER.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

from src.evaluation.metrics.wer import add_counts, corpus_wer, empty_counts
from src.evaluation.provenance import summarize_record_independence
from src.evaluation.results.reader import read_json, read_jsonl, read_record_files
from src.evaluation.results.writer import atomic_write_json, atomic_write_text

SUMMARY_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
CONDITIONS = ("clean", "noisy", "enhanced")
SIGNAL_METRICS = ("pesq", "stoi", "estoi", "si_sdr")
# JSONL for these run statuses was written after every utterance had been visited.
_FINISHED_RUN_STATUSES = {"completed", "incomplete", "validation_failed"}


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completeness = assess_completeness(records)
    groups = _groups(records, official_run=bool(completeness["official_wer"]))
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "wer_aggregation": "micro",
        "official_wer": completeness["official_wer"],
        "num_records": len(records),
        "num_ok": sum(record.get("status") == "ok" for record in records),
        "num_error": sum(record.get("status") == "error" for record in records),
        "num_skipped": sum(record.get("status") == "skipped" for record in records),
        "num_validation_failed": sum(record.get("status") == "validation_failed" for record in records),
        "completeness": completeness,
        "groups": groups,
        "paired_signal_metrics": _paired_signal_metrics(records),
    }
    independence = summarize_record_independence(records)
    if independence is not None:
        summary["data_independence"] = independence
    return summary


def assess_completeness(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Say whether these records are a finished official WER.

    Records written before the completeness fields existed cannot show that
    every required condition was scored, so they stay unofficial.
    """
    if not records:
        return _completeness("unknown", official=False, required=None, reason="no_records", missing=[])
    if any(not _declared(record) for record in records):
        missing = [
            _missing_row(record, None, "record_schema_cannot_show_completeness")
            for record in records
            if not _declared(record)
        ]
        return _completeness(
            "unknown",
            official=False,
            required=None,
            reason="record_schema_cannot_show_completeness",
            missing=missing,
        )
    flags = {bool(record["evaluation"]["required_wer"]) for record in records}
    if flags != {True} and flags != {False}:
        return _completeness(
            "unknown",
            official=False,
            required=None,
            reason="inconsistent_wer_requirement",
            missing=[],
        )
    if flags == {False}:
        return _completeness_without_wer(records)
    missing = _required_gaps(records)
    if missing:
        statuses = {record.get("status") for record in records}
        if "validation_failed" in statuses:
            state = "validation_failed"
            reason = "validation_failed"
        else:
            state = "incomplete"
            reason = "required_wer_missing"
        return _completeness(state, official=False, required=True, reason=reason, missing=missing)
    if all(_cohort_complete(items) for items in _cohorts(records).values()):
        return _completeness("completed", official=True, required=True, reason=None, missing=[])
    return _completeness(
        "incomplete",
        official=False,
        required=True,
        reason="required_wer_missing",
        missing=_required_gaps(records),
    )


def demote_unfinished(summary: dict[str, Any], run_status: str | None) -> None:
    """Drop an official WER when ``run.json`` does not say the run completed.

    A killed run can leave a prefix of successful records. Those records are
    not the target set, so their micro WER is not official.
    """
    if run_status == "completed":
        return
    completeness = summary.get("completeness")
    if not isinstance(completeness, dict):
        return
    groups = summary.get("groups") or []
    official = bool(completeness.get("official_wer")) or any(group.get("wer_official") for group in groups)
    looked_finished = completeness.get("status") == "completed"
    if not official and not looked_finished:
        return
    completeness["official_wer"] = False
    summary["official_wer"] = False
    if looked_finished:
        completeness["status"] = "unknown"
        completeness["reason"] = "run_not_completed"
    for group in groups:
        if not group.get("wer_official"):
            continue
        if group.get("wer_provisional") is None and group.get("wer_normalized") is not None:
            group["wer_provisional"] = group["wer_normalized"]
            group["edits_provisional"] = group.get("edits_normalized")
        group["wer_normalized"] = None
        group["edits_normalized"] = None
        group["edits_raw"] = None
        group["wer_official"] = False
        if group.get("wer_coverage") == "complete":
            group["wer_coverage"] = "partial"


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "summary.json", summary)
    _write_csv(run_dir / "summary.csv", summary["groups"])


def summarize_directory(run_dir: Path) -> dict[str, Any]:
    """Summarize a run directory again.

    A finished run is read from ``results.jsonl``. A run that stopped early
    is read from ``records/``, because the joined file can be behind it.
    A directory whose records cannot show completeness is not labeled official.
    """
    jsonl = run_dir / "results.jsonl"
    meta = read_json(run_dir / "run.json")
    run_status = meta.get("status") if isinstance(meta, dict) else None
    if jsonl.is_file() and run_status in _FINISHED_RUN_STATUSES:
        records = read_jsonl(jsonl)
    else:
        records = read_record_files(run_dir)
    if not isinstance(meta, dict) or meta.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation result schema")
    if any(not _declared(record) for record in records):
        raise ValueError("unsupported evaluation record schema")
    summary = build_summary(records)
    demote_unfinished(summary, run_status)
    write_summary(run_dir, summary)
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    completeness = summary.get("completeness") or {}
    official = bool(completeness.get("official_wer"))
    lines = [
        "records: ok={ok} error={error} skipped={skipped} validation_failed={invalid}".format(
            ok=summary.get("num_ok", 0),
            error=summary.get("num_error", 0),
            skipped=summary.get("num_skipped", 0),
            invalid=summary.get("num_validation_failed", 0),
        ),
        "wer_aggregation: micro",
        f"run_completeness: {completeness.get('status', 'unknown')}",
        f"official_wer: {str(official).lower()}",
    ]
    if completeness.get("reason"):
        lines.append(f"completeness_reason: {completeness['reason']}")
    groups = summary.get("groups") or []
    if not groups:
        lines.append("groups: 0")
    for group in groups:
        lines.append(
            " ".join(
                [
                    f"{group['condition']}",
                    f"model={group['model']}",
                    f"variant={group['variant']}",
                    f"snr={group['snr_db']}",
                    f"n={group['utterances']}",
                    _wer_label(group),
                    f"wer_n={group['wer_count']}",
                    f"pesq={_fmt(group['pesq_mean'])}",
                    f"stoi={_fmt(group['stoi_mean'])}",
                    f"si_sdr={_fmt(group['si_sdr_mean'])}",
                ]
            )
        )
    for paired in summary.get("paired_signal_metrics") or []:
        lines.append(
            f"paired {paired['metric']} model={paired['model']} variant={paired['variant']} "
            f"snr={paired['snr_db']} n={paired['count']} excluded={len(paired['excluded'])} "
            f"noisy={_fmt(paired['noisy_mean'])} enhanced={_fmt(paired['enhanced_mean'])} "
            f"difference={_fmt(paired['mean_difference'])}"
        )
    missing = completeness.get("missing") or []
    if missing:
        lines.append(f"missing_required_wer: {len(missing)}")
        for item in missing[:30]:
            lines.append(
                f"  - utterance={item.get('utterance_id')} condition={item.get('condition')} "
                f"reason={item.get('reason')}"
            )
        if len(missing) > 30:
            lines.append(f"  - ... {len(missing) - 30} more")
    return "\n".join(lines)


def _groups(records: list[dict[str, Any]], *, official_run: bool) -> list[dict[str, Any]]:
    groups = []
    for key, items in _cohorts(records).items():
        for condition in _cohort_conditions(items):
            coverage = _wer_coverage(items, condition)
            groups.append(
                _render_group(
                    (*key, condition),
                    items,
                    coverage=coverage,
                    official=official_run and coverage == "complete",
                )
            )
    groups.sort(key=_group_sort)
    return groups


def _render_group(
    key: tuple,
    items: list[dict[str, Any]],
    *,
    coverage: str,
    official: bool,
) -> dict[str, Any]:
    config_hash, model, variant, snr_db, condition = key
    count, raw, normalized = _consume_wer(items, condition)
    # Official requires the full target set. A shorter count is a subset.
    if official and count != len(items):
        official = False
        coverage = "partial" if count else "absent"
    scored = count > 0 and coverage not in {"not_requested", "not_required"}
    value = corpus_wer(normalized) if scored else None
    payload: dict[str, Any] = {
        "config_hash": config_hash,
        "model": model,
        "variant": variant,
        "snr_db": snr_db,
        "condition": condition,
        "utterances": len(items),
        "target_utterance_ids": [record.get("utterance_id") for record in items],
        "wer_aggregation": "micro",
        "wer_official": official,
        "wer_coverage": "complete" if official else coverage,
        "wer_normalized": value if official else None,
        "wer_raw": corpus_wer(raw) if official else None,
        "wer_provisional": None if official or not scored else value,
        "wer_count": count if scored else 0,
        "edits_normalized": normalized if official else None,
        "edits_raw": raw if official else None,
        "edits_provisional": None if official or not scored else normalized,
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


def _consume_wer(items: list[dict[str, Any]], condition: str) -> tuple[int, dict[str, int], dict[str, int]]:
    raw = empty_counts()
    normalized = empty_counts()
    count = 0
    for record in items:
        bucket = (record.get("wer") or {}).get("values", {}).get(condition)
        if not isinstance(bucket, dict) or bucket.get("status") != "ok":
            continue
        raw_counts = bucket.get("raw")
        norm_counts = bucket.get("normalized")
        if not isinstance(raw_counts, dict) or not isinstance(norm_counts, dict):
            continue
        if "N" not in norm_counts or "N" not in raw_counts:
            continue
        count += 1
        add_counts(raw, raw_counts)
        add_counts(normalized, norm_counts)
    return count, raw, normalized


def _wer_coverage(items: list[dict[str, Any]], condition: str) -> str:
    if any(not _declared(record) for record in items):
        return "unknown"
    flags = {bool(record["evaluation"]["required_wer"]) for record in items}
    if flags == {False}:
        return "not_requested"
    if flags != {True}:
        return "unknown"
    required = _agreed_required(items)
    if required is None:
        return "unknown"
    if condition not in required:
        return "not_required"
    count, _raw, _normalized = _consume_wer(items, condition)
    if (
        count == len(items)
        and all(record.get("status") == "ok" and record["evaluation"].get("status") == "ok" for record in items)
    ):
        return "complete"
    if count:
        return "partial"
    return "absent"


def _cohort_complete(items: list[dict[str, Any]]) -> bool:
    required = _agreed_required(items)
    if not required:
        return False
    return all(_wer_coverage(items, condition) == "complete" for condition in required)


def _cohort_conditions(items: list[dict[str, Any]]) -> list[str]:
    required: list[str] = []
    seen: list[str] = []
    for record in items:
        for condition in (record.get("evaluation") or {}).get("required_conditions") or []:
            if condition not in required:
                required.append(condition)
        for condition in CONDITIONS:
            if _has_condition(record, condition) and condition not in seen:
                seen.append(condition)
    selected = list(required)
    for condition in seen:
        if condition not in selected:
            selected.append(condition)
    return selected


def _cohorts(records: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_cohort_key(record), []).append(record)
    return grouped


def _cohort_key(record: dict[str, Any]) -> tuple:
    identity = record.get("identity") or {}
    model = record.get("model") or {}
    return (
        identity.get("config_hash"),
        model.get("name"),
        model.get("variant"),
        record.get("snr_db"),
    )


def _declared(record: dict[str, Any]) -> bool:
    evaluation = record.get("evaluation")
    return record.get("schema_version") == RECORD_SCHEMA_VERSION and isinstance(evaluation, dict) and "required_wer" in evaluation


def _agreed_required(items: list[dict[str, Any]]) -> tuple[str, ...] | None:
    found = {
        tuple(record["evaluation"].get("required_conditions") or [])
        for record in items
    }
    if len(found) != 1:
        return None
    required = next(iter(found))
    if not required:
        return None
    return required


def _required_gaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        evaluation = record.get("evaluation") or {}
        required = list(evaluation.get("required_conditions") or [])
        declared = evaluation.get("conditions") or {}
        values = (record.get("wer") or {}).get("values") or {}
        for condition in required:
            cell = values.get(condition) or {}
            norm = cell.get("normalized")
            # A sibling condition can fail while this cell is a real score.
            # The score stays available for a provisional figure. Official WER
            # still requires the utterance itself to be ok; see _wer_coverage.
            scored = cell.get("status") == "ok" and isinstance(norm, dict) and "N" in norm
            if scored:
                continue
            reason = (declared.get(condition) or {}).get("reason") or cell.get("reason")
            if not reason:
                reasons = evaluation.get("reasons") or []
                reason = reasons[0] if reasons else record.get("status") or "missing"
            rows.append(_missing_row(record, condition, reason))
    return rows


def _completeness_without_wer(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {record.get("status") for record in records}
    if statuses <= {"ok", "skipped"}:
        return _completeness(
            "completed",
            official=False,
            required=False,
            reason="wer_not_requested",
            missing=[],
        )
    if "validation_failed" in statuses:
        state = "validation_failed"
        reason = "validation_failed"
    else:
        state = "incomplete"
        reason = "utterance_error"
    missing = [
        _missing_row(record, None, _failure_reason(record))
        for record in records
        if record.get("status") not in {"ok", "skipped"}
    ]
    return _completeness(state, official=False, required=False, reason=reason, missing=missing)


def _completeness(
    status: str,
    *,
    official: bool,
    required: bool | None,
    reason: str | None,
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "official_wer": official,
        "required_wer": required,
        "reason": reason,
        "missing": missing,
    }


def _failure_reason(record: dict[str, Any]) -> Any:
    reasons = (record.get("evaluation") or {}).get("reasons") or []
    if reasons:
        return reasons[0]
    return record.get("status") or "error"


def _missing_row(record: dict[str, Any], condition: str | None, reason: Any) -> dict[str, Any]:
    return {
        "utterance_id": record.get("utterance_id"),
        "record_id": record.get("record_id"),
        "condition": condition,
        "reason": reason,
    }


def _paired_signal_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare noisy and enhanced only on records valid for both conditions.

    Auxiliary scores keep their previous membership: an utterance that was
    scored, including one whose required WER was refused, can still contribute
    a finite PESQ or STOI. Enhancement and recognition failures do not.
    """
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") not in {"ok", "validation_failed"}:
            continue
        identity = record.get("identity") or {}
        key = (
            identity.get("config_hash"), record.get("model", {}).get("name"),
            record.get("model", {}).get("variant"), record.get("snr_db"),
        )
        grouped.setdefault(key, []).append(record)
    result = []
    for key, items in sorted(grouped.items(), key=lambda item: _pair_sort(item[0])):
        for metric in SIGNAL_METRICS:
            if not all(
                any(condition in row.get(metric, {}).get("values", {}) for row in items)
                for condition in ("noisy", "enhanced")
            ):
                continue
            pairs, case_ids, excluded = [], [], []
            for record in items:
                values = record.get(metric, {}).get("values", {})
                cells = [values.get(condition) or {} for condition in ("noisy", "enhanced")]
                if all(_pair_exclusion(cell) is None for cell in cells):
                    pairs.append((float(cells[0]["value"]), float(cells[1]["value"])))
                    case_ids.append(record.get("utterance_id"))
                else:
                    excluded.append({
                        "utterance_id": record.get("utterance_id"),
                        "noisy_reason": _pair_exclusion(cells[0]),
                        "enhanced_reason": _pair_exclusion(cells[1]),
                    })
            if not pairs and not excluded:
                continue
            count = len(pairs)
            result.append({
                "config_hash": key[0], "model": key[1], "variant": key[2],
                "snr_db": key[3], "metric": metric, "count": count,
                "case_ids": case_ids,
                "noisy_mean": sum(p[0] for p in pairs) / count if count else None,
                "enhanced_mean": sum(p[1] for p in pairs) / count if count else None,
                "mean_difference": sum(p[1] - p[0] for p in pairs) / count if count else None,
                "excluded": excluded,
            })
    return result


def _pair_sort(key: tuple) -> tuple:
    config_hash, model, variant, snr_db = key
    return (
        config_hash or "",
        model or "",
        variant or "",
        snr_db is None,
        -1 if snr_db is None else float(snr_db),
    )


def _pair_exclusion(cell: dict[str, Any]) -> str | None:
    value = cell.get("value")
    if cell.get("status") == "ok" and isinstance(value, (int, float)) and math.isfinite(value):
        return None
    if cell.get("status") == "ok":
        return "nonfinite_or_invalid"
    return cell.get("reason") or cell.get("status") or "missing"


def _metric_values(items: list[dict[str, Any]], metric: str, condition: str) -> list[float]:
    values = []
    for record in items:
        # Signal means stay on finite ok cells. Utterance WER failure does not
        # delete a PESQ that was actually computed, and it does not invent one.
        cell = record.get(metric, {}).get("values", {}).get(condition)
        if not cell or cell.get("status") != "ok" or cell.get("value") is None:
            continue
        number = float(cell["value"])
        if not math.isfinite(number):
            continue
        values.append(number)
    return values


def _has_condition(record: dict[str, Any], condition: str) -> bool:
    if condition in record.get("wer", {}).get("values", {}):
        return True
    for metric in SIGNAL_METRICS:
        if condition in record.get(metric, {}).get("values", {}):
            return True
    return False


def _group_sort(group: dict[str, Any]) -> tuple:
    snr_db = group.get("snr_db")
    return (
        group.get("config_hash") or "",
        group.get("model") or "",
        group.get("variant") or "",
        snr_db is None,
        -1 if snr_db is None else float(snr_db),
        group.get("condition") or "",
    )


def _wer_label(group: dict[str, Any]) -> str:
    if group.get("wer_official"):
        value = group.get("wer_normalized")
        shown = "undefined" if value is None else _fmt(value)
        return f"wer=official:{shown}"
    coverage = group.get("wer_coverage")
    if coverage == "not_requested":
        return "wer=not_requested"
    if coverage == "not_required":
        return "wer=not_required"
    if group.get("wer_provisional") is not None:
        return f"wer=provisional:{_fmt(group['wer_provisional'])}"
    return "wer=withheld"


def _write_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config_hash",
        "model",
        "variant",
        "snr_db",
        "condition",
        "utterances",
        "wer_aggregation",
        "wer_official",
        "wer_coverage",
        "wer_normalized",
        "wer_provisional",
        "wer_raw",
        "wer_count",
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
                "wer_official": group.get("wer_official"),
                "wer_coverage": group.get("wer_coverage"),
                "wer_normalized": group["wer_normalized"],
                "wer_provisional": group.get("wer_provisional"),
                "wer_raw": group["wer_raw"],
                "wer_count": group["wer_count"],
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
