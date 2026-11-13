"""Audit speaker splits and shared noise spans in canonical VoiceBank recipes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from src.experiments.data.errors import ReproductionError

RATE = 48000
HOLDOUT = frozenset(("p286", "p287"))
SPLITS = ("train", "valid", "test")


def _span(row: Mapping[str, Any]) -> tuple[int, int]:
    try:
        raw_start, raw_end = row["noise_start_48k"], row["noise_end_exclusive_48k"]
        if any(isinstance(v, bool) or not isinstance(v, (str, int)) for v in (raw_start, raw_end)):
            raise ValueError("indices must be integers")
        start, end = int(raw_start), int(raw_end)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReproductionError("invalid noise span") from exc
    if start < 0 or end <= start:
        raise ReproductionError("invalid noise span")
    return start, end


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def audit_recipes(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count overlap without counting a target sample twice within one utterance.

    Noise is identified by SHA-256, so aliases cannot hide shared material.
    Target totals count each utterance separately, even when targets overlap.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in SPLITS}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        utterance = row.get("utterance_id")
        source_split = row.get("split")
        if row.get("status") != "ready" or source_split not in ("train", "test"):
            raise ReproductionError("recipe is not ready or has an invalid split")
        if not isinstance(utterance, str) or "_" not in utterance:
            raise ReproductionError("recipe has an invalid utterance id")
        key = (source_split, utterance)
        if key in seen:
            raise ReproductionError(f"duplicate recipe identity: {key}")
        seen.add(key)
        digest = row.get("noise_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ReproductionError("invalid noise sha256")
        _span(row)
        speaker = utterance.split("_", 1)[0]
        split = "valid" if source_split == "train" and speaker in HOLDOUT else source_split
        grouped[split].append(row)

    identities = {name: {str(r["utterance_id"]) for r in group} for name, group in grouped.items()}
    speakers = {name: {u.split("_", 1)[0] for u in ids} for name, ids in identities.items()}
    split_report = {
        name: {"utterances": len(grouped[name]), "speakers": len(speakers[name])}
        for name in SPLITS
    }
    intersections: dict[str, Any] = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        left_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in grouped[left]:
            left_spans[str(row["noise_sha256"])].append(_span(row))
        merged = {key: _merge(spans) for key, spans in left_spans.items()}
        right_hashes: set[str] = set()
        total = overlap = affected = 0
        for row in grouped[right]:
            key = str(row["noise_sha256"])
            right_hashes.add(key)
            start, end = _span(row)
            samples = sum(max(0, min(end, b) - max(start, a)) for a, b in merged.get(key, []))
            total += end - start
            overlap += samples
            affected += int(samples > 0)
        intersections[f"{left}_{right}"] = {
            "shared_utterance_ids": sorted(identities[left] & identities[right]),
            "shared_speakers": sorted(speakers[left] & speakers[right]),
            "shared_noise_sha256_count": len(set(left_spans) & right_hashes),
            "target_utterances_with_noise_overlap": affected,
            "target_noise_samples": total,
            "overlap_samples": overlap,
            "overlap_seconds": overlap / RATE,
            "target_noise_overlap_fraction": overlap / total if total else None,
            "target": right,
        }
    valid = intersections["train_valid"]
    independent = None if not valid["target_noise_samples"] else valid["overlap_samples"] == 0
    return {"sample_rate": RATE, "splits": split_report, "comparisons": intersections,
            "validation_noise_independent": independent}
