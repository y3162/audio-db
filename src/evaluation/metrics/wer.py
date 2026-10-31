"""Corpus-oriented word error rate.

Utterance records store substitution, deletion, insertion, and reference length.
A corpus WER is (S + D + I) / N over those counts. It is not the mean of
per-utterance ratios. Normalization version ``wer-norm-v1`` is applied to both
the reference and the hypothesis.
"""

from __future__ import annotations

import unicodedata

from src.evaluation.errors import ConfigError

WER_NORM_V1 = "wer-norm-v1"


def tokenize_raw(text: str) -> list[str]:
    return text.split()


def normalize_text(text: str, version: str = WER_NORM_V1) -> str:
    if version != WER_NORM_V1:
        raise ConfigError(f"unknown WER normalization: {version}")
    normalized = unicodedata.normalize("NFKC", text).lower()
    characters = []
    for char in normalized:
        if unicodedata.category(char).startswith("P"):
            characters.append(" ")
        else:
            characters.append(char)
    return " ".join("".join(characters).split())


def tokenize(text: str, version: str | None) -> list[str]:
    if version is None:
        return tokenize_raw(text)
    return normalize_text(text, version).split()


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int | float | None]:
    """Levenshtein counts. Substitution is preferred when its cost ties."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i
    for j in range(1, cols):
        costs[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
            else:
                costs[i][j] = 1 + min(
                    costs[i - 1][j - 1],
                    costs[i - 1][j],
                    costs[i][j - 1],
                )
    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and costs[i][j] == costs[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and costs[i][j] == costs[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and costs[i][j] == costs[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            raise RuntimeError("word-error alignment failed")
    reference_length = len(reference)
    total = substitutions + deletions + insertions
    wer = None if reference_length == 0 else total / reference_length
    return {
        "S": substitutions,
        "D": deletions,
        "I": insertions,
        "N": reference_length,
        "wer": wer,
    }


def score_pair(reference: str, hypothesis: str, normalization: str) -> dict[str, object]:
    return {
        "status": "ok",
        "raw": edit_counts(tokenize_raw(reference), tokenize_raw(hypothesis)),
        "normalized": edit_counts(
            tokenize(reference, normalization),
            tokenize(hypothesis, normalization),
        ),
    }


def empty_counts() -> dict[str, int]:
    return {"S": 0, "D": 0, "I": 0, "N": 0, "empty_reference": 0}


def add_counts(total: dict[str, int], item: dict[str, object]) -> None:
    if int(item["N"]) == 0:
        total["empty_reference"] += 1
        return
    for key in ("S", "D", "I", "N"):
        total[key] += int(item[key])


def corpus_wer(total: dict[str, int]) -> float | None:
    if total["N"] == 0:
        return None
    return (total["S"] + total["D"] + total["I"]) / total["N"]
