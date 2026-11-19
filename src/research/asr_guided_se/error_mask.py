"""Token-id alignment and the clean-side substitution/deletion mask.

Insertion is not a clean-reference window. Ties break in the order C, S, D, I.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def align_token_ids(
    clean_ids: Sequence[int],
    hypothesis_ids: Sequence[int],
) -> list[tuple[str, int | None, int | None]]:
    """Token-id Levenshtein. Equal cost prefers C, then S, then D, then I."""

    clean = [int(token) for token in clean_ids]
    hypothesis = [int(token) for token in hypothesis_ids]
    n = len(clean)
    m = len(hypothesis)
    inf = n + m + 1
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"
    rank = {"C": 0, "S": 1, "D": 2, "I": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = []
            if clean[i - 1] == hypothesis[j - 1]:
                candidates.append((dp[i - 1][j - 1], rank["C"], "C"))
            else:
                candidates.append((dp[i - 1][j - 1] + 1, rank["S"], "S"))
            candidates.append((dp[i - 1][j] + 1, rank["D"], "D"))
            candidates.append((dp[i][j - 1] + 1, rank["I"], "I"))
            cost, _rank, move = min(candidates)
            dp[i][j] = cost
            back[i][j] = move
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "C":
            ops.append(("C", i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "S":
            ops.append(("S", i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "D":
            ops.append(("D", i - 1, None))
            i -= 1
        elif move == "I":
            ops.append(("I", None, j - 1))
            j -= 1
        else:
            raise RuntimeError("alignment traceback failed")
    ops.reverse()
    return ops


def error_frames_from_alignment(
    clean_tokens: Sequence[Any],
    alignment: Sequence[tuple[str, int | None, int | None]],
) -> list[int]:
    """Union of clean-token frames for substitution and deletion."""

    frames: set[int] = set()
    for operation, clean_index, _hypothesis_index in alignment:
        if operation not in ("S", "D") or clean_index is None:
            continue
        start = _offset(clean_tokens[clean_index], "start_offset")
        end = _offset(clean_tokens[clean_index], "end_offset")
        if end < start:
            raise ValueError(f"token window ends before it starts: {start}, {end}")
        frames.update(range(start, end))
    return sorted(frames)


def sample_intervals_from_frames(
    frames: Sequence[int],
    n_samples: int,
    hop: int,
) -> list[tuple[int, int]]:
    """Encoder frame ``f`` covers ``[f*hop, (f+1)*hop)`` clipped to the utterance."""

    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}")
    intervals: list[tuple[int, int]] = []
    for frame in frames:
        start = max(0, int(frame) * hop)
        end = min(int(n_samples), (int(frame) + 1) * hop)
        if start < end:
            intervals.append((start, end))
    return intervals


def overlapping_tokens(
    tokens: Sequence[Any],
    crop_start: int,
    content_length: int,
    hop: int,
) -> list[Any]:
    """Keep tokens whose sample span overlaps the crop content. Order is unchanged."""

    begin = int(crop_start)
    end = begin + int(content_length)
    kept = []
    for token in tokens:
        span_start = _offset(token, "start_offset") * hop
        span_end = _offset(token, "end_offset") * hop
        if span_end > begin and span_start < end:
            kept.append(token)
    return kept


def crop_error_mask(
    clean_tokens: Sequence[Any],
    hypothesis_tokens: Sequence[Any],
    *,
    crop_start: int,
    content_length: int,
    window: int,
    hop: int,
    full_length: int,
) -> list[bool]:
    """Sample mask on the crop window. Padding and insertion stay false."""

    alignment = align_token_ids(
        [_token_id(token) for token in clean_tokens],
        [_token_id(token) for token in hypothesis_tokens],
    )
    frames = error_frames_from_alignment(clean_tokens, alignment)
    intervals = sample_intervals_from_frames(frames, full_length, hop)
    mask = [False] * int(window)
    content_end = int(crop_start) + int(content_length)
    for start, end in intervals:
        left = max(start, int(crop_start))
        right = min(end, content_end)
        for index in range(left, right):
            mask[index - int(crop_start)] = True
    return mask


def _token_id(token: Any) -> int:
    if isinstance(token, dict):
        return int(token["token_id"])
    return int(token.token_id)


def _offset(token: Any, name: str) -> int:
    if isinstance(token, dict):
        return int(token[name])
    return int(getattr(token, name))
