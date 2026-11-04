"""Chunk batches for fixed-length, single-category ESPnet-style inputs.

The planner follows the pinned ESPnet ``ChunkIterFactory`` for one sample rate,
one category, and discarding utterances that are not longer than the chunk.
It does not load audio. The trainer materializes the planned slices.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chunk:
    case_id: str
    start: int
    stop: int


def chunk_batches(samples, *, epoch, seed, chunk_length, batch_size,
                  cache_size=1024, shuffle=True, shift_ratio=0.5):
    """Plan chunks from ``(case_id, sample_length)`` in input order.

    Covers fixed sample rate, one category and discard-short semantics only.
    Yields complete batches; leftover chunks are discarded at epoch end.
    """
    if chunk_length < 2 or batch_size < 1 or cache_size < 1:
        raise ValueError('invalid chunk or batch size')
    shift = int(chunk_length * shift_ratio)
    if shift < 1:
        raise ValueError('chunk shift must be positive')
    rng = np.random.RandomState(epoch + seed)
    cache = []
    limit = max(cache_size, batch_size)

    def flush():
        nonlocal cache
        if shuffle:
            order = np.arange(len(cache))
            rng.shuffle(order)
            cache = [cache[i] for i in order]
        while len(cache) >= batch_size:
            yield cache[:batch_size]
            cache = cache[batch_size:]

    for case_id, length in samples:
        if length <= chunk_length:
            continue
        # Match ESPnet's random choice even for a single candidate.
        width = int(rng.choice([chunk_length], size=1).item())
        count = (length - width) // shift + 1
        offset = rng.randint(0, (length - width) % shift + 1) if shuffle else 0
        cache.extend(Chunk(case_id, offset + i * shift, offset + i * shift + width)
                     for i in range(count))
        if len(cache) > limit:
            yield from flush()
    yield from flush()


def select_utterances(case_ids, *, epoch, seed, num_samples, shuffle=True):
    """Select an epoch's utterances using ESPnet RawSampler ordering.

    Epochs are one-based, as in ESPnet's trainer. This assumes its input
    sampler generates a fixed, unshuffled singleton list for every seed.
    """
    if epoch < 1 or num_samples < 1 or not case_ids:
        raise ValueError('epoch, num_samples and input must be positive')
    count = len(case_ids)
    cursor = num_samples * (epoch - 1)
    result = []
    remaining = num_samples
    while remaining:
        cycle, offset = divmod(cursor, count)
        indices = list(range(count))
        if shuffle:
            np.random.RandomState(cycle + seed).shuffle(indices)
        take = min(count - offset, remaining)
        result.extend(case_ids[i] for i in indices[offset:offset + take])
        remaining -= take
        cursor += take
    return result


def epoch_chunk_batches(
    case_ids,
    lengths,
    *,
    espnet_epoch,
    seed,
    num_samples,
    chunk_length,
    batch_size,
    cache_size=1024,
    shuffle=True,
    shift_ratio=0.5,
):
    """Plan one ESPnet epoch of chunk batches.

    ``espnet_epoch`` is one-based, matching ``build_iter``. ``num_samples``
    is the number of source utterances. ``None`` uses each utterance once,
    which is the validation iterator (``num_iters_per_epoch`` is unset).
    """
    if len(case_ids) != len(lengths):
        raise ValueError('case ids and lengths differ')
    if num_samples is None:
        selected = list(case_ids)
        if shuffle:
            order = list(range(len(selected)))
            np.random.RandomState(espnet_epoch + seed).shuffle(order)
            selected = [selected[i] for i in order]
    else:
        selected = select_utterances(
            list(case_ids), epoch=espnet_epoch, seed=seed,
            num_samples=num_samples, shuffle=shuffle,
        )
    length_of = dict(zip(case_ids, lengths))
    samples = [(case_id, int(length_of[case_id])) for case_id in selected]
    return list(chunk_batches(
        samples, epoch=espnet_epoch, seed=seed, chunk_length=chunk_length,
        batch_size=batch_size, cache_size=cache_size, shuffle=shuffle,
        shift_ratio=shift_ratio,
    ))
