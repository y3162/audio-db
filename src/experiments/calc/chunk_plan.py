"""Chunk batches for fixed-length, single-category ESPnet-style inputs.

World size 1 is the single-rank case of
``src.experiments.data.render.chunking.plan_chunk_epoch``. The yielded chunks
match the previous one-process planner. Multi-rank plans are built there.
"""
from dataclasses import dataclass

from src.experiments.data.render.chunking import iterate_chunks, plan_chunk_epoch, select_source_ids


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
    batches, _tail = iterate_chunks(
        list(samples),
        epoch=epoch,
        seed=seed,
        chunk_length=chunk_length,
        batch_size=batch_size,
        cache_size=cache_size,
        shuffle=shuffle,
        shift_ratio=shift_ratio,
    )
    return [[Chunk(chunk.case_id, chunk.start, chunk.stop) for chunk in batch] for batch in batches]


def select_utterances(case_ids, *, epoch, seed, num_samples, shuffle=True):
    """Select an epoch's utterances using ESPnet RawSampler ordering.

    Epochs are one-based, as in ESPnet's trainer. This assumes its input
    sampler generates a fixed, unshuffled singleton list for every seed.
    """
    return select_source_ids(
        list(case_ids),
        epoch=epoch,
        seed=seed,
        num_samples=num_samples,
        shuffle=shuffle,
    )


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
    """Plan one ESPnet epoch of chunk batches for a single process.

    ``espnet_epoch`` is one-based, matching ``build_iter``. ``num_samples``
    is the number of source utterances. ``None`` uses each utterance once,
    which is the validation iterator (``num_iters_per_epoch`` is unset).
    """
    if len(case_ids) != len(lengths):
        raise ValueError("case ids and lengths differ")
    plan = plan_chunk_epoch(
        list(zip(case_ids, (int(length) for length in lengths))),
        espnet_epoch=espnet_epoch,
        seed=seed,
        world_size=1,
        num_samples=num_samples,
        chunk_length=chunk_length,
        global_batch=batch_size,
        num_cache_chunks=cache_size,
        shuffle=shuffle,
        shift_ratio=shift_ratio,
    )
    return [
        [Chunk(chunk.case_id, chunk.start, chunk.stop) for chunk in batch]
        for batch in plan.rank_batches[0]
    ]
