"""Resolve epoch-dependent training batch limits."""


def scheduled_batch_size(epoch: int, maximum: int, schedule: list[tuple[int, int]]) -> int:
    """Apply DeepFilterNet's epoch thresholds and configured maximum."""
    if epoch < 0 or maximum < 1 or not schedule or schedule[0][0] != 0:
        raise ValueError('invalid epoch, batch maximum, or schedule')
    previous = -1
    size = maximum
    for start, limit in schedule:
        if start <= previous or limit < 1:
            raise ValueError('schedule must contain ordered nonnegative epochs and positive sizes')
        previous = start
        if start <= epoch:
            size = min(limit, maximum)
    return size
