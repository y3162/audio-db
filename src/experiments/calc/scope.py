"""DDP accumulation scope used by the reference model steps."""

from __future__ import annotations

import contextlib

import torch
from torch.nn.parallel import DistributedDataParallel


@contextlib.contextmanager
def accumulation_scope(modules: list[torch.nn.Module], *, last: bool):
    """Skip the DDP all-reduce on every micro-batch except the last.

    DistributedDataParallel reads ``require_backward_grad_sync`` during
    forward and installs the all-reduce there. A ``no_sync`` that starts only
    at ``loss.backward()`` does not cancel that reduction. Enter this context
    before the forward that should participate in the optimizer step. The
    matching backward may run after the context exits: the reducer already
    recorded the decision. Non-distributed modules are unchanged. The caller
    divides the loss by the accumulation count so the reduced gradient matches
    the mean over the global batch.
    """

    with contextlib.ExitStack() as stack:
        if not last:
            for module in modules:
                if isinstance(module, DistributedDataParallel):
                    stack.enter_context(module.no_sync())
        yield


def backward(loss: torch.Tensor, modules: list[torch.nn.Module], *, last: bool) -> None:
    """Backpropagate one scaled micro-batch.

    Prefer :func:`accumulation_scope` around the forward. This function only
    runs ``loss.backward()``. Wrapping it in ``no_sync`` cannot undo an
    all-reduce that the forward already scheduled, so the arguments are
    retained for existing callers and are not a synchronization point.
    """

    del modules, last
    loss.backward()
