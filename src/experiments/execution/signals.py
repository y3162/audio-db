"""Stop an experiment at the next safe trainer boundary.

``SIGTERM`` sets a flag. Trainers notice it only where they already decide to
stop, which is outside a GAN optimizer pair. Ranks combine the flag with a
max reduction so one rank's signal does not leave the others in a collective.
"""

from __future__ import annotations

import signal

_REQUESTED = False
_INSTALLED = False


def install_interrupt_handler() -> None:
    """Replace ``SIGTERM`` with a flag. ``SIGINT`` is left alone."""

    global _INSTALLED
    if _INSTALLED:
        return
    signal.signal(signal.SIGTERM, _on_sigterm)
    _INSTALLED = True


def request_interrupt() -> None:
    """Set the stop flag. Tests use this instead of signalling the runner."""

    global _REQUESTED
    _REQUESTED = True


def clear_interrupt() -> None:
    global _REQUESTED
    _REQUESTED = False


def interrupt_requested() -> bool:
    return _REQUESTED


def ranks_should_stop(runtime: object | None, *, extra: bool = False) -> bool:
    """True when this rank's cap or signal, or any other rank's, says stop."""

    local = bool(extra) or interrupt_requested()
    world = 1 if runtime is None else int(getattr(runtime, "world_size", 1) or 1)
    if world <= 1:
        return local
    gather = getattr(runtime, "gather_objects", None)
    if not callable(gather):
        return local
    flags = gather(bool(local))
    return any(bool(item) for item in flags)


def _on_sigterm(signum: int, frame: object) -> None:
    del signum, frame
    global _REQUESTED
    _REQUESTED = True
