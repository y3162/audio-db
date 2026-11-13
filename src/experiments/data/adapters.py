"""Dataset adapters that emit a noisy/target pair.

Selection lives here, while a chunk plan is being prepared. Samplers,
validation, and trainers do not call these classes by dataset name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.experiments.data.fixed_pair import SnapshotSource, _call
from src.experiments.errors import DataError
from src.experiments.data.mix.dns_ondemand import dns_rendering, render_dns_pair
from src.experiments.data.mix.ondemand import ms_snsd_rendering, render_ms_snsd_pair

_MS = "pregenerated.ms_snsd_tfgridnet"
_DNS = "pregenerated.dns_tflocoformer"


class ChunkAdapter:
    """One realization renderer for a chunk plan."""

    def prepare(self) -> None:
        raise NotImplementedError

    def render_span(self, case_id: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, int]:
        raise NotImplementedError


class MsSnsdChunkAdapter(ChunkAdapter):
    def __init__(self, source: SnapshotSource, *, verify_on_read: bool) -> None:
        self.source = source
        self.verify_on_read = verify_on_read
        self._speech: dict[str, Any] | None = None
        self._noise: dict[str, Any] | None = None

    def prepare(self) -> None:
        slot = "test.cases" if self.source.split == "test" else "train.cases"
        speech, noise = _call(ms_snsd_rendering, self.source.store, slot)
        self._speech = speech
        self._noise = noise

    def render_span(self, case_id: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, int]:
        if self._speech is None or self._noise is None:
            raise DataError("MS-SNSD adapter is not prepared")
        clean, noisy = _call(
            render_ms_snsd_pair,
            self.source.store,
            case_id,
            speech=self._speech,
            noise=self._noise,
            verify_on_read=self.verify_on_read,
        )
        _span_in_range(case_id, start, stop, int(clean.shape[0]))
        return noisy[start:stop].copy(), clean[start:stop].copy(), stop - start


class DnsChunkAdapter(ChunkAdapter):
    def __init__(self, source: SnapshotSource, *, verify_on_read: bool) -> None:
        self.source = source
        self.verify_on_read = verify_on_read
        self._speech: dict[str, Any] | None = None
        self._noise: dict[str, Any] | None = None
        self._renderer: str | None = None
        self._retry: str | None = None

    def prepare(self) -> None:
        speech, noise, renderer, retry = _call(dns_rendering, self.source.store)
        self._speech = speech
        self._noise = noise
        self._renderer = renderer
        self._retry = retry

    def render_span(self, case_id: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, int]:
        if self._speech is None or self._noise is None:
            raise DataError("DNS adapter is not prepared")
        clean, noisy = _call(
            render_dns_pair,
            self.source.store,
            case_id,
            speech=self._speech,
            noise=self._noise,
            retry_policy=str(self._retry),
            renderer=str(self._renderer),
            verify_on_read=self.verify_on_read,
        )
        _span_in_range(case_id, start, stop, int(clean.shape[0]))
        return (
            np.ascontiguousarray(noisy[start:stop]),
            np.ascontiguousarray(clean[start:stop]),
            stop - start,
        )


def open_chunk_adapter(source: SnapshotSource, *, verify_on_read: bool) -> ChunkAdapter:
    """Bind the chunk renderer named by the snapshot's adapter id."""

    algorithm = str(source.dataset().get("algorithm"))
    adapters = {
        _MS: MsSnsdChunkAdapter,
        _DNS: DnsChunkAdapter,
    }
    factory = adapters.get(algorithm)
    if factory is None:
        raise DataError(f"no chunk adapter is registered for {algorithm}")
    return factory(source, verify_on_read=verify_on_read)


def _span_in_range(case_id: str, start: int, stop: int, length: int) -> None:
    if start < 0 or stop > length or stop <= start:
        raise DataError(f"chunk {start}:{stop} is outside {case_id} ({length} samples)")
