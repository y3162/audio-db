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
from src.experiments.data.sampling.planner import chunk_case_length, fixed_pair_case_length

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


class RecordedChunkAdapter(ChunkAdapter):
    """Slice a recorded pair after the pipeline resample, without a second normalize."""

    def __init__(
        self,
        source: SnapshotSource,
        *,
        verify_on_read: bool,
        sample_rate: int,
        resample: dict | None,
    ) -> None:
        self.source = source
        self.verify_on_read = verify_on_read
        self.sample_rate = int(sample_rate)
        self.resample = resample

    def prepare(self) -> None:
        _require_resample_fit(recorded_native_rate(self.source), self.sample_rate, self.resample)

    def render_span(self, case_id: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, int]:
        from src.experiments.data.render.renderer import load_recorded_pair

        noisy, clean = _call(
            load_recorded_pair,
            self.source.store,
            case_id,
            sample_rate=self.sample_rate,
            verify_on_read=self.verify_on_read,
            resample=self.resample,
        )
        _span_in_range(case_id, start, stop, int(noisy.shape[0]))
        return noisy[start:stop].copy(), clean[start:stop].copy(), stop - start


class PairAdapter:
    """Full noisy/target utterance before a model pipeline crops or normalizes."""

    recorded = False

    def prepare(self) -> None:
        raise NotImplementedError

    def render_full(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def case_length(self, case_id: str, *, normalize: str, resampled: bool) -> int:
        raise NotImplementedError


class RecordedPairAdapter(PairAdapter):
    recorded = True

    def __init__(self, source: SnapshotSource, *, verify_on_read: bool, sample_rate: int, resample: dict | None) -> None:
        self.source = source
        self.verify_on_read = verify_on_read
        self.sample_rate = int(sample_rate)
        self.resample = resample

    def prepare(self) -> None:
        _require_resample_fit(recorded_native_rate(self.source), self.sample_rate, self.resample)

    def render_full(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        from src.experiments.data.render.renderer import load_recorded_pair

        return _call(
            load_recorded_pair,
            self.source.store,
            case_id,
            sample_rate=self.sample_rate,
            verify_on_read=self.verify_on_read,
            resample=self.resample,
        )

    def case_length(self, case_id: str, *, normalize: str, resampled: bool) -> int:
        return _call(
            fixed_pair_case_length,
            self.source.store,
            case_id,
            normalize,
            resampled=resampled,
        )


class _GeneratedPairAdapter(PairAdapter):
    def __init__(self, source: SnapshotSource, *, verify_on_read: bool, sample_rate: int) -> None:
        self.source = source
        self.verify_on_read = verify_on_read
        self.sample_rate = int(sample_rate)
        self._ready = False

    def prepare(self) -> None:
        native = generated_native_rate(self.source)
        if native != self.sample_rate:
            raise DataError(
                f"dataset rate {native} does not match pipeline rate {self.sample_rate}; "
                "resampling was not requested"
            )
        self._bind()
        self._ready = True

    def _bind(self) -> None:
        raise NotImplementedError

    def case_length(self, case_id: str, *, normalize: str, resampled: bool) -> int:
        del normalize, resampled
        return _call(chunk_case_length, self.source.store, case_id)


class MsSnsdPairAdapter(_GeneratedPairAdapter):
    def __init__(self, source: SnapshotSource, *, verify_on_read: bool, sample_rate: int) -> None:
        super().__init__(source, verify_on_read=verify_on_read, sample_rate=sample_rate)
        self._speech: dict | None = None
        self._noise: dict | None = None

    def _bind(self) -> None:
        slot = "test.cases" if self.source.split == "test" else "train.cases"
        speech, noise = _call(ms_snsd_rendering, self.source.store, slot)
        self._speech = speech
        self._noise = noise

    def render_full(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        if not self._ready or self._speech is None or self._noise is None:
            raise DataError("MS-SNSD adapter is not prepared")
        clean, noisy = _call(
            render_ms_snsd_pair,
            self.source.store,
            case_id,
            speech=self._speech,
            noise=self._noise,
            verify_on_read=self.verify_on_read,
        )
        return noisy, clean


class DnsPairAdapter(_GeneratedPairAdapter):
    def __init__(self, source: SnapshotSource, *, verify_on_read: bool, sample_rate: int) -> None:
        super().__init__(source, verify_on_read=verify_on_read, sample_rate=sample_rate)
        self._speech: dict | None = None
        self._noise: dict | None = None
        self._renderer: str | None = None
        self._retry: str | None = None

    def _bind(self) -> None:
        speech, noise, renderer, retry = _call(dns_rendering, self.source.store)
        self._speech = speech
        self._noise = noise
        self._renderer = renderer
        self._retry = retry

    def render_full(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        if not self._ready or self._speech is None or self._noise is None:
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
        return noisy, clean


def open_chunk_adapter(
    source: SnapshotSource,
    *,
    verify_on_read: bool,
    sample_rate: int,
    resample: dict | None = None,
) -> ChunkAdapter:
    """Bind the chunk renderer named by the snapshot's generation algorithm."""

    algorithm = str(source.dataset().get("algorithm"))
    if algorithm in {_MS, _DNS}:
        if resample is not None:
            raise DataError("generated chunk audio declares its own rate; resample was not requested for this adapter")
        native = generated_native_rate(source)
        if native != int(sample_rate):
            raise DataError(
                f"dataset rate {native} does not match pipeline rate {sample_rate}; resampling was not requested"
            )
        factory = MsSnsdChunkAdapter if algorithm == _MS else DnsChunkAdapter
        return factory(source, verify_on_read=verify_on_read)
    if algorithm.startswith("fixed_pair."):
        return RecordedChunkAdapter(
            source,
            verify_on_read=verify_on_read,
            sample_rate=sample_rate,
            resample=resample,
        )
    raise DataError(f"no chunk adapter is registered for {algorithm}")


def open_pair_adapter(
    source: SnapshotSource,
    *,
    verify_on_read: bool,
    sample_rate: int,
    resample: dict | None,
) -> PairAdapter:
    """Bind the full-utterance renderer. The model pipeline is applied by the caller."""

    algorithm = str(source.dataset().get("algorithm"))
    if algorithm == _MS:
        if resample is not None:
            raise DataError("generated audio is already at its declared rate; resample does not apply")
        return MsSnsdPairAdapter(source, verify_on_read=verify_on_read, sample_rate=sample_rate)
    if algorithm == _DNS:
        if resample is not None:
            raise DataError("generated audio is already at its declared rate; resample does not apply")
        return DnsPairAdapter(source, verify_on_read=verify_on_read, sample_rate=sample_rate)
    if algorithm.startswith("fixed_pair."):
        return RecordedPairAdapter(
            source,
            verify_on_read=verify_on_read,
            sample_rate=sample_rate,
            resample=resample,
        )
    raise DataError(f"no pair adapter is registered for {algorithm}")


def recorded_native_rate(source: SnapshotSource) -> int:
    """File rate of one recorded pair. This is not the model pipeline rate."""

    case_ids = source.case_ids(f"{source.split}.cases")
    if not case_ids:
        raise DataError("snapshot has no cases in this split")
    case = source.store.case(case_ids[0])
    noisy = source.store.asset(str(case["noisy_asset_id"]))
    clean = source.store.asset(str(case["clean_asset_id"]))
    rate = int(noisy["sample_rate"])
    if int(clean["sample_rate"]) != rate or rate < 1:
        raise DataError("recorded pair files do not share a sample rate")
    return rate


def generated_native_rate(source: SnapshotSource) -> int:
    """Sample rate the generator declares for the mixture, before a model pipeline."""

    rule = source.dataset().get("rule_json")
    if not isinstance(rule, dict):
        raise DataError("snapshot rule_json is missing")
    rate = rule.get("sample_rate")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1:
        raise DataError("dataset sample_rate is missing")
    return int(rate)


def _require_resample_fit(native_rate: int, sample_rate: int, resample: dict | None) -> None:
    if resample is None:
        if native_rate != int(sample_rate):
            raise DataError(
                f"dataset rate {native_rate} does not match pipeline rate {sample_rate}; "
                "resampling was not requested"
            )
        return
    from src.experiments.data.ids import _check_resample

    try:
        _check_resample(resample)
    except Exception as exc:
        raise DataError(str(exc)) from exc
    if int(resample["source_rate"]) != int(native_rate) or int(resample["target_rate"]) != int(sample_rate):
        raise DataError(
            f"resample {resample['source_rate']} to {resample['target_rate']} "
            f"does not match dataset {native_rate} and pipeline {sample_rate}"
        )


def _span_in_range(case_id: str, start: int, stop: int, length: int) -> None:
    if start < 0 or stop > length or stop <= start:
        raise DataError(f"chunk {start}:{stop} is outside {case_id} ({length} samples)")
