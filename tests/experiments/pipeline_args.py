"""Pass a snapshot's registered preset in as explicit pipeline arguments.

The training classes do not read normalize, crop, order, or resample from the
snapshot. Tests that want that historical preset copy it here and hand it over.
"""

from __future__ import annotations

from typing import Any

from src.experiments.data.fixed_pair import FixedPairProtocol, OrderSampler, SnapshotSource


def fixed_protocol(source: SnapshotSource, verify_on_read: bool = False) -> FixedPairProtocol:
    source.open()
    rule = source.dataset()["rule_json"]
    resample = rule.get("resample")
    return FixedPairProtocol(
        source,
        verify_on_read,
        normalize=str(rule["normalize"]),
        window=int(rule["window"]),
        sample_rate=int(rule["sample_rate"]),
        validation_crop=str(rule["validation"]["crop"]),
        resample=resample if isinstance(resample, dict) else None,
    )


def order_kwargs(source: SnapshotSource) -> dict[str, Any]:
    source.open()
    dataset = source.dataset()
    rule = dataset["rule_json"]
    order = rule["order"]
    kwargs: dict[str, Any] = {
        "order": order,
        "validation_scope": "per_rank" if order == "rank_mod_fixed_v1" else "rank0",
    }
    if str(dataset.get("algorithm")) == "online.semambapp_gsr":
        kwargs["validation_batch"] = int(rule["validation"]["batch"])
    return kwargs


def order_sampler(
    source: SnapshotSource,
    protocol: object,
    *,
    seed: int,
    global_batch: int | None = None,
) -> OrderSampler:
    kwargs = order_kwargs(source)
    if global_batch is None:
        return OrderSampler(source, protocol, seed=seed, **kwargs)  # type: ignore[arg-type]
    return OrderSampler(source, protocol, seed=seed, global_batch=global_batch, **kwargs)  # type: ignore[arg-type]
