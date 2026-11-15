"""MossFormerGAN validation, plateau decision, and the bound spectral view.

Failed PESQ batches stay in the denominator and are not stored as loss 0.
Alignment forwards keep SyncBatchNorm in step and stay outside the denominator.
The comparison task scores with its own methods. This module does not call it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from src.experiments.calc.distributed import unwrap
from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mossformergan import loss as moss_loss
from src.experiments.runtime import ProcessRuntime

def local_validation_totals(reports: list[dict[str, Any]]) -> dict[str, float]:
    """Official per-rank validation totals.

    Alignment forwards that exist only to keep SyncBatchNorm in step are not
    part of the denominator. PESQ failures stay in the denominator and do not
    add a numeric loss. Their contribution is absence, not a stored zero.
    """
    loss_sum = 0.0
    batches = 0
    failed = 0
    included = 0
    alignment = 0
    for report in reports:
        if report.get("alignment_pad"):
            alignment += 1
            continue
        batches += 1
        if report.get("generator_loss_included") and report["losses"]["generator"]["status"] == "ok":
            loss_sum += float(report["losses"]["generator"]["value"])
            included += 1
        else:
            failed += 1
    mean = 0.0 if batches == 0 else loss_sum / batches
    return {
        "sum": loss_sum,
        "batches": float(batches),
        "failed": float(failed),
        "included": float(included),
        "alignment": float(alignment),
        "mean": mean,
    }

def official_validation_mean(rank_means: list[float]) -> float:
    """Mean of per-rank means, matching ``Solver._reduce_tensor``."""
    if not rank_means:
        raise TrainingError("validation produced no batches")
    return float(sum(rank_means) / len(rank_means))

def mossformer_validation_decision(best: float, stalls: int, loss: float) -> tuple[float, int, str]:
    """Return the next best, stall count, and action for one validation.

    Actions are ``improve``, ``hold``, ``halve``, and ``stop``. This follows
    ``Solver.train`` in ClearerVoice ``train/speech_enhancement/solver.py``:
    a loss that is not strictly lower replaces the stored best and increments
    the stall count. The fifth stall halves the generator learning rate. The
    tenth stall stops. The stall count is not cleared by the halving.
    """
    if loss >= best:
        best = float(loss)
        stalls += 1
        if stalls == 5:
            return best, stalls, "halve"
        if stalls >= 10:
            return best, stalls, "stop"
        return best, stalls, "hold"
    return float(loss), 0, "improve"

def _component(value: torch.Tensor, included: bool) -> dict[str, Any]:
    if not included:
        return _metric(status="na", reason="batch excluded by the pesq gate")
    return _metric(float(value.detach()))

def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}


def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
    """Official validation generator loss for one already-cropped batch.

    The 2 second crop is applied by the dataset rule before this method
    runs. This method does not step either optimizer. PESQ itself is not
    the plateau metric.

    Official ``Solver._run_one_epoch_mossformergan_se_16k`` adds the loss
    only when it is not None, then returns ``total_loss / (i+1)``. A
    missing loss is not written as zero. The batch stays in the
    denominator. This report keeps that batch out of the numeric values
    and marks it for the denominator. ``reduce_validation`` divides the
    included sum by every non-padding batch. A non-finite generator loss
    is the same exclusion. It does not raise, so the next rank can still
    enter the following forward.
    """
    alignment = bool(batch.get("validation_alignment_pad"))
    clean = batch["clean"].to(self.device)
    noisy = batch["noisy"].to(self.device)
    generator = unwrap(self.generator)
    discriminator = unwrap(self.discriminator)
    generator.eval()
    discriminator.train()
    included = False
    reason = "pesq failed; batch stays in the official denominator and is not a loss of 0"
    terms: dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        terms = moss_loss.generator_terms(self, generator, discriminator, clean, noisy)
        finite = bool(torch.isfinite(terms["generator"]).all())
        targets = moss_loss.metric_targets(terms["teacher"], terms["audio"], self.sample_rate) if finite else None
    if alignment:
        included = False
        reason = "alignment forward keeps SyncBatchNorm in step and is outside the denominator"
    elif not finite:
        included = False
        reason = "non-finite generator loss; official validation leaves it out of the sum and keeps the batch in the denominator"
    elif targets is None:
        included = False
        reason = "pesq failed; batch stays in the official denominator and is not a loss of 0"
    else:
        included = True
        reason = "official generator loss; pesq gate included this batch"
    assert terms is not None
    if included:
        generator_metric = _metric(float(terms["generator"].detach()), reason=reason)
    else:
        generator_metric = _metric(status="na", reason=reason)
    losses = {
        "generator": generator_metric,
        "ri": _component(terms["ri"], included),
        "magnitude": _component(terms["magnitude"], included),
        "time": _component(terms["time"], included),
        "gan": _component(terms["gan"], included),
        "discriminator": _metric(
            status="na",
            reason="validation does not step the discriminator",
        ),
    }
    return {
        "losses": losses,
        "pesq": {
            "status": "na",
            "value": None,
            "reason": "pesq gates inclusion and is not the plateau metric",
            "count": 0,
        },
        "stoi": {"status": "na", "value": None, "reason": "not computed", "count": 0},
        "pesq_gate": "included" if included else "denominator_only",
        "generator_loss_included": included and not alignment,
        "alignment_pad": alignment,
    }

def reduce_validation(self, reports: list[dict[str, Any]], *, collective: bool | None = None) -> dict[str, Any]:
    """Official validation loss: the mean of the per-rank means.

    Each rank's mean divides by its own batch count, including PESQ
    failures and official sampler padding, and excluding alignment pads.
    Ranks are then averaged with equal weight. Equal batch counts make
    this the same number as one global quotient. Unequal counts keep the
    official unweighted mean of means.
    """
    local = local_validation_totals(reports)
    rows = [local]
    if collective is None:
        collective = bool(self.validation_requires_all_ranks)
    if (
        collective
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        payload = torch.tensor(
            [local["sum"], local["batches"], local["failed"], local["included"], local["alignment"], local["mean"]],
            dtype=torch.float64,
        )
        if self.device.type == "cuda":
            payload = payload.to(self.device)
        gathered = [torch.empty_like(payload) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, payload)
        names = ("sum", "batches", "failed", "included", "alignment", "mean")
        rows = [dict(zip(names, [float(value) for value in item.tolist()], strict=True)) for item in gathered]
    active = [row for row in rows if row["batches"] > 0]
    if not active:
        raise TrainingError("validation produced no batches")
    official = official_validation_mean([row["mean"] for row in active])
    failed = int(sum(row["failed"] for row in rows))
    included = int(sum(row["included"] for row in rows))
    batches = int(sum(row["batches"] for row in rows))
    alignment = int(sum(row["alignment"] for row in rows))
    per_rank_batches = [int(row["batches"]) for row in rows]
    return {
        "losses": {
            "generator": {
                "status": "ok",
                "value": official,
                "reason": (
                    "mean of per-rank means; failed PESQ batches stay in each rank's "
                    "denominator and are not stored as loss 0"
                ),
                "count": batches,
            }
        },
        "pesq": {
            "status": "na",
            "value": None,
            "reason": "pesq gates inclusion and is not the plateau metric",
            "count": 0,
            "failed_batches": failed,
            "included_batches": included,
        },
        "stoi": {"status": "na", "value": None, "reason": "not computed", "count": 0},
        "validation_coverage": {
            "batches": batches,
            "included_batches": included,
            "pesq_failed_batches": failed,
            "alignment_pad_forwards": alignment,
            "per_rank_batches": per_rank_batches,
            "denominator": "per-rank batch count, then unweighted mean across ranks",
        },
    }


class OfficialCall:
    """Binds the production spectral and validation calls.

    ``_generator_terms`` and ``validation_report`` call sibling methods. A
    ``SimpleNamespace`` does not bind those functions. The functions are the
    production copies. The comparison task keeps its own.
    """

    _forward = moss_loss.forward_spectra
    _generator_parts = moss_loss.generator_parts
    _generator_terms = moss_loss.generator_terms
    validation_report = validation_report
    reduce_validation = reduce_validation

    def __init__(
        self,
        *,
        n_fft: int,
        hop: int,
        win: int,
        sample_rate: int,
        generator: torch.nn.Module | None,
        discriminator: torch.nn.Module | None,
        device: torch.device,
        validation_requires_all_ranks: bool,
    ) -> None:
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.sample_rate = sample_rate
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.validation_requires_all_ranks = validation_requires_all_ranks

# --- Rank-aware validation. The plateau metric is the generator loss, not PESQ. ---

def evaluate(
    runtime: ProcessRuntime,
    data: Any,
    generator: Any,
    discriminator: Any,
    *,
    trainer_epoch: int,
    n_fft: int,
    hop: int,
    win: int,
    sample_rate: int,
) -> dict[str, Any]:
    cursor = (int(data.epoch), int(data.update_index))
    rng = runtime.capture_rng()
    summary: dict[str, Any] | None = None
    try:
        produced = list(data.validation_epoch(trainer_epoch))
        aligned = align_validation(runtime, produced, discriminator)
        proxy = OfficialCall(
            n_fft=n_fft,
            hop=hop,
            win=win,
            sample_rate=sample_rate,
            generator=generator.execution(),
            discriminator=discriminator.execution(),
            device=runtime.device,
            validation_requires_all_ranks=_syncbn_active(discriminator),
        )
        reports: list[dict[str, Any]] = []
        for batch, alignment in aligned:
            if not isinstance(batch, Batch):
                raise RuntimeFailure("validation did not return a batch")
            moved = runtime.move(batch)
            payload = {"clean": moved.targets, "noisy": moved.inputs, "validation_alignment_pad": alignment}
            try:
                reports.append(validation_report(proxy, payload))
            except TrainingError as exc:
                raise RuntimeFailure(f"validation report failed: {exc}") from exc
        try:
            summary = reduce_validation(proxy, reports, collective=runtime.world_size > 1)
        except TrainingError as exc:
            raise RuntimeFailure(f"validation reduction failed: {exc}") from exc
    finally:
        runtime.restore_rng(rng)
        generator.train()
        discriminator.train()
    runtime.broadcast_buffers(generator)
    runtime.broadcast_buffers(discriminator)
    moved = (int(data.epoch), int(data.update_index))
    if moved != cursor:
        raise RuntimeFailure("validation advanced the training cursor")
    if not isinstance(summary, dict):
        raise RuntimeFailure("validation did not return a summary")
    return summary

def validation_proxy(runtime: ProcessRuntime, generator: Any, discriminator: Any, *, n_fft: int, hop: int, win: int, sample_rate: int) -> OfficialCall:
    return OfficialCall(
        n_fft=n_fft, hop=hop, win=win, sample_rate=sample_rate,
        generator=generator.execution(), discriminator=discriminator.execution(),
        device=runtime.device, validation_requires_all_ranks=_syncbn_active(discriminator),
    )

def align_validation(
    runtime: ProcessRuntime,
    batches: list[Batch],
    discriminator: Any,
) -> list[tuple[Batch, bool]]:
    if runtime.world_size == 1 or not _syncbn_active(discriminator):
        return [(batch, False) for batch in batches]
    counts = runtime.gather_objects(len(batches))
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in counts):
        raise RuntimeFailure("validation batch counts are not integers")
    if any(item == 0 for item in counts):
        rendered = ", ".join(str(item) for item in counts)
        raise RuntimeFailure(f"validation rank has no batch to repeat for SyncBatchNorm alignment: {rendered}")
    target = max(int(item) for item in counts)
    if all(int(item) == int(counts[0]) for item in counts):
        return [(batch, False) for batch in batches]
    padded = [(batch, False) for batch in batches]
    source = 0
    while len(padded) < target:
        padded.append((batches[source % len(batches)], True))
        source += 1
    return padded

def _syncbn_active(discriminator: Any) -> bool:
    return any(isinstance(module, torch.nn.SyncBatchNorm) for module in discriminator.unwrapped().modules())
