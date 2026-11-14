"""TF-Locoformer official SI-SDR ranking.

Loss ranking and patience stay in the TF-GridNet selection module. This
module ranks the positive SI-SDR monitor only.
"""

from __future__ import annotations

from typing import Any

from src.experiments.errors import RuntimeFailure


def rank_validation_si_snr(history: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Highest ``si_snr`` first. Equal scores keep the earlier epoch.

    Rows without ``si_snr`` are omitted. The value is SI-SDR in dB.
    """

    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise RuntimeFailure("keep_nbest must be a positive integer")
    ranked = [row for row in history if row.get("si_snr") is not None]
    ordered = sorted(ranked, key=lambda row: (-float(row["si_snr"]), int(row["epoch"])))
    return ordered[:keep]
