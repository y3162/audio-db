"""Decision rules for the SEMamba++ and MP-SENet numerics gates.

Thresholds are the existing compare_semamba_step / compare_mp_senet allowances.
They were written down before the measurement in this session and are not
fitted to those results. A strict native 3-step failure stays a failure even
when the compatibility gate is GO.
"""

from __future__ import annotations

import statistics
from pathlib import Path

LOSS_ATOL = 1e-5
GRAD_ATOL = 1e-4
COSINE_MIN = 0.999
WEIGHT_ATOL = 1e-4
TINY_GRAD = 1e-6
# MP-SENet's published allowance for a near-zero AdamW step is O(lr).
MP_TINY_WEIGHT_ATOL = 5e-4
FORMULA_ATOL = 1e-4
OPTIM_STATE_ATOL = 1e-4
SEEDS = (101, 202, 303)
STEPS = 3
SYSTEMATIC_RATIO = 10.0
SYSTEMATIC_FLOOR = 1e-6
SGD_MOMENTUM = 0.0
# torch.optim.SGD couples weight decay into the gradient. The control uses
# zero decay so the weight gap is lr times the gradient gap.
SGD_WEIGHT_DECAY = 0.0
SGD_LRS = {
    "semamba": {"matched": 2e-4, "magnified": 1e-2},
    "mp": {"matched": 5e-4, "magnified": 1e-2},
}
DISTRIBUTION_METRICS = (
    "generator_grad_max_abs",
    "generator_grad_real_max_abs",
    "weight_real_max_abs",
    "loss_generator_gap",
    "wave_max_abs",
    "optimizer_max_abs",
    "discriminator_grad_max_abs",
)
ROOT = Path(__file__).resolve().parents[3]
SEMAMBA_TRAIN = ROOT / "src" / "se" / "official" / "semambapp" / "train.py"
MP_TRAIN = ROOT / "src" / "se" / "official" / "mp-senet" / "train.py"
SEMAMBA_ORDER = (
    "optim_d.zero_grad()",
    "loss_disc_all.backward()",
    "optim_d.step()",
    "optim_g.zero_grad()",
    "loss_gen_all.backward()",
    "optim_g.step()",
)
MP_ORDER = SEMAMBA_ORDER


def update_order(text: str, markers: tuple[str, ...]) -> dict:
    """Return whether the first occurrence of each marker is in order."""

    positions = []
    cursor = 0
    for marker in markers:
        found = text.find(marker, cursor)
        if found < 0:
            return {"ok": False, "missing": marker, "positions": positions}
        positions.append(found)
        cursor = found + 1
    return {"ok": True, "missing": None, "positions": positions}


def source_inventory(text: str, markers: tuple[str, ...]) -> dict:
    order = update_order(text, markers)
    return {
        "update_order_ok": bool(order["ok"]),
        "missing_marker": order["missing"],
        "mentions_clip_grad": "clip_grad" in text,
        "mentions_grad_scaler": "GradScaler" in text,
        "mentions_autocast": "autocast" in text,
        "sets_cudnn_benchmark_true": "torch.backends.cudnn.benchmark = True" in text,
        "adamw_calls": text.count("torch.optim.AdamW("),
        "passes_weight_decay_keyword": "weight_decay" in text,
    }


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def systematic_oa(self_values: list[float], oa_values: list[float]) -> dict:
    """OA is a porting gap when its median clears every self-rerun.

    Same order of magnitude is not enough to call the gap harmless, and one
    maximum is not enough to call it a porting bug. The self envelope is the
    O/O and A/A values supplied by the caller (all steps, or the final step).
    """

    if not self_values or not oa_values:
        return {"status": "UNKNOWN", "reason": "missing values", "median_self": None, "median_oa": None, "max_self": None}
    med_self = median(self_values)
    med_oa = median(oa_values)
    max_self = max(self_values)
    triggered = med_oa > max_self and med_oa > SYSTEMATIC_RATIO * max(med_self, SYSTEMATIC_FLOOR) and med_oa > SYSTEMATIC_FLOOR
    return {
        "status": "SYSTEMATIC" if triggered else "WITHIN_SELF_ENVELOPE",
        "median_self": med_self,
        "median_oa": med_oa,
        "max_self": max_self,
        "max_oa": max(oa_values),
        "min_oa": min(oa_values),
        "count_self": len(self_values),
        "count_oa": len(oa_values),
    }


def _metric(rows: list[dict], key: str) -> list[float]:
    found = []
    for row in rows:
        if key not in row or row[key] is None:
            continue
        value = float(row[key])
        if value == value and value not in (float("inf"), float("-inf")):
            found.append(value)
    return found


def strict_reasons(model: str, row: dict) -> list[str]:
    """Existing absolute allowances. Tiny-grad SEMamba weights are exempt."""

    reasons = []
    if row.get("finite") is False:
        reasons.append("nonfinite")
    for name, gap in (row.get("loss_gap") or {}).items():
        if float(gap) > LOSS_ATOL:
            reasons.append(f"loss {name} {float(gap):.3e}")
    grad = row.get("generator_grad_max_abs")
    if grad is not None and float(grad) > GRAD_ATOL:
        reasons.append(f"grad {float(grad):.3e}")
    cosine = row.get("large_cosine_min")
    if cosine is not None and float(cosine) < COSINE_MIN:
        reasons.append(f"cosine {float(cosine):.6f}")
    weight = row.get("weight_real_max_abs")
    if weight is not None and float(weight) > WEIGHT_ATOL:
        reasons.append(f"weight_real {float(weight):.3e}")
    if model == "mp":
        tiny = row.get("weight_tiny_max_abs")
        if tiny is not None and float(tiny) > MP_TINY_WEIGHT_ATOL:
            reasons.append(f"weight_tiny {float(tiny):.3e}")
    return reasons


def strict_native_label(model: str, oa_rows_by_seed: dict[str, list[dict]]) -> dict:
    failures = []
    checked = 0
    for seed, rows in oa_rows_by_seed.items():
        if len(rows) < STEPS:
            failures.append(f"seed {seed} has {len(rows)} steps")
            continue
        for row in rows:
            checked += 1
            for reason in strict_reasons(model, row):
                failures.append(f"seed {seed} step {row.get('step')}: {reason}")
    if len(oa_rows_by_seed) < len(SEEDS):
        failures.append(f"seeds {sorted(oa_rows_by_seed)} incomplete")
    return {
        "label": "FAIL" if failures else "PASS",
        "checked_steps": checked,
        "failures": failures[:24],
        "failure_count": len(failures),
    }


def _rows(payload: dict, seed: str, optim: str, kind: str) -> list[dict]:
    return list(((payload.get(seed) or {}).get(optim) or {}).get(kind) or [])


def _collect(payload: dict, optim: str, kind: str, key: str, *, final_only: bool) -> list[float]:
    values: list[float] = []
    for seed in payload:
        rows = _rows(payload, seed, optim, kind)
        if final_only:
            rows = rows[-1:]
        values.extend(_metric(rows, key))
    return values


def distribution_gate(payload: dict) -> dict:
    per_metric = {}
    systematic = []
    for key in DISTRIBUTION_METRICS:
        pooled_self = _collect(payload, "adamw", "oo", key, final_only=False) + _collect(payload, "adamw", "aa", key, final_only=False)
        pooled_oa = _collect(payload, "adamw", "oa", key, final_only=False)
        final_self = _collect(payload, "adamw", "oo", key, final_only=True) + _collect(payload, "adamw", "aa", key, final_only=True)
        final_oa = _collect(payload, "adamw", "oa", key, final_only=True)
        pooled = systematic_oa(pooled_self, pooled_oa)
        final = systematic_oa(final_self, final_oa)
        per_metric[key] = {"pooled": pooled, "final_step": final}
        if pooled["status"] == "SYSTEMATIC" or final["status"] == "SYSTEMATIC":
            systematic.append(key)
        if pooled["status"] == "UNKNOWN" or final["status"] == "UNKNOWN":
            systematic.append(f"unknown:{key}")
    if any(item.startswith("unknown:") for item in systematic):
        status = "UNKNOWN"
    elif systematic:
        status = "FAIL"
    else:
        status = "GO"
    return {"status": status, "systematic_metrics": systematic, "metrics": per_metric}


def _formula_rows(payload: dict, kind: str) -> list[float]:
    values = []
    for seed in payload:
        rows = _rows(payload, seed, "adamw", kind)
        if not rows:
            continue
        value = rows[0].get("formula_residual_max_abs")
        if value is not None:
            values.append(float(value))
    return values


def same_state_gate(model: str, payload: dict) -> dict:
    """Resynced O/A must sit inside the existing absolute allowances."""

    failures = []
    checked = 0
    residuals = _formula_rows(payload, "oa")
    for seed in payload:
        rows = _rows(payload, seed, "adamw", "oa_resync")
        if len(rows) < STEPS:
            failures.append(f"seed {seed} resync incomplete")
            continue
        for row in rows:
            checked += 1
            failures.extend(f"seed {seed} step {row.get('step')}: {reason}" for reason in strict_reasons(model, row))
    formula_ok = bool(residuals) and max(residuals) <= FORMULA_ATOL
    if not residuals:
        failures.append("missing first-step AdamW residual")
    elif not formula_ok:
        failures.append(f"formula residual {max(residuals):.3e}")
    return {
        "status": "FAIL" if failures else "GO",
        "checked_steps": checked,
        "formula_residual_max": None if not residuals else max(residuals),
        "failures": failures[:24],
        "failure_count": len(failures),
    }


def sgd_gate(payload: dict) -> dict:
    """SGD support is required and is not itself a compatibility GO."""

    grad_self = _collect(payload, "sgd_matched", "oo", "generator_grad_real_max_abs", final_only=False)
    grad_self += _collect(payload, "sgd_matched", "aa", "generator_grad_real_max_abs", final_only=False)
    grad_oa = _collect(payload, "sgd_matched", "oa", "generator_grad_real_max_abs", final_only=False)
    weight_oa = _collect(payload, "adamw", "oa", "weight_real_max_abs", final_only=False)
    sgd_weight = _collect(payload, "sgd_matched", "oa", "weight_real_max_abs", final_only=False)
    grad = systematic_oa(grad_self, grad_oa)
    if grad["status"] == "UNKNOWN" or not weight_oa or not sgd_weight:
        return {"status": "UNKNOWN", "grad": grad, "reason": "incomplete sgd_matched"}
    grad_median = float(grad["median_oa"])
    if grad["status"] == "SYSTEMATIC" and grad_median > GRAD_ATOL:
        return {"status": "FAIL", "grad": grad, "reason": "sgd gradient gap is systematic"}
    adam = median(weight_oa)
    sgd = median(sgd_weight)
    shrunk = adam > WEIGHT_ATOL and sgd <= max(WEIGHT_ATOL, adam / SYSTEMATIC_RATIO)
    both_small = adam <= WEIGHT_ATOL and sgd <= WEIGHT_ATOL and grad["status"] != "SYSTEMATIC"
    if shrunk or both_small:
        reason = "adamw weight gap shrinks under sgd" if shrunk else "both weight gaps are inside the weight atol"
        return {"status": "GO", "grad": grad, "median_adamw_weight_real": adam, "median_sgd_weight_real": sgd, "reason": reason}
    if grad["status"] == "SYSTEMATIC":
        return {"status": "FAIL", "grad": grad, "reason": "sgd gradient gap is systematic"}
    return {
        "status": "CONDITIONAL",
        "grad": grad,
        "median_adamw_weight_real": adam,
        "median_sgd_weight_real": sgd,
        "reason": "sgd did not separate an AdamW weight gap",
    }


def trainer_status(
    *,
    run_called: bool,
    logical_steps: int,
    scheduler_steps: int,
    native_failure_count: int,
    tight_failure_count: int | None,
    graph_retained: bool = True,
) -> str:
    """Trainer.run must be invoked. Native same-state uses the absolute allowances.

    Tight flags are a control. They can explain a native miss as nondeterminism,
    which is CONDITIONAL, and they cannot turn that miss into GO. Accumulation 1
    has to keep the generator graph, matching official train.py.
    """

    if not run_called or logical_steps != STEPS:
        return "UNKNOWN"
    if scheduler_steps != 0 or not graph_retained:
        return "FAIL"
    if native_failure_count == 0:
        return "GO"
    if tight_failure_count == 0:
        return "CONDITIONAL"
    return "FAIL"


def combine_gates(gates: dict[str, str]) -> str:
    """GO only when every required gate is GO. Any FAIL stays FAIL."""

    values = [gates[name] for name in ("init_and_same_state", "distribution", "sgd", "trainer", "regression")]
    if any(value == "FAIL" for value in values):
        return "FAIL"
    if all(value == "GO" for value in values):
        return "GO"
    return "CONDITIONAL GO"


def evaluate_campaign(model: str, payload: dict) -> dict:
    oa = {seed: _rows(payload, seed, "adamw", "oa") for seed in payload}
    strict = strict_native_label(model, oa)
    same = same_state_gate(model, payload)
    dist = distribution_gate(payload)
    sgd = sgd_gate(payload)
    init_ok = all(bool((payload.get(seed) or {}).get("init_ok")) for seed in payload) and len(payload) >= len(SEEDS)
    if not init_ok:
        same = {**same, "status": "FAIL" if same["status"] == "FAIL" else "UNKNOWN", "init_ok": False}
    elif same["status"] == "GO":
        same = {**same, "init_ok": True}
    return {
        "model": model,
        "STRICT_NATIVE_3STEP_ABSOLUTE_THRESHOLD": strict["label"],
        "strict": strict,
        "init_and_same_state": same,
        "distribution": dist,
        "sgd": sgd,
        "gates_without_trainer": {
            "init_and_same_state": "FAIL" if not init_ok else same["status"],
            "distribution": dist["status"],
            "sgd": sgd["status"],
        },
    }
