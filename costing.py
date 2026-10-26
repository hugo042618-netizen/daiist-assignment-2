"""
Cost model shared by train.py and app.py.

One camera snapshot is one case. The model gives each snapshot a violation
score: the highest confidence of any bare-head box (class `head`) it finds. Two
thresholds turn that score into one of three outcomes:

    score <  t_low             -> PASS    (no action)
    t_low <= score < t_high    -> REVIEW  (the monitoring operator looks at the snapshot)
    score >= t_high            -> FAIL    (alert: the supervisor goes to stop the worker)

With t_low == t_high there is no review band and the model decides alone.

The rates (what share of violation and of compliant snapshots end up in each
outcome) come from the evaluated split. The mix of violation and compliant
snapshots comes from the production base rate, not from the split, because the
dataset shows bare heads far more often than a real site camera would. Keeping
both in one place means the app and the training pipeline can never disagree
about what a threshold costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

POSITIVE_LABEL = "violation"   # at least one bare head in the snapshot
NEGATIVE_LABEL = "compliant"   # every head in the snapshot wears a helmet


@dataclass
class Costs:
    """Euro cost per snapshot of each outcome. REPORT.md defends every figure."""

    fn: float = 50.0          # bare head missed: expected fine plus expected injury cost
    fp: float = 4.0           # supervisor walks out to a worker who is wearing a helmet
    tp: float = 4.0           # supervisor walks out to a bare-headed worker: the same walk, now useful
    tn: float = 0.0           # compliant snapshot passed: nothing happens
    hourly_cost: float = 20.0  # loaded cost of one hour of monitoring operator time
    review_seconds: float = 10.0  # operator looks at one snapshot
    base_rate: float = 0.05   # share of snapshots that really show a bare head
    cases_per_day: int = 6000  # snapshots per day
    reviewer_hours_per_day: float = 6.5  # productive hours per operator shift

    @property
    def review(self) -> float:
        """Euro cost of one snapshot review."""
        return self.hourly_cost * self.review_seconds / 3600

    def to_dict(self) -> dict:
        return {**asdict(self), "review": self.review}

    @classmethod
    def from_dict(cls, data: dict) -> "Costs":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


def outcome_rates(scores: np.ndarray, labels: np.ndarray, t_low: float, t_high: float) -> dict:
    """Share of violation (pos) and compliant (neg) snapshots sent to each outcome."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    t_high = max(t_high, t_low)
    fail = scores >= t_high
    passed = scores < t_low
    review = ~fail & ~passed
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = max(pos.sum(), 1), max(neg.sum(), 1)
    return {
        "pos_pass": (passed & pos).sum() / n_pos,
        "pos_review": (review & pos).sum() / n_pos,
        "pos_fail": (fail & pos).sum() / n_pos,
        "neg_pass": (passed & neg).sum() / n_neg,
        "neg_review": (review & neg).sum() / n_neg,
        "neg_fail": (fail & neg).sum() / n_neg,
    }


def outcome_counts(scores: np.ndarray, labels: np.ndarray, t_low: float, t_high: float) -> pd.DataFrame:
    """The three-outcome confusion matrix as counts on the evaluated split."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    t_high = max(t_high, t_low)
    outcome = np.where(scores >= t_high, "FAIL", np.where(scores < t_low, "PASS", "REVIEW"))
    table = pd.crosstab(
        pd.Series(np.where(labels == 1, POSITIVE_LABEL, NEGATIVE_LABEL), name="actual"),
        pd.Series(outcome, name="decision"),
    )
    return table.reindex(index=[POSITIVE_LABEL, NEGATIVE_LABEL], columns=["PASS", "REVIEW", "FAIL"], fill_value=0)


def cost_per_case(rates: dict, c: Costs) -> float:
    """Expected euro cost of one snapshot at the production base rate."""
    pos = (
        rates["pos_pass"] * c.fn
        + rates["pos_fail"] * c.tp
        + rates["pos_review"] * (c.review + c.tp)  # the operator spots it, then the supervisor goes out
    )
    neg = (
        rates["neg_fail"] * c.fp
        + rates["neg_pass"] * c.tn
        + rates["neg_review"] * (c.review + c.tn)  # the operator clears it
    )
    return c.base_rate * pos + (1 - c.base_rate) * neg


def review_share(rates: dict, c: Costs) -> float:
    """Share of all production snapshots that land in the review band."""
    return c.base_rate * rates["pos_review"] + (1 - c.base_rate) * rates["neg_review"]


def staffing(share: float, c: Costs) -> dict:
    """What a review share means for monitoring operators."""
    per_day = share * c.cases_per_day
    hours = per_day * c.review_seconds / 3600
    return {
        "reviews_per_day": per_day,
        "review_hours_per_day": hours,
        "reviewers_needed": hours / c.reviewer_hours_per_day,
    }


def evaluate_band(scores, labels, t_low: float, t_high: float, c: Costs) -> dict:
    """Everything the report and the dashboard show for one pair of thresholds."""
    rates = outcome_rates(scores, labels, t_low, t_high)
    share = review_share(rates, c)
    out = {
        "t_low": float(t_low),
        "t_high": float(max(t_high, t_low)),
        "cost_per_case": cost_per_case(rates, c),
        "cost_per_1000": 1000 * cost_per_case(rates, c),
        "review_share": share,
        "recall_violation": rates["pos_fail"] + rates["pos_review"],  # caught by model or operator
        "false_alert_rate": rates["neg_fail"],
        **{f"rate_{k}": float(v) for k, v in rates.items()},
        **staffing(share, c),
    }
    return out


def threshold_sweep(scores, labels, c: Costs, thresholds=None) -> pd.DataFrame:
    """Single threshold (model decides alone): cost per 1,000 snapshots at each threshold."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rows = []
    for t in thresholds:
        fail = scores >= t
        rows.append({
            "threshold": float(t),
            "tp": int((fail & (labels == 1)).sum()),
            "fn": int((~fail & (labels == 1)).sum()),
            "fp": int((fail & (labels == 0)).sum()),
            "tn": int((~fail & (labels == 0)).sum()),
            "cost_per_1000": evaluate_band(scores, labels, t, t, c)["cost_per_1000"],
        })
    return pd.DataFrame(rows)


def best_single_threshold(sweep: pd.DataFrame) -> float:
    """Lowest cost; ties go to the highest threshold (fewer supervisors sent out for nothing)."""
    best = sweep["cost_per_1000"].min()
    tied = sweep.loc[np.isclose(sweep["cost_per_1000"], best), "threshold"]
    return float(tied.max())


# Most compliant snapshots score below 0.01, so a few finer steps there let the review
# band grow in small pieces instead of jumping from "review 30%" to "review everything".
LOW_THRESHOLDS = (0.001, 0.002, 0.003, 0.005, 0.0075)


def threshold_grid(step: float = 0.01) -> np.ndarray:
    return np.unique(np.round(np.concatenate([LOW_THRESHOLDS, np.arange(0.0, 1.0001, step)]), 4))


def band_search(scores, labels, c: Costs, max_reviewers: float | None = None, step: float = 0.01) -> dict:
    """Cheapest (t_low, t_high) pair, optionally capped by how many operators exist."""
    grid = threshold_grid(step)
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            result = evaluate_band(scores, labels, lo, hi, c)
            if max_reviewers is not None and result["reviewers_needed"] > max_reviewers + 1e-9:
                continue
            key = (round(result["cost_per_1000"], 6), result["review_share"], -hi)
            if best is None or key < best[0]:
                best = (key, result)
    return best[1]


def alternatives(scores, labels, c: Costs, t_single: float, t_low: float, t_high: float) -> pd.DataFrame:
    """The chosen design priced against the three policies the brief asks about."""
    n_rev_all = staffing(1.0, c)
    rows = [
        {
            "policy": "Do nothing (no camera monitoring)",
            "cost_per_1000": 1000 * c.base_rate * c.fn + 1000 * (1 - c.base_rate) * c.tn,
            "review_share": 0.0,
            "reviewers_needed": 0.0,
        },
        {
            "policy": "Operator checks every snapshot",
            "cost_per_1000": 1000 * (c.review + c.base_rate * c.tp + (1 - c.base_rate) * c.tn),
            "review_share": 1.0,
            "reviewers_needed": n_rev_all["reviewers_needed"],
        },
    ]
    alone = evaluate_band(scores, labels, t_single, t_single, c)
    rows.append({
        "policy": f"Model decides alone (threshold {t_single:.3g})",
        "cost_per_1000": alone["cost_per_1000"],
        "review_share": 0.0,
        "reviewers_needed": 0.0,
    })
    band = evaluate_band(scores, labels, t_low, t_high, c)
    rows.append({
        "policy": f"Model + review band ({t_low:.3g} to {max(t_high, t_low):.3g})",
        "cost_per_1000": band["cost_per_1000"],
        "review_share": band["review_share"],
        "reviewers_needed": band["reviewers_needed"],
    })
    out = pd.DataFrame(rows)
    out["cost_per_day"] = out["cost_per_1000"] * c.cases_per_day / 1000
    return out
