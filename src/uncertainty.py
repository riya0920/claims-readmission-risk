"""Confidence intervals, and a paired test for "is this model actually better?".

WHY THIS FILE EXISTS
--------------------
The first build of this project reported AUROC 0.6519 for logistic and 0.6467
for gradient boosting, said the difference was "inside the noise", and did not
quantify the noise. That is a hedge, not a result. A screener is entitled to
ask "how do you know?" and the honest answer has to be a number.

Two distinct questions, and they need different machinery:

  1. HOW PRECISE IS THIS ESTIMATE?  -> bootstrap percentile interval.
  2. IS MODEL A BETTER THAN MODEL B? -> PAIRED bootstrap on the DIFFERENCE.

Question 2 is not answered by looking at whether the two intervals from
question 1 overlap. Overlapping intervals are compatible with a highly
significant difference, because the two models are scored on the SAME patients
and their errors are correlated -- when a hard case appears in a resample, both
models do badly on it, so the difference is far more stable than either level.
Comparing intervals instead of bootstrapping the difference is one of the most
common statistical errors in ML write-ups and it systematically UNDERSTATES
evidence for a difference.

CLUSTERING
----------
Resampling is by MEMBER, not by stay. A member can contribute several index
admissions, those admissions are correlated (same person, same comorbidities,
same care patterns), and resampling rows independently would treat them as
independent evidence and produce intervals that are too narrow. Cluster
bootstrap resamples whole members with replacement.

WHAT THIS DOES NOT FIX
----------------------
A confidence interval describes sampling variability under the assumption that
the data-generating process is stable and the model is fixed. It says nothing
about the model being wrong for structural reasons -- the leakage, drift and
population-shift failures in docs/CLINICAL_VALIDATION.md are all invisible to
it. A tight interval on a biased estimate is a precise wrong answer.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _capacity_sensitivity(y, p, pct):
    n = len(y)
    k = max(1, int(round(n * pct)))
    top = np.argsort(-p)[:k]
    pos = y.sum()
    return (y[top].sum() / pos) if pos else np.nan


def _capacity_ppv(y, p, pct):
    n = len(y)
    k = max(1, int(round(n * pct)))
    top = np.argsort(-p)[:k]
    return y[top].sum() / k


METRICS = {
    "auroc": lambda y, p: roc_auc_score(y, p),
    "auprc": lambda y, p: average_precision_score(y, p),
    "brier": lambda y, p: brier_score_loss(y, p),
    "sens_at_5pct": lambda y, p: _capacity_sensitivity(y, p, 0.05),
    "ppv_at_5pct": lambda y, p: _capacity_ppv(y, p, 0.05),
}


class ClusterIndex:
    """Precomputed member -> row-indices map.

    Built once and reused across replicates. Rebuilding it inside the loop
    costs an np.unique plus n boolean scans per replicate, which dominated
    runtime entirely -- the bootstrap was slower than fitting the models.
    """

    __slots__ = ("rows_by_group", "n_groups")

    def __init__(self, groups):
        order = np.argsort(groups, kind="stable")
        sorted_groups = np.asarray(groups)[order]
        boundaries = np.flatnonzero(
            np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True])
        self.rows_by_group = [order[boundaries[i]:boundaries[i + 1]]
                              for i in range(len(boundaries) - 1)]
        self.n_groups = len(self.rows_by_group)

    def resample(self, rng):
        picked = rng.integers(0, self.n_groups, self.n_groups)
        return np.concatenate([self.rows_by_group[g] for g in picked])


def cluster_bootstrap_indices(groups, rng):
    """Resample whole members with replacement, return row indices.

    Note the consequence: the resampled dataset is not the same size as the
    original, because members contribute different numbers of stays. That is
    correct -- the unit of independence is the member.
    """
    return ClusterIndex(groups).resample(rng)


def bootstrap_ci(y, p, groups, metric="auroc", n_boot=1000, alpha=0.05, seed=0):
    """Percentile CI for one metric, clustered by member."""
    rng = np.random.default_rng(seed)
    fn = METRICS[metric]
    point = float(fn(y, p))
    index = groups if isinstance(groups, ClusterIndex) else ClusterIndex(groups)
    stats = []
    for _ in range(n_boot):
        idx = index.resample(rng)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue                     # degenerate resample, no signal to score
        try:
            stats.append(fn(yb, p[idx]))
        except ValueError:
            continue
    stats = np.array(stats, dtype=float)
    stats = stats[np.isfinite(stats)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"metric": metric, "point": point, "lo": float(lo), "hi": float(hi),
            "n_boot": len(stats), "se": float(stats.std(ddof=1))}


def paired_bootstrap_difference(y, p_a, p_b, groups, metric="auroc",
                                n_boot=1000, alpha=0.05, seed=0):
    """Is model A better than model B? Bootstrap the DIFFERENCE, not the levels.

    Both models are scored on the same resample every iteration, so the
    correlation between them is preserved and cancels out of the difference.
    Returns the interval on (A - B) and a two-sided bootstrap p-value for the
    null that the difference is zero.
    """
    rng = np.random.default_rng(seed)
    fn = METRICS[metric]
    point = float(fn(y, p_a) - fn(y, p_b))
    index = groups if isinstance(groups, ClusterIndex) else ClusterIndex(groups)
    diffs = []
    for _ in range(n_boot):
        idx = index.resample(rng)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        try:
            diffs.append(fn(yb, p_a[idx]) - fn(yb, p_b[idx]))
        except ValueError:
            continue
    diffs = np.array(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value: how often does the difference cross zero
    p_val = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"metric": metric, "difference": point, "lo": float(lo),
            "hi": float(hi), "p_value": float(min(1.0, p_val)),
            "n_boot": len(diffs),
            "significant": bool(lo > 0 or hi < 0)}


def minimum_detectable_difference(y, p, groups, metric="auroc", n_boot=500,
                                  seed=0):
    """Roughly how large a difference this test set could even detect.

    Answers the question that should precede any model comparison: is this
    evaluation capable of resolving the difference I care about? If the
    smallest detectable AUROC gap is 0.02 and the candidate models differ by
    0.005, the comparison was never going to settle anything and running it
    was theatre.

    Estimated as 2 x the bootstrap standard error of the metric, which is the
    approximate half-width needed for a difference to clear zero.
    """
    ci = bootstrap_ci(y, p, groups, metric, n_boot=n_boot, seed=seed)
    return {"metric": metric, "bootstrap_se": ci["se"],
            "approx_min_detectable": 2.0 * ci["se"]}
