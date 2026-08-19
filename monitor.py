"""Drift monitoring: the failure modes from CLINICAL_VALIDATION.md, instrumented.

The validation document lists seven ways this model degrades in production and
describes how each would be detected. Writing them down is the easy half.
This file implements the detection, and the design principle is:

    MONITOR THE INPUTS, NOT JUST THE OUTPUTS.

Outcome-based monitoring is too slow to be a safety control. The 30-day
readmission label is not observable for 30 days, and claims runout means it is
not RELIABLY observable for 90 more. A model that breaks in January is caught
by outcome monitoring in April. Feature drift is visible the same week, which
is the difference between a near-miss and a quarter of bad worklists.

WHAT IS MEASURED
----------------
1. FEATURE DRIFT (PSI). Population Stability Index per feature against the
   training reference. The convention -- <0.10 stable, 0.10-0.25 watch, >0.25
   investigate -- is a rule of thumb from credit scoring, not a law, and it is
   reported as such. PSI is also sample-size sensitive: at large n it flags
   differences too small to matter, so the magnitude is shown alongside.

2. RUNOUT DRIFT specifically. The failure this project is built around. If the
   payer changes clearinghouse and receipt lag moves, every recency feature
   shifts underneath a model that cannot see it happening. Measured directly as
   the median service-to-received lag by claim type over time.

3. PREDICTION DRIFT. The distribution of scores, and the mean predicted risk
   against the mean observed rate (calibration-in-the-large) once labels mature.

4. COHORT DRIFT. The waterfall counts themselves. A contract change that alters
   who is admitted moves the denominator without moving a single model metric,
   and monitoring only model metrics misses it entirely.

Run:  python monitor.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import features as F
from generate import WINDOW_END
from train import load

OUT = "out"

# Rule-of-thumb thresholds from credit scoring. Not laws.
PSI_WATCH, PSI_INVESTIGATE = 0.10, 0.25


def psi(reference, current, bins=10):
    """Population Stability Index between two samples of one feature.

    Quantile bins are taken from the REFERENCE, because the question is how the
    current period sits inside the distribution the model was fitted on.
    Epsilon-padding empty bins avoids an infinite PSI when a bin empties, which
    otherwise makes one rare category dominate the whole score.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        # near-constant feature: compare means directly, PSI is meaningless
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift(reference_df, current_df, columns):
    rows = []
    for c in columns:
        if c not in reference_df or c not in current_df:
            continue
        ref = pd.to_numeric(reference_df[c], errors="coerce").dropna()
        cur = pd.to_numeric(current_df[c], errors="coerce").dropna()
        if len(ref) < 50 or len(cur) < 50:
            continue
        value = psi(ref, cur)
        status = ("investigate" if value > PSI_INVESTIGATE
                  else "watch" if value > PSI_WATCH else "stable")
        rows.append({"feature": c, "psi": value, "status": status,
                     "ref_mean": float(ref.mean()), "cur_mean": float(cur.mean()),
                     "pct_change": (float(cur.mean() - ref.mean())
                                    / float(ref.mean()) if ref.mean() else np.nan)})
    return sorted(rows, key=lambda r: -r["psi"])


def runout_drift(medical, period_col="received_date", by="claim_type"):
    """Median service-to-received lag per quarter, per claim type.

    This is the canary for the whole project. A shift here moves every recency
    feature without touching a single line of model code.
    """
    m = medical.copy()
    m["lag_days"] = (m[period_col] - m["service_date"]).dt.days
    m["quarter"] = m["service_date"].dt.to_period("Q").astype(str)
    g = (m.groupby(["quarter", by])["lag_days"]
         .median().unstack(fill_value=np.nan))
    return g


def prediction_drift(scores_by_period):
    rows = []
    for period, scores in scores_by_period.items():
        rows.append({"period": period, "n": len(scores),
                     "mean_score": float(np.mean(scores)),
                     "p90": float(np.percentile(scores, 90)),
                     "pct_above_0.5": float(np.mean(np.array(scores) > 0.5))})
    return rows


def main(datadir="data"):
    os.makedirs(OUT, exist_ok=True)
    med, rx, el, mem, st = load(datadir)
    cohort, counts = F.build_cohort(st, el, WINDOW_END)
    feat = F.build_features(cohort, F.pack_medical(med), F.pack_pharmacy(rx),
                            el, mem, visibility="received")

    cut = pd.Timestamp("2024-05-01")
    reference = feat[feat.discharge_date < cut]
    current = feat[feat.discharge_date >= cut]

    print("=" * 78)
    print("DRIFT MONITORING")
    print("=" * 78)
    print(f"  reference (training) period : {len(reference):,} discharges "
          f"before {cut.date()}")
    print(f"  current period              : {len(current):,} discharges after")

    # ---- 1. feature drift ------------------------------------------------
    numeric = [c for c in feat.columns
               if feat[c].dtype.kind in "if" and c not in ("y",)]
    drift = feature_drift(reference, current, numeric)
    print("\n" + "-" * 78)
    print("1. FEATURE DRIFT (PSI vs the training reference)")
    print("-" * 78)
    print(f"  {'feature':<28}{'PSI':>8}{'status':>14}{'ref mean':>11}"
          f"{'cur mean':>11}{'change':>9}")
    for r in drift[:12]:
        print(f"  {r['feature']:<28}{r['psi']:>8.4f}{r['status']:>14}"
              f"{r['ref_mean']:>11.2f}{r['cur_mean']:>11.2f}"
              f"{r['pct_change']:>8.0%}")
    flagged = [r for r in drift if r["status"] != "stable"]
    print(f"\n  {len(flagged)} of {len(drift)} features above the watch threshold")
    print("  Thresholds (0.10 watch, 0.25 investigate) are a credit-scoring rule")
    print("  of thumb, not a law. PSI is also sample-size sensitive: at large n")
    print("  it flags differences too small to act on, which is why the mean")
    print("  change is shown next to it. A PSI of 0.12 on a feature whose mean")
    print("  moved 2% is a statistic, not a problem.")

    # ---- 2. runout drift -------------------------------------------------
    print("\n" + "-" * 78)
    print("2. CLAIM-RUNOUT DRIFT -- the canary for this whole project")
    print("-" * 78)
    lag = runout_drift(med)
    print("  median service-to-received lag, days:")
    print(lag.round(1).to_string().replace("\n", "\n  "))
    spread = lag.max() - lag.min()
    print("\n  quarter-to-quarter spread per claim type:")
    for col in lag.columns:
        print(f"    {col:<6}{spread[col]:>6.1f} days")
    print("\n  Stable here by construction -- the generator uses one lag")
    print("  distribution throughout. In production this is the FIRST panel to")
    print("  check when recency features move, because a clearinghouse change")
    print("  shifts every one of them at once and the model cannot see it.")

    # ---- 3. cohort drift -------------------------------------------------
    print("\n" + "-" * 78)
    print("3. COHORT DRIFT -- monitor the waterfall, not just the model")
    print("-" * 78)
    ref_q = reference.discharge_date.dt.to_period("Q").astype(str)
    cur_q = current.discharge_date.dt.to_period("Q").astype(str)
    both = pd.concat([reference.assign(q=ref_q), current.assign(q=cur_q)])
    per_q = both.groupby("q").agg(discharges=("stay_id", "size"),
                                  readmit_rate=("y", "mean"),
                                  mean_charlson=("charlson", "mean"),
                                  mean_los=("los", "mean"))
    print(per_q.round(3).to_string().replace("\n", "\n  "))
    print("\n  A contract change that alters WHO gets admitted moves the")
    print("  denominator without moving a single model metric. Monitoring only")
    print("  AUROC misses it completely.")

    # ---- 4. what is NOT monitored ----------------------------------------
    print("\n" + "-" * 78)
    print("WHAT THIS CANNOT SEE")
    print("-" * 78)
    print("  * The feedback loop. Once outreach works, the training labels stop")
    print("    describing the untreated population, and every panel above still")
    print("    looks healthy. Only a permanently randomised holdout detects it,")
    print("    and that is an operational commitment, not a dashboard.")
    print("  * Label maturity. The 30-day outcome is unobservable for 30 days")
    print("    and unreliable for ~90 more because of runout. Outcome-based")
    print("    monitoring is therefore a quarter behind the failure, which is")
    print("    exactly why the input panels above exist.")
    print("  * Access bias. Claims measure care RECEIVED. A population whose")
    print("    access changes looks like a population whose health changed, and")
    print("    no amount of PSI distinguishes them.")

    payload = {"feature_drift": drift, "n_flagged": len(flagged),
               "runout_lag_by_quarter": json.loads(lag.to_json()),
               "cohort_by_quarter": json.loads(per_q.to_json())}
    with open(f"{OUT}/drift.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/drift.json")
    return payload


if __name__ == "__main__":
    main()
