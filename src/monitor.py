"""Drift monitoring: the failure modes from CLINICAL_VALIDATION.md, instrumented.

The validation document listed five ways this model degrades in production and
detection strategies for each, and implemented none of them. A failure mode
with a written detection strategy and no code is a plan, not a control.

WHAT IS MONITORED, AND WHY EACH ONE
-----------------------------------
INPUT DRIFT (PSI per feature). Monitoring output metrics alone cannot work in
    production, because the outcome arrives 30+ days late and, once the
    programme is running, is contaminated by the intervention itself. Inputs
    are observable immediately and are where the earliest signal is.

CLAIM-RUNOUT DRIFT. The specific input drift this model is most exposed to. If
    the payer changes clearinghouse and receipt lag shifts, every recent-
    utilisation feature changes distribution while the feature NAMES stay the
    same -- the silent train/serve skew that docs/LEAKAGE_AUDIT.md measures.
    Tracked separately because it is invisible in an overall PSI that averages
    across features.

CALIBRATION DRIFT (observed/expected). Discrimination can hold while
    calibration rots. A model that still ranks correctly but over-predicts by
    30% breaks capacity planning, which is what the probabilities are FOR.

COHORT DRIFT (the waterfall). A contract or network change alters the admitted
    population. The model metrics can look unchanged while the denominator has
    become a different group of people, so the waterfall counts are monitored,
    not just the rate.

SUBGROUP CALIBRATION, with the coverage-gap stratum pre-specified as a target
    because the validation doc found it miscalibrated at O/E 1.26 and said it
    should be watched.

WHAT NO AMOUNT OF MONITORING FIXES
----------------------------------
The feedback loop. Once outreach works, the training labels stop describing the
untreated population, and every metric computed from post-deployment data is
measuring a system that includes the intervention. A permanently randomised
holdout is the only answer, and it is a deployment decision rather than a
monitoring one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, bins=10):
    """Population Stability Index between a reference and a current sample.

    Conventional reading: <0.10 no meaningful shift, 0.10-0.25 moderate,
    >0.25 significant. Those thresholds are rules of thumb, not tests -- PSI
    has no null distribution and is sensitive to bin count, so a moving PSI is
    a prompt to look, never a verdict on its own.

    Bins come from the REFERENCE distribution's quantiles. Re-binning on the
    current sample every period would hide exactly the shift being looked for.

    BINNING IS THE WHOLE PROBLEM ON THIS DATA, and getting it wrong fails
    silently in the reassuring direction. Most utilisation features here are
    counts that are 85-99% zero. Quantile binning collapses on them: every
    decile edge lands on 0, the edges deduplicate to one, and the natural
    implementation returns 0.0 -- a confident "no shift" for a feature whose
    mass point could have moved from 85% to 99%. Both fallbacks below exist
    because a test caught that, not because it was anticipated.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins or len(actual) == 0:
        return float("nan")

    levels = np.unique(expected)
    if len(levels) <= bins:
        # discrete feature: use the values themselves as bins
        e_counts = np.array([(expected == v).sum() for v in levels], float)
        a_counts = np.array([(actual == v).sum() for v in levels], float)
        # values outside the reference support get their own bin, otherwise a
        # brand-new value is simply not counted
        outside = float((~np.isin(actual, levels)).sum())
        if outside:
            e_counts = np.append(e_counts, 0.0)
            a_counts = np.append(a_counts, outside)
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            # continuous-looking, but dominated by one value: compare the mass
            # point against everything else rather than reporting no shift
            point = edges[0]
            e_counts = np.array([(expected == point).sum(),
                                 (expected != point).sum()], float)
            a_counts = np.array([(actual == point).sum(),
                                 (actual != point).sum()], float)
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            e_counts, _ = np.histogram(expected, bins=edges)
            a_counts, _ = np.histogram(actual, bins=edges)

    e_frac = e_counts / max(1.0, e_counts.sum())
    a_frac = a_counts / max(1.0, a_counts.sum())
    # floor the zeros: an empty bin makes the log term infinite, and reporting
    # inf for a feature that merely has a sparse tail is a false alarm
    eps = 1e-6
    e_frac = np.clip(e_frac, eps, None)
    a_frac = np.clip(a_frac, eps, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def psi_report(reference_df, current_df, columns=None, moderate=0.10,
               significant=0.25):
    """PSI for every numeric feature, worst first."""
    columns = columns or [c for c in reference_df.columns
                          if reference_df[c].dtype.kind in "if"]
    rows = []
    for c in columns:
        if c not in current_df.columns:
            rows.append({"feature": c, "psi": float("nan"),
                         "verdict": "MISSING from the current window"})
            continue
        value = psi(reference_df[c].values, current_df[c].values)
        verdict = ("no meaningful shift" if value < moderate
                   else "moderate shift" if value < significant
                   else "SIGNIFICANT shift")
        rows.append({"feature": c, "psi": value, "verdict": verdict,
                     "reference_mean": float(reference_df[c].mean()),
                     "current_mean": float(current_df[c].mean())})
    return sorted(rows, key=lambda r: -(r["psi"] if r["psi"] == r["psi"] else 0))


def calibration_drift(y, p, reference_oe=1.0):
    """Observed/expected, and the direction of the miss.

    O/E is reported rather than a calibration slope because the operational
    question is "will the team be staffed for the right number of events", and
    that is a question about the total, not the spread.
    """
    obs, exp = float(np.mean(y)), float(np.mean(p))
    oe = obs / exp if exp else float("nan")
    drift = oe - reference_oe
    return {"observed_rate": obs, "predicted_rate": exp, "oe_ratio": oe,
            "drift_vs_reference": drift,
            "direction": ("under-predicting risk" if oe > 1.05 else
                          "over-predicting risk" if oe < 0.95 else
                          "calibrated within 5%")}


def cohort_drift(reference_counts, current_counts):
    """Compare two cohort waterfalls stage by stage.

    Model metrics can be unchanged while the denominator has quietly become a
    different population, so the STAGES are compared rather than only the final
    rate. A stage whose retention moves is a population change; the final rate
    moving without any stage moving is a model change. They need different
    people to look at them.
    """
    rows = []
    ref_prev = cur_prev = None
    for stage in reference_counts:
        r, c = reference_counts[stage], current_counts.get(stage)
        if c is None:
            rows.append({"stage": stage, "verdict": "stage missing"})
            continue
        r_ret = (r / ref_prev) if ref_prev else 1.0
        c_ret = (c / cur_prev) if cur_prev else 1.0
        row = {"stage": stage, "reference": r, "current": c,
               "reference_retention": r_ret, "current_retention": c_ret,
               "retention_change": c_ret - r_ret}
        # A waterfall stage cannot retain more than it received. If it does,
        # the two count dicts were not produced by the same filter chain, and
        # every retention change downstream of here is arithmetic on
        # incompatible inputs. Caught this reporting a +18.1% "finding" that
        # was purely an artefact of an inconsistent simulation.
        if r_ret > 1.0 + 1e-9 or c_ret > 1.0 + 1e-9:
            row["verdict"] = ("INCONSISTENT WATERFALL: retention above 100%. "
                              "These counts did not come from one filter "
                              "chain; the comparison is meaningless.")
        rows.append(row)
        ref_prev, cur_prev = r, c
    return rows


def runout_drift(reference_lag_days, current_lag_days):
    """Has the receipt-lag distribution moved?

    The specific failure this model is most exposed to. A clearinghouse change
    shifts receipt lag, every recent-utilisation feature changes distribution
    under an unchanged name, and nothing else in the monitoring stack
    distinguishes that from members genuinely using less care.
    """
    ref = np.asarray(reference_lag_days, dtype=float)
    cur = np.asarray(current_lag_days, dtype=float)
    out = {
        "reference_median_lag": float(np.median(ref)),
        "current_median_lag": float(np.median(cur)),
        "reference_p90_lag": float(np.percentile(ref, 90)),
        "current_p90_lag": float(np.percentile(cur, 90)),
        "psi": psi(ref, cur),
    }
    shift = out["current_median_lag"] - out["reference_median_lag"]
    out["median_shift_days"] = shift
    out["verdict"] = (
        "receipt lag has moved; recent-utilisation features are now measuring "
        "something different from what the model was fitted on"
        if abs(shift) >= 5 else "receipt lag stable")
    return out


def run_all(reference, current, *, y_current=None, p_current=None,
            reference_counts=None, current_counts=None,
            reference_lag=None, current_lag=None, feature_subset=None):
    """One call, everything the validation document promised to watch."""
    report = {"input_drift": psi_report(reference, current, feature_subset)}
    if y_current is not None and p_current is not None:
        report["calibration"] = calibration_drift(y_current, p_current)
    if reference_counts and current_counts:
        report["cohort"] = cohort_drift(reference_counts, current_counts)
    if reference_lag is not None and current_lag is not None:
        report["runout"] = runout_drift(reference_lag, current_lag)

    alerts = []
    for r in report["input_drift"]:
        if r.get("psi", 0) == r.get("psi", 0) and r.get("psi", 0) >= 0.25:
            alerts.append(f"input drift: {r['feature']} PSI {r['psi']:.3f}")
    cal = report.get("calibration")
    if cal and abs(cal["drift_vs_reference"]) > 0.10:
        alerts.append(f"calibration: O/E {cal['oe_ratio']:.2f} "
                      f"({cal['direction']})")
    ro = report.get("runout")
    if ro and abs(ro["median_shift_days"]) >= 5:
        alerts.append(f"claim runout: median lag moved "
                      f"{ro['median_shift_days']:+.0f} days")
    for r in report.get("cohort", []):
        if "INCONSISTENT" in r.get("verdict", ""):
            alerts.append(f"cohort: {r['stage']} -- {r['verdict']}")
            continue
        if abs(r.get("retention_change", 0.0)) > 0.05:
            alerts.append(f"cohort: retention at '{r['stage']}' moved "
                          f"{r['retention_change']:+.1%} -- the denominator "
                          f"is a different population, not a worse model")
    report["alerts"] = alerts
    return report


# WHY THE RUNOUT CHECK IS NOT REDUNDANT WITH THE PSI LOOP
# ------------------------------------------------------
# Measured, not assumed. Injecting a +21-day receipt-lag shift into this
# project's own data (run_monitor.py) moves the affected features by PSI
# 0.0007 to 0.036 -- the worst of them 7x below the 0.25 "significant" rule
# of thumb, and below the 0.10 "moderate" one too. A per-feature PSI dashboard
# would have shown all green through a train/serve skew severe enough to change
# what the recent-utilisation features MEAN.
#
# That is the argument for monitoring the mechanism rather than only the
# symptom. PSI on features answers "did the numbers move"; runout_drift answers
# "did the thing that generates the numbers change". The second question has
# the earlier and much louder answer -- PSI 6.06 on the lag distribution itself
# against 0.036 on the worst of the features it distorts.


# --------------------------------------------------------------------------
# Panel helpers. These produce the tables the dashboard prints; the functions
# above produce the verdicts the alerting acts on. Both share one psi(), which
# matters -- the two used to be separate implementations and only one of them
# had the zero-inflation fix.
# --------------------------------------------------------------------------

PSI_WATCH, PSI_INVESTIGATE = 0.10, 0.25


def feature_drift(reference_df, current_df, columns):
    """Per-feature PSI with the magnitude alongside.

    The magnitude is not decoration. PSI is sample-size sensitive: at large n it
    flags differences too small to act on, so a PSI of 0.12 on a feature whose
    mean moved 2% is a statistic rather than a problem, and the table has to let
    that call be made.
    """
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


def runout_lag_by_period(medical, period_col="received_date", by="claim_type"):
    """Median service-to-received lag per quarter, per claim type.

    The panel form of runout_drift(): a trend table rather than a verdict. A
    shift here moves every recency feature without touching a line of model
    code, which is why it is the first panel to check when features move.
    """
    m = medical.copy()
    m["lag_days"] = (m[period_col] - m["service_date"]).dt.days
    m["quarter"] = m["service_date"].dt.to_period("Q").astype(str)
    return (m.groupby(["quarter", by])["lag_days"]
            .median().unstack(fill_value=np.nan))


def prediction_drift(scores_by_period):
    """Score distribution over time.

    Weakest of the panels and included with that caveat: prediction drift
    without labels cannot separate "the population changed" from "the model
    broke", and it moves for both. It is a prompt to look at the input panels,
    never a finding on its own.
    """
    return [{"period": period, "n": len(scores),
             "mean_score": float(np.mean(scores)),
             "p90": float(np.percentile(scores, 90)),
             "pct_above_0.5": float(np.mean(np.array(scores) > 0.5))}
            for period, scores in scores_by_period.items()]
