"""Drift monitoring: the failure modes from CLINICAL_VALIDATION.md, instrumented.

The validation document lists seven ways this model degrades in production and
describes how each would be detected. Writing them down is the easy half. This
file implements the detection, and the design principle is:

    MONITOR THE INPUTS, NOT JUST THE OUTPUTS.

Outcome-based monitoring is too slow to be a safety control. The 30-day
readmission label is not observable for 30 days, and claims runout means it is
not RELIABLY observable for 90 more. A model that breaks in January is caught
by outcome monitoring in April. Feature drift is visible the same week, which
is the difference between a near-miss and a quarter of bad worklists.

THE DETECTORS LIVE IN src/monitor.py; THIS FILE IS THE DASHBOARD.
That split exists because the same code has to serve two callers with different
needs: a human reading a table, and an alerting path that has to return a
verdict. They previously had separate PSI implementations and only one of them
had the zero-inflation fix, which is exactly the bug that split invites.

WHAT IS MEASURED
----------------
1. FEATURE DRIFT (PSI) against the training reference.
2. CLAIM-RUNOUT DRIFT, both as a per-quarter trend and as a verdict.
3. CALIBRATION DRIFT (observed/expected), overall and for the coverage-gap
   stratum the validation document pre-specified as a concern.
4. COHORT DRIFT -- the waterfall itself, because a contract change that alters
   who is admitted moves the denominator without moving a model metric.

AND ONE INJECTED FAILURE. Panels 1-4 on this data are green by construction:
the generator uses one lag distribution throughout. A dashboard that has only
ever shown green is not evidence the monitor works, it is evidence nothing has
gone wrong yet, and those are different claims. So a +21-day clearinghouse
shift is injected and the detectors are made to catch it. That experiment
produced the most useful finding in this file -- see panel 2.

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
import monitor as M
import registry
from generate import WINDOW_END
from train import load

OUT = "out"
CUT = pd.Timestamp("2024-05-01")
MODEL_NAME = "readmission-30d"

WATCHED = ["charlson", "ed_visits_90d", "ed_visits_365d", "ip_admits_365d",
           "ip_days_365d", "office_visits_90d", "distinct_providers_180d",
           "rx_fills_180d", "los", "age", "paid_amount_365d",
           "elig_gap_days_365d"]


def main(datadir="data"):
    os.makedirs(OUT, exist_ok=True)
    med, rx, el, mem, st = load(datadir)
    cohort, counts = F.build_cohort(st, el, WINDOW_END)
    med_store, rx_store = F.pack_medical(med), F.pack_pharmacy(rx)
    feat = F.build_features(cohort, med_store, rx_store, el, mem,
                            visibility="received")

    reference = feat[feat.discharge_date < CUT]
    current = feat[feat.discharge_date >= CUT]

    print("=" * 78)
    print("DRIFT MONITORING")
    print("=" * 78)
    print(f"  reference (training) period : {len(reference):,} discharges "
          f"before {CUT.date()}")
    print(f"  current period              : {len(current):,} discharges after")

    # ---- 1. feature drift -------------------------------------------------
    numeric = [c for c in feat.columns
               if feat[c].dtype.kind in "if" and c not in ("y",)]
    drift = M.feature_drift(reference, current, numeric)
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
    print("\n  Inputs rather than outputs, because the outcome arrives 30+ days")
    print("  late and -- once the programme runs -- is contaminated by the")
    print("  intervention itself. Inputs are observable immediately.")

    # ---- 2. runout, as a trend and as an injected failure ------------------
    print("\n" + "-" * 78)
    print("2. CLAIM-RUNOUT DRIFT -- the canary for this whole project")
    print("-" * 78)
    lag_table = M.runout_lag_by_period(med)
    print("  median service-to-received lag, days:")
    print(lag_table.round(1).to_string().replace("\n", "\n  "))
    spread = lag_table.max() - lag_table.min()
    print("\n  quarter-to-quarter spread per claim type:")
    for col in lag_table.columns:
        print(f"    {col:<6}{spread[col]:>6.1f} days")
    print("\n  Stable by construction -- the generator uses one lag")
    print("  distribution throughout. Which makes this panel unfalsifiable as")
    print("  it stands, so:")

    lag = (med["received_date"] - med["service_date"]).dt.days
    ref_lag = lag[med["service_date"] < CUT].values
    cur_lag = lag[med["service_date"] >= CUT].values
    stable = M.runout_drift(ref_lag, cur_lag)
    shocked = M.runout_drift(ref_lag, cur_lag + 21)
    print("\n  INJECTED: a clearinghouse change adding 21 days of receipt lag")
    print(f"    no change   median lag {stable['reference_median_lag']:.0f}d -> "
          f"{stable['current_median_lag']:.0f}d   PSI {stable['psi']:.4f}   "
          f"{stable['verdict']}")
    print(f"    +21d shock  median lag {shocked['reference_median_lag']:.0f}d -> "
          f"{shocked['current_median_lag']:.0f}d   PSI {shocked['psi']:.4f}")
    print(f"                {shocked['verdict']}")

    shocked_med = med.copy()
    mask = shocked_med["service_date"] >= CUT
    shocked_med.loc[mask, "received_date"] = (
        shocked_med.loc[mask, "received_date"] + pd.Timedelta(21, unit="D"))
    shocked_feat = F.build_features(
        cohort[cohort.discharge_date >= CUT], F.pack_medical(shocked_med),
        rx_store, el, mem, visibility="received")

    print("\n  what that does to the FEATURES, which is the part that matters:")
    print(f"    {'feature':<28}{'before':>10}{'after':>10}{'PSI':>9}")
    feature_psis = {}
    for c in ("ed_visits_90d", "office_visits_90d", "ip_admits_365d",
              "ip_days_365d"):
        before, after = current[c].mean(), shocked_feat[c].mean()
        feature_psis[c] = M.psi(current[c].values, shocked_feat[c].values)
        print(f"    {c:<28}{before:>10.2f}{after:>10.2f}"
              f"{feature_psis[c]:>9.4f}")

    worst = max(feature_psis.values())
    print("\n  THE FINDING. The feature names are unchanged, the pipeline does")
    print("  not error, and THE PER-FEATURE PSI DOES NOT CATCH IT EITHER:")
    print(f"  the worst of those is {worst:.4f}, roughly {0.25 / worst:.0f}x below the")
    print("  0.25 investigate threshold and below the 0.10 watch threshold too.")
    print("  Panel 1 would have shown all green through a train/serve skew")
    print("  severe enough to change what the recency features MEAN.")
    print("\n  That is the argument for monitoring the mechanism and not only")
    print("  the symptom. Panel 1 asks 'did the numbers move'. This panel asks")
    print(f"  'did the thing generating them change', and answers "
          f"{shocked['psi'] / worst:.0f}x louder:")
    print(f"  PSI {shocked['psi']:.2f} on the lag distribution itself against "
          f"{worst:.4f} on the")
    print("  features it distorts. Measured here, not assumed.")

    # ---- 3. calibration ---------------------------------------------------
    print("\n" + "-" * 78)
    print("3. CALIBRATION DRIFT (observed / expected)")
    print("-" * 78)
    try:
        servable = registry.load(MODEL_NAME, 1)
    except FileNotFoundError:
        servable = None
        print("  no registered model -- run `python train.py` first")
    if servable is not None:
        p = servable.predict_proba(F.design_matrix(current,
                                                   columns=servable.columns))
        cal = M.calibration_drift(current.y.values, p)
        print(f"  observed {cal['observed_rate']:.1%}   predicted "
              f"{cal['predicted_rate']:.1%}   O/E {cal['oe_ratio']:.3f}   "
              f"{cal['direction']}")
        print("\n  O/E rather than a calibration slope, because the operational")
        print("  question is whether the team will be staffed for the right")
        print("  NUMBER of events -- a question about the total, not the spread.")
        print("\n  the stratum CLINICAL_VALIDATION.md pre-specified:")
        for label, mask in [("coverage gap", current.any_elig_gap_365d == 1),
                            ("no coverage gap", current.any_elig_gap_365d == 0)]:
            sub = current[mask]
            if len(sub) < 40:
                print(f"    {label:<18}n={len(sub)}  too few to report")
                continue
            ps = servable.predict_proba(
                F.design_matrix(sub, columns=servable.columns))
            c = M.calibration_drift(sub.y.values, ps)
            print(f"    {label:<18}n={len(sub):<6} O/E {c['oe_ratio']:.2f}  "
                  f"{c['direction']}")
        print("  Pre-specified is the operative word: it was named as a concern")
        print("  before this run, so reporting it is not a subgroup hunt.")

    # ---- 4. cohort --------------------------------------------------------
    print("\n" + "-" * 78)
    print("4. COHORT DRIFT -- monitor the waterfall, not just the model")
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

    # the waterfall comparison, with a simulated enrolment-rule change
    # Rebuilt from the reference RETENTIONS rather than by scaling each stage
    # independently, so the simulated waterfall is internally consistent. The
    # first version scaled stages separately, which produced a stage retaining
    # 117% of what it received and a confident write-up of that as a finding.
    ref_counts = dict(counts)
    stage = "after continuous enrolment (1 gap <=45d)"
    cur_counts, prev_ref, prev_cur = {}, None, None
    for name, n in ref_counts.items():
        if prev_ref is None:
            cur_counts[name] = n
        else:
            ret = n / prev_ref
            if name == stage:
                ret *= 0.845           # the simulated enrolment-rule change
            cur_counts[name] = int(round(prev_cur * ret))
        prev_ref, prev_cur = n, cur_counts[name]
    stages = M.cohort_drift(ref_counts, cur_counts)
    print("\n  stage-by-stage retention, against a simulated enrolment change:")
    print(f"  {'stage':<48}{'ref':>8}{'now':>8}{'change':>9}")
    for d in stages:
        if "reference_retention" not in d:
            continue
        mark = "  <-- moved" if abs(d["retention_change"]) > 0.05 else ""
        print(f"  {d['stage'][:46]:<48}{d['reference_retention']:>8.1%}"
              f"{d['current_retention']:>8.1%}{d['retention_change']:>+9.1%}{mark}")
    print("\n  A stage whose retention moves is a POPULATION change. The final")
    print("  rate moving with no stage moving is a MODEL change. Those need")
    print("  different people to look at them, which is the reason to monitor")
    print("  the waterfall rather than only the rate it produces.")
    print("\n  Only the enrolment stage moves here, and that is the point: the")
    print("  simulated change is a rule change at ONE stage, so exactly one stage")
    print("  should flag. An earlier version of this simulation scaled each stage")
    print("  independently, produced a stage retaining 117% of what it received,")
    print("  and got that written up as a real conditional effect. cohort_drift()")
    print("  now refuses any retention above 100% outright -- monitoring code needs")
    print("  its own sanity checks, because a plausible number from an impossible")
    print("  computation is the easiest thing in this project to believe.")

    # ---- alerts -----------------------------------------------------------
    report = M.run_all(reference, shocked_feat,
                       y_current=current.y.values,
                       p_current=p if servable is not None else None,
                       reference_counts=ref_counts, current_counts=cur_counts,
                       reference_lag=ref_lag, current_lag=cur_lag + 21,
                       feature_subset=WATCHED)
    print("\n" + "=" * 78)
    print(f"ALERTS ({len(report['alerts'])})")
    print("=" * 78)
    for a in report["alerts"]:
        print(f"  ! {a}")
    if not report["alerts"]:
        print("  none")

    # ---- what this cannot see ---------------------------------------------
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
               "runout_lag_by_quarter": json.loads(lag_table.to_json()),
               "cohort_by_quarter": json.loads(per_q.to_json()),
               "injected_shock": {"runout": shocked,
                                  "feature_psi": feature_psis},
               "alerts": report["alerts"]}
    with open(f"{OUT}/drift.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/drift.json")
    return payload


if __name__ == "__main__":
    main()
