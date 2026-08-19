"""The care-manager worklist: what the model actually produces.

A ranked probability is not a deliverable. A care manager opens a list, sees a
name, and needs to know in about four seconds why this member is on it and
what to ask about. So this script emits the artefact that a care-management
platform would consume: a capacity-bounded, ranked list with per-member reasons
in the language the caller speaks.

ON ATTRIBUTION -- READ THIS BEFORE BELIEVING THE "top drivers" COLUMN
--------------------------------------------------------------------
The spec asks for SHAP. SHAP is not installed in this offline environment, so
the drivers below are computed by OCCLUSION: set one feature to its cohort
median, re-score, and record the drop in predicted probability. This is not
SHAP and the differences matter:

  * it is not additive -- the per-feature drops do not sum to the prediction
  * it is blind to interactions -- if two features only matter together,
    occluding either alone may show nothing
  * it has no efficiency or symmetry guarantee; SHAP's whole point is that it
    does

What it is: a cheap, honest local sensitivity measure that answers "if this
member looked typical on this one axis, how much less risky would the model
call them?" For a phone-call worklist that question is close enough to the one
being asked. For anything a member could appeal, it is not, and the column
would need real Shapley values.

Every string in the output is deliberately descriptive, not diagnostic. The
model does not detect, diagnose, or predict deterioration. It ranks members by
modelled probability of an administrative event -- a readmission claim within
30 days -- so that a finite number of outreach calls go to the members most
likely to generate one.

Run:  python worklist.py [--capacity 500]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sklearn.ensemble import HistGradientBoostingClassifier

import features as F
from generate import WINDOW_END
from train import at_capacity, load

CUT = pd.Timestamp("2024-05-01")

# feature -> (template, direction) ; direction +1 means "high is risky"
EXPLAIN = {
    "ed_visits_90d": ("{v:.0f} ED visit(s) in the 90 days before admission", 1),
    "ed_visits_30d": ("{v:.0f} ED visit(s) in the 30 days before admission", 1),
    "ip_admits_365d": ("{v:.0f} prior inpatient admission(s) in the past year", 1),
    "ip_days_365d": ("{v:.0f} prior inpatient day(s) in the past year", 1),
    "los": ("{v:.0f}-day index stay", 1),
    "charlson": ("comorbidity burden (Charlson) {v:.0f}", 1),
    "age": ("age {v:.0f}", 1),
    "pdc_proxy_180d": ("medication fills cover only {pct:.0%} of the past 180 days", -1),
    "elig_gap_days_365d": ("{v:.0f}-day coverage gap in the past year -- confirm contact details", 1),
    "any_elig_gap_365d": ("coverage gap in the past year -- confirm contact details", 1),
    "distinct_providers_180d": ("{v:.0f} distinct providers in 180 days -- care may be fragmented", 1),
    "distinct_prescribers_180d": ("{v:.0f} distinct prescribers in 180 days", 1),
    "rx_classes_180d": ("{v:.0f} chronic medication classes", 1),
    "office_visits_90d": ("only {v:.0f} outpatient visit(s) in 90 days", -1),
    "paid_amount_365d": ("${v:,.0f} paid claims in the past year", 1),
}

CCSR_PLAIN = {
    "ccsr_CIR019": "heart failure", "ccsr_RSP008": "COPD",
    "ccsr_GEN003": "chronic kidney disease", "ccsr_END005": "diabetes with complications",
    "ccsr_END004": "diabetes", "ccsr_CIR009": "prior myocardial infarction",
    "ccsr_CIR017": "cardiac dysrhythmia", "ccsr_NEO070": "metastatic disease",
    "ccsr_MBD025": "dementia", "ccsr_MBD017": "alcohol-related disorder",
    "ccsr_MBD018": "opioid-related disorder", "ccsr_INF003": "prior sepsis",
    "ccsr_CIR020": "prior stroke", "ccsr_GEN002": "acute kidney injury",
}


def occlusion_drivers(model, X, medians, row_idx, top=3):
    """Per-feature drop in predicted probability when one feature is set to
    the cohort median. See the module docstring: this is NOT SHAP."""
    x = X.iloc[[row_idx]]
    base = model.predict_proba(x)[0, 1]
    cand = [c for c in X.columns
            if (c in EXPLAIN or c in CCSR_PLAIN) and x[c].iloc[0] != medians[c]]
    if not cand:
        return base, []
    probe = pd.concat([x] * len(cand), ignore_index=True)
    for i, c in enumerate(cand):
        probe.iloc[i, probe.columns.get_loc(c)] = medians[c]
    drops = base - model.predict_proba(probe)[:, 1]
    order = np.argsort(-drops)[:top]
    return base, [(cand[i], float(drops[i]), float(x[cand[i]].iloc[0]))
                  for i in order if drops[i] > 0.001]


def phrase(col, value):
    if col in CCSR_PLAIN:
        return f"history of {CCSR_PLAIN[col]}"
    tmpl, _d = EXPLAIN[col]
    return tmpl.format(v=value, pct=value)


def main(capacity=500, datadir="data"):
    med, rx, el, mem, st = load(datadir)
    cohort, _ = F.build_cohort(st, el, WINDOW_END)
    feat = F.build_features(cohort, F.pack_medical(med), F.pack_pharmacy(rx),
                            el, mem, visibility="received")
    tr, te = feat[feat.discharge_date < CUT], feat[feat.discharge_date >= CUT]
    Xtr = F.design_matrix(tr)
    Xte = F.design_matrix(te, columns=Xtr.columns)

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=7).fit(Xtr, tr.y.values)

    p = model.predict_proba(Xte)[:, 1]
    medians = Xtr.median()

    out = te.copy()
    out["risk"] = p
    out = out.sort_values("risk", ascending=False).reset_index(drop=True)
    Xsorted = F.design_matrix(out, columns=Xtr.columns)

    n_flagged = min(capacity, len(out))
    print("=" * 84)
    print("CARE MANAGEMENT WORKLIST -- discharges from "
          f"{CUT.date()} onward, ranked by modelled 30-day readmission risk")
    print("=" * 84)
    print(f"Cohort scored: {len(out):,} discharges     "
          f"Outreach capacity this cycle: {n_flagged:,}")
    print()
    print("This list ranks members by the modelled probability of an inpatient")
    print("readmission claim within 30 days. It is not a diagnosis, not a")
    print("clinical assessment, and not a prediction about any individual's")
    print("health. Members not on the list have not been cleared of risk.")
    print()

    for i in range(min(12, n_flagged)):
        r = out.iloc[i]
        base, drivers = occlusion_drivers(model, Xsorted, medians, i)
        print(f"  {i+1:>3}. {r.member_id}   risk {r.risk:.1%}   "
              f"discharged {pd.Timestamp(r.discharge_date).date()}   "
              f"LOS {int(r.los)}d")
        if drivers:
            for col, drop, val in drivers:
                print(f"         - {phrase(col, val)}  "
                      f"(risk contribution {drop:+.1%})")
        else:
            print("         - no single feature stands out; risk is the "
                  "accumulation of many typical values")
    if n_flagged > 12:
        print(f"\n  ... {n_flagged-12:,} more members on the worklist")

    # ---- the capacity conversation ----------------------------------------
    y = out.y.values
    worked = at_capacity(y, out.risk.values, n_flagged / len(out))
    print("\n" + "=" * 84)
    print("WHAT TO TELL THE CLINICAL DIRECTOR ABOUT THE MEMBERS WE DID NOT CALL")
    print("=" * 84)
    total_pos = int(y.sum())
    missed = total_pos - worked["n_caught"]
    print(f"  {len(out):,} discharges scored; {total_pos:,} of them readmitted within 30 days.")
    print(f"  With capacity for {n_flagged:,} calls we reach the top "
          f"{n_flagged/len(out):.0%} of the ranked list.")
    print(f"  Those {n_flagged:,} calls contain {worked['n_caught']:,} of the "
          f"{total_pos:,} readmissions ({worked['sensitivity']:.0%} sensitivity).")
    print(f"  {missed:,} readmissions occur among members we did not call.")
    print()
    print("  That last number is the honest headline and it should be said out")
    print("  loud, because the alternative is a clinical director who believes")
    print("  the programme covers the risk. Three things follow from it:")
    print()
    print("  1. The unflagged members are not low risk, they are lower-ranked.")
    if len(out) > n_flagged * 2:
        print(f"     The next {n_flagged:,} members below the cut still readmit at "
              f"{out.iloc[n_flagged:n_flagged*2].y.mean():.0%}"
              f" vs {out.iloc[:n_flagged].y.mean():.0%} above it -- the cliff is")
        print("     in the worklist, not in the members.")
    print("  2. Raising capacity has measurable, decreasing returns. Doubling")
    print("     calls does not double the catch:")
    for mult in (1, 2, 3):
        k = min(len(out), n_flagged * mult)
        c = at_capacity(y, out.risk.values, k / len(out))
        print(f"       {k:>5,} calls -> {c['n_caught']:>4,} readmissions reached "
              f"({c['sensitivity']:>4.0%} sensitivity, PPV {c['ppv']:.0%}, "
              f"NNC {c['nnc']:.1f})")
    print("  3. If the goal is fewer readmissions rather than more calls, the")
    print("     lever is intervention effectiveness, not list length. A better")
    print("     model moves sensitivity a few points; an intervention that")
    print("     actually works moves the outcome.")

    os.makedirs("out", exist_ok=True)
    export = out.head(n_flagged)[["member_id", "stay_id", "discharge_date",
                                  "risk", "los", "charlson", "ed_visits_90d",
                                  "ip_admits_365d", "any_elig_gap_365d"]]
    export.to_csv("out/worklist.csv", index=False)
    print(f"\nwrote out/worklist.csv ({len(export):,} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=500)
    ap.add_argument("--datadir", default="data")
    a = ap.parse_args()
    main(a.capacity, a.datadir)
