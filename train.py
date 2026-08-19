"""Train, calibrate, and evaluate FOR AN INTERVENTION.

The headline number in this file is not AUROC. It is:

    of the members we can actually call this month, how many would have
    readmitted -- and what does each avoided readmission cost us?

AUROC answers "can the model rank?", which nobody in care management asked.
Capacity is the binding constraint: a care-management team has N phone calls
per month, so the operating question is what sits in the top N. That is
sensitivity-at-capacity and PPV-at-capacity, and it is what this script leads
with.

Calibration is treated as a first-class result rather than a footnote for the
same reason. Ranking only needs monotonicity, but capacity PLANNING needs
probabilities that mean what they say: if the team is going to plan staffing
against "we expect 130 readmissions in this cohort", the predicted
probabilities have to sum to something near the truth.

Run:  python train.py [--members 50000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import uncertainty as U
from generate import TRUE_COEF, WINDOW_END

OUT = "out"
BOOT = 600      # bootstrap replicates; 600 is enough for 3-dp percentile CIs

# ---------------------------------------------------------------------------
# Economic assumptions. Every one of these is an ASSUMPTION, is named as such,
# and is varied in the break-even analysis rather than asserted.
# ---------------------------------------------------------------------------
ASSUMPTIONS = {
    "cost_per_outreach_usd": 65.0,
    # Order-of-magnitude figure consistent with AHRQ HCUP statistical briefs on
    # all-payer 30-day readmission costs (~$15K). It is deliberately NOT the
    # number the analysis leans on: the script also computes the mean paid
    # amount of readmission stays in this dataset and reports both, because a
    # real plan uses its own adjudicated paid amounts, not a national average.
    "avoided_readmission_cost_usd_literature": 15000.0,
    "capacity_pct": 0.05,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calibration_slope_intercept(y, p):
    """Cox calibration: regress y on logit(p). Slope 1 / intercept 0 is perfect.

    Slope < 1 means predictions are too extreme (over-dispersed); intercept
    below 0 means the model over-predicts risk overall (calibration-in-the-
    large). Reporting both separates "wrong on average" from "wrong in spread",
    which are different problems with different fixes.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def calibration_table(y, p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": q})
    g = df.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"),
                              observed=("y", "mean"))
    return g.reset_index(drop=True)


def at_capacity(y, p, pct):
    """Sensitivity / PPV / lift if we work the top `pct` of the ranked list."""
    n = len(y)
    k = max(1, int(round(n * pct)))
    order = np.argsort(-p)
    top = order[:k]
    tp = int(y[top].sum())
    total_pos = int(y.sum())
    prev = total_pos / n
    return {
        "capacity_pct": pct, "n_flagged": k, "n_caught": tp,
        "sensitivity": tp / total_pos if total_pos else float("nan"),
        "ppv": tp / k,
        "lift": (tp / k) / prev if prev else float("nan"),
        "nnc": k / tp if tp else float("inf"),   # number needed to contact
    }


def evaluate(y, p):
    return {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": calibration_slope_intercept(y, p)[0],
        "calibration_intercept": calibration_slope_intercept(y, p)[1],
        **{f"capacity_{int(k*100)}pct": at_capacity(y, p, k)
           for k in (0.01, 0.02, 0.05, 0.10)},
    }


# ---------------------------------------------------------------------------
def load(datadir="data"):
    med = pd.read_csv(f"{datadir}/medical_claims.csv.gz", low_memory=False,
                      parse_dates=["service_date", "service_end_date", "received_date"])
    rx = pd.read_csv(f"{datadir}/pharmacy_claims.csv.gz",
                     parse_dates=["fill_date", "received_date"])
    el = pd.read_csv(f"{datadir}/eligibility.csv.gz",
                     parse_dates=["span_start", "span_end"])
    mem = pd.read_csv(f"{datadir}/members.csv.gz")
    st = pd.read_csv(f"{datadir}/_truth_stays.csv.gz",
                     parse_dates=["admit_date", "discharge_date"])
    return med, rx, el, mem, st


def temporal_split(feat, cut):
    tr = feat[feat.discharge_date < cut]
    te = feat[feat.discharge_date >= cut]
    overlap = len(set(tr.member_id) & set(te.member_id))
    return tr, te, overlap


def main(datadir="data"):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    med, rx, el, mem, st = load(datadir)
    cohort, counts = F.build_cohort(st, el, WINDOW_END)

    print("COHORT WATERFALL")
    for k, v in counts.items():
        print(f"  {k:<46s} {v:>8,}")
    print(f"  {'observed 30-day readmission rate':<46s} "
          f"{cohort.true_readmit_30d.mean():>7.1%}")

    med_store, rx_store = F.pack_medical(med), F.pack_pharmacy(rx)
    print("\nbuilding features (serving visibility: received-by-discharge)...")
    feat = F.build_features(cohort, med_store, rx_store, el, mem,
                            visibility="received", progress=True)

    # ---- split -------------------------------------------------------------
    # Temporal, because that is how the model is deployed: fit on history,
    # score the future. A random split would let the model see discharges from
    # the same weeks it is tested on, which flatters any temporal drift away.
    cut = pd.Timestamp("2024-05-01")
    tr, te, overlap = temporal_split(feat, cut)
    Xtr = F.design_matrix(tr)
    Xte = F.design_matrix(te, columns=Xtr.columns)
    ytr, yte = tr.y.values, te.y.values
    print(f"\nSPLIT  temporal at {cut.date()}: train {len(tr):,} / test {len(te):,}")
    print(f"       members appearing in both periods: {overlap:,} "
          f"({overlap/max(1,te.member_id.nunique()):.1%} of test members)")
    print("       -- a random split would additionally leak member idiosyncrasy;"
          "\n          a temporal split leaks only the members who are admitted"
          "\n          in both periods, which is a real deployment condition.")

    # ---- models ------------------------------------------------------------
    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=1.0))
    logit.fit(Xtr, ytr)
    p_logit = logit.predict_proba(Xte)[:, 1]

    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=7)
    gbm.fit(Xtr, ytr)
    p_gbm = gbm.predict_proba(Xte)[:, 1]

    res = {"logistic": evaluate(yte, p_logit), "gbm": evaluate(yte, p_gbm)}

    print("\n" + "=" * 78)
    print("DISCRIMINATION AND CALIBRATION (test set)")
    print("=" * 78)
    print(f"{'':<22}{'logistic':>14}{'gradient boosting':>20}")
    for k, label in [("auroc", "AUROC"), ("auprc", "AUPRC"),
                     ("brier", "Brier score"),
                     ("calibration_slope", "calibration slope"),
                     ("calibration_intercept", "calibration intercept")]:
        print(f"{label:<22}{res['logistic'][k]:>14.4f}{res['gbm'][k]:>20.4f}")
    print(f"{'prevalence':<22}{res['logistic']['prevalence']:>14.4f}"
          f"{res['gbm']['prevalence']:>20.4f}")

    print("\n" + "=" * 78)
    print("THE HEADLINE: PERFORMANCE AT CARE-MANAGEMENT CAPACITY")
    print("=" * 78)
    print(f"{'capacity':<12}{'flagged':>9}{'caught':>8}{'sensitivity':>13}"
          f"{'PPV':>8}{'lift':>7}{'NNC':>7}   model")
    for name, p in [("logistic", p_logit), ("gbm", p_gbm)]:
        for pct in (0.01, 0.02, 0.05, 0.10):
            c = at_capacity(yte, p, pct)
            print(f"{'top '+str(int(pct*100))+'%':<12}{c['n_flagged']:>9,}"
                  f"{c['n_caught']:>8,}{c['sensitivity']:>12.1%}"
                  f"{c['ppv']:>8.1%}{c['lift']:>7.2f}{c['nnc']:>7.1f}   {name}")

    # ---- which model ships -------------------------------------------------
    d_auc = res["gbm"]["auroc"] - res["logistic"]["auroc"]
    d_sens = (res["gbm"]["capacity_5pct"]["sensitivity"]
              - res["logistic"]["capacity_5pct"]["sensitivity"])
    print(f"\nGBM minus logistic:  AUROC {d_auc:+.4f}   "
          f"sensitivity@5% {d_sens:+.1%}")

    # ---- uncertainty -------------------------------------------------------
    print("\n" + "=" * 78)
    print("CONFIDENCE INTERVALS (cluster bootstrap, resampled by MEMBER)")
    print("=" * 78)
    print("  Resampling is by member, not by stay: a member can contribute")
    print("  several index admissions and they are correlated, so resampling")
    print("  rows independently would treat them as independent evidence and")
    print("  produce intervals that are too narrow.")
    print()
    groups = U.ClusterIndex(te.member_id.values)   # built once, reused
    ci = {}
    print(f"  {'metric':<16}{'logistic':>28}{'gradient boosting':>28}")
    for metric in ("auroc", "auprc", "brier", "sens_at_5pct", "ppv_at_5pct"):
        a = U.bootstrap_ci(yte, p_logit, groups, metric, n_boot=BOOT, seed=1)
        b = U.bootstrap_ci(yte, p_gbm, groups, metric, n_boot=BOOT, seed=1)
        ci[metric] = {"logistic": a, "gbm": b}
        fmt = (lambda v: f"{v:.1%}") if metric.endswith("5pct") else (lambda v: f"{v:.4f}")
        print(f"  {metric:<16}"
              f"{fmt(a['point']) + ' (' + fmt(a['lo']) + '-' + fmt(a['hi']) + ')':>28}"
              f"{fmt(b['point']) + ' (' + fmt(b['lo']) + '-' + fmt(b['hi']) + ')':>28}")

    print("\n" + "=" * 78)
    print("IS THE DIFFERENCE REAL? PAIRED BOOTSTRAP ON THE DIFFERENCE")
    print("=" * 78)
    print("  NOT by checking whether the two intervals above overlap. Both")
    print("  models are scored on the SAME patients, so their errors are")
    print("  correlated and the difference is far more stable than either")
    print("  level. Comparing intervals systematically understates evidence")
    print("  for a difference; bootstrapping the difference does not.")
    print()
    print(f"  {'metric':<16}{'logistic - GBM':>18}{'95% CI':>22}{'p':>9}"
          f"   {'verdict'}")
    comparisons = {}
    for metric in ("auroc", "auprc", "sens_at_5pct"):
        d = U.paired_bootstrap_difference(yte, p_logit, p_gbm, groups, metric,
                                          n_boot=BOOT, seed=2)
        comparisons[metric] = d
        fmt = (lambda v: f"{v:+.1%}") if metric.endswith("5pct") else (lambda v: f"{v:+.4f}")
        verdict = "DISTINGUISHABLE" if d["significant"] else "indistinguishable"
        print(f"  {metric:<16}{fmt(d['difference']):>18}"
              f"{'(' + fmt(d['lo']) + ', ' + fmt(d['hi']) + ')':>22}"
              f"{d['p_value']:>9.3f}   {verdict}")

    mdd = U.minimum_detectable_difference(yte, p_logit, groups, "auroc",
                                          n_boot=BOOT, seed=3)
    print(f"\n  Smallest AUROC difference this test set could resolve: "
          f"~{mdd['approx_min_detectable']:.4f}")
    print(f"  Observed difference: {abs(d_auc):.4f}")
    if abs(d_auc) < mdd["approx_min_detectable"]:
        print("  The observed gap is SMALLER than the resolution of the")
        print("  evaluation. This comparison was never going to settle")
        print("  anything, and running it and picking a winner would have been")
        print("  theatre. The recommendation to ship logistic rests on")
        print("  transparency, not on the AUROC.")
    else:
        print("  The observed gap exceeds the resolution of the evaluation.")

    # ---- calibration table -------------------------------------------------
    best_name, best_p = ("gbm", p_gbm) if d_sens >= 0 else ("logistic", p_logit)
    ct = calibration_table(yte, best_p)
    print(f"\nCALIBRATION BY RISK DECILE ({best_name})")
    print(f"{'decile':>7}{'n':>7}{'predicted':>12}{'observed':>11}{'ratio':>8}")
    for i, r in ct.iterrows():
        ratio = r.observed / r.predicted if r.predicted else float("nan")
        print(f"{i+1:>7}{int(r.n):>7,}{r.predicted:>12.3f}{r.observed:>11.3f}"
              f"{ratio:>8.2f}")

    # ---- subgroups ---------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"SUBGROUP PERFORMANCE ({best_name}) -- the fairness analogue in clinical ML")
    print("=" * 78)
    sub = te.copy()
    sub["p"] = best_p
    sub["age_band"] = pd.cut(sub.age, [18, 35, 50, 65, 80, 120],
                             labels=["18-35", "36-50", "51-65", "66-80", "81+"])
    sub["charlson_band"] = pd.cut(sub.charlson, [-1, 0, 2, 4, 99],
                                  labels=["0", "1-2", "3-4", "5+"])
    sub["sex"] = np.where(sub.sex_f == 1, "F", "M")
    sub["coverage_gap"] = np.where(sub.any_elig_gap_365d == 1, "gap", "no gap")
    subgroup_rows = []
    for col in ["age_band", "charlson_band", "sex", "coverage_gap"]:
        print(f"\n  by {col}")
        print(f"    {'group':<10}{'n':>7}{'obs rate':>10}{'pred rate':>11}"
              f"{'O/E':>7}{'AUROC':>8}{'sens@5%':>9}")
        for g, d in sub.groupby(col, observed=True):
            if len(d) < 40 or d.y.nunique() < 2:
                print(f"    {str(g):<10}{len(d):>7,}   (too few to report)")
                continue
            auc = roc_auc_score(d.y, d.p)
            oe = d.y.mean() / d.p.mean()
            s5 = at_capacity(d.y.values, d.p.values, 0.05)["sensitivity"]
            print(f"    {str(g):<10}{len(d):>7,}{d.y.mean():>10.1%}"
                  f"{d.p.mean():>11.1%}{oe:>7.2f}{auc:>8.3f}{s5:>9.1%}")
            subgroup_rows.append({"dimension": col, "group": str(g), "n": len(d),
                                  "observed": float(d.y.mean()),
                                  "predicted": float(d.p.mean()),
                                  "oe_ratio": float(oe), "auroc": float(auc),
                                  "sens_at_5pct": float(s5)})

    worst = sorted(subgroup_rows, key=lambda r: -abs(r["oe_ratio"] - 1))[:3]
    print()
    print("  Largest calibration gaps by subgroup (observed/expected):")
    for w in worst:
        direction = "UNDER-predicts" if w["oe_ratio"] > 1 else "over-predicts"
        print(f"    {w['dimension']}={w['group']:<10} O/E {w['oe_ratio']:.2f}  "
              f"n={w['n']:,}  -- model {direction} risk for this group")
    print("  A capacity-bounded worklist ranks globally, so a group the model")
    print("  under-predicts is systematically under-represented on the call")
    print("  list relative to its actual risk. That is the clinical-ML analogue")
    print("  of a fairness finding and it belongs in front of the clinical")
    print("  director before deployment, not in an appendix.")

    # ---- coefficient recovery ---------------------------------------------
    print("\n" + "=" * 78)
    print("COEFFICIENT RECOVERY vs THE GENERATOR'S TRUTH (logistic)")
    print("=" * 78)
    print("Recovery is checked because we wrote the data-generating equation.")
    print("Do NOT expect ratios near 1, and the reasons are the interesting part:")
    print("  * ATTENUATION. Features hit hardest by claim runout (recent")
    print("    utilisation) are measured with error, and measurement error")
    print("    biases coefficients toward zero -- regression dilution.")
    print("  * INFLATION. Features that survive runout intact (comorbidity,")
    print("    which recurs on every chronic visit) absorb the signal the")
    print("    degraded features dropped, because they proxy the same latent")
    print("    morbidity. The estimate is not 'wrong'; it is answering a")
    print("    different question than the generator's coefficient.")
    print("  * The NEGATIVE CONTROL should land near zero. If the model gives")
    print("    distinct_prescribers_180d a real coefficient, it is fitting noise.")
    scaler = logit.named_steps["standardscaler"]
    coefs = logit.named_steps["logisticregression"].coef_[0]
    sd = scaler.scale_
    raw = {c: coefs[i] / sd[i] for i, c in enumerate(Xtr.columns)}
    pairs = [("charlson", "charlson"), ("ed_visits_90d", "ed_visits_90d"),
             ("ip_days_365d", "ip_days_365d"), ("los", "los"),
             ("any_elig_gap_365d", "elig_gap_365d"),
             ("ip_admits_365d", "prior_admits_365d"),
             ("distinct_prescribers_180d", "distinct_prescribers_180d")]
    print(f"\n  {'feature':<28}{'true beta':>11}{'estimated':>11}{'ratio':>8}")
    recovery = []
    for feat_name, true_name in pairs:
        if feat_name not in raw:
            continue
        tb, eb = TRUE_COEF[true_name], raw[feat_name]
        tag = "  <- negative control" if tb == 0 else ""
        ratio = f"{eb/tb:>8.2f}" if tb else f"{'n/a':>8}"
        print(f"  {feat_name:<28}{tb:>11.3f}{eb:>11.3f}{ratio}{tag}")
        recovery.append({"feature": feat_name, "true": tb, "estimated": eb,
                         "ratio": (eb / tb) if tb else None,
                         "negative_control": tb == 0})

    nc = next((r for r in recovery
               if r["feature"] == "distinct_prescribers_180d"), None)
    if nc is not None and abs(nc["estimated"]) > 0.05:
        print()
        print("  THE NEGATIVE CONTROL FAILED, and that is the finding.")
        print(f"  distinct_prescribers_180d has NO effect in the generator, and the")
        print(f"  model gave it beta = {nc['estimated']:+.3f}. It is not fitting noise")
        print("  in the random sense -- it is absorbing collinearity. Prescriber")
        print("  count moves with medication-class count and with morbidity, so in")
        print("  a correlated claims feature set the fit distributes one signal")
        print("  across several columns with partially cancelling signs.")
        print()
        print("  The consequence is concrete and it is why the worklist does not")
        print("  explain members using coefficients: in a claims model, an")
        print("  individual coefficient is not an effect size and must not be")
        print("  quoted to a clinician as one. Reason codes in worklist.py come")
        print("  from occlusion on the fitted model, which asks the question a")
        print("  care manager actually has -- and which stays honest under")
        print("  collinearity, because it never claims the parts sum to the whole.")

    # ---- economics ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("COST-EFFECTIVENESS AT CAPACITY -- arithmetic shown, assumptions named")
    print("=" * 78)
    truth = st[st.is_readmit_stay.astype(bool)]
    paid_readmit = med[med.stay_id.isin(truth.stay_id) & (med.claim_type == "IP")]
    own_cost = float(paid_readmit.paid_amount.mean()) if len(paid_readmit) else float("nan")
    cap = at_capacity(yte, best_p, ASSUMPTIONS["capacity_pct"])
    c_out = ASSUMPTIONS["cost_per_outreach_usd"]
    print(f"  assumption  cost per outreach                 ${c_out:,.0f}")
    print(f"  assumption  avoided readmission (literature)  "
          f"${ASSUMPTIONS['avoided_readmission_cost_usd_literature']:,.0f}")
    print(f"  measured    mean paid amount, readmission stays in THIS dataset "
          f"${own_cost:,.0f}")
    print("              (the simulated figure is what the arithmetic below")
    print("               uses; a real plan substitutes its own paid amounts)")
    print(f"\n  at top {ASSUMPTIONS['capacity_pct']:.0%} capacity:")
    print(f"    members flagged                  {cap['n_flagged']:,}")
    print(f"    outreach cost                    "
          f"${cap['n_flagged']*c_out:,.0f}  = {cap['n_flagged']:,} x ${c_out:,.0f}")
    print(f"    readmissions in the flagged set  {cap['n_caught']:,}")
    print(f"    number needed to contact (NNC)   {cap['nnc']:.1f}"
          f"  = {cap['n_flagged']:,} / {cap['n_caught']:,}")
    print("\n  Outreach does not prevent every readmission it touches, and the")
    print("  effect size of transitional care management is contested. So")
    print("  rather than assert one, solve for BREAK-EVEN:")
    be = (cap["n_flagged"] * c_out) / (cap["n_caught"] * own_cost) if cap["n_caught"] else float("nan")
    print(f"\n    break-even relative risk reduction = "
          f"(flagged x cost_out) / (caught x cost_readmit)")
    print(f"                                       = "
          f"({cap['n_flagged']:,} x ${c_out:,.0f}) / ({cap['n_caught']:,} x ${own_cost:,.0f})")
    print(f"                                       = {be:.1%}")
    print(f"\n  So the programme pays for itself if outreach prevents more than")
    print(f"  {be:.1%} of the readmissions it reaches. Published transitional-care")
    print("  effects vary widely and several trials are null, so whether that")
    print("  bar is clearable is an empirical question for a silent-mode trial,")
    print("  not something this model can answer.")
    for rrr in (0.05, 0.10, 0.20):
        saved = cap["n_caught"] * rrr * own_cost
        net = saved - cap["n_flagged"] * c_out
        print(f"    if RRR = {rrr:>4.0%}:  avoided ${saved:>10,.0f}   "
              f"net ${net:>+11,.0f}")

    payload = {
        "confidence_intervals": ci,
        "model_comparison": comparisons,
        "minimum_detectable_difference": mdd,
        "cohort_waterfall": counts,
        "split": {"cut": str(cut.date()), "train": len(tr), "test": len(te),
                  "member_overlap": overlap},
        "results": res, "subgroups": subgroup_rows,
        "coefficient_recovery": recovery,
        "assumptions": ASSUMPTIONS,
        "economics": {"mean_paid_readmission_stay": own_cost,
                      "break_even_rrr": be, **cap},
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(f"{OUT}/metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    ct.to_csv(f"{OUT}/calibration_{best_name}.csv", index=False)
    print(f"\nwrote {OUT}/metrics.json  ({time.time()-t0:.0f}s total)")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    a = ap.parse_args()
    main(a.datadir)
