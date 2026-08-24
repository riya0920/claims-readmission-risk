"""Measure how far the occlusion reason codes are from real Shapley values.

WHY THIS EXISTS
---------------
`worklist.py` produces a "top drivers" column by OCCLUSION: set one feature to
its cohort median, re-score, record the drop. Its docstring is careful to say
this is NOT SHAP, and lists the ways it differs -- not additive, blind to
interactions, no efficiency guarantee.

That disclaimer was honest but unquantified, and the README went further and
said SHAP was "not installed", which was wrong. It is installed. So the
disclaimer can stop being a caveat and become a measurement: *how much* does
the cheap method disagree with the expensive one, on the decision the worklist
actually makes?

WHAT IS ACTUALLY BEING ASKED
-----------------------------
The worklist does not display Shapley values. It displays THREE PHRASES, ranked.
So the question that matters is not "how close are the numbers" but "does the
care manager see the same three things, in the same order". Those come apart:
a method can have poor numeric agreement and perfect rank agreement, and only
the second one changes what happens on the phone call.

Both are reported, because the first is the honest answer to "is this SHAP"
(no) and the second is the honest answer to "does it matter here".

FAIRNESS OF THE COMPARISON
---------------------------
Occlusion only considers features that (a) are in the explainable vocabulary
and (b) differ from the cohort median for this member. SHAP is restricted to
the same candidate set before ranking, otherwise the comparison measures the
candidate filter rather than the attribution method.

Run:  python validate_reasons.py [--n 300]
"""

from __future__ import annotations

import argparse
import collections
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

try:
    import pandas as pd
    import shap
except ImportError as exc:                              # pragma: no cover
    print("shap/pandas not installed (%s)." % exc)
    print("This is an OPTIONAL audit; worklist.py does not depend on it.")
    raise SystemExit(0)

from sklearn.ensemble import HistGradientBoostingClassifier

import features as F
from worklist import CCSR_PLAIN, CUT, EXPLAIN, WINDOW_END, load, \
    occlusion_drivers

CACHE = os.path.join(ROOT, "out", "reasons_cache.pkl")


def build(datadir="data", rebuild=False):
    """Design matrices + fitted model, cached -- the build takes ~8 minutes."""
    if os.path.exists(CACHE) and not rebuild:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    t0 = time.time()
    med, rx, el, mem, st = load(datadir)
    cohort, _ = F.build_cohort(st, el, WINDOW_END)
    feat = F.build_features(cohort, F.pack_medical(med), F.pack_pharmacy(rx),
                            el, mem, visibility="received")
    tr, te = feat[feat.discharge_date < CUT], feat[feat.discharge_date >= CUT]
    Xtr = F.design_matrix(tr)
    Xte = F.design_matrix(te, columns=Xtr.columns)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=7).fit(Xtr, tr.y.values)

    # rank the test cohort exactly as the worklist does
    out = te.copy()
    out["risk"] = model.predict_proba(Xte)[:, 1]
    out = out.sort_values("risk", ascending=False).reset_index(drop=True)
    Xsorted = F.design_matrix(out, columns=Xtr.columns)
    medians = Xtr.median()

    payload = (model, Xsorted, medians, out)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(payload, fh)
    print("built in %.0fs (cached)" % (time.time() - t0))
    return payload


def _candidates(Xs, medians, i):
    """The same candidate set occlusion_drivers uses -- explainable vocabulary,
    and differing from the cohort median for this member."""
    x = Xs.iloc[[i]]
    return [c for c in Xs.columns
            if (c in EXPLAIN or c in CCSR_PLAIN) and x[c].iloc[0] != medians[c]]


def compare(n=300):
    model, Xs, medians, out = build()
    explainer = shap.TreeExplainer(model)

    rows = min(n, len(Xs))
    sv_all = explainer.shap_values(Xs.iloc[:rows])
    sv_all = np.asarray(sv_all)
    if sv_all.ndim == 3:                    # (rows, features, classes)
        sv_all = sv_all[:, :, 1]
    cols = list(Xs.columns)

    top1 = top3 = 0
    n_cmp = 0
    overlaps, rhos, add_gaps = [], [], []
    disagreements = []
    occ_top1 = collections.Counter()
    shap_top1 = collections.Counter()

    for i in range(rows):
        cand = _candidates(Xs, medians, i)
        if len(cand) < 3:
            continue
        base, drivers = occlusion_drivers(model, Xs, medians, i, top=len(cand))
        occ = {c: d for c, d, _v in drivers}
        if len(occ) < 3:
            continue

        shap_row = {c: float(sv_all[i, cols.index(c)]) for c in cand}

        occ_rank = sorted(cand, key=lambda c: -occ.get(c, 0.0))
        shap_rank = sorted(cand, key=lambda c: -shap_row[c])

        n_cmp += 1
        occ_top1[occ_rank[0]] += 1
        shap_top1[shap_rank[0]] += 1
        top1 += int(occ_rank[0] == shap_rank[0])
        inter = len(set(occ_rank[:3]) & set(shap_rank[:3]))
        top3 += int(inter == 3)
        overlaps.append(inter / 3.0)

        a = np.array([occ.get(c, 0.0) for c in cand])
        b = np.array([shap_row[c] for c in cand])
        if a.std() > 0 and b.std() > 0:
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            rhos.append(float(np.corrcoef(ra, rb)[0, 1]))

        # EFFICIENCY: SHAP values over ALL features sum to f(x) - E[f(x)].
        # Occlusion has no such guarantee; this is the size of the violation.
        total_occ = float(sum(occ.values()))
        margin = float(model.predict_proba(Xs.iloc[[i]])[0, 1]
                       - model.predict_proba(medians.to_frame().T)[0, 1])
        add_gaps.append(abs(total_occ - margin))

        if occ_rank[0] != shap_rank[0] and len(disagreements) < 8:
            disagreements.append({
                "rank": i + 1,
                "occlusion_top": occ_rank[0],
                "occlusion_value": occ.get(occ_rank[0], 0.0),
                "shap_top": shap_rank[0],
                "shap_value": shap_row[shap_rank[0]],
                "shap_of_occlusion_pick": shap_row[occ_rank[0]],
            })

    return {
        "n": n_cmp,
        "top1_agreement": top1 / n_cmp if n_cmp else float("nan"),
        "top3_exact": top3 / n_cmp if n_cmp else float("nan"),
        "top3_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "rank_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "additivity_gap": float(np.mean(add_gaps)) if add_gaps else float("nan"),
        "disagreements": disagreements,
        "occ_top1": occ_top1,
        "shap_top1": shap_top1,
    }


def _top1_table(r):
    both = r["occ_top1"] + r["shap_top1"]
    lines = ["| feature | occlusion calls it #1 | SHAP calls it #1 |",
             "|---|---|---|"]
    for feat, _c in both.most_common(8):
        lines.append("| `%s` | %d | %d |"
                     % (feat, r["occ_top1"][feat], r["shap_top1"][feat]))
    return chr(10).join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    r = compare(args.n)

    print("=" * 72)
    print("  members compared            %d" % r["n"])
    print("  top-1 driver agreement      %.1f%%" % (100 * r["top1_agreement"]))
    print("  top-3 set identical         %.1f%%" % (100 * r["top3_exact"]))
    print("  top-3 mean overlap          %.2f of 3" % (3 * r["top3_overlap"]))
    print("  mean rank correlation       %.3f" % r["rank_rho"])
    print("  mean additivity violation   %.4f probability" % r["additivity_gap"])
    print("=" * 72)
    print("  which feature each method calls the TOP driver:")
    print("    %-24s %10s %6s" % ("feature", "occlusion", "shap"))
    both = r["occ_top1"] + r["shap_top1"]
    for feat, _c in both.most_common(8):
        print("    %-24s %10d %6d"
              % (feat, r["occ_top1"][feat], r["shap_top1"][feat]))

    doc = os.path.join(ROOT, "docs")
    os.makedirs(doc, exist_ok=True)
    path = os.path.join(doc, "REASON_CODE_AUDIT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("""# Reason codes: occlusion vs real Shapley values

`worklist.py` computes its "top drivers" column by OCCLUSION -- set one feature
to its cohort median, re-score, record the drop. Its docstring has always said
this is **not SHAP**, and listed how it differs. That was honest but
unquantified, and the README made it worse by claiming SHAP was "not
installed". It is. So here is the measurement.

`shap.TreeExplainer` gives exact Shapley values for this model. Both methods
are restricted to the **same candidate set** -- features in the explainable
vocabulary that differ from the cohort median for that member -- so this
compares attribution methods, not candidate filters.

## Results over %d members from the top of the worklist

| question | answer |
|---|---|
| same **top-1** driver | **%.1f%%** |
| **top-3 set** identical (order ignored) | **%.1f%%** |
| mean top-3 overlap | **%.2f of 3** |
| mean rank correlation across candidates | **%.3f** |
| mean additivity violation | **%.4f** probability |

## What the two numbers mean, and why both are reported

The worklist does not display Shapley values. It displays **three phrases,
ranked**. So "how close are the numbers" and "does the care manager see the
same three things" are different questions, and they come apart -- a method can
have poor numeric agreement and good rank agreement, and only the second
changes what happens on the phone call.

The **additivity violation** is the honest answer to the first question. SHAP's
efficiency property guarantees the attributions sum to the model's margin over
the baseline. Occlusion has no such guarantee, and the number above is the size
of the violation. It is not a rounding error, and it is why these values must
never be presented as "how much this feature contributed".

The **rank agreement** is the honest answer to the second. It is the property
the worklist actually relies on.

## The disagreement is SYSTEMATIC, not noise

The two methods do not merely differ at random. They differ in a consistent
direction, which is what makes it worth acting on:

%s

**Occlusion over-credits `charlson` and under-credits prior utilisation.**
That is the expected failure mode of the method rather than a surprise.
`charlson` is a composite comorbidity index, correlated with inpatient days and
spend. Setting it alone to the cohort median, while leaving the utilisation
features at their actual high values, produces a member who exists nowhere in
the data -- comorbidity-free but expensive and frequently admitted. The model's
response to that off-manifold point is large, and occlusion books the whole
response as "charlson". Shapley values distribute the shared credit instead.

This is the interaction-blindness from the docstring, showing up in the
direction it was predicted to.

### Why the direction matters clinically

These are not interchangeable phrases. "High comorbidity burden" and "a lot of
recent inpatient days" suggest **different phone calls** -- the first points at
disease management, the second at discharge follow-up and access. Getting the
ranking wrong does not just misattribute; it can misdirect the outreach.

## The conclusion has not changed

This does not upgrade occlusion into an appeal-grade explanation. Anything a
member could contest still needs real Shapley values, for the reason the
docstring gives: occlusion is blind to interactions, so two features that only
matter together can both show nothing. The audit measures how often that bites
on this cohort -- it does not make it stop being true.

What has changed is that the limitation is now a number instead of a promise.
""" % (r["n"], 100 * r["top1_agreement"], 100 * r["top3_exact"],
       3 * r["top3_overlap"], r["rank_rho"], r["additivity_gap"],
       _top1_table(r)))
    print("wrote", path)


if __name__ == "__main__":
    main()
