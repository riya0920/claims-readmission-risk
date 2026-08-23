"""Fairness beyond calibration: error-rate parity, and why it conflicts.

WHAT THE README SAID WAS MISSING
--------------------------------
"Subgroup O/E and AUROC are reported; equalised odds, calibration-within-groups
over time, and the access bias that claims bake in are not addressed."

Two of those three are closed here. The third is not closeable in this project
and the reason is in `access_bias_note()`.

THE THREE CRITERIA, AND THE FACT THAT THEY CANNOT ALL HOLD
-----------------------------------------------------------
    CALIBRATION-WITHIN-GROUPS  among members given risk p, the observed rate is
                               p in every group. Already reported.

    EQUALISED ODDS             the true-positive rate and the false-positive
                               rate match across groups.

    EQUAL SELECTION            groups are outreached at the same rate.

Kleinberg-Mullainathan-Raghavan and Chouldechova (both 2016) proved these are
mutually exclusive whenever prevalence differs across groups and the model is
not perfect. Both conditions hold here. So a report that shows all three green
is a report with a bug, and one that quietly picks a favourite and calls it
"the fairness metric" is hiding the choice rather than making it.

This module computes all three and REPORTS THE CONFLICT, because the choice
between them is not a modelling decision. It depends on what the score does:

  * This model ADDS outreach and gates nothing. A false positive costs a phone
    call; a false negative costs a member the call. Under that asymmetry the
    criterion that matters is the FALSE NEGATIVE RATE gap -- who is being
    missed -- and equalising false positives has almost no welfare content.
  * The moment the same score gates a benefit, the false-positive gap becomes
    the one that matters, and the right answer flips.

`docs/CLINICAL_VALIDATION.md` already argues the safety case rests on the
deployment rather than the model. This is the same argument applied to fairness,
and it is why `recommend_criterion()` returns a criterion plus the deployment
assumption it depends on, rather than a number.

WHAT THIS IS NOT
----------------
No causal fairness, no counterfactual fairness, no path-specific effects, no
intersectional analysis beyond pairwise groups, and no mitigation --
reweighting, threshold-per-group and adversarial debiasing are all absent.
Threshold-per-group in particular is legally fraught in US healthcare and is not
something to reach for without counsel.
"""

from __future__ import annotations

import numpy as np


def _rates(y, p, threshold):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(p, dtype=float) >= threshold
    pos, neg = y == 1, y == 0
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "selection_rate": float(pred.mean()) if len(y) else float("nan"),
        "tpr": float(pred[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(pred[neg].mean()) if neg.any() else float("nan"),
        "fnr": float(1 - pred[pos].mean()) if pos.any() else float("nan"),
        "ppv": float(y[pred].mean()) if pred.any() else float("nan"),
    }


def capacity_threshold(p, capacity_fraction):
    """The score at which exactly `capacity_fraction` of members are outreached.

    THE THRESHOLD IS SET BY CAPACITY, NOT BY 0.5, and the fairness analysis has
    to use the same one the programme uses. Evaluating parity at 0.5 when the
    team actually calls the top 5% measures a decision rule nobody runs.
    """
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(np.quantile(p, 1.0 - capacity_fraction))


def group_report(y, p, groups, threshold):
    """Per-group error rates at one operating threshold."""
    y, p = np.asarray(y), np.asarray(p)
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        out[str(g)] = _rates(y[m], p[m], threshold)
    return out


def parity_gaps(report, min_n=50):
    """Largest pairwise gap on each criterion, with small groups excluded.

    EXCLUDED, NOT SILENTLY INCLUDED. A 12-member group has a TPR that swings by
    8 percentage points when one member's outcome flips, and a fairness report
    that ranks it alongside a 4,000-member group is reporting sampling noise as
    a disparity -- which discredits the whole analysis the first time someone
    checks it.
    """
    usable = {g: r for g, r in report.items() if r["n"] >= min_n}
    dropped = {g: r["n"] for g, r in report.items() if r["n"] < min_n}
    gaps = {}
    for key in ("tpr", "fpr", "fnr", "selection_rate", "ppv", "prevalence"):
        vals = {g: r[key] for g, r in usable.items()
                if r[key] == r[key]}          # drop NaN
        if len(vals) < 2:
            gaps[key] = None
            continue
        hi = max(vals, key=vals.get)
        lo = min(vals, key=vals.get)
        gaps[key] = {"max_group": hi, "max": vals[hi],
                     "min_group": lo, "min": vals[lo],
                     "gap": vals[hi] - vals[lo]}
    return {"gaps": gaps, "groups_used": sorted(usable),
            "groups_dropped_too_small": dropped, "min_n": min_n}


def impossibility_check(report, tol=0.02, min_n=50):
    """Do calibration, equalised odds and equal selection all hold at once?

    They cannot, unless prevalence is equal across groups or the model is
    perfect. This checks the premise before reporting the conclusion, because
    "all three hold" is a real possibility in a degenerate dataset and would
    otherwise look like a contradiction of the theorem rather than a sign that
    the groups are indistinguishable.
    """
    usable = {g: r for g, r in report.items() if r["n"] >= min_n}
    prevs = [r["prevalence"] for r in usable.values()]
    if len(prevs) < 2:
        return {"applicable": False,
                "why": "fewer than two groups large enough to compare"}
    prev_gap = max(prevs) - min(prevs)
    g = parity_gaps(report, min_n)["gaps"]
    eq_odds = (abs(g["tpr"]["gap"]) <= tol and abs(g["fpr"]["gap"]) <= tol
               if g["tpr"] and g["fpr"] else False)
    eq_sel = abs(g["selection_rate"]["gap"]) <= tol if g["selection_rate"] else False
    return {
        "applicable": prev_gap > tol,
        "prevalence_gap": prev_gap,
        "equalised_odds_holds": bool(eq_odds),
        "equal_selection_holds": bool(eq_sel),
        "reading": (
            f"prevalence differs by {prev_gap:.1%} across groups, so "
            f"calibration, equalised odds and equal selection cannot all hold "
            f"(Kleinberg et al. 2016; Chouldechova 2016). At most one of the "
            f"latter two can be chosen, and choosing is a deployment decision."
            if prev_gap > tol else
            f"prevalence differs by only {prev_gap:.1%}; the groups are close "
            f"enough that the impossibility result has little force here, and "
            f"any of the criteria may hold at once without contradiction."),
    }


def recommend_criterion(gates_a_benefit):
    """Which gap to act on, given what the score DOES.

    A criterion, plus the deployment assumption it rests on. Returning a bare
    number here would be the same error `docs/CLINICAL_VALIDATION.md` warns
    about for the safety case: the property belongs to the deployment, not the
    model, and it evaporates when the deployment changes.
    """
    if gates_a_benefit:
        return {
            "criterion": "false positive rate (and selection rate)",
            "because": (
                "a score that GATES a benefit turns a false positive into a "
                "denial. Equalising false negatives while one group is denied "
                "at twice the rate of another optimises the wrong thing."),
            "assumption": "the score restricts access to something",
        }
    return {
        "criterion": "false negative rate",
        "because": (
            "this model only ADDS outreach and gates nothing, so a false "
            "positive costs a phone call and a false negative costs a member "
            "the call. The FNR gap is who is being missed; the FPR gap has "
            "almost no welfare content under that asymmetry."),
        "assumption": (
            "the score never denies, delays, limits or prices care. If that "
            "changes, this recommendation inverts -- see "
            "docs/CLINICAL_VALIDATION.md, which makes the same argument for "
            "the safety case."),
    }


def calibration_within_groups(y, p, groups, n_bins=5, min_n=50):
    """Observed vs predicted per group, per risk band.

    The criterion this model already satisfies by construction -- a single
    model fitted on pooled data is close to calibrated within groups when the
    features carry the group's risk. Reported anyway, because "we did not check"
    and "we checked and it held" are different claims, and only one of them
    survives a question.
    """
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() < min_n:
            out[str(g)] = {"n": int(m.sum()), "note": "too few to bin"}
            continue
        yy, pp = y[m], p[m]
        edges = np.unique(np.quantile(pp, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            out[str(g)] = {"n": int(m.sum()), "note": "scores too concentrated"}
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.digitize(pp, edges[1:-1])
        bands = []
        for b in range(len(edges) - 1):
            bm = idx == b
            if not bm.any():
                continue
            bands.append({"band": b, "n": int(bm.sum()),
                          "predicted": float(pp[bm].mean()),
                          "observed": float(yy[bm].mean())})
        worst = max((abs(x["observed"] - x["predicted"]) for x in bands),
                    default=float("nan"))
        out[str(g)] = {"n": int(m.sum()), "bands": bands,
                       "max_abs_gap": worst}
    return out


def access_bias_note():
    """The gap that cannot be closed in this project, stated rather than dropped.

    Claims measure care RECEIVED, not health. A population with worse access
    generates fewer claims, so it looks HEALTHIER to any model fitted on claims,
    and a risk score trained that way under-refers exactly the members who most
    need the referral.

    This is not detectable with the data here, and not because the analysis is
    missing -- because `src/generate.py` has no access-inequity mechanism. Every
    member's utilisation is drawn from their true risk. There is nothing to
    find, so an analysis reporting "no access bias detected" would be reporting
    a property of the generator and passing it off as a property of the model.

    Detecting it needs an external measure of health that does NOT come through
    the claims pipeline -- registry data, survey instruments, clinical
    measurements -- which is a data-acquisition problem, not a modelling one.
    """
    return {
        "closeable_here": False,
        "why": ("the generator has no access-inequity mechanism, so an "
                "analysis would report a property of src/generate.py as a "
                "property of the model"),
        "what_it_would_need": ("an external measure of health that does not "
                               "come through the claims pipeline"),
    }
