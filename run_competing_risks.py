"""Death after discharge is a competing risk, not a non-event.

THE PROBLEM
-----------
A patient who dies at home on day 10 did not "fail to be readmitted". They were
at risk for ten of the thirty days and then not at risk at all. Labelling them
`y = 0` counts a death as a good outcome, and every model trained on that label
learns to treat "about to die" as "low risk".

Until now the generator had no post-discharge mortality, so the gap list could
only say the problem existed. It exists in the data now, drawn from the SAME
risk score as readmission -- which is what makes it a COMPETING risk rather
than nuisance censoring. The patients removed from the risk set are precisely
the ones most likely to have been readmitted.

THREE WAYS TO HANDLE IT, ALL WRONG IN DIFFERENT DIRECTIONS
-----------------------------------------------------------
  naive     count a death as `y = 0`. Understates the rate, and teaches the
            model that dying is a good outcome.
  exclude   drop the deaths. This is INFORMATIVE censoring -- deaths carry
            higher readmission risk than survivors, so dropping them fits the
            model on a healthier population than the one it will score.
  latent    what would have happened if nobody died. Not observable in real
            data; available here only because the generator recorded it, which
            is the whole reason for measuring against a planted truth.

WHAT THE ANSWER TURNS OUT TO BE
--------------------------------
Both biases are real, both point the same way, and at this cohort's mortality
they are SMALL -- a few tenths of a percentage point on an 18% rate. Reporting
that honestly matters more than making the finding sound impressive: the
correct conclusion is not "readmission models are broken", it is "this bias
scales with mortality, and here is the mortality at which it starts to matter".

So the sweep is the deliverable, not the headline number.

Run:  python run_competing_risks.py
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd

TRUTH = os.path.join(ROOT, "data", "_truth_stays.csv.gz")
PROBE = os.path.join(ROOT, "out", "cr_probe", "_truth_stays.csv.gz")


def load_index_stays(path=None):
    """Index stays that survived the admission -- the modelling cohort."""
    path = path or (TRUTH if os.path.exists(TRUTH) else PROBE)
    if not os.path.exists(path):
        raise SystemExit(
            "no generated truth file. Run `python -m src.generate` (or the "
            "generate step in the README) first.")
    stays = pd.read_csv(path)
    for col in ("is_readmit_stay", "died_inpatient",
                "died_post_discharge_30d", "latent_readmit_30d",
                "censored_by_death", "true_readmit_30d"):
        if col not in stays.columns:
            raise SystemExit(
                "%s has no `%s` column -- regenerate the data, the "
                "competing-risk fields are new." % (os.path.basename(path),
                                                    col))
        stays[col] = stays[col].astype(bool)
    return stays[~stays.is_readmit_stay & ~stays.died_inpatient].copy()


def rates(idx):
    obs = idx.true_readmit_30d
    dead = idx.died_post_discharge_30d
    return {
        "n": len(idx),
        "naive": float(obs.mean()),
        "exclude_deaths": float(obs[~dead].mean()),
        "latent": float(idx.latent_readmit_30d.mean()),
        "post_discharge_mortality": float(dead.mean()),
        "deaths": int(dead.sum()),
        "censored_readmissions": int(idx.censored_by_death.sum()),
        "share_of_deaths_that_would_have_readmitted":
            float(idx.censored_by_death.sum() / max(dead.sum(), 1)),
        "risk_of_the_dead": float(idx.loc[dead, "true_p"].mean()),
        "risk_of_the_living": float(idx.loc[~dead, "true_p"].mean()),
    }


def sweep(idx, multipliers=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0), seed=11):
    """Re-draw post-discharge mortality at different intensities.

    Done by RESAMPLING against the stored true risk rather than regenerating
    the whole corpus: the latent readmission and its day are already recorded,
    so the only thing that has to change is who dies and when. That keeps every
    other source of variation fixed, which is the point -- otherwise the sweep
    would be measuring generator noise as well as mortality.
    """
    rng = np.random.default_rng(seed)
    p = idx["true_p"].to_numpy()
    latent = idx.latent_readmit_30d.to_numpy()
    readmit_day = idx["readmit_day"].to_numpy()

    out = []
    for m in multipliers:
        p_death = np.clip(0.012 + 0.160 * m * p, 0, 0.60)
        dies = rng.random(len(idx)) < p_death
        death_day = rng.integers(1, 31, len(idx))
        censored = dies & latent & (death_day < readmit_day)
        observed = latent & ~censored

        naive = observed.mean()
        keep = ~dies
        excl = observed[keep].mean() if keep.any() else float("nan")
        out.append({
            "multiplier": m,
            "mortality": float(dies.mean()),
            "naive": float(naive),
            "exclude_deaths": float(excl),
            "latent": float(latent.mean()),
            "naive_bias_pp": float(100 * (naive - latent.mean())),
            "exclude_bias_pp": float(100 * (excl - latent.mean())),
        })
    return out


def main():
    idx = load_index_stays()
    r = rates(idx)

    print("=" * 74)
    print("  index stays that survived the admission: %d" % r["n"])
    print("=" * 74)
    print("  readmission rate")
    print("     naive        (a death counts as y=0)   %.4f" % r["naive"])
    print("     exclude      (drop the deaths)         %.4f"
          % r["exclude_deaths"])
    print("     LATENT truth (nobody dies)             %.4f" % r["latent"])
    print()
    print("     naive understates the truth by         %.2f pp"
          % (100 * (r["latent"] - r["naive"])))
    print("     excluding understates it by            %.2f pp"
          % (100 * (r["latent"] - r["exclude_deaths"])))
    print()
    print("  post-discharge mortality                  %.4f (%d stays)"
          % (r["post_discharge_mortality"], r["deaths"]))
    print("  of those, would have been readmitted      %.1f%% (%d stays)"
          % (100 * r["share_of_deaths_that_would_have_readmitted"],
             r["censored_readmissions"]))
    print()
    print("  WHY EXCLUDING IS ALSO WRONG -- the censoring is informative:")
    print("     mean true risk, died post-discharge    %.4f"
          % r["risk_of_the_dead"])
    print("     mean true risk, survived 30 days       %.4f"
          % r["risk_of_the_living"])
    print("     ratio                                  %.2fx"
          % (r["risk_of_the_dead"] / r["risk_of_the_living"]))
    print()
    print("=" * 74)
    print("  SENSITIVITY -- when does this start to matter?")
    print("  %10s %12s %10s %14s %14s"
          % ("mortality", "naive", "latent", "naive bias", "exclude bias"))
    rows = sweep(idx)
    for row in rows:
        print("  %9.1f%% %12.4f %10.4f %12.2f pp %12.2f pp"
              % (100 * row["mortality"], row["naive"], row["latent"],
                 row["naive_bias_pp"], row["exclude_bias_pp"]))
    print("=" * 74)

    _report(r, rows)


def _report(r, rows):
    doc = os.path.join(ROOT, "docs")
    os.makedirs(doc, exist_ok=True)
    path = os.path.join(doc, "COMPETING_RISKS.md")
    sweep_rows = "\n".join(
        "| %.1f%% | %.4f | %.4f | %+.2f pp | %+.2f pp |"
        % (100 * x["mortality"], x["naive"], x["latent"],
           x["naive_bias_pp"], x["exclude_bias_pp"]) for x in rows)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("""# Death after discharge is a competing risk

A patient who dies at home on day 10 did not *fail to be readmitted*. They were
at risk for ten of the thirty days and then not at risk at all. Labelling them
`y = 0` counts a death as a good outcome, and a model trained on that label
learns to treat *about to die* as *low risk*.

The gap list previously said post-discharge mortality "is not in the generator,
so there is nothing here to recover". It is now, drawn from the **same risk
score as readmission** -- which is what makes it a *competing* risk rather than
nuisance censoring. The patients removed from the risk set are precisely the
ones most likely to have been readmitted.

## Three handlings, all wrong in different directions

| | readmission rate |
|---|---|
| **naive** — a death counts as `y = 0` | %.4f |
| **exclude** — drop the deaths | %.4f |
| **latent truth** — nobody dies | **%.4f** |

Naive understates the truth by **%.2f pp**; excluding understates it by
**%.2f pp**.

`latent` is not observable in real data. It exists here only because the
generator recorded what *would* have happened, which is the entire reason for
measuring against a planted truth rather than against intuition.

## Why excluding the deaths is also wrong

Dropping them looks like the conservative move. It is **informative
censoring**: the patients who die carry higher readmission risk than those who
survive.

| | mean true risk |
|---|---|
| died post-discharge | %.4f |
| survived 30 days | %.4f |
| **ratio** | **%.2fx** |

So excluding deaths fits the model on a **healthier population than the one it
will score**. %.1f%% of the post-discharge deaths would have been readmitted.

## When does it actually matter?

This is the part worth keeping. At this cohort's mortality the bias is a few
tenths of a percentage point — real, measurable, and **small**. Saying so is
more useful than making it sound alarming.

| post-discharge mortality | naive rate | latent rate | naive bias | exclude bias |
|---|---|---|---|---|
%s

The bias scales with mortality, which is the actionable form of the finding. A
general medical cohort at ~4%% mortality can mostly ignore it. A heart-failure
or oncology cohort, where 30-day post-discharge mortality runs several times
higher, cannot — and the direction is always the same: **the model is trained
to believe the sickest patients are safer than they are.**

The sweep resamples mortality against the stored true risk rather than
regenerating the corpus, so every other source of variation is held fixed. A
regeneration sweep would measure generator noise alongside the effect.

## What this still does not do

There is no Fine-Gray or cause-specific model here — this measures the
distortion, it does not correct it. `data3-trial-survival` in this portfolio
has the Aalen-Johansen and Fine-Gray machinery, and wiring the two together is
listed as a gap rather than done. The right production answer is usually a
composite outcome (readmission **or** death), which changes what the model is
for and is a product decision, not a modelling one.
""" % (r["naive"], r["exclude_deaths"], r["latent"],
       100 * (r["latent"] - r["naive"]),
       100 * (r["latent"] - r["exclude_deaths"]),
       r["risk_of_the_dead"], r["risk_of_the_living"],
       r["risk_of_the_dead"] / r["risk_of_the_living"],
       100 * r["share_of_deaths_that_would_have_readmitted"],
       sweep_rows))
    print("wrote", path)


if __name__ == "__main__":
    main()
