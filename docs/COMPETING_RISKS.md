# Death after discharge is a competing risk

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
| **naive** — a death counts as `y = 0` | 0.1612 |
| **exclude** — drop the deaths | 0.1635 |
| **latent truth** — nobody dies | **0.1649** |

Naive understates the truth by **0.37 pp**; excluding understates it by
**0.14 pp**.

`latent` is not observable in real data. It exists here only because the
generator recorded what *would* have happened, which is the entire reason for
measuring against a planted truth rather than against intuition.

## Why excluding the deaths is also wrong

Dropping them looks like the conservative move. It is **informative
censoring**: the patients who die carry higher readmission risk than those who
survive.

| | mean true risk |
|---|---|
| died post-discharge | 0.1978 |
| survived 30 days | 0.1640 |
| **ratio** | **1.21x** |

So excluding deaths fits the model on a **healthier population than the one it
will score**. 9.9% of the post-discharge deaths would have been readmitted.

## When does it actually matter?

This is the part worth keeping. At this cohort's mortality the bias is a few
tenths of a percentage point — real, measurable, and **small**. Saying so is
more useful than making it sound alarming.

| post-discharge mortality | naive rate | latent rate | naive bias | exclude bias |
|---|---|---|---|---|
| 1.2% | 0.1638 | 0.1649 | -0.11 pp | +0.01 pp |
| 2.5% | 0.1627 | 0.1649 | -0.22 pp | -0.06 pp |
| 3.8% | 0.1608 | 0.1649 | -0.41 pp | -0.18 pp |
| 6.7% | 0.1579 | 0.1649 | -0.70 pp | -0.25 pp |
| 8.9% | 0.1557 | 0.1649 | -0.92 pp | -0.47 pp |
| 14.3% | 0.1503 | 0.1649 | -1.46 pp | -0.78 pp |

The bias scales with mortality, which is the actionable form of the finding. A
general medical cohort at ~4% mortality can mostly ignore it. A heart-failure
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
