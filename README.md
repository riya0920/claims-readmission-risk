# ML-1 — Claims-based readmission risk (first 20%)

**This is not a deployable system.** It is the load-bearing part of one: the
claims feature craft, the temporal discipline, and the intervention framing.
The API, the fairness work, the monitoring, and the real data are not here, and
[§ What is missing](#what-is-missing-the-other-80) says so specifically.

Predicts unplanned 30-day readmission from synthetic payer claims, scored at
the moment of discharge, and evaluates the result the way a care-management
programme is actually run: at capacity.

```bash
python src/generate.py --members 50000    # ~4 min, writes data/
python train.py                           # cohort, models, calibration, subgroups, economics
python leakage_audit.py                   # 3 experiments -> docs/LEAKAGE_AUDIT.md
python worklist.py --capacity 500         # the care-manager artefact
python -m pytest tests -q                 # 14 tests, all about not cheating
```

Everything runs offline. Runtime end to end is about six minutes.

---

## The four things worth reading

### 1. Claims fluency, in the features rather than the README

- **CCSR-style grouping** (`src/codes.py`). Raw ICD-10-CM is too sparse to
  learn from — 70,000 codes, most seen a handful of times. Diagnoses are
  grouped to ~27 tracked categories before modelling. The file is explicit that
  the mapping is a hand-curated subset following the CCSR convention, not the
  HCUP release, and that production means version-pinning the real file because
  category membership moves between annual releases.
- **Charlson Comorbidity Index, Quan ICD-10 adaptation** (`src/comorbidity.py`)
  with the hierarchy rules implemented: diabetes-with-complication supersedes
  diabetes-without, severe liver supersedes mild, metastatic supersedes
  malignancy. Getting the hierarchy wrong inflates the score for exactly the
  sickest members, and it is the most common quiet error in Charlson code.
  Tested in `tests/test_guard.py`.
- **Eligibility-gap features.** Coverage is spans, not a flag. Cohort entry
  uses the HEDIS allowable-gap convention (one gap ≤45 days) rather than
  requiring unbroken coverage — because requiring unbroken coverage deletes
  precisely the churning members whose gaps carry the signal. The first version
  of this file did require unbroken coverage, and shipped a gap feature that
  was identically zero; `test_gap_days_feature_is_populated_for_churning_members`
  is the regression test.

### 2. Sensitivity at capacity is the headline, not AUROC

AUROC is 0.652, which is where honest claims-only readmission models sit. The
number the programme is run on:

| capacity | flagged | readmissions caught | sensitivity | PPV | NNC |
|---|---|---|---|---|---|
| top 1% | 88 | 53 | 3.6% | 60.2% | 1.7 |
| **top 5%** | **441** | **176** | **12.0%** | **39.9%** | **2.5** |
| top 10% | 883 | 292 | 19.9% | 33.1% | 3.0 |

*Of the 441 members we can call, about 176 would have been readmitted.*

`worklist.py` prints the other half of that sentence, which is the half that
usually goes unsaid: **1,278 of the 1,466 readmissions happen among members we
did not call**, the next 500 members below the cut still readmit at 28% against
38% above it, and tripling capacity takes sensitivity from 13% to 31% — not to
39%. The cliff is in the worklist, not in the members.

Economics are computed rather than asserted, and the direction is inverted on
purpose: instead of assuming an intervention effect, the script solves for the
**break-even relative risk reduction (1.1%)** and reports the mean paid amount
of readmission stays in this dataset ($14,456) alongside the literature
order-of-magnitude figure, because a real plan uses its own adjudicated
amounts.

Calibration is treated as first-class because capacity planning runs on
probabilities: slope 0.902, intercept -0.147, O/E between 0.96 and 1.05 across
age and comorbidity bands.

### 3. The leakage audit, with a guard that has teeth

`docs/LEAKAGE_AUDIT.md` is generated, not written. Three experiments:

**The obvious leak, priced.** Discharge disposition is sitting in the claim and
is assigned at discharge coding — downstream of the outcome. Adding it:

| model | AUROC | sensitivity @ 5% |
|---|---|---|
| as specified | 0.6467 | 11.6% |
| + discharge_status | **0.9040** | **30.1%** |

The point is not that it goes up. It is that it goes up by an amount that looks
like a good week's work, which is why this failure survives review.

**The guard has teeth.** `ClaimsView` is the only way to read claims, is
constructed with an as_of, and raises `TemporalViolation` rather than returning
a number. The audit plants a post-discharge inpatient stay and shows every
numeric feature unchanged, and the direct path raising.

**The non-obvious leak: claim runout.** Every claim carries a `received_date`
as well as a `service_date`, with facility claims lagging ~41 days and
professional ~12. So a warehouse snapshot contains claims that had not *arrived*
at scoring time:

| feature | at serving time | with complete history | understated |
|---|---|---|---|
| `ed_visits_90d` | 0.19 | 0.32 | 41% |
| `office_visits_90d` | 0.75 | 0.91 | 17% |
| `ip_days_365d` | 2.94 | 3.30 | 11% |

Recency is where the signal is, and recency is what runout eats. The measured
cost of ignoring it is **-0.0046 AUROC and -0.7 points of sensitivity at
capacity** — real, in the expected direction, and smaller than leakage rhetoric
usually implies. The audit says so plainly and explains why it is modest here
(the generator's signal leans on features that survive runout) rather than
inflating it.

### 4. The humility document

[`docs/CLINICAL_VALIDATION.md`](docs/CLINICAL_VALIDATION.md) — intended use and
explicitly not-intended use, the false-negative cost analysis (the asymmetry
that makes this safe is a property of the *deployment*, not the model, and it
evaporates the moment the score gates anything), the identified subgroup
miscalibration, what synthetic data cannot tell us, seven named failure modes
including the feedback loop that degrades the model's own training labels, and
what a real deployment would require — silent-mode trial, randomised holdout,
IRB/QI determination, named clinical owner with the authority to switch it off.

---

## Two findings that were not planned

**The negative control failed, and that is the result.**
`distinct_prescribers_180d` is declared in the generator with a true coefficient
of exactly zero and never enters the risk equation. The fitted logistic gives it
β = -0.183. It is not fitting random noise — it is absorbing collinearity, since
prescriber count moves with medication-class count and with morbidity. The
consequence is concrete and is why `worklist.py` explains members with occlusion
rather than coefficients: **in a correlated claims feature set an individual
coefficient is not an effect size and must not be quoted to a clinician as one.**

**Logistic regression won.** AUROC 0.6519 vs 0.6467, sensitivity@5% 12.0% vs
11.6%. The challenger did not beat the incumbent, so the transparent model
ships. Payers deploying logistic for explainability are not being timid; on
claims features with this much collinearity there is often nothing to trade
away.

---

## What is missing (the other 80%)

Named specifically, because a list of what a project does is only half of an
honest one.

- **No serving API.** `worklist.py` writes a CSV. There is no scoring service,
  no batch scheduler, no model registry, no versioning of a deployed artefact.
- **No SHAP.** Not installed offline. Reason codes use occlusion-vs-median,
  documented at length in `worklist.py` as *not SHAP* — not additive, blind to
  interactions, no efficiency guarantee. Anything a member could appeal needs
  real Shapley values.
- **No real Synthea.** `src/generate.py` writes claims-shaped data directly.
  The docstring states what that costs: Synthea's trajectories come from curated
  disease modules with published provenance; these come from a risk equation I
  wrote, which is what makes recovery checkable and also what makes the
  clinical realism unearned.
- **No fairness analysis beyond calibration.** Subgroup O/E and AUROC are
  reported; equalised odds, calibration-within-groups over time, and the access
  bias that claims bake in are not addressed. The last is not fixable here at
  all — this generator has no access-inequity mechanism, so the analysis has
  nothing to find.
- **No monitoring implementation.** Failure modes and detection strategies are
  written down in the validation doc; none are instrumented.
- **No competing-risk handling.** Death after discharge censors the outcome and
  is treated here only by excluding in-hospital deaths. Post-discharge mortality
  is not modelled, which biases the 30-day label in the sickest stratum.
- **No hyperparameter search, no cross-validated model selection**, and no
  confidence intervals on any reported metric. The differences between logistic
  and GBM quoted above are inside the noise and are not claimed otherwise.
- **The grey-zone feature is unresolved.** Index principal diagnosis is used as
  though known at discharge; in reality the final coded diagnosis is assigned
  days later during coding. `features.py` flags this rather than fixing it.

## Files

| path | what |
|---|---|
| `src/codes.py` | ICD-10-CM / CPT / NDC tables, CCSR-style grouper |
| `src/comorbidity.py` | Charlson (Quan ICD-10) with hierarchy rules |
| `src/generate.py` | synthetic claims with runout, gaps, and a planted leak |
| `src/features.py` | cohort rules, `ClaimsView` temporal guard, feature builder |
| `train.py` | models, calibration, capacity metrics, subgroups, economics |
| `leakage_audit.py` | three experiments → `docs/LEAKAGE_AUDIT.md` |
| `worklist.py` | ranked worklist, plain-English drivers, the capacity conversation |
| `tests/test_guard.py` | 14 tests, all about the model not cheating |
