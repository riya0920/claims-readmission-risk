# ML-1 — Claims-based readmission risk (~80% build)

**This is still not a deployable system.** It is the load-bearing part of one:
the claims feature craft, the temporal discipline, the intervention framing,
and now a scoring service with a model registry that refuses to serve on a
schema mismatch. The fairness work beyond calibration, the real data, and every
operational commitment that makes a model safe rather than merely correct are
not here, and [§ What is missing](#what-is-still-missing) says so specifically.

Predicts unplanned 30-day readmission from synthetic payer claims, scored at
the moment of discharge, and evaluates the result the way a care-management
programme is actually run: at capacity.

```bash
python src/generate.py --members 50000    # ~4 min, writes data/
python train.py                           # models, calibration, subgroups, economics, CIs
python leakage_audit.py                   # 3 experiments -> docs/LEAKAGE_AUDIT.md
python monitor.py                         # drift panels + an injected failure
python worklist.py --capacity 500         # the care-manager artefact
python serve.py --register                # fit and register a servable artefact
python serve.py --demo                    # exercise the API end to end
python serve.py                           # scoring service on :8081
python -m pytest tests -q                 # 67 tests, all about not cheating
```

Everything runs offline. Runtime end to end is about ten minutes, most of
it the bootstrap.

---

## The seven things worth reading

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

### 4. Every number has an interval, and the model comparison is a test

The first build reported logistic AUROC 0.6519 against GBM 0.6467, called the
difference "inside the noise", and did not quantify the noise. That is a hedge,
not a result. Now:

| metric | logistic | gradient boosting |
|---|---|---|
| AUROC | 0.6519 (0.6358–0.6661) | 0.6467 (0.6300–0.6619) |
| AUPRC | 0.2890 (0.2641–0.3136) | 0.2850 (0.2609–0.3093) |
| sensitivity @ 5% | 12.0% (10.5–13.5%) | 11.6% (10.1–12.9%) |
| PPV @ 5% | 39.9% (34.5–44.8%) | 38.5% (33.0–43.1%) |

Intervals are a **cluster bootstrap resampled by member**, not by stay: a member
contributes several correlated index admissions, and resampling rows
independently treats them as independent evidence and produces intervals that
are too narrow. `test_clustered_interval_is_wider_than_the_naive_one` fails if
the clustering ever stops doing anything.

**Is logistic actually better?** Answered by a *paired* bootstrap on the
difference — **not** by checking whether the two intervals above overlap. Both
models are scored on the same patients, so their errors are correlated and the
difference is far more stable than either level; comparing intervals
systematically understates evidence for a difference.

| metric | logistic − GBM | 95% CI | p | verdict |
|---|---|---|---|---|
| AUROC | +0.0052 | (−0.0013, +0.0113) | 0.133 | indistinguishable |
| AUPRC | +0.0040 | (−0.0039, +0.0110) | 0.297 | indistinguishable |
| sens @ 5% | +0.4% | (−0.5%, +1.5%) | 0.330 | indistinguishable |

And the question that should have come *first*:

> Smallest AUROC difference this test set could resolve: **~0.0170**
> Observed difference: **0.0052**

**The observed gap is smaller than the resolution of the evaluation.** This
comparison was never going to settle anything, and picking a winner from it
would have been theatre. The recommendation to ship logistic rests on
transparency, not on the AUROC — which is what the first build claimed, now
with the arithmetic to back it.

### 5. Drift monitoring, and a monitor that was made to fail

`CLINICAL_VALIDATION.md` lists seven failure modes and how each would be
detected. `monitor.py` implements the detection, on the principle that
**outcome-based monitoring is too slow to be a safety control**: the 30-day
label is unobservable for 30 days and unreliable for ~90 more because of
runout, so a model that breaks in January is caught by outcome monitoring in
April.

Four panels: per-feature PSI against the training reference; **claim-runout lag
by quarter** (the canary — a clearinghouse change shifts every recency feature
at once and the model cannot see it); calibration O/E overall and for the
coverage-gap stratum the validation doc pre-specified; and the **cohort
waterfall itself**, because a contract change that alters who gets admitted
moves the denominator without touching a single model metric.

All four are green on this data, by construction — the generator uses one lag
distribution throughout. **A dashboard that has only ever shown green is not
evidence that the monitor works.** So the panel injects a +21-day clearinghouse
shift and makes the detectors catch it, which produced the most useful result
in the project:

| what moved | PSI |
|---|---|
| the receipt-lag distribution itself | **6.06** |
| `office_visits_90d` (worst affected feature) | 0.036 |
| `ed_visits_90d` | 0.015 |
| `ip_admits_365d` | 0.0008 |

**The per-feature PSI does not catch it.** The worst feature lands ~7× below
the 0.25 "investigate" threshold and below the 0.10 "watch" threshold too. A
standard PSI dashboard would have shown all green through a train/serve skew
severe enough to change what the recency features *mean* — the exact failure
`docs/LEAKAGE_AUDIT.md` measures. Monitoring the mechanism answers 168× louder
than monitoring the symptom, and that ratio was measured here rather than
asserted.

Two bugs the monitoring code found in itself, both of which are the reassuring
kind:

- **PSI silently returned 0.0 on zero-inflated features.** Most utilisation
  features here are counts that are 85–99% zero. Every decile edge lands on 0,
  the edges deduplicate to one, and the natural implementation reports "no
  shift" for a feature whose mass point could have moved from 85% to 99%.
  `psi()` now has a discrete-bin path and a mass-point fallback. Found by a
  test, not by reading the code.
- **A waterfall stage retained 117% of what it received**, and got written up
  as a real conditional effect before anyone checked the arithmetic. That came
  from a simulation that scaled stages independently. `cohort_drift()` now
  refuses any retention above 100% outright. Monitoring code needs its own
  sanity checks: a plausible number from an impossible computation is the
  easiest thing in this project to believe.

There is also a real drift finding: `elig_gap_days_365d` moves hard (PSI 0.196,
mean −87%), and it is a **boundary artefact** — coverage gaps fall inside a
fixed data window, so discharges late in the window have fewer gaps in their
365-day lookback. Distinguishing an artefact from real drift is the whole
skill; the panel reports the mean change beside the PSI so that call can be
made.

The doc is also explicit about what monitoring **cannot** see: the feedback
loop (once outreach works, the labels stop describing the untreated population
and every panel still looks healthy), label maturity, and access bias.

### 6. The humility document

[`docs/CLINICAL_VALIDATION.md`](docs/CLINICAL_VALIDATION.md) — intended use and
explicitly not-intended use, the false-negative cost analysis (the asymmetry
that makes this safe is a property of the *deployment*, not the model, and it
evaporates the moment the score gates anything), the identified subgroup
miscalibration, what synthetic data cannot tell us, seven named failure modes
including the feedback loop that degrades the model's own training labels, and
what a real deployment would require — silent-mode trial, randomised holdout,
IRB/QI determination, named clinical owner with the authority to switch it off.

### 7. A service that refuses to answer

`serve.py` is a stdlib HTTP scoring service — `GET /health`, `GET /model`,
`GET /worklist?as_of=…&capacity=N`, `POST /score`. The interesting parts are
the three places it declines.

**It refuses without an explicit `as_of`.** Both scoring routes return 400
rather than defaulting to "now". This is the whole project's thesis expressed
as a status code: a worklist is a snapshot of what had been *received* at a
moment, and silently defaulting the clock reintroduces exactly the train/serve
skew `docs/LEAKAGE_AUDIT.md` measures.

**It refuses on a feature-schema mismatch.** `src/registry.py` stores every
model with the ordered feature list it was fitted on, an order-*sensitive*
hash, the visibility mode (`received` vs complete history), and the metrics
with their intervals. `load()` returns a `ServableModel` that raises
`SchemaMismatch` rather than score. The case that justifies the hash is not a
missing column — it is the **same columns in a different order**, where a set
comparison passes, nothing raises, and `charlson` is scored through the
coefficient for `age`. That case has its own test and its own error message.

**Every response carries its own intended use**, including the sentence saying
the score must not be used to deny, delay, limit or price coverage, and that
members *not* returned have not been cleared of risk. Governance that lives
only in a README is governance the integrating team never reads.

Two bugs came out of running the service rather than reading it:

- **`/worklist` was scoring the training data.** The first version returned
  every discharge up to `as_of` — 23,402 stays — with a top risk of 98.4%.
  Those were in-sample fits being served as predictions. Fixed with a 30-day
  window (matching the outcome window, and matching what transitional-care
  outreach can actually act on) plus a per-row `in_training_set` flag and a
  response-level warning for any window that reaches back past the training
  cut.
- **An empty window returned a different response shape**, so a client reading
  `cohort_size` got a `KeyError` on a quiet day instead of a zero. An empty
  result is a normal answer and has to have a normal shape.

The 20 API tests start a real server on an ephemeral port and talk to it over
HTTP, because routing, status codes and JSON encoding are part of what is being
claimed and calling the handler methods in-process tests none of them.

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

## What is still missing

Named specifically, because a list of what a project does is only half of an
honest one.

- **The registry is not a registry.** `src/registry.py` is the minimum that
  makes serving honest: an artefact bound to its feature contract. It is not
  MLflow — no experiment tracking, no artefact store, no lineage back to the
  training run, no staging/production promotion, no approval gates, and nothing
  stopping someone registering v2 over v1.
- **The service is single-process and unauthenticated.** One stdlib
  `HTTPServer`, no auth, no rate limiting, no TLS, no request audit log — and
  an unauthenticated endpoint returning member-level risk scores is a HIPAA
  incident waiting to happen. `se1-hl7-fhir-interop` in this portfolio has the
  append-only PHI access audit this service should be writing to; they are not
  wired together.
- **No batch path.** Scoring is per-request. A real programme scores every
  discharge overnight and hands the care team a queue in the morning, which is
  a scheduler and a job table, not an API.
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
- **No monitoring *deployment*.** `monitor.py` computes the panels and
  `run_all()` emits alert strings; there is no scheduler, no alert *delivery*,
  no thresholds agreed with a named owner, and no runbook saying who does what
  when one fires. The thresholds in the code are credit-scoring rules of thumb
  I chose, which is precisely the part that has to be negotiated with the
  people who will be paged.
- **No competing-risk handling.** Death after discharge censors the outcome and
  is treated here only by excluding in-hospital deaths. Post-discharge mortality
  is not modelled, which biases the 30-day label in the sickest stratum.
- **No hyperparameter search and no cross-validated model selection.** The
  models are compared at fixed settings; a real bake-off tunes both.
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
| `src/uncertainty.py` | cluster bootstrap, paired model comparison, minimum detectable difference |
| `src/monitor.py` | drift detectors: PSI (with the zero-inflation fix), calibration, runout, cohort |
| `monitor.py` | the drift dashboard, plus the injected clearinghouse failure |
| `src/registry.py` | model + feature contract; refuses to serve on a schema mismatch |
| `serve.py` | stdlib scoring API; `--register` to fit, `--demo` to exercise |
| `tests/test_guard.py` | 20 tests: not cheating, and not overstating precision |
| `tests/test_serving.py` | 27 tests: the feature contract and the drift detectors |
| `tests/test_api.py` | 20 tests: the refusals, the windowing, the in-sample flag |
