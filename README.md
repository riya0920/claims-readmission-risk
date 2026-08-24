# ML-1 — Claims-based readmission risk — working system, 9 known gaps

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
python run_batch.py                       # overnight batch + fairness report
python -m pytest tests -q                 # 106 tests, all about not cheating
python validate_reasons.py                # occlusion vs real SHAP -> docs/
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

## The batch path, and what a rerun must not do

`src/batch.py` + `run_batch.py`. The gap list said scoring was per-request and
"a real programme scores every discharge overnight and hands the care team a
queue in the morning, which is a scheduler and a job table, not an API."

A batch is not the API in a loop. Everything interesting about it is what
happens the **second** time:

- **A rerun of an identical job returns the ORIGINAL result.** A batch fails at
  03:00 and someone reruns it at 06:00; if the run is not idempotent the team
  gets two queues that disagree and no way to tell which one to work. Jobs are
  keyed on `(model, version, as_of, lookback, capacity, cohort_hash)`.
- **The cohort is part of the key.** Keying on `(model, as_of)` alone would
  return yesterday's answer after a late claim changed the population — the
  opposite failure from duplicates, and much harder to notice.
- **A job is succeeded or it is nothing.** The queue is written in one
  transaction. A job that dies after scoring 4,000 of 5,000 discharges must not
  leave 4,000 rows readable as a worklist: the team would work a queue silently
  missing the sickest fifth of the population, with nothing about it saying so.
- **Coverage is recorded** — who was actually called, against the queue that
  told them to call. On the demo run: **50 queued, 18 worked, 36% coverage**. A
  model with excellent sensitivity attached to a queue nobody works delivers
  nothing, and that failure is invisible in every metric `train.py` reports.

## Fairness beyond calibration — and a real gap in the model

`src/fairness.py`. The threshold comes from **capacity** (top 5%), not 0.5;
evaluating parity at 0.5 measures a decision rule nobody runs.

| group | n | prevalence | selected | TPR | FPR | **FNR** | PPV |
|---|---|---|---|---|---|---|---|
| 65+ | 3,134 | 20.8% | 10.3% | 20.2% | 7.7% | **79.8%** | 40.7% |
| under 65 | 5,692 | 14.3% | 2.1% | 5.5% | 1.5% | **94.5%** | 38.1% |

**A 14.7-point false-negative gap**, and the FNR is the criterion that matters
here: this model only *adds* outreach, so a false positive costs a phone call
and a false negative costs a member the call.

The module computes all three fairness criteria and **reports the conflict**.
Prevalence differs by 6.5 points across groups, so calibration, equalised odds
and equal selection cannot all hold (Kleinberg et al. 2016; Chouldechova 2016).
The impossibility is *checked* rather than asserted — citing a theorem whose
premise was never verified is its own error, and with equal prevalence it has
no force.

`recommend_criterion()` returns a criterion **plus the deployment assumption it
rests on**, and inverts if the score ever gates a benefit. That is the same
argument `docs/CLINICAL_VALIDATION.md` makes for the safety case: the property
belongs to the deployment, not the model.

Calibration within groups holds (max band gap 0.023 and 0.016) — reported
anyway, because "we did not check" and "we checked and it held" are different
claims and only one survives a question.

## Alerting with a named owner, and a runbook that is generated

`src/alerting.py` → [`docs/RUNBOOK.md`](docs/RUNBOOK.md). Delivery is ten lines;
the agreement underneath it is the part that was missing. Every alert carries an
owner who can act, a severity, a one-line instruction written for someone
reading it at 03:00, and a `silence_if`.

**That last field keeps the system alive.** `monitor.py` already found that
`elig_gap_days_365d` drifts hard (PSI 0.196) for a boundary artefact — real,
correct and permanent. An alert firing every run for a known reason trains its
audience to mute the channel, and then the one that matters arrives muted.

Only two alerts page, and **the most important one is for something *not*
happening**: the batch producing no queue. `monitor.py` cannot detect its own
failure to run, so that check is marked `external: True` rather than
implemented — a monitoring system whose absence is silent is not a control.

A cohort alert routes to the **programme manager, not data science**: a stage
whose retention moves is a population change, and sending it to the model owner
sends it to someone who cannot fix it.

## The service now authenticates, and logs who saw whom

The gap list called the unauthenticated endpoint "a HIPAA incident waiting to
happen". It is now behind a bearer token, with `/health` exempt — a load
balancer probing it holds no credential, and it returns no member data, which
is the only reason exempting it is safe.

**The access log is the half that matters.** Authentication answers "may you
ask"; only the log answers "which members were disclosed, to whom, when", and
that is the question an investigation asks. Denied attempts are logged too. A
consequence worth stating: the log contains member ids, so *the log is PHI*,
with the same handling requirements as the scores.

## The reason codes are measured against real SHAP

`worklist.py` computes its "top drivers" by OCCLUSION — set one feature to its
cohort median, re-score, record the drop — and its docstring has always said
this is **not SHAP**. That was honest but unquantified. SHAP is installed, so
`validate_reasons.py` measures the gap on the decision the worklist actually
makes.

Both methods are restricted to the **same candidate set**, so this compares
attribution methods rather than candidate filters. Over 200 members from the
top of the worklist:

| question | answer |
|---|---|
| same **top-1** driver | **68.0%** |
| **top-3 set** identical (order ignored) | **39.5%** |
| mean top-3 overlap | **2.38 of 3** |
| mean rank correlation across candidates | **0.893** |
| mean additivity violation | **0.187** probability |

Rank correlation is high while top-3 set agreement is not, and that is not a
contradiction: most candidates are ordered the same and **the disagreements
concentrate at the top** — which is exactly the part that gets displayed.

### The disagreement is systematic, and it points one way

| feature | occlusion calls it #1 | SHAP calls it #1 |
|---|---|---|
| `ip_days_365d` | 85 | 74 |
| `paid_amount_365d` | 61 | **94** |
| `charlson` | **26** | 5 |
| `age` | 18 | 13 |
| `los` | 3 | **12** |

**Occlusion over-credits `charlson` by 5× and under-credits prior spend.**
`charlson` is a composite comorbidity index correlated with the utilisation
features. Setting it alone to the cohort median leaves a member who is
comorbidity-free but expensive and frequently admitted — a combination that
exists nowhere in the data. The model's response to that off-manifold point
gets booked entirely to `charlson`, where Shapley values split the shared
credit.

This is the interaction-blindness the docstring predicted, appearing in the
direction it predicted.

### Why the direction matters more than the percentage

"High comorbidity burden" and "a lot of recent inpatient days" are not
interchangeable phrases — they suggest **different phone calls**, the first
pointing at disease management and the second at discharge follow-up and
access. A wrong ranking does not merely misattribute; it can misdirect the
outreach.

The audit does not upgrade occlusion into an appeal-grade explanation, and the
additivity violation of 0.187 is far too large to let these be read as "how
much this feature contributed". What changed is that the limitation is a number
instead of a promise.

## What is still missing, and why it cannot be closed here

Everything below needs something this environment does not have. The gaps that
were closeable have been closed; these are named with the specific blocker
rather than left as a to-do.

- **No real Synthea.** Blocked by a **missing Java runtime**, not by the
  network — an earlier version of this list said "no network", which was wrong.
  `src/generate.py` writes
  claims-shaped data directly, so the trajectories come from a risk equation I
  wrote — which is what makes recovery checkable and also what makes the
  clinical realism unearned.
- **The shipped reason codes are still occlusion, not SHAP** — and that is now
  a *measured* limitation rather than a disclaimed one (see the audit above).
  SHAP is installed; an earlier version of this list claimed it was not. The
  worklist keeps occlusion because `worklist.py` has no third-party dependency,
  and anything a member could appeal still needs real Shapley values.
- **No experiment tracking or artefact store.** `src/registry.py` is the
  minimum that makes serving honest — an artefact bound to its feature
  contract. No lineage to the training run, no staging/production promotion, no
  approval gates. **This one is not blocked**: MLflow is installed, and this is
  a scoping decision to keep `src/` dependency-free, listed here so it is not
  mistaken for something that could not be done.
- **No scheduler.** `run_batch.py` runs one job when invoked; something else
  has to invoke it, and "the batch did not run at all" is the failure that most
  needs monitoring. It is in the policy table as `external: True` precisely
  because it cannot live here.
- **Access bias is not analysable with this data, and the reason is not that
  the analysis is missing.** Claims measure care *received*, so a population
  with worse access looks healthier to any model fitted on claims. But
  `src/generate.py` has no access-inequity mechanism — every member's
  utilisation is drawn from their true risk. An analysis reporting "no access
  bias detected" would be reporting a property of the generator and passing it
  off as a property of the model. Detecting it needs an external measure of
  health that does not come through the claims pipeline, which is a
  data-acquisition problem.
- **No competing-risk handling.** Death after discharge censors the outcome and
  is handled only by excluding in-hospital deaths. Post-discharge mortality is
  not in the generator, so as with access bias there is nothing here to
  recover; `data3-trial-survival` in this portfolio has the Aalen-Johansen and
  Fine-Gray machinery this would need.
- **Single-node, single-process.** No TLS, no rate limiting, no connection
  pooling. The bearer token authenticates a caller, not a user, and
  `se1-hl7-fhir-interop` has the SMART scope layer and append-only audit this
  service should really write to. They are not wired together.
- **No hyperparameter search and no cross-validated model selection.** The
  models are compared at fixed settings. Deliberate rather than blocked: § 4
  showed the observed gap is smaller than the evaluation can resolve, so tuning
  would produce a more precise answer to a question the data cannot settle.
- **The grey-zone feature is unresolved.** Index principal diagnosis is used as
  though known at discharge; in reality the final coded diagnosis is assigned
  days later during coding. `features.py` flags this rather than fixing it,
  because fixing it needs a coding-lag distribution the generator does not have.

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
| `src/batch.py` | job table, idempotent reruns, all-or-nothing queue writes |
| `src/fairness.py` | error-rate parity, the impossibility check, the criterion choice |
| `src/alerting.py` | thresholds with owners and silence rules; generates the runbook |
| `run_batch.py` | the overnight batch, and the fairness report |
| `docs/RUNBOOK.md` | generated from the policy, so the two cannot drift |
| `validate_reasons.py` | occlusion vs shap.TreeExplainer; found the charlson bias |
| `tests/test_reason_audit.py` | 6 tests pinning the SIZE of the gap, not its absence |
| `tests/test_complete.py` | 33 tests: reruns, partial failure, parity, alert routing, auth |
| `tests/test_guard.py` | 20 tests: not cheating, and not overstating precision |
| `tests/test_serving.py` | 27 tests: the feature contract and the drift detectors |
| `tests/test_api.py` | 20 tests: the refusals, the windowing, the in-sample flag |
