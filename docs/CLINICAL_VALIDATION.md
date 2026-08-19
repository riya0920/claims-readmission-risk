# Clinical validation and intended use

This document exists because the model in this repository would be dangerous if
someone believed the wrong things about it. It states what the model does, what
it does not do, what it gets wrong and for whom, and what would have to be true
before it touched a real member.

All figures are from `train.py` on the dataset in `data/` (50,000 synthetic
members, 30,868 index discharges, 8,826 in the temporal test set). **No real
patient data was used at any point.** See "Data provenance" below for why that
is a limitation as well as a safeguard.

---

## 1. Intended use

**The model produces a ranking of recent inpatient discharges by the modelled
probability that the member will generate an unplanned inpatient readmission
claim within 30 days, so that a care-management team with fixed outreach
capacity can decide which members to call first.**

That sentence is deliberately narrow. Each clause is load-bearing:

- **a ranking** — the output is an ordering for a queue, not a decision.
- **inpatient discharges** — members with no recent index admission are not
  scored and are not in scope.
- **an unplanned readmission *claim*** — the label is an administrative event.
  It is a proxy for "the member deteriorated", and a lossy one. A member who
  deteriorates and is treated in the ED without admission is a negative. A
  member admitted for an unrelated reason is a positive.
- **within 30 days** — a window chosen because it is the payer and CMS
  convention, not because 30 days is clinically meaningful.
- **fixed outreach capacity** — the model is only useful where capacity is the
  binding constraint. If a team can call everyone, it should call everyone and
  this model has no job.

### Not intended use

The model must not be used to:

- **make or support a clinical diagnosis.** It has no clinical findings, no
  vitals, no labs, no notes. It knows billing codes.
- **deny, delay, limit, or price coverage, benefits, or care.** Nothing here
  was designed or validated for a utilisation-management or coverage decision,
  and using a readmission-risk score that way inverts its purpose: it was built
  to route *more* attention to members, not less.
- **rank members for anything other than outreach**, including discharge
  planning, level-of-care decisions, or provider profiling.
- **predict mortality, deterioration, or any clinical outcome.** In-hospital
  deaths are excluded from the cohort entirely, so the model has never seen
  them and its behaviour on that population is undefined.
- **be shown to a member as a statement about their health.**

---

## 2. What the model is, mechanically

| | |
|---|---|
| Unit of prediction | one inpatient discharge (not one member) |
| Scoring instant | the moment of discharge |
| Inputs | claims history in the 365 days **before admission**, restricted to claims **received** by the discharge date; plus index-stay attributes from the ADT/census feed (LOS, working diagnosis) and member demographics |
| Output | a calibrated probability in [0,1] |
| Model | logistic regression (shipped) / gradient boosting (challenger) |
| Refresh | retrained per the monitoring plan in §8; not self-updating |

### Why logistic regression ships

| | logistic | gradient boosting |
|---|---|---|
| AUROC | 0.6519 | 0.6467 |
| AUPRC | 0.2890 | 0.2850 |
| Brier | 0.1319 | 0.1323 |
| calibration slope | 0.902 | 1.037 |
| sensitivity @ top 5% | 12.0% | 11.6% |

The gradient-boosted model is not better. On this data it is very slightly
worse on discrimination and marginally better calibrated in slope. When the
challenger does not win, the transparent model ships — a payer can explain a
logistic model to a clinical director, a compliance reviewer, and a regulator,
and that is worth more than a difference well inside the noise. This is a
finding, not a preference: had the GBM won by a margin that mattered, the
recommendation would have been the GBM plus a plan for explaining it.

---

## 3. Performance, stated the way it will be used

Discrimination first, because it is conventional, and then the numbers that
actually govern the programme.

| metric | value |
|---|---|
| AUROC | 0.652 |
| AUPRC | 0.289 (prevalence 16.6%) |
| Brier | 0.132 |
| calibration slope / intercept | 0.902 / -0.147 |

AUROC 0.65 is unremarkable, and it is roughly where honest claims-only
readmission models sit. Models that report substantially more than this from
claims alone are usually either using post-discharge information or evaluating
on a population that includes in-hospital deaths. The audit in
`docs/LEAKAGE_AUDIT.md` shows this same pipeline reaching AUROC 0.904 by adding
one field it should not have.

### At capacity — the operating numbers

| capacity | members flagged | readmissions in the flagged set | sensitivity | PPV | NNC |
|---|---|---|---|---|---|
| top 1% | 88 | 53 | 3.6% | 60.2% | 1.7 |
| top 2% | 177 | 89 | 6.1% | 50.3% | 2.0 |
| **top 5%** | **441** | **176** | **12.0%** | **39.9%** | **2.5** |
| top 10% | 883 | 292 | 19.9% | 33.1% | 3.0 |

Read the top 5% row as: *of the 441 members we can call, about 176 of them
would have been readmitted, and we contact about 2.5 members for each
readmission in the list.*

---

## 4. False-negative cost analysis

This is the section that matters most and it is the one most often missing.

At top-5% capacity, **1,290 of the 1,466 readmissions in the test set (88%)
occur among members the model did not flag.** That is not a model defect. It is
arithmetic: 1,466 readmissions cannot fit inside 441 phone calls. But it has
consequences that must be stated to whoever signs off:

**The cost of a false negative here is the cost of NOT ADDING a service.** A
missed member receives exactly the care they would have received without the
model. There is no withheld treatment, no delayed diagnosis, no false
reassurance recorded in a chart. This asymmetry is the single most important
safety property of the design, and it is a property of the *deployment*, not of
the model: it holds only while the score is used to add outreach. The moment
the same score is used to gate anything — a benefit, an authorisation, a
placement — a false negative becomes a denial of service to a high-risk member,
and every risk statement in this document becomes void.

**The residual danger is a clinician's mental model.** If discharge planners
learn that "the algorithm flags the risky ones", an unflagged member can
receive less scrutiny than they would have before the model existed. That is
how an additive tool becomes a subtractive one without anybody changing the
code. Mitigations, all of which are process rather than software:

1. the worklist states on every page that unflagged members have not been
   cleared of risk (implemented in `worklist.py`);
2. the model never appears in the discharge-planning workflow, only in the
   care-management queue;
3. existing risk protocols continue to run unchanged and are audited for drift
   in referral volume after go-live.

**False positives** cost a phone call — about $65, plus the member's time and
some nuisance. At top-5% PPV of 39.9%, roughly 3 in 5 calls go to a member who
would not have been readmitted. For an outreach intervention this is a normal
and acceptable ratio. For anything invasive or costly it would not be, which is
another way of saying the operating point is tied to the intervention and must
be re-derived if the intervention changes.

---

## 5. Subgroup performance and where it fails

Calibration is good overall and holds across age and comorbidity bands
(observed/expected between 0.96 and 1.05). One subgroup does not:

| subgroup | n | observed | predicted | O/E | AUROC |
|---|---|---|---|---|---|
| coverage gap in past year | 115 | 21.7% | 17.2% | **1.26** | 0.617 |
| no coverage gap | 8,711 | 16.5% | 16.5% | 1.00 | 0.652 |
| age 18-35 | 833 | 11.2% | 11.5% | 0.97 | 0.609 |
| age 81+ | 846 | 24.7% | 24.7% | 1.00 | 0.639 |
| Charlson 0 | 4,982 | 12.4% | 13.0% | 0.96 | 0.612 |
| Charlson 5+ | 116 | 37.9% | 37.7% | 1.01 | 0.646 |

**The model under-predicts risk for members with a coverage gap by about 26%
relative.** In a capacity-bounded worklist, systematic under-prediction means
systematic under-representation: this group is admitted to the call list less
often than its true risk warrants. Members with coverage gaps are, on the
available evidence, more likely to be exactly the people who benefit from a
phone call — they are harder to reach, more likely to have lost a usual source
of care, and more likely to be lost to follow-up.

Three honest caveats: n=115 is small and the interval is wide; the cohort rule
(one enrolment gap of ≤45 days, HEDIS convention) has already removed the most
severely churning members before the model sees them; and this is synthetic
data, so the finding demonstrates that the *analysis* would catch such a
disparity, not that this disparity exists in any real population.

The remedy is not to ship and monitor. It is to decide, before go-live, whether
to recalibrate within this stratum or to reserve a fixed share of outreach
capacity for it — and that decision belongs to the clinical director, not the
modeller.

---

## 6. Data provenance and what it invalidates

The data is **synthetic**, generated by `src/generate.py`. No real member data,
no PHI, no limited data set, no BAA, no IRB. That is the correct posture for a
portfolio project and it is also a hard ceiling on every claim here.

What synthetic data cannot tell us, in descending order of importance:

1. **Whether the features exist as assumed.** Real claims have duplicate
   submissions, adjustment and void pairs, coordination of benefits with a
   second payer, coding drift across contract years, and providers whose
   ICD-10 habits change when the EHR is replaced. None of that is emitted here.
2. **Whether the model would be fair.** Claims measure *care received*, not
   *illness*. Access, insurance design, referral patterns, and clinician
   behaviour are all baked into the substrate, so a claims-trained model learns
   who has been *seen* as much as who is *sick*. Members with less access
   generate fewer claims and can therefore look lower-risk while being higher
   risk. That bias cannot be detected in synthetic data, because this generator
   has no access-inequity mechanism in it.
3. **Whether the effect sizes transfer.** The coefficients here recover a
   risk equation that was written down, not discovered.
4. **What the drift looks like.** No pandemic, no benefit redesign, no
   provider-network change, no new HCC coding initiative.

Consequently: **no number in this repository is evidence about any real
population, and none of them should be quoted as such.**

---

## 7. Known failure modes

| failure | mechanism | detection |
|---|---|---|
| Claim runout shifts | payer changes clearinghouse; receipt lag distribution moves | monitor feature means at scoring time vs training; see `docs/LEAKAGE_AUDIT.md` §3 |
| Coding intensity change | risk-adjustment coding initiative raises recorded comorbidity without changing health | Charlson distribution drift with flat outcome rate |
| Cohort drift | contract or network change alters the admitted population | monitor the cohort waterfall counts, not just model metrics |
| Feedback loop | outreach reduces readmission for flagged members, so the model's own success degrades its apparent performance and its future training labels | hold out a randomised control fraction of the worklist from go-live; this is the only reliable answer |
| Grey-zone feature drift | index working diagnosis diverges from final coded principal diagnosis | compare working vs final dx agreement quarterly |

The feedback loop deserves emphasis: once outreach works, the training labels
stop describing the untreated population. A permanently randomised holdout is
not a nicety, it is the only way the model can continue to be evaluated at all.

---

## 8. What a real deployment would require

Nothing in this repository is deployable. In rough order:

**Before any real data**
- IRB or QI determination. Operational quality improvement and human-subjects
  research have different obligations, and which one this is depends on intent
  and dissemination plan, not on the code. The determination is made by the
  IRB, not the modeller.
- Minimum-necessary review of the feature set; a BAA where a vendor is
  involved; documented access controls and an audit trail on the scoring data.
- A written data-use agreement covering the retention of scores, which are
  themselves member-level health information.

**Before any clinical contact**
- **A silent-mode trial.** Score prospectively, produce the worklist, show it
  to nobody, and compare predicted against observed for at least one full
  outcome cycle. Silent mode is where claim-runout skew, cohort drift, and
  integration defects surface, and none of them are visible in backtest.
- Clinician review of a sample of flagged and unflagged members, with attention
  to whether the reasons shown are clinically coherent.
- Re-derivation of the operating point against the *actual* staffed capacity.

**At go-live**
- A randomised holdout fraction, permanently.
- Subgroup calibration monitoring, with the coverage-gap stratum in §5 named
  explicitly as a pre-specified monitoring target.
- A named clinical owner with the authority to switch it off, and a documented
  procedure for doing so.

**Standing**
- Scheduled recalibration; drift monitoring on inputs, not just outputs;
  annual review of intended use against actual use, because scope creep from
  "outreach prioritisation" to "utilisation management" is the realistic path
  by which this becomes harmful.

---

## 9. Summary judgement

The model ranks discharges better than chance and about as well as claims-only
readmission models generally do. Used as specified — to order a capacity-bound
outreach queue, with unflagged members explicitly not cleared — its worst
realistic outcome is a wasted phone call. Used to gate anything, it is
inappropriate and unvalidated.

It has been shown to be free of the specific leaks tested for, calibrated
overall, and miscalibrated for one identified subgroup. It has not been shown
to work on real data, to be fair, or to change any outcome.
