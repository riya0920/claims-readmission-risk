# Reason codes: occlusion vs real Shapley values

`worklist.py` computes its "top drivers" column by OCCLUSION -- set one feature
to its cohort median, re-score, record the drop. Its docstring has always said
this is **not SHAP**, and listed how it differs. That was honest but
unquantified, and the README made it worse by claiming SHAP was "not
installed". It is. So here is the measurement.

`shap.TreeExplainer` gives exact Shapley values for this model. Both methods
are restricted to the **same candidate set** -- features in the explainable
vocabulary that differ from the cohort median for that member -- so this
compares attribution methods, not candidate filters.

## Results over 200 members from the top of the worklist

| question | answer |
|---|---|
| same **top-1** driver | **68.0%** |
| **top-3 set** identical (order ignored) | **39.5%** |
| mean top-3 overlap | **2.38 of 3** |
| mean rank correlation across candidates | **0.893** |
| mean additivity violation | **0.1870** probability |

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

| feature | occlusion calls it #1 | SHAP calls it #1 |
|---|---|---|
| `ip_days_365d` | 85 | 74 |
| `paid_amount_365d` | 61 | 94 |
| `charlson` | 26 | 5 |
| `age` | 18 | 13 |
| `los` | 3 | 12 |
| `ccsr_CIR019` | 2 | 1 |
| `distinct_providers_180d` | 3 | 0 |
| `rx_classes_180d` | 1 | 0 |

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
