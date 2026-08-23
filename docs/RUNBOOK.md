# Alert runbook

**Generated from `src/alerting.py`. Do not edit by hand.**

Every alert has an owner who can act, a severity that is a claim about somebody's night, and a one-line instruction written for someone reading it at 03:00.

Most alerts are a **digest** or a **ticket**. Only two page, and one of those is for something *not happening* — see the last row.

## `input_drift` — TICKET

**Fires when.** PSI >= 0.25 on a watched feature

**Owner.** model owner (data science)

**Do this.** check whether the mean moved with the PSI. A PSI of 0.12 on a feature whose mean moved 2% is a statistic, not a problem. If both moved, check claim runout FIRST.

**Do NOT alert when.** feature is elig_gap_days_365d -- it drifts by construction (coverage gaps fall inside a fixed data window, so late discharges have fewer gaps in their 365-day lookback). Permanent, correct, and not news.

## `claim_runout` — TICKET

**Fires when.** median receipt lag moves >= 5 days vs the reference

**Owner.** claims operations, with the model owner informed

**Do this.** confirm with claims ops whether a clearinghouse or submitter changed. If yes, the model's recent-utilisation features are measuring something different from what it was fitted on -- suppress the worklist rather than serve a degraded one, and refit against the new lag.

## `calibration` — DIGEST

**Fires when.** O/E outside 0.90-1.10 on matured labels

**Owner.** model owner (data science)

**Do this.** check label maturity before anything else. A recent period always looks under-predicted because the outcomes have not arrived yet; O/E on a period with less than 90 days of runout is not evidence of anything.

**Do NOT alert when.** the evaluated period has < 90 days of runout

## `cohort` — TICKET

**Fires when.** any waterfall stage retention moves > 5 points

**Owner.** programme manager, not data science

**Do this.** this is a POPULATION change, not a model change -- a contract, network or enrolment-rule change altered who is admitted. The model is fine and the denominator is not. Route to whoever owns the population.

## `cohort_inconsistent` — TICKET

**Fires when.** a waterfall stage retains more than 100%

**Owner.** model owner (data science)

**Do this.** the two count sets did not come from one filter chain. Do not interpret any retention change in the same report -- they are arithmetic on incompatible inputs.

## `schema_mismatch` — PAGE

**Fires when.** the served feature contract does not match the manifest

**Owner.** model owner (data science)

**Do this.** the service is already refusing to score, which is the correct behaviour -- there is no worklist rather than a wrong one. Do NOT 'fix' it by relaxing the check. Find what changed in the feature builder.

## `batch_did_not_run` — PAGE

**Fires when.** no succeeded batch job for today's as_of by 07:00

**Owner.** programme manager and on-call engineer

**Do this.** the care team has no queue this morning. Rerun with yesterday's as_of -- run_batch.py is idempotent on identical inputs, so a rerun cannot produce a second queue that disagrees with one already being worked.

> **This check cannot live here.** `monitor.py` cannot detect its own failure to run, and a monitoring system whose absence is silent is not a control. Something outside this repository has to assert that the batch produced a queue.
