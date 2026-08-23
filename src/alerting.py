"""Alert routing and thresholds with a named owner.

WHAT THE README SAID WAS MISSING
--------------------------------
"`monitor.py` computes the panels and `run_all()` emits alert strings; there is
no scheduler, no alert *delivery*, no thresholds agreed with a named owner, and
no runbook saying who does what when one fires. The thresholds in the code are
credit-scoring rules of thumb I chose, which is precisely the part that has to
be negotiated with the people who will be paged."

THE PART THAT IS ACTUALLY MISSING IS NOT THE CODE
--------------------------------------------------
A delivery mechanism is ten lines. What makes alerting work is the agreement
underneath it, and an alert with no owner is a notification that gets muted. So
the threshold table below carries, for every alert:

    owner        a role that can act, not a mailing list
    severity     page / ticket / digest -- and MOST alerts are digest
    runbook      what to do, in one line, at the moment of being woken
    silence_if   the condition under which this alert is EXPECTED and should
                 not fire at all

That last field is the one that keeps an alerting system alive. `monitor.py`
already found that `elig_gap_days_365d` drifts hard (PSI 0.196) for a boundary
artefact -- a real, correct, permanent property of a fixed data window. An
alert that fires every single run for a known reason trains its audience to
ignore the channel, and then the one that matters arrives in a muted channel.

SEVERITY IS A CLAIM ABOUT SOMEBODY'S NIGHT
-------------------------------------------
Only one alert here pages: the batch not running at all. Everything else is a
ticket or a digest, because nothing else here is both urgent and actionable at
03:00. A drifting feature at 3am is the same problem at 9am, and waking someone
for it is how the page gets ignored the week it matters.

Note what that implies: THE MOST IMPORTANT ALERT IS FOR SOMETHING NOT
HAPPENING. `monitor.py` cannot detect its own failure to run, and a monitoring
system whose absence is silent is not a control. That check has to live
outside, which is why it is listed here with `external: True` rather than
implemented.

WHAT THIS IS NOT
----------------
No PagerDuty, no Slack, no email, no SMS -- `Router` takes a sink callable and
the sinks here are a list and a print. No deduplication window, no flap
detection, no escalation policy, no on-call rota, no acknowledgement tracking,
and no scheduler to run any of it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

PAGE, TICKET, DIGEST = "page", "ticket", "digest"

POLICY = {
    "input_drift": {
        "threshold": "PSI >= 0.25 on a watched feature",
        "owner": "model owner (data science)",
        "severity": TICKET,
        "runbook": ("check whether the mean moved with the PSI. A PSI of 0.12 "
                    "on a feature whose mean moved 2% is a statistic, not a "
                    "problem. If both moved, check claim runout FIRST."),
        "silence_if": ("feature is elig_gap_days_365d -- it drifts by "
                       "construction (coverage gaps fall inside a fixed data "
                       "window, so late discharges have fewer gaps in their "
                       "365-day lookback). Permanent, correct, and not news."),
    },
    "claim_runout": {
        "threshold": "median receipt lag moves >= 5 days vs the reference",
        "owner": "claims operations, with the model owner informed",
        "severity": TICKET,
        "runbook": ("confirm with claims ops whether a clearinghouse or "
                    "submitter changed. If yes, the model's recent-utilisation "
                    "features are measuring something different from what it "
                    "was fitted on -- suppress the worklist rather than serve "
                    "a degraded one, and refit against the new lag."),
        "silence_if": None,
    },
    "calibration": {
        "threshold": "O/E outside 0.90-1.10 on matured labels",
        "owner": "model owner (data science)",
        "severity": DIGEST,
        "runbook": ("check label maturity before anything else. A recent "
                    "period always looks under-predicted because the outcomes "
                    "have not arrived yet; O/E on a period with less than 90 "
                    "days of runout is not evidence of anything."),
        "silence_if": "the evaluated period has < 90 days of runout",
    },
    "cohort": {
        "threshold": "any waterfall stage retention moves > 5 points",
        "owner": "programme manager, not data science",
        "severity": TICKET,
        "runbook": ("this is a POPULATION change, not a model change -- a "
                    "contract, network or enrolment-rule change altered who is "
                    "admitted. The model is fine and the denominator is not. "
                    "Route to whoever owns the population."),
        "silence_if": None,
    },
    "cohort_inconsistent": {
        "threshold": "a waterfall stage retains more than 100%",
        "owner": "model owner (data science)",
        "severity": TICKET,
        "runbook": ("the two count sets did not come from one filter chain. "
                    "Do not interpret any retention change in the same report "
                    "-- they are arithmetic on incompatible inputs."),
        "silence_if": None,
    },
    "schema_mismatch": {
        "threshold": "the served feature contract does not match the manifest",
        "owner": "model owner (data science)",
        "severity": PAGE,
        "runbook": ("the service is already refusing to score, which is the "
                    "correct behaviour -- there is no worklist rather than a "
                    "wrong one. Do NOT 'fix' it by relaxing the check. Find "
                    "what changed in the feature builder."),
        "silence_if": None,
    },
    "batch_did_not_run": {
        "threshold": "no succeeded batch job for today's as_of by 07:00",
        "owner": "programme manager and on-call engineer",
        "severity": PAGE,
        "runbook": ("the care team has no queue this morning. Rerun with "
                    "yesterday's as_of -- run_batch.py is idempotent on "
                    "identical inputs, so a rerun cannot produce a second "
                    "queue that disagrees with one already being worked."),
        "silence_if": None,
        "external": True,
    },
}


class Router:
    """Route alerts by severity to sinks, applying the silence rules.

    A sink is any callable taking one dict. Nothing here knows about PagerDuty
    or email, deliberately -- the interesting content is the POLICY table, and
    binding it to a vendor would bury that under an integration.
    """

    def __init__(self, sinks=None, policy=None):
        self.policy = policy or POLICY
        self.sinks = sinks or {}
        self.sent = []
        self.silenced = []

    def _kind(self, alert_text):
        """Map a monitor.run_all() alert string to a policy entry."""
        t = alert_text.lower()
        if t.startswith("input drift"):
            return "input_drift"
        if t.startswith("claim runout"):
            return "claim_runout"
        if t.startswith("calibration"):
            return "calibration"
        if "inconsistent" in t:
            return "cohort_inconsistent"
        if t.startswith("cohort"):
            return "cohort"
        if "schema" in t:
            return "schema_mismatch"
        return None

    def _is_silenced(self, kind, alert_text):
        rule = self.policy.get(kind, {}).get("silence_if")
        if not rule:
            return None
        # The one rule that can be evaluated mechanically. The others are
        # conditions a human confirms, and are printed as guidance rather than
        # applied -- a silence rule applied automatically on a condition nobody
        # verified is just a suppressed alert.
        if kind == "input_drift" and "elig_gap_days_365d" in alert_text:
            return rule
        return None

    def route(self, alerts, now=None):
        now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
        out = []
        for text in alerts:
            kind = self._kind(text)
            if kind is None:
                # UNROUTED, NOT DROPPED. An alert with no policy entry is a
                # gap in the policy, and swallowing it hides the gap.
                rec = {"at": now, "kind": "UNROUTED", "severity": TICKET,
                       "owner": "model owner (data science)", "text": text,
                       "runbook": ("no policy entry matches this alert. Add "
                                   "one to alerting.POLICY -- an alert nobody "
                                   "owns is a notification that gets muted.")}
                out.append(rec)
                self._deliver(rec)
                continue

            silence = self._is_silenced(kind, text)
            p = self.policy[kind]
            rec = {"at": now, "kind": kind, "severity": p["severity"],
                   "owner": p["owner"], "runbook": p["runbook"], "text": text}
            if silence:
                rec["silenced_because"] = silence
                self.silenced.append(rec)
                out.append(rec)
                continue
            out.append(rec)
            self._deliver(rec)
        return out

    def _deliver(self, rec):
        self.sent.append(rec)
        sink = self.sinks.get(rec["severity"])
        if sink:
            sink(rec)

    def summary(self):
        by_sev = {}
        for r in self.sent:
            by_sev[r["severity"]] = by_sev.get(r["severity"], 0) + 1
        return {"delivered": len(self.sent), "silenced": len(self.silenced),
                "by_severity": by_sev}


def render_runbook():
    """Generate docs/RUNBOOK.md from the policy, so the two cannot drift."""
    L = ["# Alert runbook", "",
         "**Generated from `src/alerting.py`. Do not edit by hand.**", "",
         "Every alert has an owner who can act, a severity that is a claim "
         "about somebody's night, and a one-line instruction written for "
         "someone reading it at 03:00.", "",
         "Most alerts are a **digest** or a **ticket**. Only two page, and one "
         "of those is for something *not happening* — see the last row.", ""]
    for kind, p in POLICY.items():
        L += [f"## `{kind}` — {p['severity'].upper()}", "",
              f"**Fires when.** {p['threshold']}", "",
              f"**Owner.** {p['owner']}", "",
              f"**Do this.** {p['runbook']}", ""]
        if p.get("silence_if"):
            L += [f"**Do NOT alert when.** {p['silence_if']}", ""]
        if p.get("external"):
            L += ["> **This check cannot live here.** `monitor.py` cannot "
                  "detect its own failure to run, and a monitoring system "
                  "whose absence is silent is not a control. Something outside "
                  "this repository has to assert that the batch produced a "
                  "queue.", ""]
    return "\n".join(L)
