"""Tests for the batch path, fairness, alerting, and access control.

The batch tests are about the SECOND run. A batch that works once is a script;
what makes it a batch is that a rerun after a 3am failure cannot hand the care
team a second queue that disagrees with the one they already started working.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from http.server import HTTPServer

import alerting as AL
import batch as B
import fairness as F


# --------------------------------------------------------------------------
# batch: idempotency and partial failure
# --------------------------------------------------------------------------

@pytest.fixture
def con():
    return B.connect(":memory:")


ARGS = dict(model_name="m", model_version=1, as_of="2024-08-01",
            lookback_days=30, capacity=10)


def _rows(n=3):
    return [{"member_id": f"M{i}", "stay_id": f"S{i}",
             "discharge_date": "2024-07-20", "risk": 0.9 - i * 0.1,
             "in_training_set": False, "reasons": ["a"]} for i in range(n)]


def test_a_rerun_of_an_identical_job_returns_the_original(con):
    """The failure this exists for: a rerun at 06:00 after a 03:00 failure must
    not produce a second queue that disagrees with the one being worked."""
    jid, reused = B.start(con, job_id="j1", cohort_hash_value="abc", **ARGS)
    assert not reused
    B.finish(con, jid, _rows(), n_scored=3)
    again, reused2 = B.start(con, job_id="j2", cohort_hash_value="abc", **ARGS)
    assert reused2 is True and again == "j1"


def test_a_changed_cohort_is_a_new_job(con):
    """Keying only on (model, as_of) would return the OLD answer after a late
    claim changed the cohort -- the opposite failure from duplicates, and much
    harder to notice."""
    B.finish(con, B.start(con, job_id="j1", cohort_hash_value="abc",
                          **ARGS)[0], _rows(), n_scored=3)
    jid, reused = B.start(con, job_id="j2", cohort_hash_value="DIFFERENT",
                          **ARGS)
    assert reused is False and jid == "j2"


def test_the_cohort_hash_ignores_order(con):
    assert B.cohort_hash(["a", "b", "c"]) == B.cohort_hash(["c", "a", "b"])
    assert B.cohort_hash(["a", "b"]) != B.cohort_hash(["a", "b", "c"])


def test_two_concurrent_runs_of_the_same_job_are_refused(con):
    B.start(con, job_id="j1", cohort_hash_value="abc", **ARGS)
    with pytest.raises(B.BatchError) as e:
        B.start(con, job_id="j2", cohort_hash_value="abc", **ARGS)
    assert "already running" in str(e.value)


def test_a_failed_job_may_be_retried(con):
    jid, _r = B.start(con, job_id="j1", cohort_hash_value="abc", **ARGS)
    B.fail(con, jid, RuntimeError("boom"))
    new, reused = B.start(con, job_id="j2", cohort_hash_value="abc", **ARGS)
    assert reused is False and new == "j2"


def test_a_partial_write_leaves_no_visible_queue(con):
    """4,000 of 5,000 discharges left visible is a queue silently missing the
    sickest fifth of the population, with nothing about it saying so."""
    jid, _r = B.start(con, job_id="j1", cohort_hash_value="abc", **ARGS)
    bad = _rows(2) + [{"member_id": "M9"}]          # missing required keys
    with pytest.raises(Exception):
        B.finish(con, jid, bad, n_scored=3)
    B.fail(con, jid, KeyError("stay_id"))
    assert B.worklist(con, jid) == []
    assert B.history(con)[0]["status"] == "failed"


def test_a_succeeded_job_records_what_it_scored(con):
    jid, _r = B.start(con, job_id="j1", cohort_hash_value="abc", **ARGS)
    B.finish(con, jid, _rows(3), n_scored=1200)
    h = B.history(con)[0]
    assert h["status"] == "succeeded" and h["n_scored"] == 1200
    assert h["n_queued"] == 3


def test_the_queue_preserves_rank_order(con):
    jid, _r = B.start(con, job_id="j1", cohort_hash_value="abc", **ARGS)
    B.finish(con, jid, _rows(5), n_scored=5)
    wl = B.worklist(con, jid)
    assert [r["rank"] for r in wl] == [1, 2, 3, 4, 5]
    assert wl[0]["risk"] > wl[-1]["risk"]


def test_coverage_reports_how_much_of_the_queue_was_worked(con):
    """The number a model dashboard never shows. Excellent sensitivity attached
    to a queue nobody works delivers nothing."""
    jid, _r = B.start(con, job_id="j1", cohort_hash_value="abc", **ARGS)
    B.finish(con, jid, _rows(4), n_scored=4)
    B.record_outreach(con, jid, "S0", "cm-1", "reached")
    B.record_outreach(con, jid, "S1", "cm-1", "no answer")
    cov = B.coverage(con, jid)
    assert cov["queued"] == 4 and cov["worked"] == 2
    assert cov["coverage"] == 0.5
    assert cov["by_outcome"]["reached"] == 1


def test_outreach_is_idempotent_per_stay(con):
    jid, _r = B.start(con, job_id="j1", cohort_hash_value="abc", **ARGS)
    B.finish(con, jid, _rows(2), n_scored=2)
    B.record_outreach(con, jid, "S0", "cm-1", "no answer")
    B.record_outreach(con, jid, "S0", "cm-1", "reached")
    assert B.coverage(con, jid)["worked"] == 1


# --------------------------------------------------------------------------
# fairness
# --------------------------------------------------------------------------

def _planted(n=4000, seed=0):
    """Two groups with DIFFERENT prevalence, which is the condition that makes
    the impossibility result bite."""
    rng = np.random.default_rng(seed)
    g = np.where(rng.random(n) < 0.4, "A", "B")
    base = np.where(g == "A", 0.30, 0.10)
    p = np.clip(base + rng.normal(0, 0.08, n), 0.01, 0.99)
    y = (rng.random(n) < p).astype(float)
    return y, p, g


def test_the_threshold_comes_from_capacity_not_from_one_half():
    """Evaluating parity at 0.5 measures a decision rule nobody runs."""
    _y, p, _g = _planted()
    thr = F.capacity_threshold(p, 0.05)
    assert 0 < thr < 1
    assert abs((p >= thr).mean() - 0.05) < 0.01


def test_group_report_computes_the_error_rates():
    y, p, g = _planted()
    rep = F.group_report(y, p, g, F.capacity_threshold(p, 0.10))
    assert set(rep) == {"A", "B"}
    for r in rep.values():
        assert r["tpr"] + r["fnr"] == pytest.approx(1.0)


def test_small_groups_are_excluded_from_the_gaps():
    """A 12-member group's TPR swings 8 points when one outcome flips. Ranking
    it beside a 4,000-member group reports sampling noise as a disparity."""
    y, p, g = _planted()
    # object dtype on purpose: np.where(cond, "A", "B") yields '<U1', so
    # assigning "TINY" into it silently truncates to "T" and creates a group
    # nobody named. A real trap when group labels come from a numpy pipeline.
    g = g.astype(object)
    g[:12] = "TINY"
    rep = F.group_report(y, p, g, F.capacity_threshold(p, 0.10))
    out = F.parity_gaps(rep, min_n=50)
    assert "TINY" in out["groups_dropped_too_small"]
    assert "TINY" not in out["groups_used"]


def test_the_impossibility_result_is_checked_not_asserted():
    """It only bites when prevalence differs. Reporting it unconditionally
    would be citing a theorem whose premise was never verified."""
    y, p, g = _planted()
    rep = F.group_report(y, p, g, F.capacity_threshold(p, 0.10))
    imp = F.impossibility_check(rep)
    assert imp["applicable"] is True
    assert imp["prevalence_gap"] > 0.05
    assert "cannot all hold" in imp["reading"]


def test_with_equal_prevalence_the_impossibility_has_no_force():
    rng = np.random.default_rng(3)
    n = 4000
    g = np.where(rng.random(n) < 0.5, "A", "B")
    p = np.clip(0.2 + rng.normal(0, 0.05, n), 0.01, 0.99)
    y = (rng.random(n) < p).astype(float)
    rep = F.group_report(y, p, g, F.capacity_threshold(p, 0.10))
    assert F.impossibility_check(rep)["applicable"] is False


def test_the_recommended_criterion_depends_on_what_the_score_does():
    """The property belongs to the deployment, not the model -- the same
    argument CLINICAL_VALIDATION.md makes for the safety case."""
    adds = F.recommend_criterion(gates_a_benefit=False)
    gates = F.recommend_criterion(gates_a_benefit=True)
    assert "false negative" in adds["criterion"]
    assert "false positive" in gates["criterion"]
    assert adds["criterion"] != gates["criterion"]
    assert "never denies" in adds["assumption"]


def test_calibration_within_groups_is_reported_per_band():
    y, p, g = _planted()
    cal = F.calibration_within_groups(y, p, g, n_bins=4)
    for r in cal.values():
        assert r["max_abs_gap"] < 0.10       # drawn from p, so calibrated


def test_access_bias_is_declared_unclosable_rather_than_reported_clean():
    """An analysis reporting 'no access bias detected' would be reporting a
    property of src/generate.py as a property of the model."""
    note = F.access_bias_note()
    assert note["closeable_here"] is False
    assert "generator" in note["why"]


# --------------------------------------------------------------------------
# alerting
# --------------------------------------------------------------------------

def test_every_policy_entry_has_an_owner_and_a_runbook():
    """An alert with no owner is a notification that gets muted."""
    for kind, p in AL.POLICY.items():
        assert p["owner"] and p["runbook"] and p["threshold"], kind
        assert p["severity"] in (AL.PAGE, AL.TICKET, AL.DIGEST), kind


def test_almost_nothing_pages():
    """Severity is a claim about somebody's night. A drifting feature at 3am is
    the same problem at 9am."""
    paging = [k for k, p in AL.POLICY.items() if p["severity"] == AL.PAGE]
    assert len(paging) <= 2
    assert "batch_did_not_run" in paging


def test_the_most_important_alert_is_for_something_not_happening():
    """monitor.py cannot detect its own failure to run, and a monitoring system
    whose absence is silent is not a control."""
    assert AL.POLICY["batch_did_not_run"].get("external") is True


def test_the_known_boundary_artefact_is_silenced():
    """elig_gap_days_365d drifts by construction. An alert that fires every run
    for a known reason trains its audience to mute the channel."""
    r = AL.Router()
    out = r.route(["input drift: elig_gap_days_365d PSI 0.196"])
    assert "silenced_because" in out[0]
    assert r.summary()["delivered"] == 0


def test_a_real_drift_alert_is_delivered():
    r = AL.Router()
    out = r.route(["input drift: charlson PSI 0.412"])
    assert "silenced_because" not in out[0]
    assert out[0]["owner"].startswith("model owner")
    assert r.summary()["delivered"] == 1


def test_a_cohort_alert_goes_to_the_programme_manager_not_data_science():
    """A stage whose retention moves is a POPULATION change. Routing it to the
    model owner sends it to someone who cannot fix it."""
    r = AL.Router()
    out = r.route(["cohort: retention at 'x' moved -10.1% -- the denominator "
                   "is a different population, not a worse model"])
    assert "programme manager" in out[0]["owner"]


def test_an_unrouted_alert_is_surfaced_not_swallowed():
    """An alert with no policy entry is a gap in the policy."""
    r = AL.Router()
    out = r.route(["something nobody wrote a rule for"])
    assert out[0]["kind"] == "UNROUTED"
    assert r.summary()["delivered"] == 1


def test_sinks_receive_only_their_own_severity():
    pages, tickets = [], []
    r = AL.Router(sinks={AL.PAGE: pages.append, AL.TICKET: tickets.append})
    r.route(["input drift: charlson PSI 0.4",
             "schema mismatch: feature contract does not match"])
    assert len(tickets) == 1 and len(pages) == 1


def test_the_runbook_is_generated_from_the_policy():
    md = AL.render_runbook()
    for kind, p in AL.POLICY.items():
        assert f"`{kind}`" in md
        assert p["owner"] in md


# --------------------------------------------------------------------------
# access control on the service
# --------------------------------------------------------------------------

@pytest.fixture
def api():
    import registry
    import serve as S

    class _Dummy:
        def predict_proba(self, X):
            p = np.linspace(0.1, 0.9, len(X))
            return np.column_stack([1 - p, p])

    cols = ["age", "charlson", "los", "ed_visits_90d"]
    manifest = {"name": "t", "version": 1, "feature_columns": cols,
                "feature_hash": registry.feature_hash(cols),
                "n_features": 4, "feature_visibility": "received",
                "intended_use": "test"}
    rng = np.random.default_rng(0)
    n = 20
    feat = pd.DataFrame({
        "stay_id": [f"S{i}" for i in range(n)],
        "member_id": [f"M{i}" for i in range(n)],
        "discharge_date": [S.CUT + pd.Timedelta(d + 1, unit="D")
                           for d in range(n)],
        "age": rng.integers(40, 85, n).astype(float),
        "charlson": rng.integers(0, 5, n).astype(float),
        "los": rng.integers(1, 12, n).astype(float),
        "ed_visits_90d": rng.integers(0, 4, n).astype(float),
        "index_ccsr": rng.choice(["CIR", "RSP"], n),
        "pdc_proxy_180d": rng.choice([0.4, 0.8, np.nan], n),
        "y": rng.integers(0, 2, n)})
    S._STATE.update({"model": registry.ServableModel(_Dummy(), manifest),
                     "features": feat, "columns": cols,
                     "medians": feat[cols].median()})
    S.ACCESS.token = "secret-token"
    S.ACCESS.records.clear()
    httpd = HTTPServer(("127.0.0.1", 0), S.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", S
    httpd.shutdown()
    httpd.server_close()
    S.ACCESS.token = None


def _get(base, path, token=None):
    r = urllib.request.Request(base + path)
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_an_unauthenticated_worklist_is_refused(api):
    """An unauthenticated endpoint returning member-level risk scores is a bulk
    PHI disclosure to anyone who can reach the port."""
    base, _S = api
    code, body = _get(base, "/worklist?as_of=2024-06-01&capacity=5")
    assert code == 401
    assert "PHI disclosure" in body["why"]


def test_a_wrong_token_is_refused(api):
    base, _S = api
    code, _b = _get(base, "/worklist?as_of=2024-06-01", "wrong")
    assert code == 401


def test_the_right_token_is_accepted(api):
    base, _S = api
    code, _b = _get(base, "/worklist?as_of=2024-06-01&capacity=3",
                    "secret-token")
    assert code == 200


def test_health_needs_no_token(api):
    """A load balancer probing it holds no credential. Requiring one there
    removes a working service from rotation. Safe only because it returns no
    member data."""
    base, _S = api
    code, body = _get(base, "/health")
    assert code == 200
    assert "member" not in json.dumps(body).lower()


def test_a_disclosure_is_logged_with_the_members_it_disclosed(api):
    """Authentication answers 'may you ask'. Only the log answers 'which
    members were disclosed, to whom, when' -- the question an investigation
    asks."""
    base, S = api
    _c, body = _get(base, "/worklist?as_of=2024-06-01&capacity=4",
                    "secret-token")
    ids = [r["member_id"] for r in body["worklist"]]
    assert ids
    rec = S.ACCESS.records[-1]
    assert rec["outcome"] == "success"
    assert set(rec["member_ids"]) == set(ids)
    assert S.ACCESS.who_saw(ids[0])


def test_a_denied_attempt_is_logged_too(api):
    base, S = api
    _get(base, "/worklist?as_of=2024-06-01", "wrong")
    assert S.ACCESS.records[-1]["outcome"].startswith("denied")
