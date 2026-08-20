"""Tests for the scoring API.

These start a real HTTPServer on an ephemeral port and talk to it over HTTP,
rather than calling the handler methods directly. Routing, status codes and
JSON encoding are part of what is being claimed, and calling `_worklist()`
in-process tests none of them.

The state is injected rather than warmed from `data/`, so the suite runs in
under a second and does not require the generator to have been run.
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

import registry
import serve as S


# --------------------------------------------------------------------------
# a small injected state: 40 discharges either side of the training cut
# --------------------------------------------------------------------------

FEATURES = ["age", "charlson", "los", "ed_visits_90d"]


class _Dummy:
    """Scores rise with charlson, so ordering is checkable."""

    def predict_proba(self, X):
        p = 0.05 + 0.10 * np.asarray(X["charlson"], dtype=float)
        p = np.clip(p, 0.01, 0.99)
        return np.column_stack([1 - p, p])


def _frame():
    rng = np.random.default_rng(0)
    n = 40
    # half before the training cut (in-sample), half after
    dates = ([S.CUT - pd.Timedelta(d, unit="D") for d in range(1, n // 2 + 1)]
             + [S.CUT + pd.Timedelta(d, unit="D") for d in range(1, n // 2 + 1)])
    return pd.DataFrame({
        "stay_id": [f"S{i:03d}" for i in range(n)],
        "member_id": [f"M{i:03d}" for i in range(n)],
        "discharge_date": dates,
        "age": rng.integers(40, 85, n).astype(float),
        "charlson": rng.integers(0, 5, n).astype(float),
        "los": rng.integers(1, 12, n).astype(float),
        "ed_visits_90d": rng.integers(0, 4, n).astype(float),
        # design_matrix() one-hot encodes index_ccsr and adds a missingness
        # indicator for pdc_proxy_180d, so both have to be present even though
        # this fixture's model does not use them
        "index_ccsr": rng.choice(["CIR", "RSP", "END"], n),
        "pdc_proxy_180d": rng.choice([0.4, 0.8, np.nan], n),
        "y": rng.integers(0, 2, n),
    })


@pytest.fixture(scope="module")
def api():
    manifest = {"name": "test-readmission", "version": 7,
                "feature_columns": list(FEATURES),
                "feature_hash": registry.feature_hash(FEATURES),
                "n_features": len(FEATURES),
                "feature_visibility": "received",
                "intended_use": "test fixture"}
    feat = _frame()
    S._STATE.update({
        "model": registry.ServableModel(_Dummy(), manifest),
        "features": feat, "columns": list(FEATURES),
        "medians": feat[FEATURES].median(),
    })
    httpd = HTTPServer(("127.0.0.1", 0), S.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()
    httpd.server_close()


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# --------------------------------------------------------------------------
# the refusals -- the point of the service
# --------------------------------------------------------------------------

def test_worklist_refuses_without_an_explicit_as_of(api):
    """Defaulting as_of to 'now' would silently change which claims are
    visible, which is the exact train/serve skew this project measures."""
    status, body = _get(api, "/worklist?capacity=5")
    assert status == 400
    assert body["error"] == "as_of is required"
    assert "guess" in body["why"]


def test_score_refuses_without_an_explicit_as_of(api):
    status, body = _post(api, "/score", {"stay_ids": ["S000"]})
    assert status == 400
    assert body["error"] == "as_of is required"
    assert "LEAKAGE_AUDIT" in body["see"]


def test_score_refuses_without_stay_ids(api):
    status, body = _post(api, "/score", {"as_of": "2024-05-15"})
    assert status == 400


def test_bad_json_is_a_400_not_a_500(api):
    req = urllib.request.Request(
        api + "/score", data=b"{not json",
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("should have failed")
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert "bad JSON" in json.loads(e.read())["error"]


def test_unknown_route_lists_the_real_ones(api):
    status, body = _get(api, "/predict")
    assert status == 404
    assert "/worklist" in body["routes"]


# --------------------------------------------------------------------------
# as_of actually changes what is visible
# --------------------------------------------------------------------------

def test_as_of_windows_the_cohort_rather_than_returning_all_history(api):
    """The bug this was written for: the first version returned every discharge
    up to as_of -- 23,402 stays at as_of=2024-06-01. A worklist is a snapshot,
    not an archive."""
    _s, wide = _get(api, "/worklist?as_of=2024-05-20&capacity=5&lookback_days=60")
    _s, narrow = _get(api, "/worklist?as_of=2024-05-20&capacity=5&lookback_days=7")
    assert narrow["cohort_size"] < wide["cohort_size"]
    assert narrow["lookback_days"] == 7


def test_a_later_as_of_sees_different_discharges(api):
    _s, early = _get(api, "/worklist?as_of=2024-04-20&capacity=40")
    _s, late = _get(api, "/worklist?as_of=2024-05-25&capacity=40")
    early_ids = {r["stay_id"] for r in early["worklist"]}
    late_ids = {r["stay_id"] for r in late["worklist"]}
    assert early_ids and late_ids and early_ids != late_ids


def test_worklist_never_returns_a_discharge_after_as_of(api):
    _s, body = _get(api, "/worklist?as_of=2024-05-10&capacity=40&lookback_days=365")
    cut = pd.Timestamp("2024-05-10")
    frame = S._STATE["features"].set_index("stay_id")
    for row in body["worklist"]:
        assert frame.loc[row["stay_id"], "discharge_date"] <= cut


# --------------------------------------------------------------------------
# the in-sample flag
# --------------------------------------------------------------------------

def test_worklist_flags_rows_the_model_was_fitted_on(api):
    """A 30-day window ending just after the training cut necessarily reaches
    back into training data. Those scores are fits, not predictions, and the
    API says so rather than passing them off as forecasts."""
    _s, body = _get(api, "/worklist?as_of=2024-05-10&capacity=40")
    assert body["in_training_set_count"] > 0
    assert "in-sample" in body["in_training_set_warning"]
    assert any(r["in_training_set"] for r in body["worklist"])


def test_a_window_entirely_after_the_cut_carries_no_warning(api):
    _s, body = _get(api, "/worklist?as_of=2024-06-05&capacity=40&lookback_days=20")
    assert body["in_training_set_count"] == 0
    assert body["in_training_set_warning"] is None
    assert not any(r["in_training_set"] for r in body["worklist"])


def test_score_warns_per_row_not_just_per_response(api):
    _s, body = _post(api, "/score", {"as_of": "2024-05-10",
                                     "stay_ids": ["S000", "S039"],
                                     "lookback_days": 365})
    by_id = {r["stay_id"]: r for r in body["scored"]}
    assert by_id["S000"]["in_training_set"] is True
    assert "IN-SAMPLE" in by_id["S000"]["warning"]


# --------------------------------------------------------------------------
# what every response has to carry
# --------------------------------------------------------------------------

def test_every_scoring_response_carries_intended_use(api):
    _s, wl = _get(api, "/worklist?as_of=2024-05-20&capacity=3")
    _s, sc = _post(api, "/score", {"as_of": "2024-05-20", "stay_ids": ["S030"]})
    for body in (wl, sc):
        assert "not a diagnosis" in body["intended_use"]
        assert "deny, delay, limit or price" in body["intended_use"]


def test_worklist_reports_who_did_not_make_the_cut(api):
    """Capacity is the whole framing: the members below the line were not
    cleared of risk, they were ranked below the resource limit."""
    _s, body = _get(api, "/worklist?as_of=2024-05-10&capacity=3&lookback_days=365")
    assert len(body["worklist"]) == 3
    assert body["not_on_this_list"] == body["cohort_size"] - 3


def test_reason_codes_are_labelled_as_not_shap(api):
    _s, body = _post(api, "/score", {"as_of": "2024-05-20",
                                     "stay_ids": ["S030"]})
    assert "NOT SHAP" in body["scored"][0]["reason_method"]


def test_worklist_is_ordered_by_descending_risk(api):
    _s, body = _get(api, "/worklist?as_of=2024-05-20&capacity=20&lookback_days=365")
    risks = [r["risk"] for r in body["worklist"]]
    assert risks == sorted(risks, reverse=True)
    assert [r["rank"] for r in body["worklist"]] == list(range(1, len(risks) + 1))


def test_score_reports_ids_it_could_not_find(api):
    _s, body = _post(api, "/score", {"as_of": "2024-05-20",
                                     "stay_ids": ["S030", "NOPE"]})
    assert "NOPE" in body["not_found"]


def test_empty_window_returns_the_same_response_shape_as_a_populated_one(api):
    """A quiet day is a normal answer, not a different schema. The first
    version returned a three-key object here, so any client reading
    cohort_size got a KeyError instead of a zero."""
    _s, empty = _get(api, "/worklist?as_of=2020-01-01&capacity=5")
    _s, full = _get(api, "/worklist?as_of=2024-05-20&capacity=5")
    assert empty["worklist"] == []
    assert set(empty) == set(full)
    assert empty["cohort_size"] == 0


# --------------------------------------------------------------------------
# metadata routes
# --------------------------------------------------------------------------

def test_health_names_the_model_and_version(api):
    status, body = _get(api, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["version"] == 7


def test_model_route_exposes_the_full_manifest(api):
    status, body = _get(api, "/model")
    assert status == 200
    assert body["feature_columns"] == FEATURES
    assert body["feature_visibility"] == "received"


def test_service_refuses_to_score_on_a_schema_mismatch(api):
    """The dangerous case, end to end: the feature builder changes, the columns
    still all exist, and only the order moved. The service must fail loudly
    rather than return confident numbers computed from the wrong coefficients."""
    original = S._STATE["columns"]
    S._STATE["columns"] = ["charlson", "age", "los", "ed_visits_90d"]
    try:
        with pytest.raises(registry.SchemaMismatch):
            S.Handler._worklist(S.Handler.__new__(S.Handler),
                                "2024-05-20", 5, 365)
    finally:
        S._STATE["columns"] = original
