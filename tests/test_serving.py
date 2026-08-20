"""Tests for the serving layer: registry contract, drift detectors, API refusals.

The registry tests are the ones that matter. A schema check that only catches
missing columns is not a schema check -- the dangerous case is the SAME columns
in a different order, because nothing raises and every number is wrong.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import monitor as M
import registry


# --------------------------------------------------------------------------
# registry: the feature contract
# --------------------------------------------------------------------------

class _Dummy:
    """Minimal fitted-estimator stand-in; returns a fixed column of scores."""

    def predict_proba(self, X):
        p = np.linspace(0.1, 0.9, len(X))
        return np.column_stack([1 - p, p])


COLS = ["age", "charlson", "los", "ed_visits_90d"]


def _servable(columns=COLS):
    manifest = {
        "name": "test-model", "version": 1,
        "feature_columns": list(columns),
        "feature_hash": registry.feature_hash(columns),
        "n_features": len(columns),
    }
    return registry.ServableModel(_Dummy(), manifest)


def test_feature_hash_is_order_sensitive():
    # the whole reason the hash exists: a reordered design matrix scores
    # charlson through the coefficient for age, and raises nothing
    assert registry.feature_hash(["a", "b"]) != registry.feature_hash(["b", "a"])


def test_feature_hash_is_stable_across_calls():
    assert registry.feature_hash(COLS) == registry.feature_hash(list(COLS))


def test_matching_schema_passes():
    _servable().check_schema(COLS)


def test_missing_column_is_refused():
    with pytest.raises(registry.SchemaMismatch) as e:
        _servable().check_schema(["age", "charlson", "los"])
    assert "missing" in str(e.value)


def test_extra_column_is_refused():
    with pytest.raises(registry.SchemaMismatch) as e:
        _servable().check_schema(COLS + ["surprise_feature"])
    assert "unexpected" in str(e.value)


def test_reordered_columns_are_refused_and_named_as_the_dangerous_case():
    """The case a set-comparison check would pass. Nothing is missing, nothing
    is extra, and every prediction would be silently wrong."""
    reordered = ["charlson", "age", "ed_visits_90d", "los"]
    assert set(reordered) == set(COLS)
    with pytest.raises(registry.SchemaMismatch) as e:
        _servable().check_schema(reordered)
    assert "DIFFERENT ORDER" in str(e.value)


def test_predict_proba_enforces_the_schema():
    X = pd.DataFrame(np.zeros((3, 4)), columns=["charlson", "age", "los",
                                                "ed_visits_90d"])
    with pytest.raises(registry.SchemaMismatch):
        _servable().predict_proba(X)


def test_save_and_load_roundtrip_carries_the_contract(tmp_path):
    d = str(tmp_path)
    registry.save(_Dummy(), COLS, name="rt", version=3,
                  cohort={"n": 10}, metrics={"auroc": 0.65},
                  visibility="received", registry_dir=d)
    got = registry.load("rt", 3, registry_dir=d)
    assert got.columns == COLS
    assert got.manifest["feature_visibility"] == "received"
    assert got.manifest["metrics"]["auroc"] == 0.65
    # intended use travels with the artefact rather than living only in a README
    assert "NOT for coverage" in got.manifest["intended_use"]


def test_manifest_records_visibility_mode(tmp_path):
    """A model fitted on complete-history features must never be served against
    runout-limited ones. The manifest is where that becomes checkable."""
    d = str(tmp_path)
    registry.save(_Dummy(), COLS, name="v", version=1, cohort={}, metrics={},
                  visibility="complete", registry_dir=d)
    registry.save(_Dummy(), COLS, name="v", version=2, cohort={}, metrics={},
                  visibility="received", registry_dir=d)
    modes = {r["version"]: r["feature_visibility"]
             for r in registry.list_versions(registry_dir=d)}
    assert modes == {1: "complete", 2: "received"}


def test_list_versions_on_empty_registry():
    assert registry.list_versions(registry_dir="does-not-exist") == []


# --------------------------------------------------------------------------
# monitor: the detectors
# --------------------------------------------------------------------------

def test_psi_of_a_distribution_against_itself_is_about_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert M.psi(x, x) < 1e-9


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=5000)
    small = M.psi(ref, rng.normal(0.2, 1, 5000))
    large = M.psi(ref, rng.normal(1.5, 1, 5000))
    assert small < large
    assert large > 0.25


def test_psi_bins_come_from_the_reference_not_the_current_sample():
    """Re-binning on the current sample each period hides the shift being
    looked for: any distribution is uniform in its own quantiles."""
    rng = np.random.default_rng(2)
    ref = rng.normal(size=5000)
    shifted = rng.normal(3.0, 1.0, 5000)
    assert M.psi(ref, shifted) > 1.0


def test_psi_tolerates_an_empty_bin_instead_of_returning_inf():
    ref = np.concatenate([np.zeros(900), np.arange(100)])
    cur = np.zeros(1000)                      # the tail vanished entirely
    v = M.psi(ref, cur)
    assert np.isfinite(v) and v > 0


def test_psi_report_flags_a_missing_column():
    ref = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    cur = pd.DataFrame({"a": np.arange(100.0)})
    rows = {r["feature"]: r for r in M.psi_report(ref, cur, ["a", "b"])}
    assert "MISSING" in rows["b"]["verdict"]


def test_calibration_drift_names_the_direction():
    y = np.zeros(1000)
    y[:100] = 1                                         # 10% observed
    over = M.calibration_drift(y, np.full(1000, 0.20))  # predicts 20%
    under = M.calibration_drift(y, np.full(1000, 0.05))
    assert over["oe_ratio"] == pytest.approx(0.5)
    assert over["direction"] == "over-predicting risk"
    assert under["direction"] == "under-predicting risk"
    ok = M.calibration_drift(y, np.full(1000, 0.10))
    assert ok["direction"] == "calibrated within 5%"


def test_runout_drift_detects_a_clearinghouse_shift():
    rng = np.random.default_rng(3)
    ref = rng.gamma(2.0, 10.0, 5000)
    out = M.runout_drift(ref, ref + 21)
    assert out["median_shift_days"] == pytest.approx(21, abs=0.5)
    assert "has moved" in out["verdict"]


def test_runout_drift_is_quiet_when_lag_is_stable():
    rng = np.random.default_rng(4)
    ref = rng.gamma(2.0, 10.0, 5000)
    cur = rng.gamma(2.0, 10.0, 5000)
    assert M.runout_drift(ref, cur)["verdict"] == "receipt lag stable"


def test_cohort_drift_separates_a_population_change_from_a_model_change():
    ref = {"members": 1000, "with an admission": 200, "after enrolment": 180}
    cur = {"members": 1000, "with an admission": 200, "after enrolment": 120}
    rows = {r["stage"]: r for r in M.cohort_drift(ref, cur)}
    assert rows["with an admission"]["retention_change"] == pytest.approx(0.0)
    assert rows["after enrolment"]["retention_change"] < -0.30


def test_cohort_drift_reports_a_stage_that_disappeared():
    rows = M.cohort_drift({"a": 10, "b": 5}, {"a": 10})
    assert rows[1]["verdict"] == "stage missing"


def test_run_all_alerts_on_runout_even_when_feature_psi_stays_quiet():
    """The finding from run_monitor.py, pinned as a regression test.

    A +21-day receipt-lag shift moves the downstream features by PSI ~0.03 --
    below even the 'moderate' 0.10 threshold. If runout were folded into the
    per-feature PSI loop instead of checked separately, this would be silent."""
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"ed_visits_90d": rng.poisson(0.16, 4000).astype(float)})
    cur = pd.DataFrame({"ed_visits_90d": rng.poisson(0.11, 4000).astype(float)})
    lag = rng.gamma(2.0, 10.0, 4000)
    report = M.run_all(ref, cur, reference_lag=lag, current_lag=lag + 21,
                       feature_subset=["ed_visits_90d"])
    assert all(r["psi"] < 0.25 for r in report["input_drift"])
    assert any("claim runout" in a for a in report["alerts"])


def test_run_all_alerts_when_a_waterfall_stage_moves():
    counts = {"members": 5000, "with an admission": 900, "eligible": 800}
    moved = {"members": 5000, "with an admission": 900, "eligible": 600}
    report = M.run_all(pd.DataFrame({"x": np.arange(100.0)}),
                       pd.DataFrame({"x": np.arange(100.0)}),
                       reference_counts=counts, current_counts=moved,
                       feature_subset=["x"])
    assert any("cohort" in a and "different population" in a
               for a in report["alerts"])


def test_run_all_is_silent_when_nothing_moved():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"x": rng.normal(size=3000)})
    lag = rng.gamma(2.0, 10.0, 3000)
    y = (rng.random(3000) < 0.15).astype(float)
    counts = {"members": 3000, "eligible": 2000}
    report = M.run_all(df, df.copy(), y_current=y,
                       p_current=np.full(3000, float(y.mean())),
                       reference_counts=counts, current_counts=dict(counts),
                       reference_lag=lag, current_lag=lag.copy(),
                       feature_subset=["x"])
    assert report["alerts"] == []


def test_cohort_drift_refuses_a_waterfall_that_retains_more_than_it_received():
    """Regression. An earlier simulation scaled each stage independently, so a
    stage 'retained' 117% of what it received, and that impossible number was
    written up as a real conditional effect before the guard existed."""
    ref = {"a": 100, "b": 80}
    bad = {"a": 100, "b": 117}
    rows = M.cohort_drift(ref, bad)
    assert "INCONSISTENT" in rows[1]["verdict"]


def test_run_all_alerts_on_an_inconsistent_waterfall_instead_of_a_change():
    df = pd.DataFrame({"x": np.arange(100.0)})
    report = M.run_all(df, df.copy(),
                       reference_counts={"a": 100, "b": 80},
                       current_counts={"a": 100, "b": 117},
                       feature_subset=["x"])
    assert any("INCONSISTENT" in a for a in report["alerts"])
    assert not any("different population" in a for a in report["alerts"])


def test_feature_drift_skips_columns_with_too_little_data():
    ref = pd.DataFrame({"a": np.arange(200.0), "tiny": np.arange(200.0)})
    cur = pd.DataFrame({"a": np.arange(200.0), "tiny": np.arange(10.0).tolist()
                        + [np.nan] * 190})
    names = [r["feature"] for r in M.feature_drift(ref, cur, ["a", "tiny"])]
    assert names == ["a"]


def test_runout_lag_by_period_medians_by_quarter_and_claim_type():
    med = pd.DataFrame({
        "service_date": pd.to_datetime(["2024-01-05", "2024-01-06",
                                        "2024-04-05", "2024-04-06"]),
        "received_date": pd.to_datetime(["2024-02-15", "2024-02-16",
                                         "2024-04-17", "2024-04-18"]),
        "claim_type": ["IP", "IP", "PROF", "PROF"]})
    table = M.runout_lag_by_period(med)
    assert table.loc["2024Q1", "IP"] == 41.0
    assert table.loc["2024Q2", "PROF"] == 12.0
