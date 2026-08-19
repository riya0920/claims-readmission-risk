"""Tests for the things that would fail silently.

None of these test that the model is good. They test that the model cannot
cheat, which is the only property here worth a test suite. A readmission model
with a leak still scores well, still trains, still deploys, and is still
wrong -- so the failure mode has to be caught structurally or not at all.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import codes as C
import comorbidity as CM
import features as F
import uncertainty as U
from features import ClaimsView, TemporalViolation


# ---------------------------------------------------------------------------
# Fixtures: a two-member world, hand-built so every expected value is countable.
# ---------------------------------------------------------------------------
D = lambda s: pd.Timestamp(s)  # noqa: E731


def _medical(rows):
    cols = ["claim_id", "member_id", "service_date", "claim_type", "dx_code",
            "provider_id", "paid_amount", "los", "received_date"]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture
def world():
    med = _medical([
        # member A: an ED visit 40 days pre-admission, received promptly
        ("c1", "A", D("2023-05-01"), "ED", "J18.9", "P1", 800.0, 0, D("2023-05-10")),
        # an inpatient stay 200 days pre-admission, received late but in time
        ("c2", "A", D("2022-12-01"), "IP", "I50.9", "P2", 9000.0, 5, D("2023-01-20")),
        # an ED visit 20 days pre-admission whose claim has NOT arrived by
        # the scoring instant -- the runout case
        ("c3", "A", D("2023-05-21"), "ED", "N17.9", "P3", 1200.0, 0, D("2023-08-30")),
        # the index stay's own facility claim: service before discharge,
        # received long after. Must never count as prior utilisation.
        ("c4", "A", D("2023-06-10"), "IP", "A41.9", "P4", 22000.0, 4, D("2023-08-01")),
    ])
    rx = pd.DataFrame([
        ("r1", "A", D("2023-04-01"), "00093-0742-01", "loop_diuretic", 30, "P9",
         D("2023-04-01")),
    ], columns=["rx_claim_id", "member_id", "fill_date", "ndc",
                "therapeutic_class", "days_supply", "prescriber_id",
                "received_date"])
    elig = pd.DataFrame([("A", D("2021-01-01"), D("2024-12-31"), "HMO-01")],
                        columns=["member_id", "span_start", "span_end", "plan_id"])
    members = pd.DataFrame([("A", 70, "F", "TX", "HMO-01")],
                           columns=["member_id", "age", "sex", "state", "plan_id"])
    stays = pd.DataFrame([{
        "member_id": "A", "admit_date": D("2023-06-10"),
        "discharge_date": D("2023-06-14"), "los": 4, "principal_dx": "A41.9",
        "planned": False, "died_inpatient": False, "true_readmit_30d": False,
        "is_readmit_stay": False, "discharge_status": "01", "stay_id": "S1",
    }])
    return med, rx, elig, members, stays


def build(world, **kw):
    med, rx, elig, members, stays = world
    return F.build_features(stays, F.pack_medical(med), F.pack_pharmacy(rx),
                            elig, members, **kw)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
def test_verifier_raises_on_dates_at_or_after_as_of(world):
    med, rx, *_ = world
    v = ClaimsView(F.pack_medical(med), F.pack_pharmacy(rx), D("2023-01-01"))
    with pytest.raises(TemporalViolation):
        v._verify(np.array(["2023-06-10"], dtype="datetime64[D]"), "medical")


def test_future_claim_cannot_change_a_single_feature(world):
    med, rx, elig, members, stays = world
    before = build(world)
    poisoned = pd.concat([med, _medical([
        ("evil", "A", D("2023-06-20"), "IP", "I50.9", "P7", 30000.0, 7,
         D("2023-06-21")),
    ])], ignore_index=True)
    after = F.build_features(stays, F.pack_medical(poisoned),
                             F.pack_pharmacy(rx), elig, members)
    num = [c for c in before.columns if before[c].dtype.kind in "if"]
    pd.testing.assert_frame_equal(before[num], after[num])


def test_index_stay_is_not_its_own_prior_admission(world):
    """c4 is the index stay's own claim, service-dated before discharge.

    Under a naive `service_date < discharge_date` filter it counts as a prior
    inpatient admission -- the model learns that being admitted predicts being
    admitted. History must stop at the ADMISSION.
    """
    f = build(world, visibility="service")   # most permissive visibility
    assert int(f.ip_admits_365d.iloc[0]) == 1, "only c2 is a prior admission"
    assert int(f.ip_days_365d.iloc[0]) == 5, "only c2's 5 days are prior"


# ---------------------------------------------------------------------------
# Claim runout
# ---------------------------------------------------------------------------
def test_unreceived_claim_is_invisible_at_serving_time(world):
    """c3: ED visit on 2023-05-21 (before admission), claim received
    2023-08-30 (after discharge). Real at serving time? No -- it has not
    arrived. Present in a warehouse snapshot? Yes. That is the skew."""
    serving = build(world, visibility="received")
    warehouse = build(world, visibility="service")
    assert int(serving.ed_visits_90d.iloc[0]) == 1     # only c1 has arrived
    assert int(warehouse.ed_visits_90d.iloc[0]) == 2   # c1 and c3
    assert int(serving.ed_visits_30d.iloc[0]) == 0
    assert int(warehouse.ed_visits_30d.iloc[0]) == 1


def test_late_received_but_in_time_claim_is_visible(world):
    """c2: served 2022-12-01, received 2023-01-20, scored 2023-06-14.
    Late is fine; late-but-before-scoring is still knowledge."""
    f = build(world, visibility="received")
    assert int(f.ip_admits_365d.iloc[0]) == 1


# ---------------------------------------------------------------------------
# The excluded field
# ---------------------------------------------------------------------------
def test_discharge_status_never_reaches_the_design_matrix(world):
    f = build(world, include_leaky=True)
    assert "LEAK_discharge_status" in f.columns
    X = F.design_matrix(f, leaky=False)
    assert not [c for c in X.columns if "discharge_status" in c]
    assert all(X.dtypes != object), "design matrix must be fully numeric"


def test_design_matrix_drops_exactly_redundant_columns(world):
    f = build(world)
    X = F.design_matrix(f)
    for c in F.REDUNDANT_IN_DESIGN:
        assert c not in X.columns


# ---------------------------------------------------------------------------
# Clinical code handling
# ---------------------------------------------------------------------------
def test_ccsr_longest_prefix_wins():
    assert C.ccsr_category("E11.22") == "END005"   # with complication
    assert C.ccsr_category("E11.9") == "END004"    # without
    assert C.ccsr_category("ZZZ.9") == "XXX000"


def test_charlson_hierarchy_does_not_double_count():
    assert CM.charlson_score(["E11.9"]) == 1
    assert CM.charlson_score(["E11.22"]) == 2
    assert CM.charlson_score(["E11.9", "E11.22"]) == 2      # not 3
    assert CM.charlson_score(["C34.90"]) == 2
    assert CM.charlson_score(["C34.90", "C78.00"]) == 6     # not 8
    assert CM.charlson_score(["K70.30", "K72.90"]) == 3     # not 4


def test_charlson_reads_dotted_and_undotted_codes():
    assert CM.charlson_score(["I50.9"]) == CM.charlson_score(["I509"])


# ---------------------------------------------------------------------------
# Cohort rules
# ---------------------------------------------------------------------------
def _stay(**kw):
    base = dict(member_id="A", admit_date=D("2023-06-10"),
                discharge_date=D("2023-06-14"), los=4, principal_dx="A41.9",
                planned=False, died_inpatient=False, true_readmit_30d=False,
                is_readmit_stay=False, discharge_status="01", stay_id="S")
    base.update(kw)
    return base


def test_cohort_exclusions_are_applied_and_counted():
    elig = pd.DataFrame([("A", D("2021-01-01"), D("2024-12-31"), "HMO-01")],
                        columns=["member_id", "span_start", "span_end", "plan_id"])
    stays = pd.DataFrame([
        _stay(stay_id="keep"),
        _stay(stay_id="dead", died_inpatient=True),
        _stay(stay_id="planned", planned=True),
        _stay(stay_id="readmit", is_readmit_stay=True),
    ])
    cohort, counts = F.build_cohort(stays, elig, D("2024-12-31"))
    assert list(cohort.stay_id) == ["keep"]
    assert counts["all inpatient stays"] == 4
    assert counts["after requiring 30d observable follow-up"] == 1


def test_cohort_requires_observable_followup():
    """Discharged 10 days before coverage ends: the 30-day outcome cannot be
    observed, so the stay cannot be a training row."""
    elig = pd.DataFrame([("A", D("2021-01-01"), D("2023-06-24"), "HMO-01")],
                        columns=["member_id", "span_start", "span_end", "plan_id"])
    stays = pd.DataFrame([_stay(stay_id="unobservable")])
    cohort, counts = F.build_cohort(stays, elig, D("2024-12-31"))
    assert len(cohort) == 0


def test_cohort_allows_one_short_enrolment_gap():
    """HEDIS-style allowable gap: one gap of <=45 days does not disqualify.
    Requiring unbroken coverage would delete the churning members whose gaps
    are the predictive signal."""
    ok = pd.DataFrame([("A", D("2021-01-01"), D("2023-02-01"), "H"),
                       ("A", D("2023-03-10"), D("2024-12-31"), "H")],
                      columns=["member_id", "span_start", "span_end", "plan_id"])
    too_long = pd.DataFrame([("A", D("2021-01-01"), D("2023-01-01"), "H"),
                             ("A", D("2023-04-01"), D("2024-12-31"), "H")],
                            columns=["member_id", "span_start", "span_end", "plan_id"])
    stays = pd.DataFrame([_stay(stay_id="S1")])
    assert len(F.build_cohort(stays, ok, D("2024-12-31"))[0]) == 1
    assert len(F.build_cohort(stays, too_long, D("2024-12-31"))[0]) == 0


def test_gap_days_feature_is_populated_for_churning_members(world):
    """Regression test. The first version of build_cohort required unbroken
    365-day enrolment, which excluded every member with a gap and left
    elig_gap_days_365d identically zero -- a feature that looked fine and
    carried no information."""
    med, rx, _, members, stays = world
    churn = pd.DataFrame([("A", D("2021-01-01"), D("2023-01-05"), "H"),
                          ("A", D("2023-02-10"), D("2024-12-31"), "H")],
                         columns=["member_id", "span_start", "span_end", "plan_id"])
    f = F.build_features(stays, F.pack_medical(med), F.pack_pharmacy(rx),
                         churn, members)
    assert int(f.elig_gap_days_365d.iloc[0]) == 36
    assert int(f.any_elig_gap_365d.iloc[0]) == 1


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def test_cluster_index_partitions_every_row_exactly_once():
    groups = np.array(["a", "b", "a", "c", "b", "a"])
    idx = U.ClusterIndex(groups)
    assert idx.n_groups == 3
    allrows = np.sort(np.concatenate(idx.rows_by_group))
    assert list(allrows) == list(range(len(groups)))


def test_cluster_resample_keeps_members_intact():
    """The whole point: a member's rows travel together, so correlated stays
    are never treated as independent evidence."""
    groups = np.array(["a", "a", "a", "b", "c"])
    idx = U.ClusterIndex(groups)
    rng = np.random.default_rng(0)
    for _ in range(20):
        rows = idx.resample(rng)
        picked = groups[rows]
        # every appearance of 'a' comes in a block of 3
        assert (picked == "a").sum() % 3 == 0


def test_clustered_interval_is_wider_than_the_naive_one():
    """Ignoring clustering understates uncertainty. If this ever fails, the
    clustering is not doing anything and the intervals are too narrow."""
    rng = np.random.default_rng(3)
    n_members, per = 300, 4
    groups = np.repeat([f"m{i}" for i in range(n_members)], per)
    member_effect = rng.normal(0, 1.2, n_members).repeat(per)
    y = (rng.random(n_members * per) < 0.2).astype(int)
    p = 1 / (1 + np.exp(-(member_effect + y * 0.8)))
    clustered = U.bootstrap_ci(y, p, groups, "auroc", n_boot=200, seed=1)
    naive = U.bootstrap_ci(y, p, np.arange(len(y)).astype(str), "auroc",
                           n_boot=200, seed=1)
    assert (clustered["hi"] - clustered["lo"]) > (naive["hi"] - naive["lo"])


def test_paired_difference_detects_a_genuinely_better_model():
    rng = np.random.default_rng(5)
    n = 3000
    groups = np.arange(n).astype(str)
    y = (rng.random(n) < 0.3).astype(int)
    good = np.clip(y * 0.5 + rng.normal(0.3, 0.15, n), 0.001, 0.999)
    bad = np.clip(y * 0.05 + rng.normal(0.3, 0.15, n), 0.001, 0.999)
    d = U.paired_bootstrap_difference(y, good, bad, groups, "auroc",
                                      n_boot=200, seed=1)
    assert d["difference"] > 0
    assert d["significant"] is True
    assert d["p_value"] < 0.05


def test_paired_difference_finds_nothing_between_identical_models():
    rng = np.random.default_rng(7)
    n = 1500
    groups = np.arange(n).astype(str)
    y = (rng.random(n) < 0.25).astype(int)
    p = np.clip(y * 0.3 + rng.normal(0.3, 0.2, n), 0.001, 0.999)
    d = U.paired_bootstrap_difference(y, p, p.copy(), groups, "auroc",
                                      n_boot=100, seed=1)
    assert d["difference"] == 0.0
    assert d["significant"] is False


def test_point_estimate_lies_inside_its_own_interval():
    rng = np.random.default_rng(11)
    n = 1200
    groups = np.repeat([f"m{i}" for i in range(300)], 4)
    y = (rng.random(n) < 0.2).astype(int)
    p = np.clip(y * 0.4 + rng.normal(0.25, 0.2, n), 0.001, 0.999)
    for metric in ("auroc", "auprc", "brier", "sens_at_5pct", "ppv_at_5pct"):
        ci = U.bootstrap_ci(y, p, groups, metric, n_boot=150, seed=2)
        assert ci["lo"] <= ci["point"] <= ci["hi"], metric
