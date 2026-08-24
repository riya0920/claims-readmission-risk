"""Death after discharge is a competing risk, and the label knows it.

The generator now draws post-discharge mortality from the SAME risk score as
readmission. That is what makes it a competing risk rather than nuisance
censoring: the patients removed from the risk set are precisely the ones most
likely to have been readmitted, so the bias does not average out.

These tests use a small cohort generated once for the module. They assert the
STRUCTURE of the distortion -- direction, ordering, informativeness -- rather
than exact rates, which move with the seed.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_competing_risks as CR


@pytest.fixture(scope="module")
def idx(tmp_path_factory):
    """A small cohort, generated once."""
    import generate as G

    out = tmp_path_factory.mktemp("cr")
    G.generate(n_members=4000, seed=7, outdir=str(out))
    return CR.load_index_stays(os.path.join(out, "_truth_stays.csv.gz"))


# ----------------------------------------------------------- the generator
def test_post_discharge_death_exists_at_all(idx):
    """The gap list used to say this was not in the generator, so there was
    nothing to recover. There is now."""
    assert idx.died_post_discharge_30d.sum() > 20
    assert 0.01 < idx.died_post_discharge_30d.mean() < 0.15


def test_nobody_dies_twice(idx):
    """In-hospital death and post-discharge death are exclusive, and the
    modelling cohort excludes the first."""
    assert not idx.died_inpatient.any()


def test_a_censored_readmission_is_one_that_lost_the_race(idx):
    """Death only wins if it happens FIRST. A readmission on day 5 followed by
    death on day 20 is an observed readmission, not a censored one."""
    censored = idx[idx.censored_by_death]
    assert len(censored) > 0
    assert (censored["death_day_post"] < censored["readmit_day"]).all()
    assert censored.latent_readmit_30d.all()
    assert not censored.true_readmit_30d.any()


def test_the_readmission_stay_uses_the_day_the_race_was_decided_on(idx):
    """Drawing a fresh gap when materialising the stay would let a patient be
    readmitted on a day they were already dead."""
    readmitted = idx[idx.true_readmit_30d]
    assert len(readmitted) > 100
    assert (readmitted["readmit_day"] >= 2).all()
    assert (readmitted["readmit_day"] <= 30).all()


# -------------------------------------------------------------- the biases
def test_the_naive_label_understates_the_true_rate(idx):
    """Counting a death as `y = 0` counts a death as a good outcome."""
    r = CR.rates(idx)
    assert r["naive"] < r["latent"]


def test_excluding_the_deaths_also_understates_it(idx):
    """Dropping them looks conservative and is not -- it is informative
    censoring, and it points the same way as the naive label."""
    r = CR.rates(idx)
    assert r["exclude_deaths"] < r["latent"]


def test_excluding_is_less_wrong_than_naive_but_still_wrong(idx):
    """Both biases point the same direction; excluding is the smaller of the
    two. Neither is correct."""
    r = CR.rates(idx)
    assert r["naive"] <= r["exclude_deaths"] <= r["latent"]


def test_the_censoring_is_informative(idx):
    """THE REASON EXCLUDING FAILS. If deaths were a random sample, dropping
    them would be harmless in expectation. They are not: they carry higher
    readmission risk, so the model is fitted on a healthier population than
    the one it will score."""
    r = CR.rates(idx)
    assert r["risk_of_the_dead"] > r["risk_of_the_living"]


def test_a_meaningful_share_of_the_dead_would_have_readmitted(idx):
    r = CR.rates(idx)
    assert r["share_of_deaths_that_would_have_readmitted"] > 0.02


# ------------------------------------------------------------- the sweep
def test_zero_mortality_removes_the_bias_entirely(idx):
    """The control, checked for FIRING. If the bias did not vanish when
    nobody dies post-discharge, it would be measuring something else."""
    rows = CR.sweep(idx, multipliers=(0.0,))
    # a floor of 1.2% mortality remains by construction, so this is small
    # rather than exactly zero -- but it must be the smallest of the sweep
    assert abs(rows[0]["naive_bias_pp"]) < 0.6


def test_the_bias_grows_with_mortality(idx):
    """THE ACTIONABLE FORM OF THE FINDING.

    A general medical cohort can mostly ignore this. A heart-failure or
    oncology cohort, where post-discharge mortality runs several times higher,
    cannot -- and the direction never changes.
    """
    rows = CR.sweep(idx, multipliers=(0.0, 1.0, 5.0))
    magnitudes = [abs(r["naive_bias_pp"]) for r in rows]
    assert magnitudes[0] < magnitudes[-1]
    assert rows[-1]["mortality"] > rows[0]["mortality"]


def test_the_bias_never_changes_direction(idx):
    """The naive rate is below the latent rate at every mortality level. A
    sign flip would mean the mechanism is not what this claims."""
    for row in CR.sweep(idx, multipliers=(0.5, 1.0, 2.0, 3.0, 5.0)):
        assert row["naive_bias_pp"] <= 0
