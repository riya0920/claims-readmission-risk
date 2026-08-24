"""Pins what the SHAP audit found about the occlusion reason codes.

SKIPS when `shap` is absent, and ALSO when the cached model is absent: building
it takes about eight minutes, which does not belong in a test run. Generate it
once with `python validate_reasons.py`.

These tests deliberately assert the LIMITATION rather than the agreement. The
worklist ships occlusion, not SHAP, and the honest thing to protect is the size
of the gap -- if someone later "improves" occlusion until it matches SHAP, that
is a real change to what care managers are told and the test should notice.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("shap", reason="reason-code audit only")

import validate_reasons as V                                  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(V.CACHE),
    reason="run `python validate_reasons.py` once to build the cached model")


@pytest.fixture(scope="module")
def result():
    return V.compare(n=200)


def test_the_two_methods_disagree_on_the_top_driver_often_enough_to_matter(
        result):
    """THE HEADLINE. Roughly a third of members are shown a different primary
    reason than Shapley values would give.

    Asserted as a band, not a point: the exact rate depends on the cohort, but
    "materially below perfect and well above chance" is the claim, and it is
    what makes the docstring's disclaimer a measurement rather than a caveat.
    """
    assert 0.50 < result["top1_agreement"] < 0.85


def test_the_three_phrases_shown_are_usually_not_the_shap_three(result):
    """The worklist displays three ranked phrases. The set of three matches
    Shapley's set well under half the time."""
    assert result["top3_exact"] < 0.60
    assert result["top3_overlap"] > 0.60      # but they do overlap substantially


def test_rank_agreement_is_high_even_though_set_agreement_is_not(result):
    """These come apart, which is why both are reported.

    High rank correlation with mediocre top-3 set agreement is not a
    contradiction: most candidates are ordered the same, and the disagreements
    concentrate at the top, which is exactly the part that gets displayed.
    """
    assert result["rank_rho"] > 0.80


def test_occlusion_violates_additivity_by_a_lot(result):
    """SHAP's efficiency property guarantees attributions sum to the model's
    margin over the baseline. Occlusion has no such guarantee.

    The violation is far too large to present these as "how much this feature
    contributed" -- which is why `worklist.py` phrases them as sensitivity
    rather than contribution.
    """
    assert result["additivity_gap"] > 0.05


def test_the_disagreement_is_directional_not_random(result):
    """THE FINDING WORTH ACTING ON.

    Occlusion over-credits `charlson` -- a composite comorbidity index
    correlated with the utilisation features. Setting it alone to the cohort
    median leaves a member who is comorbidity-free but expensive and frequently
    admitted, which exists nowhere in the data; the model's response to that
    off-manifold point gets booked entirely to `charlson`.

    This is the interaction-blindness the docstring predicted, appearing in the
    predicted direction. It matters because "high comorbidity burden" and "a lot
    of recent inpatient days" suggest DIFFERENT phone calls.
    """
    occ = result["occ_top1"]
    shp = result["shap_top1"]
    assert occ["charlson"] > 2 * max(shp["charlson"], 1)


def test_occlusion_under_credits_prior_spend(result):
    """The other side of the same effect: credit taken by `charlson` has to
    come from somewhere, and it comes from the correlated utilisation
    features."""
    assert result["shap_top1"]["paid_amount_365d"] > \
        result["occ_top1"]["paid_amount_365d"]
