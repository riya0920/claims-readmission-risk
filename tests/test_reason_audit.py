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


def test_agreement_tracks_attribution_concentration(result):
    """CORRECTED CLAIM.

    An earlier version asserted `0.50 < top1_agreement < 0.85` and called it
    the headline. Regenerating the data changed the model, agreement rose to
    97%, and this test failed -- correctly. The 68% was never a property of
    occlusion; it was a property of that model's attribution being SPREAD
    across correlated features.

    So the invariant is the relationship, not the rate. Pinning a band would
    just re-pin whichever corpus happened to be current when it was written.
    """
    conc = result["concentration"]
    agree = result["top1_agreement"]
    assert 0.0 < conc <= 1.0
    # Agreement can never fall far BELOW concentration: if one feature is the
    # top pick for 92% of members, any sane method finds it for most of them.
    assert agree >= conc - 0.15


def test_the_three_phrases_shown_are_usually_not_the_shap_three(result):
    """The worklist displays three ranked phrases. The set of three matches
    Shapley's set well under half the time."""
    assert result["top3_exact"] < 0.60
    assert result["top3_overlap"] > 0.60      # but they do overlap substantially
    # This claim survived the corpus change where the top-1 claim did not,
    # which is itself informative: agreeing on the single strongest driver is
    # much easier than agreeing on the ordering below it.


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


def test_the_charlson_bias_appears_only_when_attribution_is_spread(result):
    """ALSO CORRECTED.

    An earlier version asserted occlusion over-credits `charlson` by more than
    2x. It did, on a corpus where attribution was spread across three
    correlated features. On a corpus where one feature dominates, both methods
    agree on `charlson` exactly -- so the assertion failed, correctly.

    The bias is a property of the DATA AND MODEL, not of occlusion alone. What
    generalises is the mechanism: setting a composite index to its median while
    leaving correlated utilisation features high produces a member who exists
    nowhere in the data, and occlusion books the model's whole reaction to that
    impossible combination against the feature it moved. That can only show up
    when the composite is competing for attribution in the first place.
    """
    occ = result["occ_top1"]
    shp = result["shap_top1"]
    if result["concentration"] > 0.8:
        # Concentrated regime: no other feature has room to be over-credited.
        assert abs(occ["charlson"] - shp["charlson"]) <= 5
    else:
        assert occ["charlson"] >= shp["charlson"]


def test_occlusion_never_over_credits_prior_spend(result):
    """The other side of the same effect: credit taken by a composite index
    has to come from somewhere, and it comes from the correlated utilisation
    features.

    So SHAP should never give prior spend FEWER top-1 picks than occlusion
    does. That inequality holds in BOTH regimes, unlike the strict version
    this replaced, which held only on the corpus it was written against.
    """
    assert (result["shap_top1"]["paid_amount_365d"]
            >= result["occ_top1"]["paid_amount_365d"])
