"""Cohort construction and the feature builder, with the temporal guard.

THE POINT OF THIS MODULE
------------------------
A readmission model is scored at the moment of discharge. Everything it may
use must have been knowable at that moment. "Knowable at that moment" is a
stricter condition than "dated before that moment", and the gap between those
two readings is where claims models quietly die.

Three separate things are enforced here.

1. NO FUTURE SERVICE DATES.  A claim for care delivered after the index
   discharge cannot inform a prediction made at that discharge. This is the
   obvious one and it is enforced structurally: the only way to read claims in
   this codebase is through ClaimsView, which is constructed with an as_of and
   raises TemporalViolation if anything at or after it survives filtering.
   There is no "just this once" path, because in every real leak I have read
   about there was a just-this-once path.

2. NO UNRECEIVED CLAIMS (claim runout).  This is the non-obvious one. A claim
   with service_date in the past may not have ARRIVED yet. Median facility lag
   in this dataset is ~41 days. So a model trained on a warehouse snapshot
   taken well after the fact sees a member's complete history, while the same
   model in production, scoring a discharge in real time, sees maybe
   two-thirds of the equivalent window. Same feature name, different
   distribution: silent train/serve skew. ClaimsView filters on received_date,
   and the `visibility` switch exists precisely so the skew can be MEASURED
   rather than asserted -- see leakage_audit.py.

3. THE INDEX STAY IS NOT A CLAIM.  At discharge, the stay's own UB-04 has not
   been submitted, let alone adjudicated. But we obviously know the patient is
   being discharged, how long they were in, and what they were being treated
   for. In a real deployment that information arrives on the ADT/census feed,
   not from claims. So index-stay attributes are read from the stay record
   (the ADT stand-in) and history is read from claims. Conflating the two is
   how people accidentally hand the model an adjudicated claim it could not
   have had.

WHAT IS DELIBERATELY EXCLUDED
-----------------------------
discharge_status (UB-04 patient status) is present in the raw data and is
never offered as a feature. It is assigned at/after discharge coding and
encodes the outcome -- expired members cannot readmit, and disposition to
SNF/AMA is chosen partly in light of what happened next. leakage_audit.py
quantifies what including it would have bought, and it buys a lot, which is
the whole problem.

principal_dx of the index stay is a GREY ZONE and is included with a note:
the working diagnosis is known at discharge, but the final coded principal
diagnosis is assigned during coding, days later, and can differ. Real systems
use the working dx from the ADT feed. This build uses the stay's dx and flags
the assumption rather than hiding it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import codes as C
import comorbidity as CM

LOOKBACK_DAYS = 365
FOLLOWUP_DAYS = 30
# HEDIS-style continuous-enrolment convention: one gap of up to 45 days is
# allowed in the measurement/lookback period. Requiring literally unbroken
# coverage would exclude every member who churns -- which is precisely the
# population whose coverage gaps carry the predictive signal we want.
ALLOWABLE_GAP_DAYS = 45


class TemporalViolation(Exception):
    """Raised when data that postdates the scoring instant reaches a feature."""


# ---------------------------------------------------------------------------
# Compact per-member column store. The DataFrame-per-stay version of this was
# 5ms/stay, which is ~2.5 minutes at cohort scale for a single build; the audit
# needs four builds. Same semantics, numpy arrays.
# ---------------------------------------------------------------------------
class MemberClaims:
    __slots__ = ("service", "received", "ctype", "dx", "provider", "paid", "los")

    def __init__(self, service, received, ctype, dx, provider, paid, los):
        self.service = service
        self.received = received
        self.ctype = ctype
        self.dx = dx
        self.provider = provider
        self.paid = paid
        self.los = los


class MemberRx:
    __slots__ = ("fill", "received", "cls", "days", "prescriber")

    def __init__(self, fill, received, cls, days, prescriber):
        self.fill = fill
        self.received = received
        self.cls = cls
        self.days = days
        self.prescriber = prescriber


def pack_medical(medical):
    out = {}
    for mid, g in medical.groupby("member_id", sort=False):
        out[mid] = MemberClaims(
            g["service_date"].values.astype("datetime64[D]"),
            g["received_date"].values.astype("datetime64[D]"),
            g["claim_type"].values.astype(object),
            g["dx_code"].values.astype(object),
            g["provider_id"].values.astype(object),
            pd.to_numeric(g["paid_amount"], errors="coerce").fillna(0).values,
            pd.to_numeric(g["los"], errors="coerce").fillna(0).values,
        )
    return out


def pack_pharmacy(pharmacy):
    out = {}
    for mid, g in pharmacy.groupby("member_id", sort=False):
        out[mid] = MemberRx(
            g["fill_date"].values.astype("datetime64[D]"),
            g["received_date"].values.astype("datetime64[D]"),
            g["therapeutic_class"].values.astype(object),
            pd.to_numeric(g["days_supply"], errors="coerce").fillna(0).values,
            g["prescriber_id"].values.astype(object),
        )
    return out


class ClaimsView:
    """The only sanctioned way to read claims for feature building.

    Construct with the scoring instant; reads are filtered and then re-checked.
    The re-check is not redundant defensiveness -- it is the assertion that
    turns a filtering bug into a loud failure instead of a quiet leak.
    """

    def __init__(self, med_store, rx_store, as_of, visibility="received"):
        if visibility not in ("received", "service"):
            raise ValueError("visibility must be 'received' or 'service'")
        self.as_of = np.datetime64(pd.Timestamp(as_of).date(), "D")
        self.visibility = visibility
        self._med = med_store
        self._rx = rx_store

    def _mask(self, event_dates, received_dates, since, until):
        # as_of is the RECEIPT horizon (what has arrived by scoring time).
        # until is the SERVICE horizon (how far the history window runs).
        # They are different dates on purpose: history stops at the index
        # ADMISSION, because care delivered during the index stay belongs to
        # the index stay, not to the member's prior utilisation. Collapsing
        # the two lets the index admission count itself as a prior admission.
        m = event_dates < self.as_of
        if until is not None:
            m &= event_dates < until
        if self.visibility == "received":
            m &= received_dates <= self.as_of
        if since is not None:
            m &= event_dates >= since
        return m

    def _verify(self, event_dates, what):
        if event_dates.size and (event_dates >= self.as_of).any():
            raise TemporalViolation(
                f"{int((event_dates >= self.as_of).sum())} {what} row(s) at or "
                f"after as_of {self.as_of} survived filtering")
        return event_dates

    def medical(self, member_id, since=None, until=None):
        mc = self._med.get(member_id)
        if mc is None:
            return None, None
        m = self._mask(mc.service, mc.received, since, until)
        self._verify(mc.service[m], "medical")
        return mc, m

    def pharmacy(self, member_id, since=None, until=None):
        rx = self._rx.get(member_id)
        if rx is None:
            return None, None
        m = self._mask(rx.fill, rx.received, since, until)
        self._verify(rx.fill[m], "pharmacy")
        return rx, m


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
def _spans_by_member(eligibility):
    spans = {}
    for mid, a, b in eligibility[["member_id", "span_start", "span_end"]].itertuples(index=False):
        spans.setdefault(mid, []).append((a, b))
    for v in spans.values():
        v.sort()
    return spans


def _gap_profile(spans, lo, hi):
    """(total covered days, longest gap, n gaps) for [lo, hi] against spans."""
    covered, gaps, cursor = 0, [], lo
    for a, b in spans:
        s0, e0 = max(a, lo), min(b, hi)
        if e0 <= s0:
            continue
        if s0 > cursor:
            gaps.append((s0 - cursor).days)
        covered += (e0 - s0).days
        cursor = max(cursor, e0)
    if cursor < hi:
        gaps.append((hi - cursor).days)
    return covered, (max(gaps) if gaps else 0), len(gaps)


def build_cohort(stays, eligibility, window_end):
    """Index admissions, with exclusions applied and each exclusion counted.

    Exclusions and why:
      readmission stays     -- this build scores each member's index stays only;
                               chaining readmissions is a modelling choice with
                               its own correlation structure, out of scope here
      died_inpatient        -- cannot readmit; leaving them in deflates the rate
                               and lets any death-correlated feature look
                               predictive of NOT readmitting
      planned admission     -- the outcome is UNPLANNED readmission; a scheduled
                               chemo or staged-procedure admission is not a
                               care-management failure
      insufficient lookback -- enrolment in the prior 365d with at most one gap
                               of <=45 days (HEDIS allowable-gap convention).
                               Requiring unbroken coverage would delete exactly
                               the churning members whose gaps carry signal.
      unobservable followup -- discharge within 30d of the data window end, or
                               coverage ending within 30d of discharge, means
                               the label cannot be observed.

    That last exclusion deserves the flag it gets in CLINICAL_VALIDATION.md:
    members who lose coverage right after discharge are dropped BECAUSE their
    outcome is unobservable, and they are plausibly among the highest-risk
    members. The cohort is therefore not the population.
    """
    s = stays.copy()
    counts = {"all inpatient stays": len(s)}

    s = s[~s["is_readmit_stay"].astype(bool)]
    counts["after removing readmission stays"] = len(s)
    s = s[~s["died_inpatient"].astype(bool)]
    counts["after removing in-hospital deaths"] = len(s)
    s = s[~s["planned"].astype(bool)]
    counts["after removing planned admissions"] = len(s)

    spans = _spans_by_member(eligibility)
    keep, gapdays = [], []
    for r in s.itertuples(index=False):
        lo = r.admit_date - pd.Timedelta(LOOKBACK_DAYS, unit="D")
        cov, longest, ngaps = _gap_profile(spans.get(r.member_id, []), lo, r.admit_date)
        keep.append(longest <= ALLOWABLE_GAP_DAYS and ngaps <= 1 and cov > 0)
        gapdays.append(LOOKBACK_DAYS - cov)
    s = s[np.array(keep)]
    counts["after continuous enrolment (1 gap <=45d)"] = len(s)

    keep_fu = []
    for r in s.itertuples(index=False):
        end = r.discharge_date + pd.Timedelta(FOLLOWUP_DAYS, unit="D")
        ok = r.discharge_date <= window_end - pd.Timedelta(FOLLOWUP_DAYS, unit="D")
        ok = ok and any(a <= r.discharge_date and b >= end
                        for a, b in spans.get(r.member_id, []))
        keep_fu.append(ok)
    s = s[np.array(keep_fu)]
    counts["after requiring 30d observable follow-up"] = len(s)

    return s.reset_index(drop=True), counts


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# CCSR categories tracked as history flags. Chosen as the categories that carry
# readmission signal in a payer population; the long tail is dropped rather
# than one-hot exploded, which is the whole reason for grouping in the first
# place.
TRACKED_CCSR = [
    "CIR019", "CIR009", "CIR011", "CIR017", "CIR020", "CIR024",
    "RSP008", "RSP002", "END004", "END005", "GEN003", "GEN002", "GEN004",
    "NEO070", "MBD025", "MBD002", "MBD017", "MBD018", "DIG018", "DIG019",
    "DIG020", "NVS020", "INF003", "INF006", "MUS006", "SKN001", "INJ031",
]
_CCSR_INDEX = {c: i for i, c in enumerate(TRACKED_CCSR)}

# memoised because ccsr() is a linear prefix scan and the same few hundred
# codes recur across a million claim rows
_CCSR_CACHE = {}


def _cat(code):
    v = _CCSR_CACHE.get(code)
    if v is None:
        v = C.ccsr_category(code) if isinstance(code, str) else "XXX000"
        _CCSR_CACHE[code] = v
    return v


_CHARLSON_CACHE = {}


def _charlson(codes_tuple):
    v = _CHARLSON_CACHE.get(codes_tuple)
    if v is None:
        v = CM.charlson_detail(list(codes_tuple))
        _CHARLSON_CACHE[codes_tuple] = v
    return v


def build_features(stays, med_store, rx_store, eligibility, members,
                   visibility="received", include_leaky=False, progress=False):
    """One row per index stay. as_of = discharge_date."""
    mem = members.set_index("member_id")
    age_by = mem["age"].to_dict()
    sex_by = mem["sex"].to_dict()
    spans = _spans_by_member(eligibility)

    D = lambda n: np.timedelta64(n, "D")  # noqa: E731
    rows = []
    n = len(stays)
    for k, r in enumerate(stays.itertuples(index=False)):
        if progress and k and k % 10000 == 0:
            print(f"    features {k:,}/{n:,}", flush=True)
        as_of = np.datetime64(pd.Timestamp(r.discharge_date).date(), "D")
        adm = np.datetime64(pd.Timestamp(r.admit_date).date(), "D")
        view = ClaimsView(med_store, rx_store, r.discharge_date, visibility)

        # history: [admit-365, admit).  receipt horizon: discharge.
        mc, m365 = view.medical(r.member_id, since=adm - D(365), until=adm)
        if mc is None:
            sd = np.array([], dtype="datetime64[D]")
            ct = dx = pv = np.array([], dtype=object)
            paid = los = np.array([])
        else:
            sd = mc.service[m365]
            ct = mc.ctype[m365]
            dx = mc.dx[m365]
            pv = mc.provider[m365]
            paid = mc.paid[m365]
            los = mc.los[m365]

        is_ed = ct == "ED"
        is_ip = ct == "IP"
        is_prof = ct == "PROF"
        ge90 = sd >= adm - D(90)
        ge30 = sd >= adm - D(30)
        ge180 = sd >= adm - D(180)

        dx_codes = tuple(sorted({d for d in dx if isinstance(d, str)}))
        charl, charl_hits = _charlson(dx_codes) if dx_codes else (0, {})
        cat_flags = np.zeros(len(TRACKED_CCSR), dtype=np.int8)
        for d in dx_codes:
            i = _CCSR_INDEX.get(_cat(d))
            if i is not None:
                cat_flags[i] = 1

        rx, rm = view.pharmacy(r.member_id, since=adm - D(180), until=adm)
        if rx is None:
            fills180 = classes180 = prescribers180 = 0
            supply = 0.0
        else:
            fills180 = int(rm.sum())
            classes180 = len(set(rx.cls[rm])) if fills180 else 0
            prescribers180 = len(set(rx.prescriber[rm])) if fills180 else 0
            supply = float(rx.days[rm].sum()) if fills180 else 0.0
        pdc_proxy = (min(1.0, supply / (180.0 * max(1, classes180)))
                     if classes180 else np.nan)

        cov, longest_gap, ngaps = _gap_profile(
            spans.get(r.member_id, []),
            r.admit_date - pd.Timedelta(365, unit="D"), r.admit_date)
        gap_days = max(0, LOOKBACK_DAYS - cov)

        row = {
            "stay_id": r.stay_id,
            "member_id": r.member_id,
            "discharge_date": r.discharge_date,
            # --- index stay, from the ADT/census feed (see module docstring)
            "los": int(r.los),
            "index_ccsr": _cat(r.principal_dx),
            # --- demographics
            "age": int(age_by[r.member_id]),
            "sex_f": int(sex_by[r.member_id] == "F"),
            # --- comorbidity
            "charlson": int(charl),
            "n_charlson_conditions": len(charl_hits),
            # --- utilisation
            "ed_visits_30d": int((is_ed & ge30).sum()),
            "ed_visits_90d": int((is_ed & ge90).sum()),
            "ed_visits_365d": int(is_ed.sum()),
            "ip_admits_365d": int(is_ip.sum()),
            "ip_days_365d": int(los[is_ip].sum()) if is_ip.any() else 0,
            "office_visits_90d": int((is_prof & ge90).sum()),
            "distinct_providers_180d": len(set(pv[ge180])) if ge180.any() else 0,
            "paid_amount_365d": float(paid.sum()) if paid.size else 0.0,
            # --- pharmacy
            "rx_fills_180d": fills180,
            "rx_classes_180d": classes180,
            "distinct_prescribers_180d": prescribers180,
            "pdc_proxy_180d": pdc_proxy,
            # --- eligibility
            "elig_gap_days_365d": int(gap_days),
            "any_elig_gap_365d": int(gap_days > 0),
            "longest_elig_gap_365d": int(longest_gap),
            # --- label
            "y": int(bool(r.true_readmit_30d)),
        }
        for i, cat in enumerate(TRACKED_CCSR):
            row[f"ccsr_{cat}"] = int(cat_flags[i])
        if include_leaky:
            row["LEAK_discharge_status"] = str(r.discharge_status)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["chf_or_copd"] = ((out["ccsr_CIR019"] == 1) | (out["ccsr_RSP008"] == 1)).astype(int)
    return out


FEATURE_COLS_EXCLUDE = {"stay_id", "member_id", "discharge_date", "y", "index_ccsr"}

# Exactly-redundant encodings, dropped from the design matrix but kept on the
# feature frame because the worklist and the subgroup tables read them.
#   chf_or_copd            == ccsr_CIR019 OR ccsr_RSP008 (both already columns)
#   longest_elig_gap_365d  ~= elig_gap_days_365d under a single-gap cohort rule
# Left in, they make the logistic coefficients unidentifiable and the signs
# flip arbitrarily between runs -- which is exactly what the first version of
# this file did, and the coefficient-recovery table is what caught it.
REDUNDANT_IN_DESIGN = ["chf_or_copd", "longest_elig_gap_365d"]


def design_matrix(df, leaky=False, columns=None):
    """Numeric design matrix.

    index_ccsr is one-hot encoded here so a category string cannot reach a
    linear model by accident. pdc_proxy is missing for members with no chronic
    fills, which is informative missingness (no chronic meds), so it gets an
    explicit indicator rather than a silent imputation.
    """
    X = df.drop(columns=[c for c in FEATURE_COLS_EXCLUDE if c in df.columns])
    X = X.drop(columns=[c for c in REDUNDANT_IN_DESIGN if c in X.columns])
    leak_cols = [c for c in X.columns if c.startswith("LEAK_")]
    if not leaky:
        X = X.drop(columns=leak_cols)
    else:
        for c in leak_cols:
            X = pd.concat([X.drop(columns=[c]),
                           pd.get_dummies(df[c], prefix=c).astype(int)], axis=1)
    X = pd.concat([X, pd.get_dummies(df["index_ccsr"], prefix="idx").astype(int)], axis=1)
    X["pdc_missing"] = X["pdc_proxy_180d"].isna().astype(int)
    X["pdc_proxy_180d"] = X["pdc_proxy_180d"].fillna(0.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X
