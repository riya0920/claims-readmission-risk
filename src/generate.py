"""Synthetic payer claims generator.

WHAT THIS IS NOT
----------------
This is not Synthea. Synthea is a Java application that simulates patients
through disease-progression modules and exports FHIR/CCDA/CSV; it is the right
tool and the spec names it. It is not runnable in this offline build, so this
module generates claims-shaped data directly.

That substitution costs something and the cost is stated: Synthea's clinical
trajectories come from curated disease modules with published provenance,
whereas this generator's trajectories come from a hand-specified risk equation.
The upside is that the risk equation is KNOWN, which is what makes the leakage
audit and the calibration checks verifiable rather than merely plausible --
we can ask whether the model recovered the truth, because we wrote the truth.

Both share the same headline limitation, and it is the one that matters:
synthetic claims are far cleaner than real claims. No duplicate submissions, no
adjustment/void pairs, no coding drift across a contract year, no provider
whose ICD-10 habits change when a new EHR goes live, no COB with a second
payer. A model that works here has not been shown to work on real claims.

WHAT IS DELIBERATELY MESSY
--------------------------
Three things, because they are the three that change modelling decisions:

1. CLAIM RUNOUT. Every claim carries service_date AND received_date, and the
   lag between them is long and right-skewed (facility claims lag more than
   professional). This is what creates train/serve skew in claims models and
   it is invisible in any dataset that ships one date column.
2. ELIGIBILITY GAPS. Members churn. Coverage is spans, not a boolean, and
   "no claims in that window" is ambiguous between healthy and not-enrolled.
3. A LEAKING FIELD. Discharge disposition is populated from the outcome,
   exactly as it is in reality (it is assigned at discharge/adjudication and
   encodes what happened next). It is left in the raw data on purpose so the
   leakage audit has something to catch.

Usage:  python src/generate.py [--members 50000] [--seed 7]
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

import codes as C

WINDOW_START = pd.Timestamp("2022-01-01")
WINDOW_END = pd.Timestamp("2024-12-31")

# ---------------------------------------------------------------------------
# TRUE risk equation for 30-day unplanned readmission.
# These coefficients are the ground truth the model is trying to recover.
# Written on the raw/centred inputs noted per term.
# ---------------------------------------------------------------------------
TRUE_COEF = {
    "intercept": -2.62,
    "charlson": 0.115,          # per point
    "ed_visits_90d": 0.240,     # per visit
    "ip_days_365d": 0.021,      # per prior inpatient day
    "nonadherence": 0.75,       # per unit of (1 - PDC), range 0..1
    "elig_gap_365d": 0.42,      # binary: any coverage gap in prior year
    "los": 0.031,               # per day of the index stay
    "age_z": 0.155,             # per SD of age
    # NEGATIVE CONTROL: declared here at zero and deliberately never entered
    # into the risk equation below. A model that assigns it a large coefficient
    # is inventing signal, and the recovery table will show that.
    "distinct_prescribers_180d": 0.0,
    "chf_or_copd": 0.33,        # binary: heart failure or COPD in history
    "prior_admits_365d": 0.30,  # per prior admission
}

CHRONIC = [
    "chf", "copd", "diabetes", "diabetes_c", "ckd", "cad_mi", "afib",
    "cancer", "mets", "cva", "dementia", "liver_mild", "liver_sev", "pvd",
    "pud", "rheum", "hemiplegia", "hiv", "psych", "sud",
]

# baseline log-odds and age slope for each chronic condition
CHRONIC_PREV = {
    "chf": (-3.4, 1.05), "copd": (-3.0, 0.55), "diabetes": (-1.9, 0.45),
    "diabetes_c": (-3.5, 0.60), "ckd": (-3.6, 0.95), "cad_mi": (-3.2, 0.90),
    "afib": (-3.9, 1.10), "cancer": (-3.9, 0.75), "mets": (-5.2, 0.55),
    "cva": (-4.0, 0.85), "dementia": (-5.0, 1.40), "liver_mild": (-4.2, 0.15),
    "liver_sev": (-6.0, 0.20), "pvd": (-4.0, 0.80), "pud": (-4.6, 0.30),
    "rheum": (-4.3, 0.25), "hemiplegia": (-5.4, 0.55), "hiv": (-5.6, -0.10),
    "psych": (-2.4, -0.15), "sud": (-3.3, -0.35),
}

ACUTE_DX = ["pneumonia", "sepsis", "aki", "uti", "cellulitis", "fall",
            "chest_pain", "syncope"]


def _days(n):
    """_days(...) trips a numpy 2.5 deprecation; this is the
    non-deprecated spelling, kept in one place."""
    return pd.Timedelta(int(n), unit="D")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _runout_lag(rng, n, facility):
    """Days between service and receipt. Right-skewed mixture.

    Facility (inpatient/outpatient hospital) claims bill later and slower than
    professional claims -- a UB-04 goes out after the stay is coded, a CMS-1500
    for the same day's rounding can go out that week. This asymmetry is why
    "claims from the last 60 days" means different things for different claim
    types, and it is the source of the train/serve skew this project measures.
    """
    u = rng.random(n)
    if facility:
        lag = np.where(u < 0.35, rng.integers(14, 31, n),
              np.where(u < 0.80, rng.integers(31, 61, n),
                                 rng.integers(61, 121, n)))
    else:
        lag = np.where(u < 0.60, rng.integers(3, 15, n),
              np.where(u < 0.90, rng.integers(15, 46, n),
                                 rng.integers(46, 121, n)))
    return lag.astype(int)


def generate(n_members=50000, seed=7, outdir="data"):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)

    # -- members ------------------------------------------------------------
    member_id = np.array([f"M{100000 + i}" for i in range(n_members)])
    age = np.clip(rng.normal(58, 17, n_members), 19, 95).astype(int)
    age_z = (age - age.mean()) / age.std()
    sex = rng.choice(["F", "M"], n_members, p=[0.53, 0.47])
    state = rng.choice(
        ["TX", "FL", "OH", "PA", "NY", "CA", "GA", "NC", "MI", "AZ"],
        n_members,
        p=[0.14, 0.13, 0.10, 0.10, 0.10, 0.13, 0.08, 0.08, 0.07, 0.07],
    )
    plan = rng.choice(["HMO-01", "PPO-02", "PPO-03", "HMO-04"], n_members,
                      p=[0.3, 0.35, 0.2, 0.15])
    # latent morbidity, not observable -- drives utilisation intensity
    morbidity = rng.gamma(2.0, 0.5, n_members)

    members = pd.DataFrame({
        "member_id": member_id, "age": age, "sex": sex, "state": state,
        "plan_id": plan,
    })

    # -- chronic conditions -------------------------------------------------
    cond = {}
    for c in CHRONIC:
        b0, b_age = CHRONIC_PREV[c]
        p = _sigmoid(b0 + b_age * age_z + 0.55 * (morbidity - morbidity.mean()))
        cond[c] = rng.random(n_members) < p
    cond["mets"] = cond["mets"] & cond["cancer"]           # mets implies cancer
    cond["liver_sev"] = cond["liver_sev"] & cond["liver_mild"]
    cond["diabetes_c"] = cond["diabetes_c"] & cond["diabetes"]
    n_chronic = np.sum([cond[c] for c in CHRONIC], axis=0)

    # -- eligibility spans --------------------------------------------------
    # 68% enrolled the whole window; the rest start late, end early, or churn
    # out and back in (the gap that predicts loss to follow-up).
    elig_rows = []
    pattern = rng.choice(["full", "late_start", "early_end", "gap"],
                         n_members, p=[0.68, 0.11, 0.09, 0.12])
    for i in range(n_members):
        mid = member_id[i]
        if pattern[i] == "full":
            elig_rows.append((mid, WINDOW_START, WINDOW_END, plan[i]))
        elif pattern[i] == "late_start":
            s = WINDOW_START + _days(int(rng.integers(30, 500)))
            elig_rows.append((mid, s, WINDOW_END, plan[i]))
        elif pattern[i] == "early_end":
            e = WINDOW_END - _days(int(rng.integers(30, 500)))
            elig_rows.append((mid, WINDOW_START, e, plan[i]))
        else:
            g0 = WINDOW_START + _days(int(rng.integers(120, 700)))
            g1 = g0 + _days(int(rng.integers(30, 210)))
            if g1 >= WINDOW_END:
                g1 = WINDOW_END - _days(15)
            elig_rows.append((mid, WINDOW_START, g0, plan[i]))
            elig_rows.append((mid, g1, WINDOW_END, plan[i]))
    eligibility = pd.DataFrame(
        elig_rows, columns=["member_id", "span_start", "span_end", "plan_id"])

    covered = {}
    for mid, s, e in eligibility[["member_id", "span_start", "span_end"]].itertuples(index=False):
        covered.setdefault(mid, []).append((s, e))

    def is_covered(mid, d):
        return any(s <= d <= e for s, e in covered[mid])

    # -- claims -------------------------------------------------------------
    med, rx, admits = [], [], []
    pdc_list = []
    claim_seq = 0
    window_days = (WINDOW_END - WINDOW_START).days

    for i in range(n_members):
        mid = member_id[i]
        m = morbidity[i]
        nc = n_chronic[i]
        my_conditions = [c for c in CHRONIC if cond[c][i]]
        hx_codes = [_pick(rng, C.CONDITION_CODES[c]) for c in my_conditions]

        # ---- chronic maintenance visits
        n_office = rng.poisson(max(0.2, (1.4 + 1.5 * nc) * 3))
        for _ in range(int(n_office)):
            d = WINDOW_START + _days(int(rng.integers(window_days)))
            if not is_covered(mid, d):
                continue
            dx = _pick(rng, hx_codes) if hx_codes else "R07.9"
            claim_seq += 1
            med.append((f"C{claim_seq:08d}", mid, d, None, "PROF",
                        _pick(rng, C.CPT["office_visit"]), C.POS["office"],
                        dx, None, None, round(float(rng.gamma(3, 45)), 2),
                        f"P{int(rng.integers(1, 900)):04d}", None, None, None))

        # ---- ED visits
        lam_ed = 0.20 + 0.55 * m + 0.22 * nc + (0.5 if cond["sud"][i] else 0)
        for _ in range(int(rng.poisson(lam_ed * 3))):
            d = WINDOW_START + _days(int(rng.integers(window_days)))
            if not is_covered(mid, d):
                continue
            dx = _pick(rng, C.CONDITION_CODES[_pick(rng, ACUTE_DX)])
            claim_seq += 1
            med.append((f"C{claim_seq:08d}", mid, d, None, "ED",
                        _pick(rng, C.CPT["ed_visit"]), C.POS["emergency_room"],
                        dx, None, None, round(float(rng.gamma(4, 320)), 2),
                        f"P{int(rng.integers(1, 900)):04d}", None, None, None))

        # ---- inpatient admissions
        lam_ip = 0.05 + 0.30 * m + 0.22 * (cond["chf"][i] or cond["copd"][i]
                                           or cond["ckd"][i])
        n_adm = int(rng.poisson(lam_ip * 3))
        for _ in range(n_adm):
            adm = WINDOW_START + _days(int(rng.integers(window_days - 40)))
            if not is_covered(mid, adm):
                continue
            los = int(np.clip(rng.gamma(2.0, 2.2) + 1, 1, 45))
            dis = adm + _days(los)
            planned = rng.random() < 0.09
            dx = (_pick(rng, C.CONDITION_CODES["cancer"]) if planned and cond["cancer"][i]
                  else _pick(rng, C.CONDITION_CODES[_pick(rng, ACUTE_DX)]))
            admits.append({
                "member_id": mid, "admit_date": adm, "discharge_date": dis,
                "los": los, "principal_dx": dx, "planned": planned,
                "idx": i,
            })

        # ---- pharmacy
        classes = sorted({cl for c in my_conditions
                          for cl in C.CHRONIC_CLASSES.get(c, [])})
        pdc = float(np.clip(rng.beta(5, 2) - 0.25 * (m > 1.5), 0.05, 1.0))
        for cl in classes:
            ndc = _pick(rng, C.NDC[cl])
            d = WINDOW_START + _days(int(rng.integers(60)))
            while d < WINDOW_END:
                if is_covered(mid, d) and rng.random() < pdc:
                    claim_seq += 1
                    rx.append((f"R{claim_seq:08d}", mid, d, ndc, cl, 30,
                               round(float(rng.gamma(2, 22)), 2),
                               f"P{int(rng.integers(1, 900)):04d}"))
                d += _days(30)
        pdc_list.append(pdc)

    members["true_pdc"] = pdc_list
    admits = pd.DataFrame(admits).sort_values(["member_id", "admit_date"])
    admits = admits.reset_index(drop=True)

    # -- readmission outcome, from the TRUE risk equation --------------------
    # Features here are computed from the generator's own state (i.e. from
    # complete knowledge), which is exactly what makes them ground truth.
    adm_by_member = {mid: g for mid, g in admits.groupby("member_id")}
    ed_dates = {}
    for row in med:
        if row[4] == "ED":
            ed_dates.setdefault(row[1], []).append(row[2])

    rows = []
    for r in admits.itertuples(index=False):
        i = r.idx
        prior = adm_by_member[r.member_id]
        prior_admits = int(((prior.discharge_date < r.admit_date) &
                            (prior.discharge_date >= r.admit_date - _days(365))).sum())
        ip_days_365 = int(prior.loc[(prior.discharge_date < r.admit_date) &
                                    (prior.discharge_date >= r.admit_date - _days(365)),
                                    "los"].sum())
        eds = ed_dates.get(r.member_id, [])
        ed90 = sum(1 for d in eds if r.admit_date - _days(90) <= d < r.admit_date)
        gap = pattern[i] == "gap"
        z = (TRUE_COEF["intercept"]
             + TRUE_COEF["charlson"] * 0.0   # filled below
             + TRUE_COEF["ed_visits_90d"] * ed90
             + TRUE_COEF["ip_days_365d"] * ip_days_365
             + TRUE_COEF["nonadherence"] * (1 - members.true_pdc.iloc[i])
             + TRUE_COEF["elig_gap_365d"] * gap
             + TRUE_COEF["los"] * r.los
             + TRUE_COEF["age_z"] * age_z[i]
             + TRUE_COEF["prior_admits_365d"] * prior_admits
             + TRUE_COEF["chf_or_copd"] * (cond["chf"][i] or cond["copd"][i]))
        rows.append((z, ed90, ip_days_365, prior_admits, gap))
    admits["_z"] = [r[0] for r in rows]

    # charlson from the member's chronic conditions (true, not code-derived)
    import comorbidity as CM
    charl = np.array([
        CM.charlson_score([_pick(rng, C.CONDITION_CODES[c])
                           for c in CHRONIC if cond[c][i]] or ["R07.9"])
        for i in range(n_members)
    ])
    admits["_z"] += TRUE_COEF["charlson"] * charl[admits["idx"].values]

    p = _sigmoid(admits["_z"].values)
    died = rng.random(len(admits)) < np.clip(0.010 + 0.055 * p, 0, 0.30)
    readmit = (rng.random(len(admits)) < p) & (~died)
    admits["died_inpatient"] = died
    admits["true_readmit_30d"] = readmit
    admits["true_p"] = p

    # -- materialise the readmission stays and the discharge disposition -----
    extra = []
    disp = np.full(len(admits), "01", dtype=object)
    for j, r in enumerate(admits.itertuples(index=False)):
        if r.died_inpatient:
            disp[j] = "20"
            continue
        if r.true_readmit_30d:
            # the readmit stay itself
            gapdays = int(rng.integers(2, 30))
            adm2 = r.discharge_date + _days(gapdays)
            los2 = int(np.clip(rng.gamma(2.0, 2.4) + 1, 1, 40))
            extra.append({
                "member_id": r.member_id, "admit_date": adm2,
                "discharge_date": adm2 + _days(los2), "los": los2,
                "principal_dx": _pick(rng, C.CONDITION_CODES[_pick(rng, ACUTE_DX)]),
                "planned": False, "idx": r.idx, "_z": np.nan,
                "died_inpatient": False, "true_readmit_30d": False,
                "true_p": np.nan, "is_readmit_stay": True,
            })
            # LEAK: disposition is drawn from the outcome
            disp[j] = _pick(rng, ["03", "07", "03", "06", "01"])
        else:
            disp[j] = _pick(rng, ["01", "01", "01", "06", "62"])
    admits["discharge_status"] = disp
    admits["is_readmit_stay"] = False
    if extra:
        admits = pd.concat([admits, pd.DataFrame(extra)], ignore_index=True)
        admits["discharge_status"] = admits["discharge_status"].fillna("01")
        admits["is_readmit_stay"] = admits["is_readmit_stay"].fillna(False).astype(bool)
        admits["planned"] = admits["planned"].fillna(False).astype(bool)
        admits["died_inpatient"] = admits["died_inpatient"].fillna(False).astype(bool)
        admits["true_readmit_30d"] = admits["true_readmit_30d"].fillna(False).astype(bool)
    admits = admits.sort_values(["member_id", "admit_date"]).reset_index(drop=True)
    admits["stay_id"] = [f"S{i:08d}" for i in range(len(admits))]

    # -- inpatient facility + professional claims for every stay ------------
    for r in admits.itertuples(index=False):
        claim_seq += 1
        med.append((f"C{claim_seq:08d}", r.member_id, r.admit_date,
                    r.discharge_date, "IP",
                    _pick(rng, C.CPT["chemo"] if r.planned else C.CPT["ip_initial"]),
                    C.POS["inpatient_hospital"], r.principal_dx,
                    r.discharge_status, r.los,
                    round(float(rng.gamma(3, 3200) + 900 * r.los), 2),
                    f"P{int(rng.integers(1, 900)):04d}", r.stay_id,
                    bool(r.planned), None))
        claim_seq += 1
        med.append((f"C{claim_seq:08d}", r.member_id, r.discharge_date, None,
                    "PROF", _pick(rng, C.CPT["ip_discharge"]),
                    C.POS["inpatient_hospital"], r.principal_dx, None, None,
                    round(float(rng.gamma(3, 90)), 2),
                    f"P{int(rng.integers(1, 900)):04d}", r.stay_id, None, None))

    cols = ["claim_id", "member_id", "service_date", "service_end_date",
            "claim_type", "procedure_code", "place_of_service", "dx_code",
            "discharge_status", "los", "paid_amount", "provider_id",
            "stay_id", "planned_admission", "_unused"]
    medical = pd.DataFrame(med, columns=cols).drop(columns=["_unused"])
    pharmacy = pd.DataFrame(rx, columns=[
        "rx_claim_id", "member_id", "fill_date", "ndc", "therapeutic_class",
        "days_supply", "paid_amount", "prescriber_id"])

    # -- claim runout: received_date -----------------------------------------
    facility = medical["claim_type"].isin(["IP", "ED"]).values
    lag = np.empty(len(medical), dtype=int)
    lag[facility] = _runout_lag(rng, int(facility.sum()), True)
    lag[~facility] = _runout_lag(rng, int((~facility).sum()), False)
    medical["received_date"] = medical["service_date"] + pd.to_timedelta(lag, unit="D")
    pharmacy["received_date"] = pharmacy["fill_date"] + pd.to_timedelta(
        _runout_lag(rng, len(pharmacy), False) // 4, unit="D")  # rx adjudicates at POS

    members["morbidity_latent"] = morbidity
    for c in CHRONIC:
        members[f"true_{c}"] = cond[c]
    members["true_charlson"] = charl

    medical.to_csv(f"{outdir}/medical_claims.csv.gz", index=False)
    pharmacy.to_csv(f"{outdir}/pharmacy_claims.csv.gz", index=False)
    eligibility.to_csv(f"{outdir}/eligibility.csv.gz", index=False)
    members.to_csv(f"{outdir}/members.csv.gz", index=False)
    admits.drop(columns=["idx"]).to_csv(f"{outdir}/_truth_stays.csv.gz", index=False)

    idx = admits[(~admits.is_readmit_stay)]
    print(f"members            {len(members):,}")
    print(f"medical claims     {len(medical):,}")
    print(f"pharmacy claims    {len(pharmacy):,}")
    print(f"eligibility spans  {len(eligibility):,}")
    print(f"inpatient stays    {len(admits):,}")
    print(f"  index-eligible   {len(idx):,}")
    print(f"  readmit rate     {idx.true_readmit_30d.mean():.1%}")
    print(f"  died inpatient   {idx.died_inpatient.mean():.1%}")
    print(f"median runout lag  facility {np.median(lag[facility]):.0f}d / "
          f"professional {np.median(lag[~facility]):.0f}d")
    return medical, pharmacy, eligibility, members, admits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    generate(a.members, a.seed, a.outdir)
