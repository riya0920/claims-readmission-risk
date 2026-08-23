"""Run the overnight batch, and demonstrate the failure modes it exists for.

Run:  python run_batch.py
"""

from __future__ import annotations

import json
import os
import secrets
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import batch as B
import fairness as F
import features as FE
import registry
from generate import WINDOW_END
from serve import CUT, MODEL_NAME, occlusion_drivers, phrase
from train import load

OUT = "out"
DB = "out/batch.db"


def score_window(servable, feat, as_of, lookback_days):
    cut = pd.Timestamp(as_of)
    start = cut - pd.Timedelta(lookback_days, unit="D")
    frame = feat[(feat.discharge_date <= cut) & (feat.discharge_date > start)]
    if frame.empty:
        return frame, np.array([]), None
    X = FE.design_matrix(frame, columns=servable.columns)
    servable.check_schema(X.columns)
    return frame, servable.model.predict_proba(X)[:, 1], X


def main(datadir="data", as_of="2024-08-01", lookback_days=30, capacity=50):
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    con = B.connect(DB)

    servable = registry.load(MODEL_NAME, 1)
    med, rx, el, mem, st = load(datadir)
    cohort, _counts = FE.build_cohort(st, el, WINDOW_END)
    feat = FE.build_features(cohort, FE.pack_medical(med), FE.pack_pharmacy(rx),
                             el, mem, visibility="received")

    frame, p, X = score_window(servable, feat, as_of, lookback_days)
    ch = B.cohort_hash(frame.stay_id)

    print("=" * 76)
    print("OVERNIGHT BATCH")
    print("=" * 76)
    print(f"  as_of {as_of}   window {lookback_days}d   cohort {len(frame):,} "
          f"discharges   hash {ch}")

    job_id = "job-" + secrets.token_hex(4)
    job_id, reused = B.start(con, job_id=job_id, model_name=MODEL_NAME,
                             model_version=1, as_of=as_of,
                             lookback_days=lookback_days, capacity=capacity,
                             cohort_hash_value=ch)
    order = np.argsort(-p)[:capacity]
    medians = FE.design_matrix(feat[feat.discharge_date < CUT],
                               columns=servable.columns).median()
    Xr = X.reset_index(drop=True)
    rows = []
    for rank, i in enumerate(order):
        r = frame.iloc[i]
        _b, drivers = occlusion_drivers(servable.model, Xr, medians, int(i))
        rows.append({"member_id": r.member_id, "stay_id": r.stay_id,
                     "discharge_date": pd.Timestamp(r.discharge_date).date(),
                     "risk": float(p[i]),
                     "in_training_set": bool(pd.Timestamp(r.discharge_date) < CUT),
                     "reasons": [phrase(c, v) for c, _d, v in drivers]})
    B.finish(con, job_id, rows, n_scored=len(frame))
    print(f"  job {job_id}: scored {len(frame):,}, queued {len(rows)}")

    # ---- idempotency --------------------------------------------------------
    print("\n" + "-" * 76)
    print("RERUNNING THE SAME JOB")
    print("-" * 76)
    again_id, reused = B.start(con, job_id="job-" + secrets.token_hex(4),
                               model_name=MODEL_NAME, model_version=1,
                               as_of=as_of, lookback_days=lookback_days,
                               capacity=capacity, cohort_hash_value=ch)
    print(f"  returned {again_id} reused={reused}")
    assert reused and again_id == job_id
    print("  The ORIGINAL job, not a recomputation. A rerun at 06:00 after a")
    print("  03:00 failure must not hand the team a second queue that")
    print("  disagrees with the one they already started working.")

    # ---- a changed cohort is a NEW job --------------------------------------
    changed = B.cohort_hash(list(frame.stay_id)[:-1])
    new_id, reused2 = B.start(con, job_id="job-" + secrets.token_hex(4),
                              model_name=MODEL_NAME, model_version=1,
                              as_of=as_of, lookback_days=lookback_days,
                              capacity=capacity, cohort_hash_value=changed)
    print(f"\n  one discharge removed -> new job {new_id} reused={reused2}")
    print("  Keying idempotency on (model, as_of) alone would return the OLD")
    print("  answer after a late claim changed the cohort -- the opposite")
    print("  failure from duplicates, and much harder to notice.")
    B.fail(con, new_id, RuntimeError("demo"))

    # ---- partial failure ----------------------------------------------------
    print("\n" + "-" * 76)
    print("A JOB THAT DIES MID-WRITE")
    print("-" * 76)
    dead = "job-" + secrets.token_hex(4)
    B.start(con, job_id=dead, model_name=MODEL_NAME, model_version=1,
            as_of="2024-07-01", lookback_days=lookback_days, capacity=capacity,
            cohort_hash_value="deadbeefdeadbeef")
    try:
        B.finish(con, dead, [{"member_id": "M1", "stay_id": "S1",
                              "discharge_date": "2024-07-01", "risk": 0.5},
                             {"member_id": "M2"}], n_scored=2)   # malformed
    except Exception as exc:                                      # noqa: BLE001
        B.fail(con, dead, exc)
        print(f"  failed with {type(exc).__name__}")
    left = B.worklist(con, dead)
    print(f"  rows visible from the failed job: {len(left)}")
    assert left == []
    print("  A job is succeeded or it is nothing. 4,000 of 5,000 discharges")
    print("  left visible would be a queue silently missing the sickest fifth")
    print("  of the population, with nothing about it saying so.")

    # ---- the queue is worked ------------------------------------------------
    print("\n" + "-" * 76)
    print("COVERAGE -- the number a model dashboard never shows")
    print("-" * 76)
    wl = B.worklist(con, job_id)
    for r in wl[:18]:
        B.record_outreach(con, job_id, r["stay_id"], "care-manager-1",
                          "reached" if r["rank"] % 3 else "no answer")
    cov = B.coverage(con, job_id)
    print(f"  queued {cov['queued']}   worked {cov['worked']}   "
          f"coverage {cov['coverage']:.0%}   {cov['by_outcome']}")
    print("  A model with excellent sensitivity attached to a queue nobody")
    print("  works delivers nothing, and that failure is invisible in every")
    print("  metric train.py reports.")

    # ---- fairness -----------------------------------------------------------
    print("\n" + "=" * 76)
    print("FAIRNESS BEYOND CALIBRATION")
    print("=" * 76)
    test = feat[feat.discharge_date >= CUT]
    Xt = FE.design_matrix(test, columns=servable.columns)
    pt = servable.model.predict_proba(Xt)[:, 1]
    groups = np.where(test.age.values >= 65, "65+", "under 65")
    thr = F.capacity_threshold(pt, 0.05)
    rep = F.group_report(test.y.values, pt, groups, thr)

    print(f"  threshold set by CAPACITY (top 5%) = {thr:.4f}, not 0.5 --")
    print("  evaluating parity at 0.5 measures a rule nobody runs.\n")
    print(f"  {'group':<12}{'n':>7}{'prev':>8}{'sel':>8}{'TPR':>8}{'FPR':>8}"
          f"{'FNR':>8}{'PPV':>8}")
    for g, r in rep.items():
        print(f"  {g:<12}{r['n']:>7,}{r['prevalence']:>8.1%}"
              f"{r['selection_rate']:>8.1%}{r['tpr']:>8.1%}{r['fpr']:>8.1%}"
              f"{r['fnr']:>8.1%}{r['ppv']:>8.1%}")

    gaps = F.parity_gaps(rep)
    print("\n  largest pairwise gaps:")
    for k in ("tpr", "fpr", "fnr", "selection_rate"):
        g = gaps["gaps"][k]
        if g:
            print(f"    {k:<16}{g['gap']:+.1%}  ({g['max_group']} vs "
                  f"{g['min_group']})")

    imp = F.impossibility_check(rep)
    print(f"\n  {imp['reading']}")

    rec = F.recommend_criterion(gates_a_benefit=False)
    print(f"\n  criterion to act on: {rec['criterion'].upper()}")
    print(f"    {rec['because']}")
    print(f"    ASSUMPTION: {rec['assumption'][:200]}")

    cal = F.calibration_within_groups(test.y.values, pt, groups)
    print("\n  calibration within groups (max |observed - predicted| per band):")
    for g, r in cal.items():
        if "max_abs_gap" in r:
            print(f"    {g:<12}n={r['n']:<6}{r['max_abs_gap']:.3f}")

    ab = F.access_bias_note()
    print(f"\n  access bias: NOT closeable here -- {ab['why']}")

    payload = {"job": job_id, "coverage": cov, "history": B.history(con),
               "fairness": {"threshold": thr, "groups": rep, "gaps": gaps,
                            "impossibility": imp, "recommendation": rec,
                            "calibration_within_groups": cal,
                            "access_bias": ab}}
    with open(f"{OUT}/batch.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    con.close()
    print(f"\nwrote {OUT}/batch.json")
    return payload


if __name__ == "__main__":
    main()
