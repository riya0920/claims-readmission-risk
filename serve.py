"""Scoring service. Standard library only.

THREE DESIGN POSITIONS THIS API TAKES
-------------------------------------

1. IT REFUSES TO SCORE WITHOUT AN EXPLICIT `as_of`.
   "Score this member now" is an ambiguous request, and the ambiguity is the
   one this whole project is about: `now` does not say which claims had been
   RECEIVED at the moment being scored. A caller replaying a discharge from
   last month wants last month's visibility, not today's. Defaulting to
   `datetime.now()` would be convenient and would silently reintroduce the
   train/serve skew that docs/LEAKAGE_AUDIT.md exists to measure, so the API
   returns 400 instead.

2. IT REFUSES TO SERVE ON A SCHEMA MISMATCH.
   The model is loaded through `registry.ServableModel`, which compares the
   offered feature contract against the one the model was fitted on and raises
   rather than scoring. A model that declines to answer is recoverable; one
   that answers wrongly is not.

3. EVERY SCORING RESPONSE CARRIES ITS INTENDED USE.
   Not decoration. This score's safety property -- that its worst realistic
   failure is a wasted phone call -- holds only while it is used to ADD
   outreach. The moment it gates a benefit or an authorisation, a false
   negative becomes a denial of service to a high-risk member. An API that
   ships the score without that sentence is inviting the misuse, because the
   next system to consume it will not have read the validation document.

WHAT THIS IS NOT
----------------
No auth, no TLS, no rate limiting, no request log, no horizontal scaling, and
`http.server` is single-threaded-per-request by design here. It demonstrates
the serving CONTRACT, not a production service. A real deployment puts this
behind a gateway that does all of the above, and the scores themselves are
member-level health information subject to the same access controls as any
other PHI.

Run:  python serve.py --register     # train, register v1, exit
      python serve.py                # serve on 127.0.0.1:8081
      python serve.py --demo         # register, serve, exercise, exit
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import registry
import uncertainty as U
from generate import WINDOW_END
from train import at_capacity, load
from worklist import EXPLAIN, CCSR_PLAIN, occlusion_drivers, phrase

MODEL_NAME = "readmission-30d"
CUT = pd.Timestamp("2024-05-01")

_STATE = {"model": None, "features": None, "medians": None, "columns": None}


# ---------------------------------------------------------------------------
def train_and_register(version=1, datadir="data"):
    """Fit the shipped model and record the contract it was fitted under."""
    med, rx, el, mem, st = load(datadir)
    cohort, counts = F.build_cohort(st, el, WINDOW_END)
    feat = F.build_features(cohort, F.pack_medical(med), F.pack_pharmacy(rx),
                            el, mem, visibility="received")
    tr, te = feat[feat.discharge_date < CUT], feat[feat.discharge_date >= CUT]
    Xtr = F.design_matrix(tr)
    Xte = F.design_matrix(te, columns=Xtr.columns)

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=1.0))
    model.fit(Xtr, tr.y.values)
    p = model.predict_proba(Xte)[:, 1]

    groups = te.member_id.values
    auroc = U.bootstrap_ci(te.y.values, p, groups, "auroc", n_boot=300)
    sens = U.bootstrap_ci(te.y.values, p, groups, "sens_at_5pct", n_boot=300)
    cap = at_capacity(te.y.values, p, 0.05)

    stem = registry.save(
        model, list(Xtr.columns),
        name=MODEL_NAME, version=version,
        cohort={"waterfall": counts, "train_n": len(tr), "test_n": len(te),
                "temporal_cut": str(CUT.date())},
        metrics={"auroc": auroc, "sens_at_5pct": sens,
                 "ppv_at_5pct": cap["ppv"], "nnc": cap["nnc"]},
        visibility="received",
        calibration={"observed": float(te.y.mean()), "predicted": float(p.mean())},
        notes=("Logistic regression shipped over gradient boosting: the paired "
               "difference does not clear zero and the transparent model is "
               "worth more than a difference inside the noise."))
    print(f"registered {stem}")
    print(f"  AUROC       {auroc['point']:.4f} ({auroc['lo']:.4f}-{auroc['hi']:.4f})")
    print(f"  sens@5%     {sens['point']:.1%} ({sens['lo']:.1%}-{sens['hi']:.1%})")
    print(f"  features    {len(Xtr.columns)} ({registry.feature_hash(Xtr.columns)})")
    return stem


def warm(version=1, datadir="data"):
    """Load the model and the feature frame the API scores against."""
    servable = registry.load(MODEL_NAME, version)
    med, rx, el, mem, st = load(datadir)
    cohort, _counts = F.build_cohort(st, el, WINDOW_END)
    feat = F.build_features(cohort, F.pack_medical(med), F.pack_pharmacy(rx),
                            el, mem, visibility="received")
    cols = servable.columns
    _STATE.update({
        "model": servable, "features": feat, "columns": cols,
        "medians": F.design_matrix(feat[feat.discharge_date < CUT],
                                   columns=cols).median(),
    })
    return servable


# ---------------------------------------------------------------------------
# access control
# ---------------------------------------------------------------------------
# THE README CALLED THIS "a HIPAA incident waiting to happen", and it was
# right: an unauthenticated endpoint returning member-level risk scores is a
# bulk PHI disclosure to anyone who can reach the port.
#
# What is here is deliberately minimal -- a shared bearer token and an access
# log. It authenticates the CALLER, not a user, and it is not an identity
# system; se1-hl7-fhir-interop in this portfolio has the SMART scope layer and
# the append-only audit this service should really be writing to. The point of
# adding it is that "no auth at all" and "auth a reviewer can criticise" are
# different states, and only the second one is a starting position.

class _Access:
    """Bearer-token check plus an access log that records WHO SAW WHOM.

    The log is the half that matters. Authentication answers "may you ask";
    only the log answers "which members were disclosed, to whom, when" -- and
    that second question is the one an OCR investigation asks. A service that
    authenticates and does not log has a control and no evidence.

    Member ids are recorded because that is the point of the log. That makes
    the log itself PHI, with the same handling requirements as the scores --
    which is a real consequence and is why it is a bounded in-memory list here
    rather than something written to disk beside the code.
    """

    def __init__(self, token=None, limit=5000):
        self.token = token
        self.records = []
        self.limit = limit

    def check(self, header):
        if self.token is None:
            return True, None
        if not header or not header.startswith("Bearer "):
            return False, "no bearer token"
        if not hmac.compare_digest(header[7:], self.token):
            return False, "token does not match"
        return True, None

    def log(self, *, route, caller, as_of, member_ids, outcome):
        rec = {"at": time.time(), "route": route, "caller": caller,
               "as_of": str(as_of), "n_members": len(member_ids),
               "member_ids": list(member_ids)[:200], "outcome": outcome}
        self.records.append(rec)
        del self.records[:-self.limit]
        return rec

    def who_saw(self, member_id):
        return [r for r in self.records if member_id in r["member_ids"]]


ACCESS = _Access()


INTENDED_USE = (
    "Ranks discharges by modelled probability of an unplanned readmission "
    "CLAIM within 30 days, to order a capacity-bound outreach queue. This is "
    "not a diagnosis, not a clinical assessment, and not a prediction about "
    "any individual's health. Members not returned have NOT been cleared of "
    "risk. Must not be used to deny, delay, limit or price coverage, benefits "
    "or care: the safety argument for this model depends on it only ever "
    "ADDING outreach."
)


# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass                                  # quiet; the demo prints its own

    def _send(self, code, payload):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- GET ------------------------------------------------------------
    def _authorised(self, route):
        """401 unless the caller presents the token. /health is exempt.

        /health is exempt on purpose: a load balancer probing it does not hold
        a credential, and requiring one there means the probe fails and the
        service is removed from rotation while working perfectly. It returns no
        member data, which is the only reason exempting it is safe.
        """
        ok, why = ACCESS.check(self.headers.get("Authorization"))
        if ok:
            return True
        ACCESS.log(route=route, caller="<unauthenticated>", as_of=None,
                   member_ids=[], outcome=f"denied: {why}")
        self._send(401, {"error": "authentication required",
                         "detail": why,
                         "why": ("this endpoint returns member-level risk "
                                 "scores. Unauthenticated access to it is a "
                                 "bulk PHI disclosure.")})
        return False

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        model = _STATE["model"]

        if url.path not in ("/health",) and not self._authorised(url.path):
            return

        if url.path == "/health":
            return self._send(200, {
                "status": "ok" if model else "no model loaded",
                "model": MODEL_NAME if model else None,
                "version": model.manifest["version"] if model else None,
            })

        if url.path == "/model":
            if not model:
                return self._send(503, {"error": "no model loaded"})
            return self._send(200, model.manifest)

        if url.path == "/worklist":
            if not model:
                return self._send(503, {"error": "no model loaded"})
            as_of = (q.get("as_of") or [None])[0]
            if not as_of:
                return self._send(400, {
                    "error": "as_of is required",
                    "why": ("a worklist is a snapshot of what was known at a "
                            "moment. Without as_of the service would have to "
                            "guess, and guessing 'today' silently changes "
                            "which claims are visible."),
                    "example": "/worklist?as_of=2024-06-01&capacity=50"})
            capacity = int((q.get("capacity") or ["50"])[0])
            lookback = int((q.get("lookback_days") or ["30"])[0])
            body = self._worklist(as_of, capacity, lookback)
            ACCESS.log(route="/worklist",
                       caller=self.headers.get("X-Caller", "anonymous-token"),
                       as_of=as_of,
                       member_ids=[r["member_id"] for r in body["worklist"]],
                       outcome="success")
            return self._send(200, body)

        return self._send(404, {"error": f"no route {url.path}",
                                "routes": ["/health", "/model", "/worklist",
                                           "POST /score"]})

    # -- POST -----------------------------------------------------------
    def do_POST(self):
        url = urlparse(self.path)
        if not self._authorised(url.path):
            return
        if url.path != "/score":
            return self._send(404, {"error": f"no route {url.path}"})
        model = _STATE["model"]
        if not model:
            return self._send(503, {"error": "no model loaded"})

        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            return self._send(400, {"error": f"bad JSON: {exc}"})

        as_of = body.get("as_of")
        if not as_of:
            return self._send(400, {
                "error": "as_of is required",
                "why": ("'score this member now' does not say which claims had "
                        "been RECEIVED at the moment being scored. Defaulting "
                        "to the current time would silently reintroduce the "
                        "train/serve skew this project measures."),
                "see": "docs/LEAKAGE_AUDIT.md"})

        ids = body.get("stay_ids")
        if not ids:
            return self._send(400, {"error": "stay_ids is required"})
        lookback = int(body.get("lookback_days", 30))
        out = self._score(ids, as_of, lookback)
        ACCESS.log(route="/score",
                   caller=self.headers.get("X-Caller", "anonymous-token"),
                   as_of=as_of,
                   member_ids=[r["member_id"] for r in out["scored"]],
                   outcome="success")
        return self._send(200, out)

    # -- work -----------------------------------------------------------
    def _frame_as_of(self, as_of, lookback_days=30):
        """Discharges in the working window ending at as_of.

        TWO THINGS THIS FIXES, both found by running the API rather than by
        reading it.

        A WINDOW, NOT ALL HISTORY. The first version returned every discharge
        up to as_of, which at as_of=2024-06-01 was 23,402 stays. That is not a
        worklist -- a care-management team calls people discharged in the last
        few weeks, because transitional-care outreach has a shelf life. The
        default is 30 days, matching the outcome window.

        IN-SAMPLE ROWS ARE FLAGGED. Because the window can reach back before
        the training cut, some scored rows may be ones the model was FITTED
        on. Their scores are in-sample fits, not predictions, and they are
        systematically optimistic -- the first run returned risks of 98.4%
        against a test-set maximum near 73%, which is what an in-sample
        logistic fit looks like. The API now marks them rather than passing
        them off as forecasts.
        """
        feat = _STATE["features"]
        cut = pd.Timestamp(as_of)
        start = cut - pd.Timedelta(lookback_days, unit="D")
        return feat[(feat.discharge_date <= cut)
                    & (feat.discharge_date > start)]

    def _score(self, stay_ids, as_of, lookback_days=30):
        model = _STATE["model"]
        frame = self._frame_as_of(as_of, lookback_days)
        rows = frame[frame.stay_id.isin(stay_ids)]
        missing = sorted(set(stay_ids) - set(rows.stay_id))

        if rows.empty:
            return {"as_of": as_of, "scored": [], "not_found": missing,
                    "intended_use": INTENDED_USE}

        X = F.design_matrix(rows, columns=_STATE["columns"])
        model.check_schema(X.columns)          # refuses rather than guesses
        p = model.model.predict_proba(X)[:, 1]

        scored = []
        for i, (_idx, r) in enumerate(rows.iterrows()):
            _base, drivers = occlusion_drivers(model.model, X.reset_index(drop=True),
                                               _STATE["medians"], i)
            in_sample = pd.Timestamp(r.discharge_date) < CUT
            scored.append({
                "stay_id": r.stay_id, "member_id": r.member_id,
                "discharge_date": str(pd.Timestamp(r.discharge_date).date()),
                "risk": round(float(p[i]), 4),
                "in_training_set": bool(in_sample),
                "warning": ("this discharge predates the training cut, so the "
                            "score is an IN-SAMPLE fit and is optimistic -- it "
                            "is not a prediction") if in_sample else None,
                "reasons": [phrase(c, v) for c, _d, v in drivers],
                "reason_method": ("occlusion vs cohort median -- NOT SHAP; not "
                                  "additive and blind to interactions"),
            })
        return {"as_of": as_of, "model_version": model.manifest["version"],
                "scored": sorted(scored, key=lambda s: -s["risk"]),
                "not_found": missing, "intended_use": INTENDED_USE}

    def _worklist(self, as_of, capacity, lookback_days=30):
        model = _STATE["model"]
        frame = self._frame_as_of(as_of, lookback_days)
        if frame.empty:
            # SAME KEYS AS THE POPULATED RESPONSE. The first version returned a
            # three-key object here, so any client that read cohort_size or
            # model_version got a KeyError on a quiet day rather than a zero.
            # An empty result is a normal answer and has to have a normal shape.
            return {"as_of": as_of, "capacity": capacity,
                    "lookback_days": lookback_days, "cohort_size": 0,
                    "model_version": model.manifest["version"],
                    "worklist": [], "not_on_this_list": 0,
                    "in_training_set_count": 0,
                    "in_training_set_warning": None,
                    "intended_use": INTENDED_USE}
        X = F.design_matrix(frame, columns=_STATE["columns"])
        model.check_schema(X.columns)
        p = model.model.predict_proba(X)[:, 1]
        order = np.argsort(-p)[:capacity]
        rows = frame.iloc[order]
        n_in_sample = int((frame.discharge_date < CUT).sum())
        return {
            "as_of": as_of, "capacity": capacity,
            "lookback_days": lookback_days,
            "cohort_size": int(len(frame)),
            "model_version": model.manifest["version"],
            "worklist": [
                {"rank": i + 1, "member_id": r.member_id, "stay_id": r.stay_id,
                 "risk": round(float(p[order[i]]), 4),
                 "in_training_set": bool(pd.Timestamp(r.discharge_date) < CUT)}
                for i, (_ix, r) in enumerate(rows.iterrows())],
            "not_on_this_list": int(len(frame) - len(rows)),
            "in_training_set_count": n_in_sample,
            "in_training_set_warning": (
                f"{n_in_sample} of {len(frame)} discharges in this window "
                f"predate the training cut; their scores are in-sample fits "
                f"and are optimistic") if n_in_sample else None,
            "intended_use": INTENDED_USE,
        }


def serve(port=8081, version=1, datadir="data"):
    warm(version, datadir)
    httpd = HTTPServer(("127.0.0.1", port), Handler)
    print(f"serving {MODEL_NAME} v{version} on http://127.0.0.1:{port}")
    print("  GET  /health   GET /model   GET /worklist?as_of=...&capacity=N")
    print("  POST /score    {\"as_of\": \"...\", \"stay_ids\": [...]}")
    return httpd


def demo(port=8082, datadir="data"):
    """Register, serve, exercise every route, and prove the refusals."""
    import urllib.error
    import urllib.request

    if not os.path.exists(os.path.join("models", f"{MODEL_NAME}-v1.json")):
        train_and_register(1, datadir)
    httpd = serve(port, 1, datadir)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    base = f"http://127.0.0.1:{port}"

    def get(path):
        try:
            with urllib.request.urlopen(base + path) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(path, payload):
        req = urllib.request.Request(
            base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    print("\n" + "=" * 76)
    print("EXERCISING THE API")
    print("=" * 76)

    code, body = get("/health")
    print(f"  GET /health -> {code} {body['status']}, "
          f"{body['model']} v{body['version']}")

    code, body = get("/model")
    print(f"  GET /model  -> {code}, {body['n_features']} features, "
          f"hash {body['feature_hash']}")
    print(f"    AUROC {body['metrics']['auroc']['point']:.4f} "
          f"({body['metrics']['auroc']['lo']:.4f}-"
          f"{body['metrics']['auroc']['hi']:.4f})")

    print("\n  the two refusals:")
    code, body = post("/score", {"stay_ids": ["S00000001"]})
    print(f"    POST /score without as_of -> {code} {body['error']}")
    print(f"      {body['why'][:72]}...")
    code, body = get("/worklist?capacity=5")
    print(f"    GET /worklist without as_of -> {code} {body['error']}")

    code, body = get("/worklist?as_of=2024-06-01&capacity=5")
    print(f"\n  GET /worklist?as_of=2024-06-01&capacity=5 -> {code}")
    print(f"    cohort {body['cohort_size']:,}, returned {len(body['worklist'])}, "
          f"{body['not_on_this_list']:,} not on the list")
    for w in body["worklist"][:3]:
        print(f"      {w['rank']}. {w['member_id']}  risk {w['risk']:.1%}")

    ids = [w["stay_id"] for w in body["worklist"][:2]]
    code, body = post("/score", {"as_of": "2024-06-01", "stay_ids": ids})
    print(f"\n  POST /score -> {code}")
    for s in body["scored"]:
        print(f"    {s['member_id']} risk {s['risk']:.1%}")
        for reason in s["reasons"][:2]:
            print(f"      - {reason}")

    print(f"\n  every response carries intended use:")
    print(f"    \"{body['intended_use'][:68]}...\"")

    print("\n  as_of changes what is visible:")
    for d in ("2024-04-20", "2024-05-15", "2024-08-01"):
        _c, b = get(f"/worklist?as_of={d}&capacity=1")
        flag = ""
        if b.get("in_training_set_count"):
            flag = (f"  <- {b['in_training_set_count']} IN-SAMPLE, flagged "
                    f"(scores are fits, not predictions)")
        print(f"    as_of={d} -> {b['cohort_size']:,} discharges in the "
              f"{b['lookback_days']}-day window{flag}")
    print("    A worklist is a snapshot. The same request on two dates is two")
    print("    different questions, which is why as_of cannot be optional.")

    httpd.shutdown()
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--datadir", default="data")
    a = ap.parse_args()
    if a.register:
        train_and_register(a.version, a.datadir)
    elif a.demo:
        demo(a.port, a.datadir)
    else:
        serve(a.port, a.version, a.datadir).serve_forever()
