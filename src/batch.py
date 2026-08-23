"""The overnight batch path, with a job table and idempotent reruns.

WHAT THE README SAID WAS MISSING
--------------------------------
"Scoring is per-request. A real programme scores every discharge overnight and
hands the care team a queue in the morning, which is a scheduler and a job
table, not an API."

WHY BATCH IS NOT "THE API IN A LOOP"
-------------------------------------
It has failure modes the request path does not, and every one of them is about
what happens the second time:

  RERUNS ARE NORMAL, NOT EXCEPTIONAL. A batch fails at 03:00 and someone reruns
  it at 06:00. If the run is not idempotent the care team gets a queue with
  duplicates, or worse, two queues that disagree. Every job here is keyed on
  (model, as_of, cohort_hash) and a rerun of an identical job returns the
  ORIGINAL result rather than recomputing -- so a rerun cannot produce a
  different answer from the one the team already worked.

  PARTIAL FAILURE IS THE COMMON CASE. A job that dies after scoring 4,000 of
  5,000 discharges must not leave 4,000 rows visible as if they were a
  worklist. The queue is written in one transaction at the end and a job is
  `succeeded` or it is nothing -- no partial queue is ever readable.

  THE as_of IS THE JOB'S, NOT THE CLOCK'S. A batch that runs at 03:00 and asks
  for "now" gets a different answer from the same batch rerun at 06:00, because
  three more hours of claims arrived. The job records its as_of and the rerun
  uses the RECORDED one. This is the same discipline `serve.py` enforces by
  refusing an implicit as_of, applied to the path where the temptation is
  strongest.

  A QUEUE HAS A LIFECYCLE. Rows are worked, and a batch that overwrites
  yesterday's queue destroys the record of who was called. Each run writes a
  new queue and the previous one is retained, which is also what makes "did we
  call the people we said we would" answerable.

WHAT THIS IS NOT
----------------
Not a scheduler. There is no cron, no Airflow, no retry policy, no alerting on
a missed run, no backfill orchestration, no distributed execution and no
concurrency control beyond SQLite's. `run_batch.py` runs one job when invoked;
something else has to invoke it, and "the batch did not run at all" is the
failure that needs monitoring most and is not monitored here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_job (
    job_id        TEXT PRIMARY KEY,
    model_name    TEXT NOT NULL,
    model_version INTEGER NOT NULL,
    as_of         TEXT NOT NULL,
    lookback_days INTEGER NOT NULL,
    capacity      INTEGER NOT NULL,
    cohort_hash   TEXT NOT NULL,
    status        TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    n_scored      INTEGER,
    n_queued      INTEGER,
    error_class   TEXT,
    UNIQUE (model_name, model_version, as_of, lookback_days, capacity,
            cohort_hash)
);

CREATE TABLE IF NOT EXISTS worklist_row (
    job_id        TEXT NOT NULL,
    rank          INTEGER NOT NULL,
    member_id     TEXT NOT NULL,
    stay_id       TEXT NOT NULL,
    discharge_date TEXT NOT NULL,
    risk          REAL NOT NULL,
    in_training_set INTEGER NOT NULL DEFAULT 0,
    reasons       TEXT,
    PRIMARY KEY (job_id, rank)
);

CREATE TABLE IF NOT EXISTS outreach (
    job_id      TEXT NOT NULL,
    stay_id     TEXT NOT NULL,
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    note        TEXT,
    PRIMARY KEY (job_id, stay_id)
);
"""

TERMINAL = ("succeeded", "failed")


class BatchError(Exception):
    pass


def connect(path=":memory:"):
    con = sqlite3.connect(path, isolation_level=None)
    con.executescript(SCHEMA)
    return con


def cohort_hash(stay_ids):
    """Identity of the population, so a rerun on CHANGED data is a new job.

    Idempotency keyed only on (model, as_of) would return yesterday's answer
    after a late claim changed the cohort -- which is the opposite failure from
    duplicates, and harder to notice. Hashing the cohort makes "same inputs" the
    condition rather than "same request".
    """
    joined = "|".join(sorted(str(s) for s in stay_ids))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_existing(con, *, model_name, model_version, as_of, lookback_days,
                  capacity, cohort_hash_value):
    row = con.execute(
        "SELECT job_id, status FROM batch_job WHERE model_name=? AND "
        "model_version=? AND as_of=? AND lookback_days=? AND capacity=? AND "
        "cohort_hash=?",
        (model_name, model_version, str(as_of), lookback_days, capacity,
         cohort_hash_value)).fetchone()
    return row


def start(con, *, job_id, model_name, model_version, as_of, lookback_days,
          capacity, cohort_hash_value):
    """Claim a job. Returns (job_id, reused) -- reused means do not recompute."""
    existing = find_existing(
        con, model_name=model_name, model_version=model_version, as_of=as_of,
        lookback_days=lookback_days, capacity=capacity,
        cohort_hash_value=cohort_hash_value)
    if existing:
        prior_id, status = existing
        if status == "succeeded":
            return prior_id, True
        if status == "running":
            raise BatchError(
                f"job {prior_id} for the same inputs is already running. Two "
                f"concurrent batches would produce two queues that disagree, "
                f"and the care team has no way to tell which one to work.")
        # a previously FAILED job for identical inputs is replaced, not reused
        con.execute("DELETE FROM batch_job WHERE job_id=?", (prior_id,))
        con.execute("DELETE FROM worklist_row WHERE job_id=?", (prior_id,))

    con.execute(
        "INSERT INTO batch_job (job_id, model_name, model_version, as_of, "
        "lookback_days, capacity, cohort_hash, status, started_at) "
        "VALUES (?,?,?,?,?,?,?,'running',?)",
        (job_id, model_name, model_version, str(as_of), lookback_days,
         capacity, cohort_hash_value, _now()))
    return job_id, False


def finish(con, job_id, rows, n_scored):
    """Write the queue and mark the job succeeded, in ONE transaction.

    ALL OR NOTHING. A job that dies after scoring 4,000 of 5,000 discharges
    must not leave 4,000 rows visible as a worklist -- the team would work a
    queue that silently omits the sickest fifth of the population, and nothing
    about the queue would say so.
    """
    try:
        con.execute("BEGIN IMMEDIATE")
        con.executemany(
            "INSERT INTO worklist_row (job_id, rank, member_id, stay_id, "
            "discharge_date, risk, in_training_set, reasons) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(job_id, i + 1, r["member_id"], r["stay_id"],
              str(r["discharge_date"]), float(r["risk"]),
              int(r.get("in_training_set", 0)),
              json.dumps(r.get("reasons", [])))
             for i, r in enumerate(rows)])
        con.execute(
            "UPDATE batch_job SET status='succeeded', finished_at=?, "
            "n_scored=?, n_queued=? WHERE job_id=?",
            (_now(), int(n_scored), len(rows), job_id))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def fail(con, job_id, exc):
    """Record the failure CLASS, and leave no partial queue behind."""
    con.execute("DELETE FROM worklist_row WHERE job_id=?", (job_id,))
    con.execute(
        "UPDATE batch_job SET status='failed', finished_at=?, error_class=? "
        "WHERE job_id=?", (_now(), type(exc).__name__, job_id))


def worklist(con, job_id):
    rows = con.execute(
        "SELECT rank, member_id, stay_id, discharge_date, risk, "
        "in_training_set, reasons FROM worklist_row WHERE job_id=? "
        "ORDER BY rank", (job_id,)).fetchall()
    return [{"rank": r[0], "member_id": r[1], "stay_id": r[2],
             "discharge_date": r[3], "risk": r[4],
             "in_training_set": bool(r[5]), "reasons": json.loads(r[6] or "[]")}
            for r in rows]


def record_outreach(con, job_id, stay_id, actor, outcome, note=""):
    """Who was actually called, against the queue that told them to call.

    Kept per JOB rather than per member, which is what makes "did we work the
    queue we published" answerable. A batch that overwrote yesterday's queue
    would destroy the only record of what the team was asked to do.
    """
    con.execute(
        "INSERT OR REPLACE INTO outreach VALUES (?,?,?,?,?,?)",
        (job_id, stay_id, _now(), actor, outcome, note))


def coverage(con, job_id):
    """What fraction of the published queue was actually worked?

    The number a programme manager needs and a model dashboard never shows. A
    model with excellent sensitivity attached to a queue nobody works delivers
    nothing, and that failure is invisible in every metric in train.py.
    """
    total = con.execute(
        "SELECT COUNT(*) FROM worklist_row WHERE job_id=?", (job_id,)).fetchone()[0]
    worked = con.execute(
        "SELECT COUNT(*) FROM outreach WHERE job_id=?", (job_id,)).fetchone()[0]
    by_outcome = dict(con.execute(
        "SELECT outcome, COUNT(*) FROM outreach WHERE job_id=? GROUP BY outcome",
        (job_id,)).fetchall())
    return {"queued": total, "worked": worked,
            "coverage": (worked / total) if total else 0.0,
            "by_outcome": by_outcome}


def history(con, limit=20):
    rows = con.execute(
        "SELECT job_id, as_of, status, started_at, finished_at, n_scored, "
        "n_queued, error_class FROM batch_job ORDER BY started_at DESC LIMIT ?",
        (limit,)).fetchall()
    return [dict(zip(("job_id", "as_of", "status", "started_at", "finished_at",
                      "n_scored", "n_queued", "error_class"), r)) for r in rows]
