"""Model registry: an artefact you can serve, and refuse to serve.

WHY A REGISTRY IS NOT JUST pickle.dump
--------------------------------------
A pickled estimator is not a deployable model. It is half of one. The other
half is the contract it was fitted under, and without that contract the most
dangerous failure in production ML is silent: the feature builder changes, the
columns arrive in a different order or with a different meaning, and the model
keeps returning confident probabilities computed from the wrong numbers.

Nothing raises. The API stays up. The scores are garbage.

So every artefact here carries a MANIFEST recording what it was fitted against:

  * the exact ordered feature list, plus a hash of it
  * the training cohort definition and size
  * the metrics it achieved, with the confidence intervals from uncertainty.py
  * the feature-visibility mode ("received" -- see features.py; a model trained
    on complete-history features must never be served against runout-limited
    ones, and the manifest is where that becomes checkable)
  * the calibration it was measured at

`load()` then REFUSES to score anything whose feature schema does not match.
That refusal is the point of the file. A model that declines to answer is
recoverable; one that answers wrongly is not.

WHAT THIS IS NOT
----------------
MLflow, or a real registry. No experiment tracking, no artefact store, no
lineage to the training run, no staging/production promotion workflow, no
approval gates. It is the minimum that makes serving honest rather than the
system a platform team would run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time


class SchemaMismatch(Exception):
    """Raised when the features offered do not match what the model was fitted
    on. Deliberately fatal: the alternative is a confident wrong answer."""


REGISTRY_DIR = "models"


def feature_hash(columns):
    """Order-sensitive hash of the feature contract.

    Order-SENSITIVE on purpose. A tree model does not care about column order,
    but a numpy array fed to a fitted pipeline absolutely does, and the way that
    breaks is by silently scoring `charlson` through the coefficient for `age`.
    """
    joined = "|".join(str(c) for c in columns)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def save(model, columns, *, name, version, cohort, metrics, visibility,
         calibration=None, notes="", registry_dir=REGISTRY_DIR):
    """Persist a model plus the contract it was fitted under."""
    os.makedirs(registry_dir, exist_ok=True)
    manifest = {
        "name": name,
        "version": version,
        "created_at": time.time(),
        "feature_columns": list(columns),
        "n_features": len(columns),
        "feature_hash": feature_hash(columns),
        "cohort": cohort,
        "metrics": metrics,
        "feature_visibility": visibility,
        "calibration": calibration or {},
        "notes": notes,
        "intended_use": (
            "Rank recent inpatient discharges by modelled probability of an "
            "unplanned readmission claim within 30 days, so a capacity-bound "
            "care-management team can order its outreach queue. NOT for "
            "coverage, benefit, authorisation or level-of-care decisions. "
            "See docs/CLINICAL_VALIDATION.md."),
    }
    stem = os.path.join(registry_dir, f"{name}-v{version}")
    with open(stem + ".pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(stem + ".json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return stem


class ServableModel:
    """A model bound to its contract. The only way to score in this codebase."""

    def __init__(self, model, manifest):
        self.model = model
        self.manifest = manifest

    @property
    def columns(self):
        return self.manifest["feature_columns"]

    def check_schema(self, columns):
        """Raise unless the offered features match exactly, in order."""
        offered = list(columns)
        if feature_hash(offered) == self.manifest["feature_hash"]:
            return
        expected = self.columns
        missing = [c for c in expected if c not in offered]
        extra = [c for c in offered if c not in expected]
        if not missing and not extra:
            detail = ("same columns in a DIFFERENT ORDER -- this is the "
                      "dangerous case, because the arithmetic still works")
        else:
            detail = f"missing={missing[:5]} unexpected={extra[:5]}"
        raise SchemaMismatch(
            f"{self.manifest['name']} v{self.manifest['version']} was fitted on "
            f"{len(expected)} features ({self.manifest['feature_hash']}); "
            f"offered {len(offered)} ({feature_hash(offered)}). {detail}")

    def predict_proba(self, X):
        self.check_schema(X.columns)
        return self.model.predict_proba(X)[:, 1]


def load(name, version, registry_dir=REGISTRY_DIR):
    stem = os.path.join(registry_dir, f"{name}-v{version}")
    with open(stem + ".json") as fh:
        manifest = json.load(fh)
    with open(stem + ".pkl", "rb") as fh:
        model = pickle.load(fh)
    return ServableModel(model, manifest)


def list_versions(registry_dir=REGISTRY_DIR):
    if not os.path.isdir(registry_dir):
        return []
    out = []
    for fn in sorted(os.listdir(registry_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(registry_dir, fn)) as fh:
                m = json.load(fh)
            out.append({k: m.get(k) for k in
                        ("name", "version", "created_at", "n_features",
                         "feature_hash", "feature_visibility")})
    return out
