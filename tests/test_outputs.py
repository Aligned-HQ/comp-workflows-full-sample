"""Verifier for the viral-vs-bacterial classifier task (verifier-owned CV).

Imports the agent's /workspace/output/pipeline.py, builds the feature
matrix via the agent's build_features, derives the true labels itself,
and runs its OWN StratifiedKFold to score the mean held-out AUC-ROC. The
agent cannot inflate the score by training on held-out data: the verifier
controls the cohort, the labels, the splits, and the scoring. The agent
controls only feature engineering and model choice.
"""
from __future__ import annotations

import importlib.util
import sys
import os
from pathlib import Path

import GEOparse
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DEFAULT_DATA_DIR = Path("/workspace/data")
DEFAULT_OUTPUT_DIR = Path("/workspace/output")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR)))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
PIPELINE_PATH = Path(os.environ.get("PIPELINE_PATH", str(OUTPUT_DIR / "pipeline.py")))
GEO_SOFT_PATH = Path(os.environ.get("GEO_SOFT_PATH", str(DATA_DIR / "GSE6269_family.soft.gz")))

SEED = 123
N_SPLITS = 5
COHORT_SIZE = 91

# Pass bar for the mean cross-validated AUC. Calibrated below the reference
# pipeline's honest-CV score and well above chance (0.5); see
# verification_explanation in task.toml.
AUC_THRESHOLD = 0.90

VIRAL_TOKENS = ("influenza a",)
BACTERIAL_TOKENS = ("s. aureus", "e. coli", "s. pneumoniae")


@pytest.fixture(scope="module")
def gse():
    assert GEO_SOFT_PATH.exists(), f"GEO data not found at {GEO_SOFT_PATH}"
    return GEOparse.get_GEO(filepath=str(GEO_SOFT_PATH))


@pytest.fixture(scope="module")
def truth(gse) -> dict:
    """Verifier-owned ground truth: {gsm_id: label} for the GPL96 cohort."""
    labels: dict[str, int] = {}
    for gsm_id, gsm in gse.gsms.items():
        if gsm.metadata.get("platform_id", [None])[0] != "GPL96":
            continue
        meta = " ".join(gsm.metadata.get("characteristics_ch1", [])).lower()
        if any(t in meta for t in VIRAL_TOKENS):
            labels[gsm_id] = 1
        elif any(t in meta for t in BACTERIAL_TOKENS):
            labels[gsm_id] = 0
    return labels


@pytest.fixture(scope="module")
def pipeline_module():
    assert PIPELINE_PATH.exists(), f"agent pipeline not found at {PIPELINE_PATH}"
    spec = importlib.util.spec_from_file_location("agent_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_pipeline"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def features(pipeline_module, gse):
    assert hasattr(pipeline_module, "build_features"), (
        "pipeline.py must define build_features(gse)"
    )
    out = pipeline_module.build_features(gse)
    assert isinstance(out, tuple) and len(out) == 2, (
        "build_features must return a (X, sample_ids) tuple"
    )
    X, sample_ids = out
    X = np.asarray(X, dtype=float)
    sample_ids = list(sample_ids)
    assert X.ndim == 2, f"X must be 2-D, got shape {X.shape}"
    assert X.shape[0] == len(sample_ids), "X rows must align with sample_ids"
    assert np.isfinite(X).all(), "X must not contain NaN/inf"
    return X, sample_ids


def test_truth_cohort_size(truth: dict) -> None:
    assert len(truth) == COHORT_SIZE, (
        f"verifier derived {len(truth)} labelled samples, expected {COHORT_SIZE}"
    )


def test_cohort_matches(features, truth: dict) -> None:
    _, sample_ids = features
    assert len(sample_ids) == COHORT_SIZE, (
        f"build_features returned {len(sample_ids)} samples, expected {COHORT_SIZE}"
    )
    assert set(sample_ids) == set(truth), (
        "build_features must return exactly the labelled cohort samples"
    )


def test_make_estimator_contract(pipeline_module) -> None:
    assert hasattr(pipeline_module, "make_estimator"), (
        "pipeline.py must define make_estimator()"
    )
    est = pipeline_module.make_estimator()
    assert hasattr(est, "fit"), "estimator must implement fit"
    assert hasattr(est, "predict_proba") or hasattr(est, "decision_function"), (
        "estimator must implement predict_proba or decision_function"
    )


def _scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def test_mean_cv_auc_meets_threshold(features, truth: dict, pipeline_module) -> None:
    X, sample_ids = features
    y = np.array([truth[s] for s in sample_ids])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        est = pipeline_module.make_estimator()
        est.fit(X[train_idx], y[train_idx])
        aucs.append(roc_auc_score(y[test_idx], _scores(est, X[test_idx])))

    mean_auc = float(np.mean(aucs))
    print(f"per-fold AUC: {[round(a, 4) for a in aucs]}")
    print(f"mean CV AUC:  {mean_auc:.4f} (threshold {AUC_THRESHOLD})")
    assert mean_auc >= AUC_THRESHOLD, (
        f"mean CV AUC {mean_auc:.4f} is below the pass threshold {AUC_THRESHOLD}"
    )
