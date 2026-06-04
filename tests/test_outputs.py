"""Verifier for the viral-vs-bacterial classifier task (verifier-owned CV).

The agent delivers two machine-graded artifacts:

  pipeline.py   build_features(gse) -> (X, sample_ids, feature_names)
                make_estimator()    -> unfitted sklearn estimator (predict_proba)
  markers.json  directional gene markers grouped viral-up / bacterial-up

(The agent also writes report.md, a model card for human review; it is a
required deliverable but is not machine-scored here.)

The verifier owns the cohort, the labels, the CV splits and the scoring, so
the AUC cannot be inflated by training on held-out data. Beyond ranking
performance it scores calibration, per-fold robustness, and whether the
agent's reported markers are (a) real model features, (b) directionally
correct, (c) biologically coherent with the known interferon / neutrophil
signatures, and (d) genuinely predictive on their own.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import GEOparse
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_DIR = Path("/workspace/data")
DEFAULT_OUTPUT_DIR = Path("/workspace/output")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR)))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
PIPELINE_PATH = Path(os.environ.get("PIPELINE_PATH", str(OUTPUT_DIR / "pipeline.py")))
MARKERS_PATH = Path(os.environ.get("MARKERS_PATH", str(OUTPUT_DIR / "markers.json")))
GEO_SOFT_PATH = Path(os.environ.get("GEO_SOFT_PATH", str(DATA_DIR / "GSE6269_family.soft.gz")))

SEED = 123
N_SPLITS = 5
COHORT_SIZE = 91

# --- pass bars (calibrated below the reference pipeline, well above chance) ---
AUC_THRESHOLD = 0.90          # mean held-out AUC (reference ~0.98)
MIN_FOLD_AUC = 0.80           # worst single fold (reference ~0.93): rewards robustness
BRIER_THRESHOLD = 0.15        # mean held-out Brier (reference ~0.04)
MARKER_AUC_THRESHOLD = 0.85   # CV AUC using ONLY the reported markers (reference ~0.96)
MIN_MARKERS_PER_CLASS = 8     # how many directional markers each class must report
MIN_SIGNATURE_HITS = 3        # canonical-signature overlap required per class
MIN_DIRECTION_FRAC = 0.80     # fraction of markers with correct mean direction

VIRAL_TOKENS = ("influenza a",)
BACTERIAL_TOKENS = ("s. aureus", "e. coli", "s. pneumoniae")

# Canonical type-I interferon-stimulated genes (up in viral / influenza).
ISG_SET = {
    "IFI27", "IFI44", "IFI44L", "IFI6", "OAS1", "OAS2", "OAS3", "OASL",
    "MX1", "MX2", "ISG15", "IFIT1", "IFIT2", "IFIT3", "IFIT5", "RSAD2",
    "SIGLEC1", "USP18", "HERC5", "HERC6", "LY6E", "SERPING1", "GBP1",
    "IFITM1", "IFITM3", "EPSTI1", "LAMP3", "SPATS2L", "DDX60", "XAF1",
    "SAMD9", "IRF7", "ZBP1", "DHX58", "LGALS3BP", "IFI35",
}
# Canonical neutrophil / myeloid granule genes (up in bacterial).
NEUTRO_SET = {
    "ELANE", "DEFA1", "DEFA3", "DEFA4", "MPO", "BPI", "LCN2", "CEACAM8",
    "CEACAM6", "LTF", "CAMP", "MMP8", "MMP9", "ARG1", "S100A12", "OLFM4",
    "CD177", "ANXA3", "HP", "CTSG", "RETN", "CRISP3", "PGLYRP1", "CHI3L1",
    "CHIT1", "CYP4F3", "MGAM", "MS4A3", "IL1R2", "CXCL8", "LCN2",
}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
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
    assert isinstance(out, tuple) and len(out) == 3, (
        "build_features must return a (X, sample_ids, feature_names) tuple"
    )
    X, sample_ids, feature_names = out
    X = np.asarray(X, dtype=float)
    sample_ids = list(sample_ids)
    feature_names = [str(f) for f in feature_names]
    assert X.ndim == 2, f"X must be 2-D, got shape {X.shape}"
    assert X.shape[0] == len(sample_ids), "X rows must align with sample_ids"
    assert X.shape[1] == len(feature_names), "feature_names must align with X columns"
    assert np.isfinite(X).all(), "X must not contain NaN/inf"
    return X, sample_ids, feature_names


@pytest.fixture(scope="module")
def markers() -> dict:
    assert MARKERS_PATH.exists(), f"markers.json not found at {MARKERS_PATH}"
    data = json.loads(MARKERS_PATH.read_text())
    assert isinstance(data, dict), "markers.json must be a JSON object"
    for key in ("viral_up_markers", "bacterial_up_markers"):
        assert key in data and isinstance(data[key], list), (
            f"markers.json must contain a list '{key}'"
        )
    return data


def _labels_for(sample_ids, truth):
    return np.array([truth[s] for s in sample_ids])


def _scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


# --------------------------------------------------------------------------- #
# contract / cohort
# --------------------------------------------------------------------------- #
def test_truth_cohort_size(truth: dict) -> None:
    assert len(truth) == COHORT_SIZE, (
        f"verifier derived {len(truth)} labelled samples, expected {COHORT_SIZE}"
    )


def test_cohort_matches(features, truth: dict) -> None:
    _, sample_ids, _ = features
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
    assert hasattr(est, "predict_proba"), (
        "estimator must implement predict_proba (probabilities are needed for "
        "the calibration score)"
    )


# --------------------------------------------------------------------------- #
# performance: ranking, per-fold robustness, calibration
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cv_results(features, truth, pipeline_module):
    X, sample_ids, _ = features
    y = _labels_for(sample_ids, truth)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs, briers = [], []
    for train_idx, test_idx in cv.split(X, y):
        est = pipeline_module.make_estimator()
        est.fit(X[train_idx], y[train_idx])
        p = _scores(est, X[test_idx])
        aucs.append(roc_auc_score(y[test_idx], p))
        # Brier needs probabilities in [0, 1]; only score it when available.
        if hasattr(est, "predict_proba"):
            briers.append(brier_score_loss(y[test_idx], est.predict_proba(X[test_idx])[:, 1]))
    return {"aucs": aucs, "briers": briers}


def test_mean_cv_auc_meets_threshold(cv_results) -> None:
    aucs = cv_results["aucs"]
    mean_auc = float(np.mean(aucs))
    print(f"per-fold AUC: {[round(a, 4) for a in aucs]}")
    print(f"mean CV AUC:  {mean_auc:.4f} (threshold {AUC_THRESHOLD}); "
          f"SD {np.std(aucs):.4f}")
    assert mean_auc >= AUC_THRESHOLD, (
        f"mean CV AUC {mean_auc:.4f} is below the pass threshold {AUC_THRESHOLD}"
    )


def test_per_fold_robustness(cv_results) -> None:
    worst = float(np.min(cv_results["aucs"]))
    print(f"worst-fold AUC: {worst:.4f} (floor {MIN_FOLD_AUC})")
    assert worst >= MIN_FOLD_AUC, (
        f"worst-fold AUC {worst:.4f} below floor {MIN_FOLD_AUC}: the model is "
        f"not robust across folds"
    )


def test_calibration(cv_results) -> None:
    briers = cv_results["briers"]
    assert briers, "estimator did not expose predict_proba; cannot score calibration"
    mean_brier = float(np.mean(briers))
    print(f"mean CV Brier: {mean_brier:.4f} (threshold {BRIER_THRESHOLD})")
    assert mean_brier <= BRIER_THRESHOLD, (
        f"mean Brier {mean_brier:.4f} exceeds {BRIER_THRESHOLD}: probabilities "
        f"are poorly calibrated"
    )


# --------------------------------------------------------------------------- #
# markers: validity, direction, biology, predictiveness
# --------------------------------------------------------------------------- #
def test_marker_counts_and_validity(markers, features) -> None:
    _, _, feature_names = features
    feat = set(feature_names)
    viral = [str(g) for g in markers["viral_up_markers"]]
    bact = [str(g) for g in markers["bacterial_up_markers"]]
    assert len(viral) >= MIN_MARKERS_PER_CLASS, (
        f"need >= {MIN_MARKERS_PER_CLASS} viral_up_markers, got {len(viral)}"
    )
    assert len(bact) >= MIN_MARKERS_PER_CLASS, (
        f"need >= {MIN_MARKERS_PER_CLASS} bacterial_up_markers, got {len(bact)}"
    )
    assert not (set(viral) & set(bact)), (
        "a gene cannot be both viral-up and bacterial-up"
    )
    missing = [g for g in viral + bact if g not in feat]
    assert not missing, (
        f"reported markers absent from build_features feature_names: {missing[:10]} "
        f"(markers must be real model features)"
    )


def test_marker_direction(markers, features, truth) -> None:
    X, sample_ids, feature_names = features
    y = _labels_for(sample_ids, truth)
    col = {g: i for i, g in enumerate(feature_names)}
    viral_mask, bact_mask = y == 1, y == 0

    def correct_fraction(genes, expect_viral_higher):
        ok = 0
        for g in genes:
            i = col[g]
            higher_in_viral = X[viral_mask, i].mean() > X[bact_mask, i].mean()
            ok += int(higher_in_viral == expect_viral_higher)
        return ok / len(genes)

    vfrac = correct_fraction([str(g) for g in markers["viral_up_markers"]], True)
    bfrac = correct_fraction([str(g) for g in markers["bacterial_up_markers"]], False)
    print(f"direction-correct fraction: viral {vfrac:.2f}, bacterial {bfrac:.2f}")
    assert vfrac >= MIN_DIRECTION_FRAC, (
        f"only {vfrac:.0%} of viral_up_markers are actually higher in viral samples"
    )
    assert bfrac >= MIN_DIRECTION_FRAC, (
        f"only {bfrac:.0%} of bacterial_up_markers are actually higher in bacterial samples"
    )


def test_marker_biology(markers) -> None:
    viral = {str(g).upper() for g in markers["viral_up_markers"]}
    bact = {str(g).upper() for g in markers["bacterial_up_markers"]}
    isg_hits = sorted(viral & ISG_SET)
    neu_hits = sorted(bact & NEUTRO_SET)
    print(f"interferon hits: {isg_hits}")
    print(f"neutrophil hits: {neu_hits}")
    assert len(isg_hits) >= MIN_SIGNATURE_HITS, (
        f"viral markers overlap the canonical interferon signature in only "
        f"{len(isg_hits)} genes (need >= {MIN_SIGNATURE_HITS})"
    )
    assert len(neu_hits) >= MIN_SIGNATURE_HITS, (
        f"bacterial markers overlap the canonical neutrophil signature in only "
        f"{len(neu_hits)} genes (need >= {MIN_SIGNATURE_HITS})"
    )


def test_markers_are_predictive(markers, features, truth) -> None:
    """A simple model on ONLY the reported markers must still generalise:
    the differential-expression markers and the classification signal agree."""
    X, sample_ids, feature_names = features
    y = _labels_for(sample_ids, truth)
    col = {g: i for i, g in enumerate(feature_names)}
    genes = [str(g) for g in markers["viral_up_markers"]] + \
            [str(g) for g in markers["bacterial_up_markers"]]
    idx = [col[g] for g in genes]
    Xr = X[:, idx]
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs = []
    for train_idx, test_idx in cv.split(Xr, y):
        est = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED)),
        ])
        est.fit(Xr[train_idx], y[train_idx])
        aucs.append(roc_auc_score(y[test_idx], est.predict_proba(Xr[test_idx])[:, 1]))
    mean_auc = float(np.mean(aucs))
    print(f"marker-only mean CV AUC: {mean_auc:.4f} (threshold {MARKER_AUC_THRESHOLD})")
    assert mean_auc >= MARKER_AUC_THRESHOLD, (
        f"a model using only the reported markers scores {mean_auc:.4f} < "
        f"{MARKER_AUC_THRESHOLD}: the markers do not carry the signal"
    )
