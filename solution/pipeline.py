"""Reference pipeline for the viral-vs-bacterial classifier task.

Delivers the two callables the verifier drives:
  build_features(gse) -> (X, sample_ids)
  make_estimator()    -> unfitted sklearn-compatible estimator

Design note: build_features receives no labels, so every label-dependent
step (the classifier) lives in make_estimator and is fit fresh by the
verifier inside each CV fold. build_features performs only
label-independent feature construction (probe->gene aggregation and a
log2 transform), leaking no information across folds. Feature scaling is
placed inside the estimator Pipeline so it, too, is fit on training folds
only.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 123

VIRAL_TOKENS = ("influenza a",)
BACTERIAL_TOKENS = ("s. aureus", "e. coli", "s. pneumoniae")


def _cohort_ids(gse) -> list[str]:
    """The labelled GPL96 samples, in deterministic (sorted) order."""
    ids = []
    for gsm_id, gsm in gse.gsms.items():
        if gsm.metadata.get("platform_id", [None])[0] != "GPL96":
            continue
        meta = " ".join(gsm.metadata.get("characteristics_ch1", [])).lower()
        if any(t in meta for t in VIRAL_TOKENS) or any(t in meta for t in BACTERIAL_TOKENS):
            ids.append(gsm_id)
    return sorted(ids)


def build_features(gse):
    # Expression matrix over GPL96 samples (probes x samples).
    gpl96_ids = [
        gsm_id for gsm_id, gsm in gse.gsms.items()
        if gsm.metadata.get("platform_id", [None])[0] == "GPL96"
    ]
    original = gse.gsms
    gse.gsms = {gsm_id: original[gsm_id] for gsm_id in gpl96_ids}
    gene_expr = gse.pivot_samples("VALUE").dropna(how="all")
    gse.gsms = original

    # Probe -> gene symbol (drop NA, first symbol when multi-mapped); mean-aggregate.
    gpl = gse.gpls["GPL96"].table[["ID", "Gene Symbol"]].dropna().copy()
    gpl["Gene Symbol"] = gpl["Gene Symbol"].str.split(" /// ").str[0]
    merged = gene_expr.merge(gpl, left_index=True, right_on="ID")
    final_df = merged.groupby("Gene Symbol").mean(numeric_only=True).T  # samples x genes

    # Restrict to the fixed 91-sample cohort.
    sample_ids = [s for s in _cohort_ids(gse) if s in final_df.index]
    X = final_df.loc[sample_ids]

    # Label-independent transforms only: clip negatives, log2 if not already log-scaled.
    p99 = np.nanpercentile(X.values, 99)
    if p99 > 100:
        X = np.log2(X.clip(lower=0) + 1)
    return X.to_numpy(dtype=float), sample_ids


def make_estimator():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])
