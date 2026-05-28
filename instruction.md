# Viral vs Bacterial Infection Classifier from Gene-Expression Profiles

Build the best binary classifier you can to distinguish viral from
bacterial infection from pediatric peripheral-blood gene-expression
profiles, and deliver it as a reusable modelling pipeline. Performance is
measured by the mean AUC-ROC under 5-fold stratified cross-validation; do
as well as you can within the time budget.

## Data

The gene-expression series is shipped with the task at
`/workspace/data/GSE6269_family.soft.gz` — the full GEO SOFT family file
for NCBI GEO series `GSE6269` (no network access required). Parse it with
`GEOparse.get_GEO(filepath="/workspace/data/GSE6269_family.soft.gz")`.

The cohort is fixed. After restricting to platform `GPL96`, samples are
labelled from their `characteristics_ch1` metadata:

- influenza A → viral, label `1` (18 samples).
- *E. coli*, *S. aureus*, or *S. pneumoniae* → bacterial, label `0`
  (73 samples).
- Healthy controls and all other phenotypes → excluded.

This yields exactly 91 labelled samples. Your pipeline must return
features for these 91 samples and no others.

## Deliverable

Write a Python module to `/workspace/output/pipeline.py` exposing exactly
two callables:

- `build_features(gse)` — receives the parsed GEOparse object for
  `GSE6269` (the return value of `GEOparse.get_GEO` on the shipped file,
  containing all platforms). Returns a tuple `(X, sample_ids)` where `X`
  is a 2-D array-like (NumPy array or pandas DataFrame) of shape
  `(91, n_features)` and `sample_ids` is a list of the 91 GSM accession
  strings, row-aligned to `X`. The set of `sample_ids` must equal the 91
  labelled samples defined above. Feature engineering is entirely up to
  you (probe-to-gene mapping, normalisation, transforms, feature
  selection, dimensionality reduction). `build_features` does not receive
  labels.

- `make_estimator()` — returns an unfitted scikit-learn-compatible
  estimator implementing `fit(X, y)` and either `predict_proba(X)` or
  `decision_function(X)`. Model family and hyperparameters are up to you.
  Put any label-dependent steps (supervised feature selection, scaling
  you want fit on training data only) inside this estimator so they are
  refit per fold.

The module must import cleanly and must not require network access.

## How you are graded

The verifier imports your module, calls `build_features` once, derives
the true labels itself, and runs its own 5-fold stratified
cross-validation (`StratifiedKFold(n_splits=5, shuffle=True,
random_state=123)`): for each fold it fits a fresh `make_estimator()` on
the training rows and scores AUC-ROC on the held-out rows. Your score is
the mean held-out AUC across the five folds. Because the verifier owns
the splits and the labels, you cannot improve your score by fitting on
held-out data — only a genuinely predictive pipeline scores well.
