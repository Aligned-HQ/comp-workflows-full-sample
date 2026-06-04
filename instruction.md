# Viral vs Bacterial Infection Classifier from Gene-Expression Profiles

Build the best binary classifier you can to distinguish viral from
bacterial infection from pediatric peripheral-blood gene-expression
profiles, **and** investigate *why* it works: identify the genes that
drive it, reconcile them with the known host-response biology, and reconcile
the bundled dataset with its published record. This is a modelling **and**
analysis task — strong cross-validated performance alone is not enough.

You will deliver three artifacts: a reusable modelling pipeline
(`pipeline.py`), a directional marker table (`markers.json`), and a model
card (`report.md`). Performance is measured by 5-fold stratified
cross-validation, but you are also scored on calibration, per-fold
robustness, and whether your reported markers are real, directionally
correct, biologically coherent, and predictive on their own.

## Data

The gene-expression series is shipped with the task at
`/workspace/data/GSE6269_family.soft.gz` — the full GEO SOFT family file
for NCBI GEO series `GSE6269`. Parse it with
`GEOparse.get_GEO(filepath="/workspace/data/GSE6269_family.soft.gz")`. The
file contains **three** Affymetrix platforms; this task uses only `GPL96`.

After restricting to platform `GPL96`, samples are labelled from their
`characteristics_ch1` metadata:

- influenza A → viral, label `1` (18 samples).
- *E. coli*, *S. aureus*, or *S. pneumoniae* → bacterial, label `0`
  (73 samples).
- Healthy controls and all other phenotypes → excluded.

This yields exactly 91 labelled samples. Your pipeline must return
features for these 91 samples and no others.

**The annotation matters.** The GPL96 platform table (`gse.gpls["GPL96"].table`)
maps probes to genes; you will need it both to engineer features and to
talk about specific genes. Inspect it rather than treating probes as opaque.

**Internet access is available.** Use it to look up the GEO series record
and its linked publication so you can reconcile the bundled SOFT file with
the external record in your report (see Deliverable 3).

## Deliverable 1 — `/workspace/output/pipeline.py`

A Python module exposing exactly two callables:

- `build_features(gse)` — receives the parsed GEOparse object for
  `GSE6269` (containing all platforms). Returns a tuple
  `(X, sample_ids, feature_names)` where `X` is a 2-D array-like
  (NumPy array or pandas DataFrame) of shape `(91, n_features)`,
  `sample_ids` is a list of the 91 GSM accession strings row-aligned to
  `X`, and `feature_names` is a list of length `n_features` giving the
  identity (gene symbol) of each column, column-aligned to `X`. The set of
  `sample_ids` must equal the 91 labelled samples defined above. Feature
  engineering is up to you (probe-to-gene mapping, normalisation,
  transforms, selection, dimensionality reduction), but `feature_names`
  must let a reader tie each column back to a gene. `build_features` does
  **not** receive labels.

- `make_estimator()` — returns an unfitted scikit-learn-compatible
  estimator implementing `fit(X, y)` and `predict_proba(X)` (probabilities
  are required so calibration can be scored). Model family and
  hyperparameters are up to you. Put any label-dependent steps (supervised
  feature selection, scaling you want fit on training data only) inside
  this estimator so they are refit per fold.

The module must import cleanly. It may use the network at import time only
if it degrades gracefully, but it must not *require* the network.

## Deliverable 2 — `/workspace/output/markers.json`

A JSON object reporting the genes that distinguish the two classes,
**with direction**:

```json
{
  "viral_up_markers": ["IFI27", "..."],
  "bacterial_up_markers": ["LTF", "..."]
}
```

- `viral_up_markers`: genes expressed **higher in viral** samples.
- `bacterial_up_markers`: genes expressed **higher in bacterial** samples.

Each list must contain at least 8 gene symbols; every symbol must appear
in your `feature_names` (i.e. be a real model feature); no gene may appear
in both lists. Derive these from your data — from the model's weights, a
differential-expression analysis, or both — and make sure the direction is
correct. You are encouraged to add other keys (e.g. a `method` or
`signature_interpretation` field); they will be ignored by the grader.

## Deliverable 3 — `/workspace/output/report.md`

A model card (a few hundred words minimum) that explains your scientific
choices. This file is a required deliverable kept for human review; it is
not machine-scored, so write it for a reviewer, not a regex. It should
cover, at least:

- **Feature engineering** — probe-to-gene mapping, normalisation, scale
  handling, any selection, and *why*; comment on robustness to the
  probe-to-gene mapping choices.
- **Leakage controls** — how you ensured no label information leaks into
  features or across folds.
- **Cross-validation variance** — per-fold AUC and its spread, not just
  the mean; discuss feature **stability** across folds.
- **Calibration** — whether the predicted probabilities are trustworthy,
  and how you checked.
- **Biological plausibility** — interpret the top predictors. Compare your
  viral markers against the known type-I **interferon** / interferon-
  stimulated-gene response and your bacterial markers against the
  **neutrophil** / myeloid granulocyte response.
- **Source reconciliation** — reconcile the bundled SOFT file with the
  external GEO/PubMed record for `GSE6269`: the linked publication, how
  many samples and platforms the full series contains, and how that relates
  to the 91-sample GPL96 cohort you actually model.

## How you are graded

The verifier owns the cohort, the labels, the cross-validation splits, and
the scoring — you cannot inflate your score by training on held-out data.
It imports your module, calls `build_features` once, derives the true
labels itself, and runs its own
`StratifiedKFold(n_splits=5, shuffle=True, random_state=123)`. You pass
only if **all** of the following hold:

1. The returned cohort equals the 91 labelled samples and the
   `(X, sample_ids, feature_names)` contract is satisfied.
2. **Mean held-out AUC-ROC ≥ 0.90**, and the **worst single fold ≥ 0.80**
   (robustness across folds).
3. **Mean held-out Brier score ≤ 0.15** (calibration).
4. Every reported marker is a real feature, directionally correct (≥ 80%
   per class match the actual viral/bacterial mean difference), and each
   class overlaps the canonical interferon / neutrophil signature in at
   least 3 genes.
5. A simple classifier built from **only** your reported markers still
   achieves mean CV AUC ≥ 0.85 — your markers must carry the signal.

`report.md` is a required deliverable but is reviewed by a human rather
than machine-scored; produce it, but it does not factor into the automated
pass/fail.
