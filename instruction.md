# Viral vs Bacterial Infection Classifier from Gene-Expression Profiles

Build a binary classifier that distinguishes viral from bacterial
infection using publicly available gene-expression array data from the
NCBI Gene Expression Omnibus (accession `GSE6269`, platform `GPL96` —
[HG-U133A] Affymetrix Human Genome U133A Array). The series contains
pediatric peripheral-blood samples from acute infections caused by
influenza A (viral), or by *E. coli*, *S. aureus*, or *S. pneumoniae*
(bacterial), plus healthy controls that must be excluded.

## Inputs

The gene-expression series is shipped with the task — no network call is
required. Load it directly from disk:

- `/workspace/data/GSE6269_family.soft.gz` — the full GEO SOFT family
  file for series `GSE6269`. Parse it with
  `GEOparse.get_GEO(filepath="/workspace/data/GSE6269_family.soft.gz")`.

After parsing, restrict to platform `GPL96` and discard samples from any
other platform.

After filtering, you should end up with 91 labeled samples:

- 73 bacterial infections (*E. coli*, *S. aureus*, *S. pneumoniae*) →
  label `0`.
- 18 viral infections (influenza A) → label `1`.
- Healthy controls and any other phenotypes → discard.

## Task

1. Load `/workspace/data/GSE6269_family.soft.gz` with `GEOparse`, then
   restrict to GPL96 samples.
2. Build the gene-expression matrix:
   - Drop probes with missing gene symbols (`NaN`).
   - For probes mapping to multiple symbols (e.g. `DDR1 /// MIR4640`),
     keep only the first symbol.
   - For multiple probes mapping to the same gene, aggregate by
     arithmetic mean.
   - Clip negative expression values to `0`.
   - If the data is not log2-scaled (99th percentile > 100), apply
     `log2(x + 1)`.
   - Z-score normalise the matrix with `sklearn.preprocessing.StandardScaler`.
3. Parse sample metadata to assign binary labels (`0` bacterial,
   `1` viral) and drop unlabelled samples.
4. Fit `sklearn.linear_model.LogisticRegression` with L1 (LASSO) penalty,
   `C=1.0`, `solver='liblinear'`, `class_weight='balanced'`, and
   `random_state=123`.
5. Evaluate with 5-fold stratified cross-validation
   (`StratifiedKFold(n_splits=5, shuffle=True, random_state=123)`).
   Compute the mean cross-validated accuracy, sensitivity (recall on the
   viral class), specificity (recall on the bacterial class), and
   AUC-ROC.
6. Refit on all 91 samples and extract the top 3 genes with the largest
   positive coefficients (viral biomarkers) and the top 3 with the most
   negative coefficients (bacterial biomarkers).

Use a seed of `123` for every non-deterministic step.

## Output

Write the result to `/workspace/output/result.json` as a single JSON
object with these fields:

- `accuracy` *(float in [0, 1])* — mean cross-validated accuracy.
- `sensitivity` *(float in [0, 1])* — mean cross-validated recall on
  the viral class (`pos_label=1`).
- `specificity` *(float in [0, 1])* — mean cross-validated recall on
  the bacterial class (`pos_label=0`).
- `auc` *(float in [0, 1])* — mean cross-validated AUC-ROC.
- `viral_biomarkers` *(list[object], length 3)* — top viral genes,
  ordered from largest to smallest coefficient. Each entry is
  `{"gene": <str>, "coefficient": <float>}`; coefficients must be
  positive.
- `bacterial_biomarkers` *(list[object], length 3)* — top bacterial
  genes, ordered from most negative to least negative coefficient. Each
  entry is `{"gene": <str>, "coefficient": <float>}`; coefficients must
  be negative.

## Constraints

- Python 3. Pre-installed packages: `GEOparse`, `numpy`, `pandas`,
  `scipy`, `scikit-learn`. No network access — load the GEO data from
  the local file at `/workspace/data/GSE6269_family.soft.gz`.
- Use the LASSO configuration specified above; do not substitute
  another model or solver. `liblinear` is required because `lbfgs` does
  not support the L1 penalty.
- Use `StratifiedKFold`, not plain `KFold` — the class distribution is
  imbalanced (~80/20 bacterial/viral) and plain k-fold will skew the
  validation folds.
- Sensitivity is `recall_score(y, y_pred, pos_label=1)` (viral);
  specificity is `recall_score(y, y_pred, pos_label=0)` (bacterial).
- Output must be valid JSON containing exactly the fields above.
