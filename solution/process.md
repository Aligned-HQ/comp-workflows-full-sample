# Process — how to solve the viral-vs-bacterial classifier task

An ordered walkthrough of the steps an expert would take to produce the
three deliverables (`pipeline.py`, `markers.json`, `report.md`). The
reference implementation in this directory follows these steps.

## 1. Orient and inspect the data
1. Parse the bundled file: `gse = GEOparse.get_GEO(filepath="/workspace/data/GSE6269_family.soft.gz")`.
2. Inspect `gse.metadata` (series title, `pubmed_id`, `overall_design`,
   `platform_id`) and note this is a multi-platform series — `GPL96`,
   `GPL570`, `GPL2507` — with 143 deposited samples.
3. List the GSMs and their `platform_id`; confirm 97 are on `GPL96`.
4. Look at a few samples' `characteristics_ch1` to see how phenotype is
   recorded as free text, and skim `gse.gpls["GPL96"].table` columns to
   find `ID` and `Gene Symbol`.

## 2. Define the labelled cohort
1. Restrict to samples whose `platform_id` is `GPL96`.
2. Lower-case the joined `characteristics_ch1` text and label: influenza A
   → viral (`1`); `e. coli` / `s. aureus` / `s. pneumoniae` → bacterial
   (`0`); everything else (healthy controls, other phenotypes) → excluded.
3. Confirm exactly 91 labelled samples (18 viral, 73 bacterial). Keep a
   deterministic (sorted) sample order so results are reproducible.
4. Note the class imbalance (~80/20) — it drives later choices (balanced
   class weights, stratified CV, AUC rather than accuracy).

## 3. Build the feature matrix (label-free)
1. Build a probes × samples expression matrix over the GPL96 samples
   (`gse.pivot_samples("VALUE")`), dropping all-NaN rows.
2. Map probes to genes using the GPL96 annotation: drop probes with no
   `Gene Symbol`; for multi-mapped probes (`A /// B`) take the first
   symbol.
3. Collapse multiple probes hitting the same gene by averaging, giving a
   samples × genes matrix (~13k genes).
4. Restrict rows to the 91-sample cohort.
5. Apply only label-independent transforms: clip negatives at 0 and, if the
   99th percentile indicates raw intensities (≫ 100), `log2(x + 1)`.
6. Return `(X, sample_ids, feature_names)` with `feature_names` the gene
   symbols, column-aligned to `X`. Do **not** use labels anywhere here —
   this prevents leakage across folds.

## 4. Choose and wrap the model (label-dependent steps fold-local)
1. Pick a model suited to p ≫ n with class imbalance: an L1-penalised
   logistic regression (sparse, regularised) with balanced class weights
   and a fixed seed.
2. Put `StandardScaler` and the classifier together in a single
   scikit-learn `Pipeline` returned by `make_estimator()`, so the scaler
   and the model are refit inside each training fold only (no scaling on
   full data).
3. Ensure the estimator exposes `predict_proba` (needed for calibration).

## 5. Validate honestly
1. Run `StratifiedKFold(n_splits=5, shuffle=True, random_state=123)`,
   refitting a fresh estimator per fold and scoring held-out AUC-ROC.
2. Record per-fold AUC, the mean, and the spread; confirm the mean clears
   0.90 and the worst fold clears 0.80 (robustness, not one lucky split).
3. Compute the held-out Brier score to check probability calibration
   (target ≤ 0.15); recalibrate (Platt/isotonic) only if needed.
4. Track which genes get non-zero weight across folds to gauge feature
   stability (a stable interferon/neutrophil core should recur).

## 6. Derive directional markers (a second computational mode)
1. Separately from the classifier, run a per-gene differential-expression
   contrast between viral and bacterial samples (standardised mean
   difference or t-test) on the log2 matrix.
2. Rank genes by effect size and split by direction: higher-in-viral vs
   higher-in-bacterial.
3. Cross-check against the model's coefficients — keep markers that are
   both differentially expressed and model-relevant.
4. Verify each chosen marker's direction against the actual group means,
   and confirm every symbol is in `feature_names`.
5. Write `markers.json` with `viral_up_markers` and `bacterial_up_markers`
   (≥ 8 each, no overlap). Sanity-check that a quick logistic model on just
   these markers still cross-validates well (≥ 0.85) — they must carry the
   signal.

## 7. Reconcile with known biology
1. Compare the viral-up markers to the canonical type-I interferon /
   interferon-stimulated-gene program (IFI27, IFI44L, IFIT1/3, ISG15, MX1,
   OAS1/2/3, RSAD2, USP18, SIGLEC1, …).
2. Compare the bacterial-up markers to the neutrophil / myeloid granule
   program (LTF, CAMP, ELANE, MPO, BPI, DEFA1/4, CEACAM8, OLFM4, MMP8, …).
3. Confirm the learned signal matches the expected host response rather
   than batch or composition artifacts; note any non-immune confounds
   (e.g. erythroid genes from blood-composition shifts in sepsis).

## 8. Reconcile the dataset with its published record
1. With internet access, look up the GEO record for `GSE6269` and its
   linked publication (PubMed 17105821).
2. Reconcile the external record with the bundled SOFT: the publication,
   the 143 deposited samples across the three platforms (GPL96/GPL570/
   GPL2507), and the 144-vs-143 design note.
3. Explain how that full series maps down to the 91-sample GPL96 cohort you
   actually model (97 GPL96 samples minus healthy/other phenotypes), and
   why you do not pool the other platforms (cross-platform batch confound).

## 9. Write the model card and finalise
1. Write `report.md` covering: feature engineering and probe-to-gene
   robustness, leakage controls, per-fold CV variance and feature
   stability, calibration, biological plausibility (interferon vs
   neutrophil), and the source reconciliation above.
2. Ensure all three artifacts are written under `/workspace/output/`.
3. Re-import the module cleanly with no network dependency and re-run the
   CV end-to-end to confirm every pass condition holds.
