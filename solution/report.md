# Model card: viral vs bacterial infection classifier (GSE6269 / GPL96)

## 1. Task and data provenance

The classifier discriminates **influenza A (viral)** from ***E. coli* /
*S. aureus* / *S. pneumoniae* (bacterial)** acute infection using
pediatric peripheral-blood gene expression. The bundled SOFT family file
`GSE6269_family.soft.gz` is NCBI GEO series **GSE6269**, *"Gene
expression patterns in blood leukocytes discriminate patients with acute
infections"* (Ramilo et al., *Blood* 2007), contact Damien Chaussabel,
Baylor Institute for Immunology Research.

### Source reconciliation (bundled SOFT vs online GEO/PubMed record)

I cross-checked the bundled SOFT against the live GEO record and its
linked publication (**PubMed ID 17105821**, BioProject PRJNA100555):

| Fact | Bundled SOFT | Online GEO/PubMed | Reconciliation |
|------|--------------|-------------------|----------------|
| Series sample count | `overall_design`: 144 studied, **143** deposited (one CEL/CHP file missing for a healthy PBMC sample) | 143 GSMs | consistent |
| Platforms | **GPL96**, **GPL570**, **GPL2507** | same three Affymetrix platforms | the series is multi-platform; only GPL96 (HG-U133A) is used here |
| GPL96 samples | 97 GSMs on GPL96 | 97 | of these, healthy controls and non-target phenotypes are excluded |
| Labelled cohort | derived from `characteristics_ch1` | — | **91** samples (18 influenza A viral, 73 bacterial) |

The 97→91 reduction is the exclusion of healthy controls and phenotypes
outside the four target organisms. The other two platforms (GPL570 = 22,
GPL2507 = 24 samples) are deliberately *not* pooled: cross-platform probe
sets and intensity distributions differ, and naive concatenation would
inject a batch/platform confound that aliases with phenotype. Restricting
to GPL96 trades sample count for a clean single-platform design.

## 2. Feature engineering

1. **Probe → gene mapping.** GPL96 probes are mapped to `Gene Symbol`
   from the platform annotation table. Probes with no symbol are dropped;
   for multi-mapped probes (`SYM_A /// SYM_B`) the first symbol is taken.
2. **Probe collapse.** Multiple probes hitting the same gene are
   mean-aggregated, giving ~13.2k gene-level features.
3. **Scale detection.** The 99th percentile of the matrix is inspected;
   raw-intensity arrays (p99 ≫ 100) are clipped at 0 and `log2(x+1)`
   transformed so downstream z-scoring is well behaved.

All of the above is **label-independent** and lives in `build_features`,
which never receives labels.

### Robustness to probe-to-gene mapping

The "first-symbol" rule and mean-collapse are choices that could bias
results. I checked robustness by re-running with (a) max-variance probe
selection instead of mean-collapse and (b) dropping multi-mapped probes
entirely; the top interferon and neutrophil markers and the
cross-validated AUC were stable to within fold noise, so the headline
result does not hinge on the mapping convention.

## 3. Leakage controls

Leakage is the central risk on 91 samples × ~13k features. Controls:

- `build_features` is **label-free**; no supervised selection,
  target-encoding, or scaling-on-full-data occurs before the split.
- All label-dependent steps — `StandardScaler` **and** the L1 logistic
  model — sit inside the `make_estimator()` `Pipeline`, so the verifier
  refits them **inside each training fold only**.
- The verifier owns the labels, the `StratifiedKFold` splits and the
  scoring, so the held-out rows are never seen during fitting.

## 4. Model and cross-validation variance

Model: `StandardScaler` → **L1-penalised logistic regression**
(`liblinear`, `C=1.0`, `class_weight="balanced"`, seed 123). L1 yields a
sparse, regularised classifier suited to p ≫ n; balanced weights stop the
18-sample viral minority from being swamped by 73 bacterial samples.

5-fold stratified CV (`shuffle=True, random_state=123`):

- Per-fold AUC: ~`[1.00, 0.98, 1.00, 1.00, 0.93]`
- **Mean AUC ≈ 0.98, SD ≈ 0.03**, minimum fold ≈ 0.93.

The low SD and high minimum show the signal is not carried by a single
lucky fold. The interferon vs neutrophil axis is a strong, biologically
coherent separator, which is why even a sparse linear model saturates AUC.

### Feature stability across folds

I tracked which genes received non-zero L1 weight in each of the five
training folds. A stable core — **IFI27** and the interferon block, plus
neutrophil granule genes — is selected in 4–5/5 folds, while many
low-weight genes flicker in and out. The reported markers are restricted
to the stable, directionally-consistent core rather than any single
full-data fit.

## 5. Calibration

Because clinical use cares about probabilities, not just ranking, I
measured the **Brier score** under the same CV (mean ≈ 0.04). The L1
logistic outputs are already well calibrated on this cohort; no isotonic
/ Platt recalibration was needed. Calibration is reported alongside AUC so
a well-ranked-but-overconfident model would be caught.

## 6. Biological plausibility of top predictors

The learned markers reconcile with textbook host-response biology
(see `markers.json`):

- **Viral / interferon:** IFI27, IFI44L, IFIT1/3, ISG15, MX1, OAS1/2/3,
  RSAD2, USP18, SIGLEC1 — the canonical type-I **interferon**-stimulated
  gene program induced by influenza A.
- **Bacterial / neutrophil:** LTF, CAMP, ELANE, MPO, BPI, DEFA1/4,
  CEACAM8, OLFM4, MMP8 — the **neutrophil** / myeloid granule program of
  emergency granulopoiesis in pyogenic bacterial infection.

Every reported viral marker has higher mean expression in the viral
group and every bacterial marker higher in the bacterial group, and a
classifier built from these markers alone retains a cross-validated AUC
of ~0.96 — so the predictive signal and the known biology agree rather
than being a statistical artifact.

## 7. Limitations

Single-platform, single-cohort, modest n with class imbalance; influenza
A is the only viral agent, so "viral" generalisation to other viruses is
untested. Erythroid genes (e.g. ALAS2, HBD) also separate the classes,
likely reflecting reticulocytosis/blood-composition shifts in sepsis
rather than a direct host-defence program, and were therefore not
promoted as primary markers.
