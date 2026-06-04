# Tool usage — viral-vs-bacterial classifier task

This task is designed to force heterogeneous tool use, multiple data
sources, and at least two distinct computational modes whose outputs must
be reconciled. This file documents the genuinely-different tools / data
sources / computational modes the expected (reference) solution uses, how
each maps to a deliverable, and whether the verifier hard-enforces it.

Summary: **5 distinct tools/modes**, of which **4 are machine-enforced** by
the verifier and **1 (web research) is requested but unscored** (it was
only ever validated through `report.md`, which is no longer machine-graded).

| # | Tool / data source / mode | Kind | Deliverable | Enforced? |
|---|---------------------------|------|-------------|-----------|
| 1 | GEOparse SOFT parsing | Local data source A | `pipeline.py` | Yes |
| 2 | GPL96 platform annotation (probe→gene) | 2nd table joined | `pipeline.py` (`feature_names`) | Yes |
| 3 | scikit-learn classifier + CV + calibration | Computational mode A (prediction) | `pipeline.py` | Yes |
| 4 | Differential-expression marker analysis | Computational mode B (inference) | `markers.json` | Yes |
| 5 | Biological-signature reconciliation (ISG / neutrophil gene sets) | Reference/domain knowledge | `markers.json` | Yes |
| 6 | Web research on NCBI GEO / PubMed | External data source B | `report.md` | No (unscored) |

## 1. GEOparse SOFT parsing — local data source A — ENFORCED
Parse the bundled `/workspace/data/GSE6269_family.soft.gz` with
`GEOparse.get_GEO`. Pull per-sample `characteristics_ch1` free-text
metadata to recover labels, restrict to platform `GPL96`, and pivot
probe-level `VALUE` measurements into a samples × probes matrix.
Verifier hook: the cohort must equal the 91 labelled GPL96 samples the
verifier independently derives.

## 2. GPL96 platform annotation (probe→gene mapping) — 2nd table — ENFORCED
Read `gse.gpls["GPL96"].table` and map probe IDs to `Gene Symbol`,
dropping unmapped probes and collapsing multi-mapped / duplicate probes
onto genes. This is a structurally different table inside the file that
must be joined to the expression matrix.
Verifier hook: `build_features` must return `feature_names` (gene symbols)
aligned to the columns of `X`, and the reported markers must appear in
them.

## 3. scikit-learn classifier + cross-validation + calibration — mode A — ENFORCED
Build the leakage-safe pipeline (`build_features` is label-free;
label-dependent steps live in `make_estimator` and are refit per fold).
Reference: `StandardScaler` → L1 logistic regression, balanced classes.
Verifier hook: verifier-owned `StratifiedKFold` scoring — mean AUC ≥ 0.90,
worst fold ≥ 0.80, mean Brier ≤ 0.15 (calibration via `predict_proba`).

## 4. Differential-expression marker analysis — mode B — ENFORCED
A per-gene viral-vs-bacterial contrast (e.g. standardised mean difference
or t-test) to derive directional markers, cross-checked against the
model's coefficients. This is a genuinely different computational mode
from the classifier, and its output must reconcile with mode A.
Why a separate mode is needed: the sparse L1 classifier selects a minimal
set (dominated by a few strong genes) and does not surface enough of the
neutrophil program on its own, so a DE analysis is required to populate
`markers.json` with ≥ 8 directional, signature-overlapping genes per class.
Verifier hook: markers must be directionally correct (≥ 80% per class) and
independently predictive (marker-only CV AUC ≥ 0.85).

## 5. Biological-signature reconciliation — reference knowledge — ENFORCED
Compare the learned viral markers against the canonical type-I
interferon / interferon-stimulated-gene (ISG) program and the bacterial
markers against the neutrophil / myeloid granulocyte program (e.g. using
MSigDB / Interferome-style reference gene sets or the literature).
Verifier hook: reported markers must overlap the verifier's own ISG and
neutrophil gene sets in ≥ 3 genes per class.

## 6. Web research on NCBI GEO / PubMed — external data source B — NOT ENFORCED
Query the live GEO record for `GSE6269` and its linked publication
(PubMed 17105821; 143 deposited samples across GPL96 / GPL570 / GPL2507)
and reconcile that external record with the bundled SOFT file in
`report.md`.
Status: requested in the instruction but **not machine-scored** — its only
verifier hooks lived in the `report.md` checks, which were removed.
Internet access remains enabled (`allow_internet = true`) so the step is
possible; it is currently honor-system / human-review only.

## Reconciliation pairs (the core requirement)
- classifier weights ↔ differential-expression markers (modes 3 ↔ 4)
- learned markers ↔ known interferon / neutrophil signatures (4 ↔ 5)
- bundled SOFT file ↔ external GEO / PubMed record (1 ↔ 6, unscored)
