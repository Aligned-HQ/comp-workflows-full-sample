"""Reference solution: viral vs bacterial infection classifier from GSE6269."""
from __future__ import annotations

import json
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

OUTPUT_PATH = Path("/workspace/output/result.json")
GEO_SOFT_PATH = Path("/workspace/data/GSE6269_family.soft.gz")
SEED = 123

# Load the gene-expression series. The file is provided by the task
# environment under /workspace/data/.
gse = GEOparse.get_GEO(filepath=str(GEO_SOFT_PATH))

# Restrict to GPL96 samples and build the expression matrix.
gpl96_ids = [
    gsm_id for gsm_id, gsm in gse.gsms.items()
    if gsm.metadata["platform_id"][0] == "GPL96"
]
original_gsms = gse.gsms
gse.gsms = {gsm_id: gse.gsms[gsm_id] for gsm_id in gpl96_ids}
gene_expr = gse.pivot_samples("VALUE").dropna(how="all")
gse.gsms = original_gsms

# Probe -> gene-symbol mapping (drop NA, keep first symbol when multi-mapped).
gpl = gse.gpls["GPL96"].table[["ID", "Gene Symbol"]].copy()
gpl = gpl.dropna()
gpl["Gene Symbol"] = gpl["Gene Symbol"].str.split(" /// ").str[0]

# Aggregate probes mapping to the same gene by arithmetic mean (samples x genes).
merged = gene_expr.merge(gpl, left_index=True, right_on="ID")
final_df = merged.groupby("Gene Symbol").mean(numeric_only=True).T

# Extract binary labels (viral=1, bacterial=0) from sample metadata.
labels: list[int] = []
sample_ids: list[str] = []
for gsm_id in final_df.index:
    metadata_str = " ".join(gse.gsms[gsm_id].metadata["characteristics_ch1"]).lower()
    if "influenza a" in metadata_str:
        labels.append(1)
        sample_ids.append(gsm_id)
    elif any(b in metadata_str for b in ("s. aureus", "e. coli", "s. pneumoniae")):
        labels.append(0)
        sample_ids.append(gsm_id)

X = final_df.loc[sample_ids]
y = np.array(labels)

# Clip negatives, log2-transform if the data isn't already log-scaled.
p99 = np.nanpercentile(X, 99)
if p99 > 100:
    print(f"Status: 99th percentile is {p99:.2f}. Applying log2 transformation...")
    X = np.log2(X.clip(lower=0) + 1)
else:
    print(f"Status: 99th percentile is {p99:.2f}. Data is already log-scaled.")
    X = X.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    class_weight="balanced",
    random_state=SEED,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {
    "accuracy": "accuracy",
    "sensitivity": "recall",
    "specificity": make_scorer(recall_score, pos_label=0),
    "auc": "roc_auc",
}
results = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)

accuracy = float(results["test_accuracy"].mean())
sensitivity = float(results["test_sensitivity"].mean())
specificity = float(results["test_specificity"].mean())
auc = float(results["test_auc"].mean())

print("--- Final Diagnostic Report ---")
print(f"Mean CV Accuracy:    {accuracy * 100:.3g}%")
print(f"Mean CV Sensitivity: {sensitivity * 100:.3g}%")
print(f"Mean CV Specificity: {specificity * 100:.3g}%")
print(f"Mean CV AUC:         {auc:.3g}")

# Refit on the full set and rank LASSO-selected biomarkers.
model.fit(X_scaled, y)
coef_series = pd.Series(model.coef_[0], index=X.columns)
selected_genes = coef_series[coef_series != 0]

print("\n--- Model Complexity ---")
print(f"LASSO selected {len(selected_genes)} genes out of {X.shape[1]}")

top_viral = coef_series.sort_values(ascending=False).head(3)
top_bacterial = coef_series.sort_values(ascending=True).head(3)

print("\n--- Top Viral Biomarkers (Positive Coefficients) ---")
print(top_viral.apply(lambda x: f"{x:.3g}"))
print("\n--- Top Bacterial Biomarkers (Negative Coefficients) ---")
print(top_bacterial.apply(lambda x: f"{x:.3g}"))

result = {
    "accuracy": accuracy,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "auc": auc,
    "viral_biomarkers": [
        {"gene": str(gene), "coefficient": float(coef)}
        for gene, coef in top_viral.items()
    ],
    "bacterial_biomarkers": [
        {"gene": str(gene), "coefficient": float(coef)}
        for gene, coef in top_bacterial.items()
    ],
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result))
