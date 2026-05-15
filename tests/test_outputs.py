"""Pytest verifier for the viral-vs-bacterial classifier task.

Loads /workspace/output/result.json and checks it against the reference
values from the task author, with tolerances matching the verification
criteria in the original task submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


OUTPUT_PATH = Path("/workspace/output/result.json")

EXPECTED_KEYS = {
    "accuracy",
    "sensitivity",
    "specificity",
    "auc",
    "viral_biomarkers",
    "bacterial_biomarkers",
}

# Reference metric values from the task author's solution.
EXPECTED_ACCURACY = 0.944
EXPECTED_SENSITIVITY = 1.000
EXPECTED_SPECIFICITY = 0.930
EXPECTED_AUC = 0.978

# Tolerances pulled from the original verificationCriteria.
METRIC_TOL = 0.05  # ±5 percentage points for accuracy/sensitivity/specificity
AUC_TOL = 0.05

EXPECTED_VIRAL_GENES = {"OTOF", "HIST1H4H", "IFI27"}
EXPECTED_BACTERIAL_GENES = {"OSBPL1A", "VGLL1", "AC004692.5"}
MIN_GENE_MATCHES = 2


@pytest.fixture(scope="module")
def output() -> dict:
    assert OUTPUT_PATH.exists(), f"output file not found at {OUTPUT_PATH}"
    return json.loads(OUTPUT_PATH.read_text())


def test_output_has_required_keys(output: dict) -> None:
    assert set(output) == EXPECTED_KEYS


@pytest.mark.parametrize(
    "key, expected",
    [
        ("accuracy", EXPECTED_ACCURACY),
        ("sensitivity", EXPECTED_SENSITIVITY),
        ("specificity", EXPECTED_SPECIFICITY),
    ],
)
def test_metric_within_tolerance(output: dict, key: str, expected: float) -> None:
    value = float(output[key])
    assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"
    assert abs(value - expected) <= METRIC_TOL, (
        f"{key}={value:.3f} differs from reference {expected} by more than ±{METRIC_TOL}"
    )


def test_auc_within_tolerance(output: dict) -> None:
    auc = float(output["auc"])
    assert 0.0 <= auc <= 1.0, f"auc={auc} out of [0, 1]"
    assert abs(auc - EXPECTED_AUC) <= AUC_TOL, (
        f"auc={auc:.3f} differs from reference {EXPECTED_AUC} by more than ±{AUC_TOL}"
    )


def _check_biomarkers(
    biomarkers: list,
    expected_genes: set[str],
    *,
    sign: str,
) -> None:
    assert isinstance(biomarkers, list), "biomarkers must be a list"
    assert len(biomarkers) == 3, f"expected 3 biomarkers, got {len(biomarkers)}"

    for entry in biomarkers:
        assert isinstance(entry, dict), f"biomarker entry must be a dict, got {type(entry)}"
        assert set(entry) == {"gene", "coefficient"}, (
            f"biomarker entry must have keys {{'gene', 'coefficient'}}, got {set(entry)}"
        )
        coef = float(entry["coefficient"])
        if sign == "positive":
            assert coef > 0, f"viral biomarker {entry['gene']!r} has non-positive coef {coef}"
        else:
            assert coef < 0, f"bacterial biomarker {entry['gene']!r} has non-negative coef {coef}"

    submitted = {entry["gene"] for entry in biomarkers}
    overlap = submitted & expected_genes
    assert len(overlap) >= MIN_GENE_MATCHES, (
        f"expected at least {MIN_GENE_MATCHES} of {expected_genes}; got {submitted}"
    )


def test_viral_biomarkers(output: dict) -> None:
    _check_biomarkers(output["viral_biomarkers"], EXPECTED_VIRAL_GENES, sign="positive")


def test_bacterial_biomarkers(output: dict) -> None:
    _check_biomarkers(
        output["bacterial_biomarkers"], EXPECTED_BACTERIAL_GENES, sign="negative"
    )


def test_viral_biomarkers_ordered_descending(output: dict) -> None:
    coefs = [float(entry["coefficient"]) for entry in output["viral_biomarkers"]]
    assert coefs == sorted(coefs, reverse=True), (
        f"viral_biomarkers must be sorted from largest to smallest coefficient; got {coefs}"
    )


def test_bacterial_biomarkers_ordered_ascending(output: dict) -> None:
    coefs = [float(entry["coefficient"]) for entry in output["bacterial_biomarkers"]]
    assert coefs == sorted(coefs), (
        f"bacterial_biomarkers must be sorted from most-negative to least-negative; got {coefs}"
    )
