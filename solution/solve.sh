#!/usr/bin/env bash
set -euo pipefail

# The deliverables are the pipeline module plus the marker table and model
# card. The verifier imports the pipeline, runs its own cross-validation, and
# checks the reported markers and report against the data and the external
# GEO record. Place all three reference artifacts where the verifier expects
# the agent's output.

OUTPUT_DIR=${OUTPUT_DIR:-/workspace/output}
SOLUTION_DIR=${SOLUTION_DIR:-/solution}

mkdir -p "${OUTPUT_DIR}"
cp "${SOLUTION_DIR}/pipeline.py" "${OUTPUT_DIR}/pipeline.py"
cp "${SOLUTION_DIR}/markers.json" "${OUTPUT_DIR}/markers.json"
cp "${SOLUTION_DIR}/report.md" "${OUTPUT_DIR}/report.md"
