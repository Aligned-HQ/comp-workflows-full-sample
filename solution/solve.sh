#!/usr/bin/env bash
set -euo pipefail

# The deliverable is the pipeline module; the verifier imports it and runs
# its own cross-validation. Place the reference pipeline where the verifier
# expects the agent's output.

OUTPUT_DIR=${OUTPUT_DIR:-/workspace/output}
PIPELINE_PATH=${PIPELINE_PATH:-${OUTPUT_DIR}/pipeline.py}
SOLUTION_DIR=${SOLUTION_DIR:-/solution}

mkdir -p "${OUTPUT_DIR}"
cp "${SOLUTION_DIR}/pipeline.py" "${PIPELINE_PATH}"
