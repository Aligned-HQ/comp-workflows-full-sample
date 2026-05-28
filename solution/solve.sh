#!/usr/bin/env bash
set -euo pipefail

# The deliverable is the pipeline module; the verifier imports it and runs
# its own cross-validation. Place the reference pipeline where the verifier
# expects the agent's output.
mkdir -p /workspace/output
cp /solution/pipeline.py /workspace/output/pipeline.py
