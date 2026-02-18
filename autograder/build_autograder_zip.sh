#!/usr/bin/env bash
# build_autograder_zip.sh — Assembles the Gradescope autograder zip.
# Run from the project root:  bash autograder/build_autograder_zip.sh
#
# Produces: autograder_upload.zip (ready to upload to Gradescope)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT="autograder_upload.zip"
rm -f "$OUTPUT"

zip -r "$OUTPUT" \
    autograder/grade_race.py \
    src/ \
    usd/ \
    pyproject.toml \
    setup.py \
    -x "src/third_parties/rsl_rl_local/.git/*" \
    -x "**/__pycache__/*" \
    -x "*.pyc"

# Add setup.sh and run_autograder at the zip root (not under autograder/)
cd autograder
zip -g "../$OUTPUT" setup.sh run_autograder
cd "$PROJECT_ROOT"

echo ""
echo "Built: $PROJECT_ROOT/$OUTPUT"
echo "Upload this file as the Gradescope autograder zip."
