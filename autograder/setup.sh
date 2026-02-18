#!/usr/bin/env bash
# setup.sh — Gradescope autograder setup script
# Assumes Isaac Lab / Isaac Sim is pre-installed on the Gradescope GPU machine.

set -euo pipefail

# Install any additional Python packages needed
pip install tensordict

# Copy project source tree into the autograder environment so that
# grade_race.py can import the task definitions and local rsl_rl.
cp -r /autograder/source/src       /autograder/source/src       2>/dev/null || true
cp -r /autograder/source/scripts   /autograder/source/scripts   2>/dev/null || true
cp -r /autograder/source/usd       /autograder/source/usd       2>/dev/null || true
