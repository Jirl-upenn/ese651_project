#!/usr/bin/env bash
# setup.sh — Gradescope autograder setup script
# Installs Isaac Sim, Isaac Lab, and all project dependencies from scratch.
# Reference: ESE651 Drone Racing Project Handout §1.1

set -euo pipefail

# ── 1. System dependencies ──────────────────────────────────────────
apt-get update && apt-get install -y cmake build-essential git

# ── 2. Upgrade pip ──────────────────────────────────────────────────
pip install --upgrade pip

# ── 3. Install CUDA-enabled PyTorch 2.7.0 (CUDA 12.8) ──────────────
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── 4. Install Isaac Sim 4.5.0 ──────────────────────────────────────
pip install "isaacsim[all,extscache]==4.5.0" \
    --extra-index-url https://pypi.nvidia.com

# ── 5. Clone & install the class fork of Isaac Lab ──────────────────
# Isaac Lab must be at the same level as the project directory.
cd /autograder
git clone https://github.com/vineetpasumarti/IsaacLab.git
cd /autograder/IsaacLab
# '--install none' skips RL libraries (project uses a local rsl_rl copy)
./isaaclab.sh --install none

# ── 6. Install the local rsl_rl library ─────────────────────────────
cd /autograder/source
pip install -e src/third_parties/rsl_rl_local

# ── 7. Install the project package itself ───────────────────────────
pip install -e .

# ── 8. Install any remaining Python deps ────────────────────────────
pip install tensordict scipy
