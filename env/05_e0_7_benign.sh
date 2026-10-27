#!/usr/bin/env bash
# 05_e0_7_benign.sh -- E0.7 benign false-trigger rate (CPU only, no GPU, no net).
#
# RUN IN AN INTERACTIVE SESSION:
#     bash env/05_e0_7_benign.sh
set -uo pipefail
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_NAME=kh-torch28

source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${CONDA_ENVS_DIR}/${ENV_NAME}"

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
mkdir -p results/e0_7

python scripts/e0_7_benign_false_trigger.py \
    --domain medical \
    --limit 0 \
    --lengths 50,100,200,500 \
    --taus 0.4,0.5,0.6,0.7,0.8 \
    --out results/e0_7/e0_7_medical.json

echo "E0_7_DONE"
