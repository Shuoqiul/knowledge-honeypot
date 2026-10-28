#!/usr/bin/env bash
# 03_download_llama31_8b.sh -- download LLaMA-3.1-8B-Instruct ONCE.
#
# RUN IN AN INTERACTIVE SESSION (no GPU needed). This is the ONLY script where
# HF_HUB_OFFLINE=0. Everything else runs with HF_HUB_OFFLINE=1 and reads the cache.
#
# NOTE: the PI already has a snapshot at
#   .../models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
# so this script will detect it and skip. `hf download` also resumes a partial.
#
# HF_TOKEN: do NOT hardcode it here. Either:
#   A) export HF_TOKEN=hf_xxx   in your shell, then run; or
#   B) write the token to ~/.hf_token (chmod 600); this script reads it if unset.
# The token is never echoed.
set -uo pipefail
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_PATH="${CONDA_ENVS_DIR}/kh-torch28"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${ENV_PATH}"

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"

SNAP_ROOT="${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
if [ -d "${SNAP_ROOT}" ] && [ -n "$(ls -A "${SNAP_ROOT}" 2>/dev/null)" ]; then
    echo "SNAPSHOT_ALREADY_PRESENT (offline, no download):"
    ls -d "${SNAP_ROOT}/"*
    exit 0
fi

export HF_HUB_OFFLINE=0

# Resolve the token without printing it (file may contain "HF_TOKEN=..." or just the token).
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
    HF_TOKEN="$(sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^HF_TOKEN=//' "$HOME/.hf_token" | head -n1)"
    export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export it, or write it to ~/.hf_token (chmod 600)."
    exit 1
fi
echo "HF_TOKEN is set (len=${#HF_TOKEN}); account must have accepted Meta's Llama-3.1 license."

# `hf` is the current CLI (huggingface-cli is deprecated).
hf download meta-llama/Llama-3.1-8B-Instruct \
    --repo-type model \
    --cache-dir "${HF_HUB_CACHE}"

echo "DOWNLOAD_OK"
echo "SNAP_DIR below (copy the hash to SNAP in the sbatch files):"
ls -d "${SNAP_ROOT}/"* || true
