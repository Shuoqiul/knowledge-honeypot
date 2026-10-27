#!/usr/bin/env bash
# 03_download_llama31_8b.sh -- download LLaMA-3.1-8B-Instruct ONCE.
#
# RUN IN AN INTERACTIVE SESSION (no GPU needed). This is the ONLY script where
# HF_HUB_OFFLINE=0. Everything else runs with HF_HUB_OFFLINE=1 and reads the cache.
#
# HF_TOKEN: do NOT hardcode it in this file (it would be committed). Two options:
#   A) export HF_TOKEN=hf_xxx   in your shell, then run this script; or
#   B) put the token on its own line in ~/.hf_token (chmod 600) and run this script;
#      it will source that file if HF_TOKEN is unset.
# The token is never echoed.
set -uo pipefail
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_NAME=kh-torch28

source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${CONDA_ENVS_DIR}/${ENV_NAME}"

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=0

# Resolve the token without printing it.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
    # file contains just the token (optionally prefixed with "HF_TOKEN=")
    HF_TOKEN="$(sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^HF_TOKEN=//' "$HOME/.hf_token" | head -n1)"
    export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it, or write it to ~/.hf_token (chmod 600)."
    exit 1
fi
# Don't print the token; just confirm it is present.
echo "HF_TOKEN is set (len=${#HF_TOKEN}); gated access must be granted to this account."

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --repo-type model \
    --cache-dir "${HF_HUB_CACHE}"

echo "DOWNLOAD_OK"
echo "SNAP_DIR below (copy the hash to SNAP in the sbatch files):"
ls -d "${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"* || true
