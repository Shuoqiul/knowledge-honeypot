#!/usr/bin/env bash
# 01_download_llama31_8b.sbatch
# Purpose: download LLaMA-3.1-8B-Instruct ONCE into the shared HF cache.
# This is the ONLY job where HF_HUB_OFFLINE must be 0.
# Actor: give this to the PI. Acceptance = "DOWNLOAD_OK" + a snapshot dir listed.
# If the job dies mid-way, resubmit the SAME script; huggingface-cli resumes.
#
# PI prerequisites:
#   - HF account has accepted Meta's Llama-3.1 license.
#   - HF_TOKEN exported in the submitting shell (never echo it).
#
#SBATCH --job-name=dl-llama31-8b
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=dl_llama31_%j.out
#SBATCH --error=dl_llama31_%j.err

set -o pipefail
set -x

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /orange/sgao1/ZhengaoLi/Envs/torch2.7.0

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=0
: "${HF_TOKEN:?HF_TOKEN not set in the submitting environment}"

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --repo-type model \
    --cache-dir "${HF_HUB_CACHE}"

echo "DOWNLOAD_OK"
ls -la "${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots" || true
