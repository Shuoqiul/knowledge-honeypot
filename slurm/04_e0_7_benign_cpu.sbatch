#!/usr/bin/env bash
# 04_e0_7_benign_cpu.sbatch
# Purpose: E0.7 benign false-trigger rate, source masked. CPU only, no GPU, no
# network. Runs on a login/CPU node.
# Actor: give to PI any time in Phase 0-2. Fast.
#
#SBATCH --job-name=e0-7-benign
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=e0_7_%j.out
#SBATCH --error=e0_7_%j.err

set -o pipefail
set -x

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /orange/sgao1/ZhengaoLi/Envs/torch2.7.0

export PYTHONUNBUFFERED=1
REPO="${REPO:-/path/to/knowledge-honeypot}"
cd "$REPO"
mkdir -p results/e0_7

python scripts/e0_7_benign_false_trigger.py \
    --domain medical \
    --limit 0 \
    --lengths 50,100,200,500 \
    --taus 0.4,0.5,0.6,0.7,0.8 \
    --out results/e0_7/e0_7_medical.json

echo "E0_7_DONE"
