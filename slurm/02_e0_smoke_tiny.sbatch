#!/usr/bin/env bash
# 02_e0_smoke_tiny.sbatch
# Purpose: validate the E0 patches + local victim wiring before spending B200
# hours on the real matrix. Uses the README's tiny model as the attacker base
# but the REAL local victim (vLLM LLaMA-3.1-8B-Instruct), budget B=20, both
# baseline and full, one seed. Cheap. Produces logs to inspect for the whole
# pipeline (defense JSONL, structured log, honeypot summary).
#
# Actor: give this to the PI after 00 and 01 pass. Ask for the .out/.err and
# the contents of the RUN_ROOT dir listing.
#
#SBATCH --job-name=e0-smoke
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=e0_smoke_%j.out
#SBATCH --error=e0_smoke_%j.err

set -o pipefail
set -x

source /orange/sgao1/sli/conda_cache/anaconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/orange/sgao1/sli/.conda/envs
conda activate /orange/sgao1/sli/.conda/envs/kh-torch28

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# ---- fill these in from 00/01 outputs (VERIFY with PI) ----
REPO="${REPO:-/path/to/knowledge-honeypot}"          # checkout of branch repro/oct-2026
SNAP="${SNAP:-${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<HASH>}"
export VICTIM_BACKEND=local_vllm
export VICTIM_BASE_URL=http://127.0.0.1:8000/v1
VICTIM_NAME=llama-3.1-8b-instruct

cd "$REPO"
run_root="${REPO}/runs/e0_smoke/${SLURM_JOB_ID}"
mkdir -p "$run_root"

# --- start the local victim (vLLM) ---
python -m vllm.entrypoints.openai.api_server \
    --model "${SNAP}" \
    --served-model-name "${VICTIM_NAME}" \
    --port 8000 \
    --gpu-memory-utilization 0.35 \
    --max-model-len 8192 \
    > "${run_root}/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "waiting for vLLM /health ..."
for i in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "vLLM is up after ${i} tries"
        break
    fi
    sleep 2
done
curl -fsS http://127.0.0.1:8000/health || { echo "vLLM FAILED TO START"; tail -n 80 "${run_root}/vllm.log"; exit 1; }

# --- provenance ---
{ nvidia-smi; python -c "import torch; print('cap', torch.cuda.get_device_capability(0))"; } > "${run_root}/gpu.txt" 2>&1 || true
python -m pip freeze > "${run_root}/pip_freeze.txt" 2>/dev/null || true
git -C "$REPO" rev-parse HEAD > "${run_root}/commit.txt" 2>/dev/null || true

# --- run baseline and full at B=20 with the tiny attacker ---
for mode in baseline full; do
  run_name="smoke_${mode}_B20"
  python trainStudent.py \
      --STUDENT_BASE=sshleifer/tiny-gpt2 \
      --domain=medical \
      --MAX_QUERIES=20 \
      --Teacher_model="${VICTIM_NAME}" \
      --defense_mode="${mode}" \
      --detection_threshold=0.6 \
      --hkg_size=1500 \
      --hkg_path=honeypot_hkg/medical_hkg_1500_repaired.json \
      --experiment_name="${run_name}" \
      --skip_final_train=true \
      --with_coq=false \
      --with_followup=false \
      --with_correction=false \
      2>&1 | tee "${run_root}/${run_name}.log"
done

echo "SMOKE_DONE"
ls -la "${REPO}/results" | tail -n 40 || true
