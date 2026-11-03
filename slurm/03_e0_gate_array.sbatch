#!/usr/bin/env bash
# 03_e0_gate_array.sbatch
# The Nov-8 gate matrix: E0.1 (source mask), E0.2 (junk control), E0.3 (oracle
# detector). One array task = one (config x budget) cell. Each task starts its
# own vLLM victim, runs trainStudent.py once, and tears the server down.
#
# Why one job per cell (not one job, many runs): the teacher-answer cache is now
# namespaced per run_name, but a fresh job also isolates GPU memory and makes a
# single OOM/crash lose one cell, not the whole array.
#
# Actor: give this to the PI. Ask them to submit the array, then paste the
# per-task .out/.err plus the RUN_ROOT listings. Analyse combined logs.
#
# PRIORITY: E0.1 first (array 0-3). If those succeed, submit the rest.
#
# Cell map (see CONFIGS below for the authoritative list):
#   0 baseline B200 | 1 full B200 | 2 full+mask B200 | 3 full+mask B500
#   4 full B500     | 5 junk B200 | 6 junk B500       | 7 oracle B200 | 8 oracle B500
#
#SBATCH --job-name=e0gate
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%1
#SBATCH --output=e0_gate_%A_%a.out
#SBATCH --error=e0_gate_%A_%a.err

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

# ============================================================================
# REPO is hardcoded to the cluster checkout (branch repro/oct-2026).
# ============================================================================
REPO="/orange/sgao1/sli/knowledge-honeypot"
SNAP_DIR_ROOT="${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
if [ -z "${SNAP:-}" ]; then
    SNAP="$(ls -d "${SNAP_DIR_ROOT}/"*/ 2>/dev/null | head -n1)"
fi
SNAP="${SNAP%/}"
if [ ! -d "$REPO" ]; then
    echo "FATAL: REPO='${REPO}' is not a directory. Edit slurm/03_e0_gate_array.sbatch and set REPO."; exit 2
fi
if [ -z "$SNAP" ] || [ ! -d "$SNAP" ]; then
    echo "FATAL: victim SNAP not found under ${SNAP_DIR_ROOT}. Run env/03_download first or set SNAP."; exit 2
fi
echo "REPO=${REPO}"; echo "SNAP=${SNAP}"

export VICTIM_BACKEND=local_vllm
export VICTIM_BASE_URL=http://127.0.0.1:8000/v1
VICTIM_NAME=llama-3.1-8b-instruct

idx="${SLURM_ARRAY_TASK_ID}"

# ---- config table: idx -> experiment name | defense | budget | extra flags ----
case "${idx}" in
  0) NAME="e0_1_baseline_B200";      MODE="baseline"; B=200; EXTRA="--mask_source=true" ;;
  1) NAME="e0_1_full_tagged_B200";   MODE="full";     B=200; EXTRA="" ;;
  2) NAME="e0_1_full_mask_B200";     MODE="full";     B=200; EXTRA="--mask_source=true" ;;
  3) NAME="e0_1_full_mask_B500";     MODE="full";     B=500; EXTRA="--mask_source=true" ;;
  4) NAME="e0_1_full_tagged_B500";   MODE="full";     B=500; EXTRA="" ;;
  5) NAME="e0_2_junk_B200";          MODE="full";     B=200; EXTRA="--junk_control=true --mask_source=true" ;;
  6) NAME="e0_2_junk_B500";          MODE="full";     B=500; EXTRA="--junk_control=true --mask_source=true" ;;
  7) NAME="e0_3_oracle_B200";        MODE="full";     B=200; EXTRA="--detector_mode=oracle --mask_source=true" ;;
  8) NAME="e0_3_oracle_B500";        MODE="full";     B=500; EXTRA="--detector_mode=oracle --mask_source=true" ;;
  *) echo "Unknown array index ${idx}"; exit 2 ;;
esac

cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 2; }
run_root="${REPO}/runs/e0/${SLURM_ARRAY_JOB_ID}_${idx}"
mkdir -p "$run_root"
echo "config: name=${NAME} mode=${MODE} B=${B} extra=[${EXTRA}]"

# --- start the local victim (vLLM) ---
# MUST use --enforce-eager: without it vLLM does torch.compile level 3 + 68 CUDA
# graph captures on cold boot, hanging >10 min at 0% GPU. Leaves ~71 GiB for the
# 8B attacker on the same B200. Plain redirection.
export VLLM_LOGGING_LEVEL=INFO
python -m vllm.entrypoints.openai.api_server \
    --model "${SNAP}" \
    --served-model-name "${VICTIM_NAME}" \
    --port 8000 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 2048 \
    --enforce-eager \
    > "${run_root}/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

up=0
for i in $(seq 1 300); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then up=1; break; fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "[wait $((i*2))s] vllm alive=$(kill -0 ${VLLM_PID} 2>/dev/null && echo yes || echo no)"
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM PROCESS EXITED before healthy"; break
    fi
    sleep 2
done
if [ "$up" -ne 1 ]; then
    echo "vLLM FAILED TO START -- full vllm.log follows"
    cat "${run_root}/vllm.log"
    exit 1
fi

# --- provenance ---
{ nvidia-smi; python -c "import torch; print('cap', torch.cuda.get_device_capability(0))"; } > "${run_root}/gpu.txt" 2>&1 || true
python -m pip freeze > "${run_root}/pip_freeze.txt" 2>/dev/null || true
git -C "$REPO" rev-parse HEAD > "${run_root}/commit.txt" 2>/dev/null || true
echo "seed=0" > "${run_root}/manifest.txt"

# --- the attacker run ---
python trainStudent.py \
    --STUDENT_BASE=meta-llama/Llama-3.1-8B-Instruct \
    --domain=medical \
    --MAX_QUERIES="${B}" \
    --Teacher_model="${VICTIM_NAME}" \
    --defense_mode="${MODE}" \
    --detection_threshold=0.6 \
    --hkg_size=1500 \
    --hkg_path=honeypot_hkg/medical_hkg_1500_repaired.json \
    --hkg_sample_strategy=prefix \
    --experiment_name="${NAME}" \
    --skip_final_train=true \
    --with_coq=false \
    --with_followup=false \
    --with_correction=false \
    ${EXTRA} \
    2>&1 | tee "${run_root}/train.log"

echo "CELL_DONE ${NAME}"
# Copy the run artifacts next to the logs so they are easy to hand back.
cp -f "${REPO}/results/${NAME}"*  "${run_root}/" 2>/dev/null || true
cp -f "${REPO}/checkpoint/${NAME}"* "${run_root}/" 2>/dev/null || true
ls -la "${run_root}"
