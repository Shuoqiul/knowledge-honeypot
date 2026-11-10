#!/usr/bin/env bash
# 06_e0_8_benign_acc.sbatch
# E0.8: accuracy loss for BENIGN users who are falsely triggered and substituted.
# Complements E0.7 (which measured only trigger probability). Uses task accuracy
# vs GOLD on MedQA. Session-length driven: queries at step >= t0 are substituted.
#
# Needs: env/08_download_datasets.sh run once (MedQA in datasets cache, offline).
# Prereq artifacts: none (does not use collected pairs). Only the victim server.
#
#SBATCH --job-name=e0-8
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=e0_8_%j.out
#SBATCH --error=e0_8_%j.err

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

REPO="/orange/sgao1/sli/knowledge-honeypot"
SNAP_DIR_ROOT="${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
SNAP="${SNAP:-$(ls -d "${SNAP_DIR_ROOT}/"*/ 2>/dev/null | head -n1)}"
SNAP="${SNAP%/}"
if [ ! -d "$REPO" ] || [ -z "$SNAP" ] || [ ! -d "$SNAP" ]; then
    echo "FATAL: REPO or SNAP missing (REPO=$REPO SNAP=$SNAP)"; exit 2
fi
cd "$REPO"
run_root="${REPO}/runs/e0_8/${SLURM_JOB_ID}"; mkdir -p "$run_root"

export VLLM_LOGGING_LEVEL=INFO
python -m vllm.entrypoints.openai.api_server \
    --model "${SNAP}" --served-model-name llama-3.1-8b-instruct \
    --port 8000 --gpu-memory-utilization 0.60 --max-model-len 2048 --enforce-eager \
    > "${run_root}/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
up=0
for i in $(seq 1 900); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then up=1; break; fi
    if [ $((i % 60)) -eq 0 ]; then echo "[wait $((i*2))s] alive=$(kill -0 ${VLLM_PID} 2>/dev/null && echo yes || echo no)"; fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then echo "vLLM EXITED"; cat "${run_root}/vllm.log"; exit 1; fi
    sleep 2
done
if [ "$up" -ne 1 ]; then echo "vLLM FAILED"; cat "${run_root}/vllm.log"; exit 1; fi
echo "vLLM up"

{ nvidia-smi; python -c "import torch; print('cap',torch.cuda.get_device_capability(0))"; } > "${run_root}/gpu.txt" 2>&1 || true
python -m pip freeze > "${run_root}/pip_freeze.txt" 2>/dev/null || true
git -C "$REPO" rev-parse HEAD > "${run_root}/commit.txt" 2>/dev/null || true

for DEF in hkg junk; do
  python scripts/e0_8_benign_accuracy.py \
      --domain medical --benchmark medqa --limit 100 \
      --lengths 50,100,200,500 --taus 0.6,0.7,0.8 \
      --defense "${DEF}" \
      --base_url http://127.0.0.1:8000/v1 \
      --victim_model llama-3.1-8b-instruct \
      --out "results/e0_8_${DEF}.json" \
      2>&1 | tee "${run_root}/e0_8_${DEF}.log"
done

cp -f results/e0_8_*.json "${run_root}/" 2>/dev/null || true
echo "E0_8_DONE"
ls -la "$run_root"
