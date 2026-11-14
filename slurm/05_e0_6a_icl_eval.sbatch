#!/usr/bin/env bash
# 05_e0_6a_icl_eval.sbatch
# E0.6a: ICL Agreement with LOCAL victim+surrogate, comparing demo selection
# (random vs relevance) across baseline / full-mask / junk pair sets, 5 seeds.
# Consumes ALREADY-COLLECTED qa_pairs from the E0.1/E0.2 runs; does NOT re-run
# the attacker. Only the victim server is needed.
#
# Actor: submit after E0.1/E0.2 pairs exist (jobs 49707321, 50380074). Paste back
# results/e0_6a_*.json and this .out.
#
#SBATCH --job-name=e0-6a
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=e0_6a_%j.out
#SBATCH --error=e0_6a_%j.err

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
run_root="${REPO}/runs/e0_6a/${SLURM_JOB_ID}"; mkdir -p "$run_root"

# --- victim server (same launch as the known-good E0 config) ---
export VLLM_LOGGING_LEVEL=INFO
python -m vllm.entrypoints.openai.api_server \
    --model "${SNAP}" --served-model-name llama-3.1-8b-instruct \
    --port 8000 --gpu-memory-utilization 0.60 --max-model-len 16384 --enforce-eager \
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

# --- provenance ---
{ nvidia-smi; python -c "import torch; print('cap',torch.cuda.get_device_capability(0))"; } > "${run_root}/gpu.txt" 2>&1 || true
python -m pip freeze > "${run_root}/pip_freeze.txt" 2>/dev/null || true
git -C "$REPO" rev-parse HEAD > "${run_root}/commit.txt" 2>/dev/null || true

# --- benchmark: medqa (full USMLE test). Requires env/08_download_datasets.sh
#     to have populated the HF datasets cache (offline). Override with BM env. ---
BM="${BM:-medqa}"
echo "benchmark=${BM}"

# Pair sets from the completed E0 runs (globs resolve to the latest snapshot).
QA_BASE="checkpoint/e0_1_baseline_B200_all_qa_pairs_200_*.json"
QA_MASK="checkpoint/e0_1_full_mask_B200_all_qa_pairs_200_*.json"
QA_JUNK="checkpoint/e0_2_junk_B200_all_qa_pairs_200_*.json"

for MODE in random relevance; do
  python evaluate_icl_local.py \
      --qa_files "$QA_BASE" "$QA_MASK" "$QA_JUNK" \
      --benchmark "$BM" --domain medical --limit 100 --max_demos 24 --max_prompt_tokens 15000 \
      --seeds 0,1,2,3,4 --demo_mode "$MODE" \
      --base_url http://127.0.0.1:8000/v1 \
      --victim_model llama-3.1-8b-instruct \
      --surrogate_model llama-3.1-8b-instruct \
      --judge_model llama-3.1-8b-instruct \
      --output "results/e0_6a_${MODE}.json" \
      2>&1 | tee "${run_root}/e0_6a_${MODE}.log"
done

cp -f results/e0_6a_*.json "${run_root}/" 2>/dev/null || true
echo "E0_6A_DONE"
ls -la "$run_root"
