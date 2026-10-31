#!/usr/bin/env bash
# 06_gpu_probe.sh -- quick GPU + vLLM sanity check. MUST run inside a GPU alloc:
#
#   srun --partition=hpg-b200 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:15:00 \
#        bash env/06_gpu_probe.sh
#
# It (1) prints torch CUDA/capability (expect (10,0) on B200), then (2) starts the
# vLLM server for ~90s on a spare port and hits /health, then kills it. This
# isolates "does vLLM start on this node" from the full smoke run.
set -uo pipefail
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/06_gpu_probe_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "logging to ${LOG_FILE}"
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_PATH="${CONDA_ENVS_DIR}/kh-torch28"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${ENV_PATH}"

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1

SNAP_DIR_ROOT="${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
SNAP="${SNAP:-$(ls -d "${SNAP_DIR_ROOT}/"*/ 2>/dev/null | head -n1)}"
SNAP="${SNAP%/}"

nvidia-smi || true

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("capability", cap, "->", "B200/sm_100 OK" if cap == (10, 0) else "NOT sm_100")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print("matmul bf16 ok, sum", float(y.float().sum().item()) is not None)
else:
    print("NO CUDA: you are probably not inside a --gres=gpu:1 allocation")
PY

if [ -z "${SNAP}" ] || [ ! -d "${SNAP}" ]; then
    echo "No victim snapshot; skipping vLLM start test."; exit 0
fi

echo "=== try to start vLLM on port 8001 for ~90s ==="
python -m vllm.entrypoints.openai.api_server \
    --model "${SNAP}" --served-model-name llama-3.1-8b-instruct \
    --port 8001 --gpu-memory-utilization 0.85 --max-model-len 2048 \
    > "${LOG_DIR}/vllm_probe.log" 2>&1 &
VPID=$!
trap 'kill ${VPID} 2>/dev/null || true' EXIT
ok=0
for i in $(seq 1 45); do
    if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 "$VPID" 2>/dev/null; then echo "vLLM exited early:"; tail -n 120 "${LOG_DIR}/vllm_probe.log"; break; fi
    sleep 2
done
if [ "$ok" = 1 ]; then
    echo "VLLM_HEALTH_OK"
    curl -fsS http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"llama-3.1-8b-instruct","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}' \
      | head -c 400; echo
else
    echo "VLLM_HEALTH_FAILED"; tail -n 120 "${LOG_DIR}/vllm_probe.log"
fi
