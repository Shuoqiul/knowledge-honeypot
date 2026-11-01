#!/usr/bin/env bash
# 06_gpu_probe.sh -- GPU + vLLM sanity check. MUST run inside a GPU allocation:
#
#   srun --partition=hpg-b200 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:20:00 \
#        bash env/06_gpu_probe.sh
#
# Steps: (1) torch CUDA/capability; (2) vLLM engine smoke via the Python API
# (fails fast and prints the real traceback); (3) if that works, start the
# OpenAI server and hit /health + one chat completion.
set -uo pipefail
ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${ENV_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/06_gpu_probe_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "logging to ${LOG_FILE}"

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_PATH="${CONDA_ENVS_DIR}/kh-torch28"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${ENV_PATH}"

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SNAP_ROOT="${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
SNAP="${SNAP:-$(ls -d "${SNAP_ROOT}/"*/ 2>/dev/null | head -n1)}"
SNAP="${SNAP%/}"
echo "SNAP=${SNAP}"

echo "================ 1) torch / GPU ================"
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("capability", cap, "->", "sm_100 OK" if cap == (10, 0) else "NOT sm_100")
PY

echo "================ 1b) plain transformers generate (weights sanity) ================"
# If this fails, the problem is the weights/transformers, not vLLM.
python - <<PY
import sys, traceback
snap = "${SNAP}"
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda:0")
    ids = tok("Reply with OK", return_tensors="pt").to("cuda:0")
    out = model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    print("HF_GENERATE_OK:", tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    del model; torch.cuda.empty_cache()
except Exception:
    print("HF_GENERATE_FAILED"); traceback.print_exc()
PY

echo "================ 2) vLLM engine import + LLM smoke ================"
# This is where the real failure shows up. Print versions first.
python - <<'PY'
import importlib.metadata as m
for p in ["vllm","torch","transformers","numpy","xformers","flashinfer","triton"]:
    try: print(p, m.version(p))
    except Exception: print(p, "MISSING")
import vllm
print("vllm import ok", vllm.__version__)
PY

if [ -z "${SNAP}" ] || [ ! -d "${SNAP}" ]; then
    echo "No snapshot; skipping vLLM engine test."; exit 0
fi

echo "--- build vllm.LLM and generate one prompt (real traceback below if it fails) ---"
python - <<PY
import sys, traceback
snap = "${SNAP}"
try:
    from vllm import LLM, SamplingParams
    llm = LLM(model=snap, gpu_memory_utilization=0.60, max_model_len=2048,
              enforce_eager=True, dtype="bfloat16")
    out = llm.generate(["Reply with OK"], SamplingParams(max_tokens=8, temperature=0.0))
    print("VLLM_ENGINE_OK:", out[0].outputs[0].text.strip())
except Exception:
    print("VLLM_ENGINE_FAILED"); traceback.print_exc(); sys.exit(3)
PY
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "ENGINE_SMOKE_FAILED rc=$rc -- server test skipped"; exit "$rc"
fi

echo "================ 3) vLLM OpenAI server ================"
VLLM_LOG="${LOG_DIR}/vllm_server_probe.log"
python -m vllm.entrypoints.openai.api_server \
    --model "${SNAP}" --served-model-name llama-3.1-8b-instruct \
    --port 8001 --gpu-memory-utilization 0.60 --max-model-len 2048 \
    > "${VLLM_LOG}" 2>&1 &
VPID=$!
trap 'kill ${VPID} 2>/dev/null || true' EXIT
ok=0
# First start can compile CUDA graphs / jit kernels; allow up to ~5 minutes.
for i in $(seq 1 150); do
    if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 "$VPID" 2>/dev/null; then echo "SERVER EXITED EARLY"; break; fi
    sleep 2
done
if [ "$ok" = 1 ]; then
    echo "VLLM_HEALTH_OK"
    curl -fsS http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"llama-3.1-8b-instruct","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}' \
      | head -c 400; echo
else
    echo "VLLM_HEALTH_FAILED -- server log follows:"
    cat "${VLLM_LOG}"
fi
