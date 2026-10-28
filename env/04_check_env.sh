#!/usr/bin/env bash
# 04_check_env.sh -- sanity-check the env: python/torch/vllm versions, imports,
# and (if a GPU is visible) the Blackwell capability.
#
# Most of this is CPU-only and can run in a normal interactive session:
#     bash env/04_check_env.sh
# To also confirm sm_100 on a B200, put the SAME command inside a GPU allocation:
#     srun --partition=hpg-b200 --gres=gpu:1 bash env/04_check_env.sh
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
export HF_HUB_OFFLINE=1

python - <<'PY'
import sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0), "(want (10, 0) for B200)")
for m in ["transformers","peft","sentence_transformers","jsonargparse","datasets","openai","sklearn","vllm"]:
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
PY

# confirm the victim weights are visible offline
SNAP_DIR="${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
echo "--- victim snapshot ---"
ls -d "${SNAP_DIR}/"* 2>/dev/null || echo "NO SNAPSHOT FOUND (run 03_download first)"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "(no nvidia-smi here; run under srun --gres=gpu:1 to check the GPU)"
