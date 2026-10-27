#!/usr/bin/env bash
# 01_create_env.sh  --  create the torch2.8 conda env and install requirements.
#
# RUN THIS IN AN INTERACTIVE SESSION (no GPU needed, no sbatch):
#     bash env/01_create_env.sh
#
# Cluster paths (given by PI; verify they exist if this fails):
#   conda base : /orange/sgao1/sli/conda_cache/anaconda3
#   envs dir   : /orange/sgao1/sli/.conda/envs
#   pkgs dir   : /orange/sgao1/sli/.conda/pkgs   (sibling of envs, under .conda)
#
# The repo's requirements.txt pins a PyTorch NIGHTLY (2.8.0.dev...+cu128) that a
# plain "pip install -r" cannot resolve without the nightly index. We try the
# exact pins first; if that fails we fall back to the stable 2.8.0+cu128 wheels
# (Blackwell/sm_100 capable) and record the deviation. Everything else stays at
# the pinned versions.
set -uo pipefail
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
CONDA_PKGS_DIR=/orange/sgao1/sli/.conda/pkgs
ENV_NAME=kh-torch28

source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Put envs and pkgs where the PI asked (not in $HOME).
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
echo "repo=${REPO}"

# 1) create env with the Python the repo targets (>=3.10)
conda create -y -p "${CONDA_ENVS_DIR}/${ENV_NAME}" python=3.10

conda activate "${CONDA_ENVS_DIR}/${ENV_NAME}"
which python
python --version

PY="${CONDA_ENVS_DIR}/${ENV_NAME}/bin/python"
pip install --upgrade pip setuptools wheel

# 2) try the exact pinned requirements first (nightly torch needs the nightly index)
echo "=== attempt 1: exact requirements.txt (with nightly torch index) ==="
pip install -r "${REPO}/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    > /tmp/kh_pip_exact.log 2>&1
if [ $? -eq 0 ]; then
    echo "EXACT_PINS_OK"
else
    echo "EXACT_PINS_FAILED -> stable-torch fallback (recorded deviation)"
    tail -n 40 /tmp/kh_pip_exact.log

    # 3) fallback: everything except torch/audio/vision/triton at pinned versions,
    #    then stable torch 2.8.0+cu128 which supports B200 (sm_100).
    grep -viE '^(torch|torchaudio|torchvision|pytorch-triton|triton)==' \
        "${REPO}/requirements.txt" > /tmp/kh_req_notorch.txt
    pip install -r /tmp/kh_req_notorch.txt

    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

# 4) verify imports + Blackwell capability (capability check only meaningful with GPU)
python - <<'PY'
import sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
for m in ["transformers","peft","sentence_transformers","jsonargparse","datasets","openai","sklearn"]:
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__, e)
PY

echo "ENV_READY ${CONDA_ENVS_DIR}/${ENV_NAME}"
echo "NOTE: vLLM is intentionally NOT in requirements.txt. If E0 runs need it,"
echo "      install separately (see env/README.md)."
