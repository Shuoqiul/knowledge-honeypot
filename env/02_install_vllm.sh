#!/usr/bin/env bash
# 02_install_vllm.sh -- optional: install vLLM (used to serve the local victim).
#
# RUN IN AN INTERACTIVE SESSION AFTER 01 succeeds (no GPU needed to install):
#     bash env/02_install_vllm.sh
#
# vLLM is NOT in requirements.txt. The local victim uses it (Supplement §4A). If
# this fails on Blackwell, tell me and we fall back to in-process transformers
# (Supplement §4B) -- that is a recorded deviation, not a blocker to hide.
set -uo pipefail
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_NAME=kh-torch28

source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${CONDA_ENVS_DIR}/${ENV_NAME}"

# Let vLLM pick a wheel matching the installed torch. Pin if it pulls a different
# torch and breaks the 2.8 pin above.
pip install "vllm>=0.10.0"

python - <<'PY'
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("vllm MISSING", type(e).__name__, e)
import torch
print("torch after vllm", torch.__version__)
PY
