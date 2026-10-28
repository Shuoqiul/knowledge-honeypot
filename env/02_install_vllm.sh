#!/usr/bin/env bash
# 02_install_vllm.sh -- verify vLLM, or install it if 01 did not.
#
# 01_create_env.sh already attempts `vllm>=0.10.0,<0.11.0`. Run this to confirm
# the import works, or to (re)install if it was skipped/failed. No GPU needed.
#     bash env/02_install_vllm.sh
set -uo pipefail
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_PATH="${CONDA_ENVS_DIR}/kh-torch28"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${ENV_PATH}"

if python -c "import vllm, sys; print('vllm', vllm.__version__)" 2>/dev/null; then
    echo "VLLM_ALREADY_OK"
else
    python -m pip install "vllm>=0.10.0,<0.11.0"
    python -c "import vllm; print('vllm', vllm.__version__)"
fi

# Make sure vLLM did not silently bump torch off 2.8.
python -c "import importlib.metadata as m; print('torch', m.version('torch'))"
