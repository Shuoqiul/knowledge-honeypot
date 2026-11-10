#!/usr/bin/env bash
# 08_download_datasets.sh -- pre-download eval datasets so offline (E0.6a) works.
#
# RUN IN AN INTERACTIVE SESSION (no GPU, needs network -> HF_HUB_OFFLINE=0).
#   bash env/08_download_datasets.sh
#
# Datasets used by evaluate_icl_local.py:
#   GBaker/MedQA-USMLE-4-options   (medqa, ~1273 test items)
#   openlifescienceai/medmcqa      (medmcqa; data/medmcqa_eval_cache.json has 50)
# FinQA / ConvFinQA already have bundled caches in data/.
set -uo pipefail
ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${ENV_DIR}/logs"; mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/08_download_datasets_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "logging to ${LOG_FILE}"; set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_PATH="${CONDA_ENVS_DIR}/kh-torch28"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${ENV_PATH}"

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=0

python - <<'PY'
from datasets import load_dataset
specs = [
    ("GBaker/MedQA-USMLE-4-options", "test"),
    ("openlifescienceai/medmcqa", "validation"),
]
for name, split in specs:
    try:
        ds = load_dataset(name, split=split)
        print(f"{name}/{split}: OK n={len(ds)}")
    except Exception as e:
        print(f"{name}/{split}: FAILED {type(e).__name__}: {e}")
PY

echo "DATASETS_DOWNLOAD_DONE"
