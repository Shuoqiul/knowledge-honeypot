#!/usr/bin/env bash
# 07_download_sbert.sh -- pre-download the small CPU embedding models the attacker
# loop uses, so offline (HF_HUB_OFFLINE=1) jobs can load them from cache.
#
# RUN IN AN INTERACTIVE SESSION (no GPU). Public models, no gated token needed,
# but outbound network IS needed for this one script -> HF_HUB_OFFLINE=0.
#
#   bash env/07_download_sbert.sh
#
# Models:
#   - sentence-transformers/all-MiniLM-L6-v2   (trainStudent cache_answer / cluster)
#   - sentence-transformers/all-mpnet-base-v2  (correct/untils is_bad_answer)
set -uo pipefail
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/07_download_sbert_$(date +%Y%m%d_%H%M%S).log"
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
export HF_HUB_OFFLINE=0

for repo in \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "sentence-transformers/all-mpnet-base-v2" ; do
    echo "=== downloading ${repo} ==="
    hf download "${repo}" --cache-dir "${HF_HUB_CACHE}"
done

echo "=== verify offline load ==="
HF_HUB_OFFLINE=1 python - <<'PY'
from sentence_transformers import SentenceTransformer
for name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
    try:
        m = SentenceTransformer(name, device="cpu")
        v = m.encode("ok", normalize_embeddings=True)
        print(f"{name}: OK dim={len(v)}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}")
PY

echo "SBERT_DOWNLOAD_DONE"
