#!/usr/bin/env bash
# 07_download_models.sh -- pre-download EVERY model an E0 job loads, so the
# offline (HF_HUB_OFFLINE=1) jobs read from the shared cache and never hit the net.
#
# RUN IN AN INTERACTIVE SESSION (no GPU). This is the one script that needs
# outbound network -> HF_HUB_OFFLINE=0. SBERT models are public; the Llama repos
# are gated (HF_TOKEN + accepted Meta license required).
#
#   bash env/07_download_models.sh
#
# Models used:
#   victim (vLLM)        : meta-llama/Llama-3.1-8B-Instruct        (already cached)
#   smoke attacker       : sshleifer/tiny-gpt2                     (public, tiny)
#   E0 real attacker     : meta-llama/Llama-3.1-8B-Instruct         (same weights as victim;
#                          see slurm/03: STUDENT_BASE uses this cached id, not the
#                          'Meta-Llama' alias, so offline resolves from cache)
#   CPU embedder         : sentence-transformers/all-MiniLM-L6-v2  (attacker loop)
#                        : sentence-transformers/all-mpnet-base-v2 (is_bad_answer)
set -uo pipefail
ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${ENV_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/07_download_models_$(date +%Y%m%d_%H%M%S).log"
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

# Optional token for the gated Llama repos (SBERT + tiny-gpt2 do not need it).
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
    HF_TOKEN="$(sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^HF_TOKEN=//' "$HOME/.hf_token" | head -n1)"
    export HF_TOKEN
fi

for repo in \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "sentence-transformers/all-mpnet-base-v2" \
    "sshleifer/tiny-gpt2" \
    "meta-llama/Llama-3.1-8B-Instruct" ; do
    echo "=== downloading ${repo} ==="
    hf download "${repo}" --cache-dir "${HF_HUB_CACHE}" || \
        echo "WARN: ${repo} failed (gated? need HF_TOKEN + Meta license). Continuing."
done

echo "=== verify offline load (HF_HUB_OFFLINE=1) ==="
HF_HUB_OFFLINE=1 python - <<'PY'
from sentence_transformers import SentenceTransformer
for name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
    try:
        m = SentenceTransformer(name, device="cpu")
        v = m.encode("ok", normalize_embeddings=True)
        print(f"{name}: OK dim={len(v)}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}")

from transformers import AutoConfig, AutoTokenizer
for name in ["sshleifer/tiny-gpt2", "meta-llama/Llama-3.1-8B-Instruct"]:
    try:
        AutoConfig.from_pretrained(name)
        AutoTokenizer.from_pretrained(name)
        print(f"{name}: CONFIG+TOKENIZER OK")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}")
PY

echo "MODELS_DOWNLOAD_DONE"
