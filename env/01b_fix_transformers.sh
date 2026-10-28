#!/usr/bin/env bash
# 01b_fix_transformers.sh -- in-place fix if 01 left transformers/tokenizers wrong.
#
# Use this when torch is already 2.8.0+cu128 and vllm works, but
# transformers/tokenizers drifted (e.g. 4.57.6 / 0.22.2) because vllm bumped them.
# Rebuilds the transformers stack to the vllm-compatible pin. No full env rebuild.
#
#     bash env/01b_fix_transformers.sh
set -uo pipefail
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/01b_fix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "logging to ${LOG_FILE}"
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
ENV_PATH="${CONDA_ENVS_DIR}/kh-torch28"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
conda activate "${ENV_PATH}"

# vllm 0.10.2 needs transformers>=4.55.2, tokenizers>=0.21.1.
# Pin transformers==4.55.2; let pip resolve a compatible tokenizers & hub.
python -m pip install "transformers==4.55.2" "tokenizers>=0.21,<0.22"

python - <<'PY'
import importlib.metadata as md, sys
from packaging.version import Version
ok = True
def check(name, want, test):
    global ok
    try: got = md.version(name)
    except Exception: got = None
    good = got is not None and test(got)
    print(f"{name:22s} want {want:12s} got {got}  {'ok' if good else 'MISMATCH'}")
    ok = ok and good
check("torch", "2.8.0", lambda v: v=="2.8.0")
check("transformers", "4.55.2", lambda v: v=="4.55.2")
check("tokenizers", "0.21.x", lambda v: Version("0.21")<=Version(v)<Version("0.22"))
check("peft", "0.16.0", lambda v: v=="0.16.0")
check("sentence-transformers", "4.1.0", lambda v: v=="4.1.0")
check("numpy", "2.2.6", lambda v: v=="2.2.6")
try:
    import transformers, torch, vllm
    print("imports ok:", transformers.__version__, torch.__version__, vllm.__version__)
except Exception as e:
    print("IMPORT_FAILED", type(e).__name__, e); ok=False
print("VERIFY_OK" if ok else "VERIFY_FAILED")
sys.exit(0 if ok else 1)
PY
