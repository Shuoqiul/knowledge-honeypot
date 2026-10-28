#!/usr/bin/env bash
# 01_create_env.sh  --  create the kh-torch28 conda env and install the repo deps.
#
# RUN IN AN INTERACTIVE SESSION (no GPU needed):
#     bash env/01_create_env.sh
#
# Cluster paths (given by PI):
#   conda base : /orange/sgao1/sli/conda_cache/anaconda3
#   envs dir   : /orange/sgao1/sli/.conda/envs
#   pkgs dir   : /orange/sgao1/sli/.conda/pkgs   (sibling of envs, under .conda)
#
# Why we do NOT run `pip install -r requirements.txt`:
#   - It pins a PyTorch NIGHTLY (2.8.0.dev20250626+cu128) that PyTorch has already
#     deleted; it cannot resolve.
#   - It also pins the cu128 `nvidia-*` wheels and `pytorch-triton`, which fight
#     with the torch wheel and with python 3.10 and abort half-way (that is how the
#     previous env ended up on torch 2.13 / transformers 5.17).
# So we: (1) install a single stable torch 2.8.0+cu128 base, (2) install every
# other pin from requirements.txt EXCEPT torch/vision/audio/triton/nvidia-*,
# (3) hard-verify the pinned versions. This is a recorded deviation
# (nightly->stable 2.8.0); torch 2.8.0 supports Blackwell/sm_100.
set -uo pipefail

# Persist a full log next to the repo so it cannot be lost in scrollback.
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/01_create_env_$(date +%Y%m%d_%H%M%S).log"
# send everything (stdout+stderr, including set -x) to the log and the terminal
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "logging to ${LOG_FILE}"
set -x

CONDA_BASE=/orange/sgao1/sli/conda_cache/anaconda3
CONDA_ENVS_DIR=/orange/sgao1/sli/.conda/envs
CONDA_PKGS_DIR=/orange/sgao1/sli/.conda/pkgs
ENV_NAME=kh-torch28
ENV_PATH="${CONDA_ENVS_DIR}/${ENV_NAME}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
echo "repo=${REPO}"

# 0) Nuke a dirty env from an earlier failed attempt, then recreate clean.
if [ -d "${ENV_PATH}" ]; then
    echo "WARNING: ${ENV_PATH} exists; removing it to guarantee a clean install."
    conda deactivate 2>/dev/null || true
    conda env remove -y -p "${ENV_PATH}"
fi
conda create -y -p "${ENV_PATH}" python=3.10
conda activate "${ENV_PATH}"
which python; python --version

export PIP_DISABLE_PIP_VERSION_CHECK=1
python -m pip install --upgrade pip setuptools wheel

# 1) torch base first (Blackwell/sm_100), from the cu128 index.
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2) derive a filtered requirements: drop torch/vision/audio/triton and every
#    nvidia-* wheel (torch brings the ones it needs).
#    Also drop ace_tools (OpenAI-internal stub, not on public PyPI, unused here)
#    and transformers/tokenizers: vLLM requires transformers>=4.55.2, which is
#    incompatible with the pinned 4.53.1. We install transformers explicitly
#    after vLLM (step 3) at a version vLLM supports. Recorded deviation.
FILTERED=/tmp/kh_req_filtered.txt
grep -viE '^(torch|torchvision|torchaudio|pytorch-triton|triton|nvidia-[a-z0-9-]+|ace_tools|transformers|tokenizers|huggingface-hub)==' \
    "${REPO}/requirements.txt" > "${FILTERED}"
echo "=== filtered requirements (torch/nvidia/triton/ace_tools/transformers/tokenizers/huggingface-hub removed) ==="
cat "${FILTERED}"

python -m pip install -r "${FILTERED}"

# 3) vLLM, then an explicit transformers/tokenizers that vLLM accepts.
#    vllm 0.10.2 requires transformers>=4.55.2; 4.55.2 is the closest to the
#    pinned 4.53.1. Install transformers AFTER vllm so pip keeps this pin.
python -m pip install "vllm>=0.10.0,<0.11.0" || {
    echo "VLLM_INSTALL_FAILED (fallback: in-process transformers, Supplement 4B)"
}
python -m pip install "transformers==4.55.2"

# 4) hard verification: fail loudly if anything drifted.
python - <<'PY'
import importlib.metadata as md, sys
from packaging.version import Version
# Strict pins (must match exactly):
want = {"torch":"2.8.0","peft":"0.16.0","sentence-transformers":"4.1.0",
        "numpy":"2.2.6","transformers":"4.55.2"}
# tokenizers: whatever transformers==4.55.2 requires (>=0.21,<0.22).
bad = []
print("python", sys.version.split()[0])
for name, ver in want.items():
    try:
        got = md.version(name)
    except Exception:
        got = None
    ok = (got == ver) if got else False
    print(f"{name:22s} want {ver:10s} got {got}  {'ok' if ok else 'MISMATCH'}")
    if not ok: bad.append(name)
try:
    tok = md.version("tokenizers")
    ok_tok = Version("0.21") <= Version(tok) < Version("0.22")
    print(f"{'tokenizers':22s} want 0.21.x     got {tok}  {'ok' if ok_tok else 'MISMATCH'}")
    if not ok_tok: bad.append("tokenizers")
except Exception:
    bad.append("tokenizers")
import torch
print("torch.cuda.is_available", torch.cuda.is_available())
try:
    import transformers, vllm
    print("transformers", transformers.__version__, "vllm", vllm.__version__)
except Exception as e:
    print("IMPORT_FAILED", type(e).__name__, e); bad.append("import")
if bad:
    print("VERIFY_FAILED:", ",".join(bad)); sys.exit(1)
print("VERIFY_OK")
PY

echo "ENV_READY ${ENV_PATH}"
