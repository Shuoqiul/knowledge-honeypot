#!/usr/bin/env bash
# 00_blackwell_check.sbatch
# Purpose: confirm the conda env has a Blackwell-compatible PyTorch (sm_100) and
# list what is installed. Run this FIRST. It costs a few seconds of B200 time.
# Actor: give this to the PI, ask them to paste back the .out log.
#
#SBATCH --job-name=b200-check
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=b200_check_%j.out
#SBATCH --error=b200_check_%j.err

set -o pipefail
set -x

# NOTE: path taken from Actor Supplement.md §2. Verify with PI; stop if wrong.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /orange/sgao1/ZhengaoLi/Envs/torch2.7.0

python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
try:
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))  # expect (10, 0) for B200
except Exception as e:
    print("cuda error", type(e).__name__, e)
for mod in ["transformers", "vllm", "peft", "sentence_transformers", "jsonargparse"]:
    try:
        m = __import__(mod)
        print(f"{mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod} MISSING ({type(e).__name__})")
PY

nvidia-smi || true
