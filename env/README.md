# Environment setup (interactive `.sh`, no sbatch, no GPU)

These are plain shell scripts you run in your own interactive session. They do
**not** use Slurm and do **not** need a GPU (except the optional capability check,
which you run under `srun` if you want to confirm sm_100).

All scripts expect the cluster paths you gave:

- conda base: `/orange/sgao1/sli/conda_cache/anaconda3`
- envs:       `/orange/sgao1/sli/.conda/envs`
- pkgs:       `/orange/sgao1/sli/.conda/pkgs`
- new env:    `kh-torch28` (Python 3.10, torch 2.8)

## Order

```bash
# 1) create env + install requirements (torch 2.8)
bash env/01_create_env.sh

# 2) (optional) install vLLM to serve the local victim
bash env/02_install_vllm.sh

# 3) download the gated weights ONCE (needs HF_TOKEN, see below)
bash env/03_download_llama31_8b.sh

# 4) sanity check (versions + snapshot present; add srun for capability)
bash env/04_check_env.sh
#   or, to also print capability (10,0):
#   srun --partition=hpg-b200 --gres=gpu:1 bash env/04_check_env.sh

# 5) E0.7 benign false-trigger (CPU, any time)
bash env/05_e0_7_benign.sh
```

## torch version and the pin

`requirements.txt` pins `torch==2.8.0.dev20250626+cu128` (a **nightly**) and
`pytorch-triton==3.3.1+git...`. A plain `pip install -r` can't resolve those.
`01_create_env.sh` first tries the exact pins with the PyTorch nightly index; if
that fails it falls back to **stable torch 2.8.0+cu128** (Blackwell/sm_100
capable) and keeps every other package at its pinned version. The fallback is a
recorded deviation; the log says `EXACT_PINS_FAILED` if it happened — send it to me.

## HF_TOKEN (do not hardcode in the script)

Choose one:

- **A. Session export** (simplest):
  ```bash
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
  bash env/03_download_llama31_8b.sh
  ```
- **B. Token file** (persists across sessions, still not in git):
  ```bash
  printf '%s\n' 'hf_xxxxxxxxxxxxxxxx' > ~/.hf_token
  chmod 600 ~/.hf_token
  bash env/03_download_llama31_8b.sh   # script reads ~/.hf_token if HF_TOKEN unset
  ```

Either way the token is never echoed. `~/.hf_token` is covered by `.gitignore`'s
`.env`/`*.key` patterns; if you name it differently, add it to `.gitignore`.
The HF account must have accepted Meta's Llama-3.1 license, or the download 403s.

## After 03 succeeds

Copy the printed snapshot hash and put it in `SNAP` in the two GPU sbatch files
(`slurm/02_e0_smoke_tiny.sbatch`, `slurm/03_e0_gate_array.sbatch`), and set `REPO`.
Those two are the only things that still go through `sbatch`.
