# Environment setup (interactive `.sh`, no sbatch, no GPU)

Plain shell scripts for your own interactive session. None need Slurm or a GPU
(except the optional capability check, which you run under `srun` if you want sm_100).

Cluster paths:

- conda base: `/orange/sgao1/sli/conda_cache/anaconda3`
- envs:       `/orange/sgao1/sli/.conda/envs`
- pkgs:       `/orange/sgao1/sli/.conda/pkgs`
- new env:    `kh-torch28` (Python 3.10, torch 2.8.0+cu128)

## Order (run all 5)

```bash
bash env/01_create_env.sh      # remove any dirty env, recreate, install deps
bash env/02_install_vllm.sh    # confirm/install vLLM (01 already tries)
bash env/03_download_llama31_8b.sh   # skips if snapshot already present
bash env/04_check_env.sh       # verify versions + imports + snapshot
bash env/05_e0_7_benign.sh     # E0.7 (CPU), any time
```

`04` is still the verification step — and it is now strict: `01` also ends with a
hard verify that fails loudly if torch/transformers/tokenizers/peft/numpy do not
match the pinned versions. If `04` reports `VERIFY_FAILED` (in `01`) or a mismatch,
send me the output and stop; do not proceed.

## How the deps are installed (why not `pip install -r`)

`requirements.txt` pins a PyTorch **nightly** (`2.8.0.dev20250626+cu128`) that
PyTorch has deleted, plus cu128 `nvidia-*` wheels and `pytorch-triton` that clash
with the torch wheel on python 3.10. A plain `-r` install aborts half-way (that is
how an earlier attempt ended on torch 2.13 / transformers 5.17).

`01` instead:
1. removes any existing `kh-torch28` (guarantees clean), creates python 3.10;
2. installs a single **stable `torch==2.8.0` + `torchvision==0.23.0` +
   `torchaudio==2.8.0` from the cu128 index** (Blackwell/sm_100);
3. installs every **other** pin from `requirements.txt`, with `torch*`, `triton`,
   `pytorch-triton`, and all `nvidia-*` lines filtered out (torch brings its own);
4. installs `vllm>=0.10.0,<0.11.0`;
5. hard-verifies: `transformers==4.53.1`, `tokenizers==0.21.2`, `peft==0.16.0`,
   `sentence-transformers==4.1.0`, `numpy==2.2.6`, `torch==2.8.0`.

**Recorded deviation:** nightly→stable `torch 2.8.0+cu128`. Functionally equivalent
for our use and Blackwell-capable.

## HF_TOKEN (do not hardcode)

- A: `export HF_TOKEN=hf_xxx` then run `03`; or
- B: `printf '%s\n' 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token`; `03` reads it.

Never echoed. The HF account must have accepted Meta's Llama-3.1 license.
The download uses `hf download` (the current CLI; `huggingface-cli` is deprecated).

## You already have the weights

A snapshot exists at
`.../models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`,
so `03` will print `SNAPSHOT_ALREADY_PRESENT` and skip. Use that hash for `SNAP` in
the two GPU sbatch files (`slurm/02_e0_smoke_tiny.sbatch`, `slurm/03_e0_gate_array.sbatch`),
and set `REPO`. Those two are the only things that go through `sbatch`.
