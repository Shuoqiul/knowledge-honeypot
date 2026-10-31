# E0 Slurm scripts (branch `repro/oct-2026`)

Only GPU runs use Slurm. Environment setup, the weight download, and E0.7 are
plain `.sh` in `env/` — see `env/README.md`.

**Prerequisites (done first, interactively, no GPU):**
`env/01_create_env.sh` → `env/02_install_vllm.sh` → `env/03_download_llama31_8b.sh`
→ `env/04_check_env.sh` (capability must be `(10, 0)`).

In both GPU sbatch files you MUST set **one** variable:

| Variable | Value |
| --- | --- |
| `REPO` | absolute path to this checkout on `repro/oct-2026` (e.g. `/orange/sgao1/sli/knowledge-honeypot`) |

`SNAP` is now **auto-resolved** from the HF cache (single snapshot dir
`.../snapshots/0e9e39f...`), so you do not need to paste the hash. Both files
hard-fail immediately if `REPO` is still `/path/to/...` or the snapshot is missing,
instead of burning a job slot. Both activate `/orange/sgao1/sli/.conda/envs/kh-torch28`.

## First, confirm the GPU side (isolates vLLM from the smoke run)

`cuda_available False` in `env/04` is expected: you ran it on a login node, which
has no GPU. To check the GPU, run the probe inside an allocation:

```bash
srun --partition=hpg-b200 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:15:00 \
     bash env/06_gpu_probe.sh
```

Expect `capability (10, 0)`, `matmul bf16 ok`, then `VLLM_HEALTH_OK` and a JSON
reply. If vLLM fails to start, its log is `env/logs/vllm_probe.log`.

## GPU jobs

1. `sbatch slurm/02_e0_smoke_tiny.sbatch`
   - Validates the E0 patches + local victim wiring with the README tiny attacker,
     B=20, `baseline` and `full`, real LLaMA-3.1-8B local victim. Cheap.
   - Acceptance: `SMOKE_DONE`, and a listing of `runs/e0_smoke/<jobid>/` and
     `results/smoke_*`. **Do not** start the gate array until this passes.

2. `sbatch slurm/03_e0_gate_array.sbatch` (the Nov-8 gate matrix)
   - Default `--array=0-3%1` runs only E0.1: baseline B200, full B200,
     full+mask B200, full+mask B500. Judge E0.1 first.
   - If clean, change the array line to `0-8%1` to add full+tagged B500 (4),
     E0.2 junk B200/B500 (5,6), E0.3 oracle B200/B500 (7,8).
   - `%1` keeps one task at a time (one B200); each task starts/kills its own vLLM.

## Cell → flag map

| Cell | Flags | Meaning |
| --- | --- | --- |
| baseline | `--defense_mode=baseline --mask_source=true` | no defense |
| full (tagged) | `--defense_mode=full` | as shipped |
| full (mask) | `+ --mask_source=true` | E0.1: detector sees `source=None` |
| junk | `+ --junk_control=true --mask_source=true` | E0.2: generic non-HKG substitute |
| oracle | `+ --detector_mode=oracle --mask_source=true` | E0.3: trigger from step 0 |

All cells: `--skip_final_train=true --with_coq=false --with_followup=false
--with_correction=false` (shipped ICL route).
