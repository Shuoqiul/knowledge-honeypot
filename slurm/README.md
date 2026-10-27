# E0 Slurm scripts (branch `repro/oct-2026`)

Only GPU runs use Slurm. Environment setup, the weight download, and E0.7 are
plain `.sh` in `env/` — see `env/README.md`.

**Prerequisites (done first, interactively, no GPU):**
`env/01_create_env.sh` → `env/02_install_vllm.sh` → `env/03_download_llama31_8b.sh`
→ `env/04_check_env.sh` (capability must be `(10, 0)`).

From `03_download` copy the printed **snapshot hash**, then in both GPU sbatch
files set:

| Variable | Value |
| --- | --- |
| `REPO` | absolute path to this checkout on `repro/oct-2026` |
| `SNAP` | `${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<HASH>` |

Both files activate `/orange/sgao1/sli/.conda/envs/kh-torch28`.

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
