# E0 Slurm scripts (branch `repro/oct-2026`)

Everything here is written for the PI to submit. The actor (me) is offline and
cannot see the cluster. Submit in this order; hand back the `.out`/`.err` logs.

**Before submitting anything**, the PI must fill in two placeholders in the
`*.sbatch` files (they are `REPO` and `SNAP`):

| Variable | Value | Where it comes from |
| --- | --- | --- |
| `REPO` | absolute path to this checkout on `repro/oct-2026` | PI knows |
| `SNAP` | `${HF_HUB_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<HASH>` | printed by `01_download` |

Also **verify** these against the reference `srun_group.sh` (not in this repo),
per Actor Supplement §2. If any differ, stop and tell me:

- partition `hpg-b200`
- conda env `/orange/sgao1/ZhengaoLi/Envs/torch2.7.0`
- `HF_HOME=/orange/sgao1/sli/hf_cache`, hub cache `${HF_HOME}/hub`

## Batch 1 (what I need first)

1. `sbatch slurm/00_blackwell_check.sbatch`
   - Acceptance: `capability (10, 0)` and `torch`/`vllm` import. If not (10,0),
     that's a blocker; send me the log.
2. `export HF_TOKEN=...` in the submitting shell (its HF account must have
   accepted Meta's Llama-3.1 license), then
   `sbatch slurm/01_download_llama31_8b.sbatch`
   - Acceptance: `DOWNLOAD_OK` and a snapshot dir listed. Send me the snapshot
     hash so I can fill `SNAP`.
3. `sbatch slurm/02_e0_smoke_tiny.sbatch` after setting `REPO` and `SNAP`
   - Acceptance: `SMOKE_DONE`, and a listing of `runs/e0_smoke/<jobid>/` and
     `results/smoke_*`. This validates the patches. Do NOT run the gate array
     until this passes.
4. `sbatch slurm/04_e0_7_benign_cpu.sbatch` (CPU, independent; can go in parallel)

## Batch 2 (the gate matrix, after 3 passes)

`sbatch slurm/03_e0_gate_array.sbatch`

The default `--array=0-3%1` runs only E0.1 (baseline B200, full B200, full+mask
B200, full+mask B500) because E0.1 is gate-critical and should be judged first.
After those come back clean, edit the array line to `0-8%1` to add:

- 4 full+tagged B500 (pairs with 3)
- 5,6 E0.2 junk B200/B500
- 7,8 E0.3 oracle B200/B500

(Actor note: on a single B200 runs are serialized; `%1` keeps one task at a time.
`%2` is possible but risks two vLLM servers each requesting 0.35 of the 180 GB,
which is fine memory-wise but doubles wall time per cell; keep `%1` unless the PI
confirms quota and wants throughput.)

## What each cell changes vs. the shipped code

| Cell | Flag(s) | Meaning |
| --- | --- | --- |
| baseline | `--defense_mode=baseline --mask_source=true` | no defense; mask is irrelevant here |
| full (tagged) | `--defense_mode=full` | as shipped (attackers still pass their `source`) |
| full (mask) | `+ --mask_source=true` | E0.1: detector sees `source=None` |
| junk | `+ --junk_control=true --mask_source=true` | E0.2: substitute generic non-HKG text |
| oracle | `+ --detector_mode=oracle --mask_source=true` | E0.3: trigger from step 0 |

All runs use `--skip_final_train=true --with_coq=false --with_followup=false
--with_correction=false`, matching the shipped ICL route.
