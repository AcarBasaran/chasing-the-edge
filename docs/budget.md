# Compute budget and project scale

Status 2026-11-15: **ESTIMATE** from the 1-GPU smoke run. It is updated with the measured 4-GPU step time
once job 2024783 (5-step 4-GPU validation run) finishes. Recompute with
`python eval/budget.py --sec-per-step-4gpu <measured>`.

## Inputs

| input | value | source |
|---|---|---|
| 1-GPU step, batch 256 × 6 rollouts | 56.6 s (gen 21, log-probs 12.5, update 22) | Gate-2 smoke, `results/gate2/fixed_band_smoke_metrics.csv` |
| paper batch | 1024 prompts × 6 rollouts | upstream `rl-200steps/*.yaml`, App. A.3 |
| paper steps | ~195 (1 epoch of 4 × 50k); text says "two epochs"; experiment named "100steps" | see docs/gate3.md |
| response-length growth | ×1.2 (assumed) | base mean 306 tokens; harder levels / RL run longer |
| 4-GPU efficiency | 0.8 (assumed, **to be measured**) | |
| tracking sweep | 19 levels × 32 × 8 samples every 10 steps (~3% overhead) | `configs/rl_base.yaml` |
| pass@128 eval per checkpoint | easy + medium + hard = 1,229 problems ≈ 0.5 GPU-h | base model on medium-200: 1014 s for 800 problems |

Per run: **100 steps ≈ 10.7 GPU-h (≈ 2.5 h on 4 GPUs); 195 steps ≈ 19.9 GPU-h (≈ 4.8 h).**

## Scales

| scale | runs | GPU-h | wall-clock, one 4-GPU job at a time, 100% / 50% of the time running |
|---|---|---|---|
| A: 6 arms × 3 seeds × 195 steps (paper) | 18 | 358 | 3.7 d / 7.5 d |
| **B: 6 arms × 3 seeds × 100 steps** | 18 | 192 (+9 to extend the baseline to 195 for Gate 3) | 2.1 d / 4.2 d |
| C: 6 arms × 2 seeds × 100 steps | 12 | 137 | 1.4 d / 2.9 d |
| D: 6 arms × 1 seed × 100 steps | 6 | 73 | 0.8 d / 1.5 d |
| E: 2 arms × 3 seeds × 195 steps | 6 | 119 | 1.2 d / 2.5 d |

## Queue reality (measured today)

* QoS `ai` per user: **8 running jobs, 8 GPUs, 350 GB memory**. CPU-only jobs from other projects
  (`restock-*`) share the 8 job slots. GPUs: up to two 4-GPU jobs at once.
* All 80 A40/A6000 GPUs were allocated all morning. 1-GPU jobs started within minutes overnight. A 4-GPU
  job needs 4 free GPUs on *one* node: the first one waited about 1.5 h (scheduler estimate).
* Assume we are running about half the time (queue waits, failed runs, debugging, evaluation): two weeks ≈ 7
  effective days of one 4-GPU job ≈ **670 GPU-h**. With two 4-GPU jobs in parallel, about double.

## Proposal

**Scale B: all six arms, 3 seeds, 100 steps each (half the paper's ~195).** About 200 GPU-h, roughly
4 days of wall-clock at a 50% duty cycle with one 4-GPU job at a time. That leaves room for reruns,
the Gate-3 extension of the baseline to 195 steps, and longer runs for the best arm if time allows.

* Order: fixed band s0 → (Gate 3 check) → threshold τ=0.3 s0 (Gate 4, Dec 1) → uniform s0, τ=0.1 s0,
  τ=0.5 s0, bandit s0 (after m is set from the measured edge speed) → seeds 1 and 2 for all arms.
* 100 steps is also what the paper uses for its own compute-matched RL runs (Sec. 5). LR is constant, so
  the baseline's step-100 checkpoint is an exact 100-step run, and extending it to 195 steps for Gate 3 is cheap.
* What to cut first if the queue is worse than assumed: seed 2, then τ=0.1 (CLAUDE.md order: seeds, test-set
  size, τ sweep, steps).
