# Compute budget and project scale

Status 2026-11-20: **actuals below** (section "Actuals and projection"). Original status 2026-11-15: ESTIMATE from the 1-GPU smoke run. It is updated with the measured 4-GPU step time
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

## Actuals and projection (2026-11-20)

Measured on one A40 per run (`sacct`, GPU-hours = elapsed × GPUs):

| item | GPU-h |
|---|---|
| spent so far, all GPU jobs since Nov 14 | **104.7** |
| ... of which the two hung jobs 2026026/2026027 (24 h each, no training) | 48.0 |
| per training run, measured (seed 0): fixed 8.4, τ=0.3 8.5, uniform 7.2, τ=0.1 9.3, τ=0.5 7.1 | mean 8.1 |
| pass@128 eval per run (3 test sets) | 0.5 |

Projection to the end of the plan, counting what is still running or queued:

| | GPU-h |
|---|---|
| bandit × 3, seed 1 × 5, seed 2 × 5 (incl. evals), minus the 9.4 h already in the running jobs | 102 |
| fixed-band checkpoint re-evaluation (informative share over training) | 2.5 |
| **control: uniform op 11–17 × 3 seeds** (~8.8 h each, between fixed band and τ=0.1) + evals | **28** |
| **total incl. control** | **≈ 237** (≈ 189 without the 48 h lost to the hang) |
| **cut τ=0.5 seeds 1–2** (it never moves; 3 seeds buy nothing) | **−15** |
| **total after the cut** | **≈ 222** (≈ 174 without the hang) |

Against the experimental plan, the control fits after the cut (174 ≤ 200). Against everything the
allocation has actually burned, it does not (222 > 200), because of the 48 h hang. DECIDED 2026-11-20:
τ=0.5 cut to one seed. The next cut, if needed, follows CLAUDE.md order (seeds of the other arms).

Added 2026-11-20: bandit ablation m = 10, one seed, 50 steps, no eval: **≈ 4 GPU-h** → ≈ 226 total
(≈ 178 without the hang). Bandit seeds 0–2 ran 7.5–8.5 h each, as projected.

## Where the remaining budget goes (2026-11-20 13:20)

The 200 GPU-h ceiling is our planning number (Scale B). SLURM enforces no GPU-hour limit on this account
(QoS `ai`: 8 GPUs and 8 running jobs per user only).

| | GPU-h |
|---|---|
| spent so far (sacct, incl. the running jobs' elapsed time) | 165.6 |
| of which lost to the Ray hang | 48 |
| left in the 8 running jobs (τ=0.1 s1+s2 12.8, uniform s2 5.0, fixed band s2 5.7, τ=0.3 s2 6.0, control s0+s1 13.7, bandit m10 3.2) | ≈ 46 |
| control seed 2 (queued, not started) | ≈ 8.8 |
| test-set evals for the 8 remaining runs (≈ 0.5 each) | ≈ 4 |
| **projected total for Dec 3** | **≈ 225** (≈ 177 without the hang) |
| January plan (2 × 200-step runs ≈ 35, τ=0.5 δ/patience rerun ≈ 8) | ≈ 43 more, not in the 225 |

Cut options if 200 is firm, in order of least scientific cost (only what has not been spent yet is saved):
fixed band s2 (−6.2; seeds 0 and 1 are identical at 48/48 newly solved), τ=0.3 s2 (−6.5; not central),
bandit m10 (−3.4), control s2 (−9.3; leaves the range-vs-schedule control at 2 seeds). All four ≈ −25,
which gives ≈ 200. τ=0.1 s1/s2 and uniform s2 are not on the list.

## Cuts and seed policy (DECIDED 2026-11-20, user)

* Cut at 17:11: fixed band seed 2 and τ=0.3 seed 2. Both were at step 73 of 100, so the cut saved only their
  last ~27 steps plus evals (≈ 6 GPU-h together), not the ≈ 12.5 estimated at 13:20.
* Kept: τ=0.1 seeds 1–2, uniform seed 2, control (uniform op 11–17) seeds 0–2, bandit m = 10 ablation (done, 3.7 GPU-h).
* **Seed policy from here: 3 seeds only for τ=0.1, uniform and the op 11–17 control**, the arms the main claim
  rests on. 2 seeds for every other arm (fixed band 48 / 48, τ=0.3 51 / 46 newly solved). τ=0.5 stays at 1.
  No third seed for any other arm without asking.
* 200 GPU-h was the planning number, not a cap. January gets its own budget.

| (17:33) | GPU-h |
|---|---|
| spent (sacct, incl. running jobs' elapsed time) | 197.4 |
| left in the 6 running runs (τ=0.1 s1 ~1.5, s2 ~2.8, uniform s2 ~0.9, control s0 ~2.2, s1 ~2.8, s2 ~8.3) | ≈ 18.5 |
| test-set evals for those 6 runs | ≈ 3 |
| **projected total for Dec 3** | **≈ 219** (≈ 171 without the 48 h hang) |
