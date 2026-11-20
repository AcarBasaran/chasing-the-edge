# First results: five samplers, seed 0 (2026-11-20)

**ONE SEED PER ARM. These are preliminary.** Seeds 1 and 2 and the Bandit are running (docs/budget.md).

Every arm uses the identical config (`configs/rl_base.yaml`): 100 GRPO steps, 1024 prompts × 6 rollouts per
step, strict process reward, 1 A40. Only `data.sampler` differs. Evaluation uses the paper's protocol: T=0.7,
n=128, strict eval criterion. Test sets: easy op2–10 (225), medium op11–14 (800), hard op15–20 (204).

## Test-set results

| arm | levels trained on | easy p@1 | medium p@1 / p@128 | **hard p@1 / p@128** | **hard newly solved** (of 204) |
|---|---|---|---|---|---|
| base model | – | 85.5 | 17.5 / 38.4 | 0 / 0 | – |
| released RL (paper) | 11–14 | 85.2 | 49.4 / 83.9 | 10.8 / 23.5 | 48 |
| fixed band (baseline) | 11–14 | 85.5 | 49.2 / 84.8 | 10.2 / 23.5 | 48 |
| uniform random | 2–20 | 86.6 | 47.3 / 82.9 | 16.3 / 37.7 | 77 |
| **threshold τ=0.1** | 11 → 17 | 84.0 | 46.9 / 84.8 | **22.9 / 46.1** | **94** |
| threshold τ=0.3 | 11 → 14 | 84.7 | 49.3 / 85.5 | 11.4 / 25.0 | 51 |
| threshold τ=0.5 | 11 only | 85.4 | 33.2 / 58.9 | 0 / 0 | 0 |

Hard problems newly solved, by level (34 per level):

| arm | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---|---|---|---|---|---|
| fixed band | 24 | 11 | 7 | 5 | 1 | 0 |
| uniform | 26 | 17 | 15 | 13 | 5 | 1 |
| threshold τ=0.1 | 29 | 20 | 18 | 16 | 9 | 2 |
| threshold τ=0.3 | 28 | 11 | 7 | 4 | 1 | 0 |
| threshold τ=0.5 | 0 | 0 | 0 | 0 | 0 | 0 |

No arm lost a problem the base model could solve on the easy or hard set (at most 5 on medium, τ=0.5).

## What the samplers did

`results/sampler_levels.png` shows, per arm, the levels it chose plotted over the measured success rate of every level.

| arm | path of the active level | mean p in batch | informative groups in batch |
|---|---|---|---|
| fixed band | 11–14 throughout | 0.44 | 0.35 |
| uniform | 2–20 throughout | 0.52 | 0.19 |
| threshold τ=0.1 | 11 →12 (step 1) →13 (3) →14 (7) →15 (15) →16 (22) →17 (51) | 0.18 | 0.25 |
| threshold τ=0.3 | 11 →12 (1) →13 (4) →14 (24), then stays | 0.35 | 0.34 |
| threshold τ=0.5 | 11 throughout | 0.57 | 0.40 |

## Reading (seed 0, to be confirmed by seeds 1–2)

1. **The τ question.** τ = 0.1 gives about 2× the fixed band on the hard set: pass@128 46.1 vs 23.5, and 94
   vs 48 problems newly solved. The arm with the most informative groups (τ=0.5, 40%) learns the least. The
   arm with fewer of them but the hardest problems (τ=0.1, 25%) learns the most. **Headroom beats gradient
   throughput here.**
2. **τ = 0.5 never moves.** The model's p at op11 sits at 0.55–0.63, so it never stays above the upper bound
   τ+δ = 0.6 for 2 steps in a row. A real result of the rule: with δ = 0.1 and patience 2, a
   target at the model's own success rate is a fixed point. Training only on op11 does not reach op15+ at all.
3. **τ = 0.3 ≈ fixed band.** It went to 14 by step 24 and stayed there. That is the paper's band, chosen by
   the sampler on its own.
4. **Uniform beats the fixed band** on hard (37.7 vs 23.5) despite 81% of its groups giving no gradient.
5. **Reliability cost.** τ=0.1 is slightly lower on medium pass@1 (46.9 vs 49.2) and easy pass@1 (84.0 vs
   85.5). medium pass@128 is equal.

## Caveat that must go in the report

τ=0.1 (op 15–17) and uniform (op 15–20) **train on problems from the hard test levels**. They are different
problems (pool vs held-out test, no overlap), but the same difficulty. For the fixed band and τ≤0.3, the hard
set is extrapolation; for these two arms it is partly in-distribution. The paper's Fig. 3 asks "does RL on
11–14 extrapolate?" Our question is "which training distribution gives the most new capability for a fixed
budget?" The answer "follow the edge into harder levels" is legitimate, but it should be stated as such,
not as better extrapolation. Possible control: evaluate on op 18–20 only, which τ=0.1 never trained on
(newly solved: τ=0.1 27, uniform 19, fixed 6 of 102).

## Cost

Training wall-clock on one A40 is 7.1–9.3 h per run, about 8 GPU-h on average. Evaluation is about 0.5 GPU-h
per run. The 18-run plan is about 155 GPU-h, within the approved 200.

Files: `results/method_comparison*.{csv,png}`, `results/newly_solved.csv`, `results/per_problem/*.csv`,
`results/sampler_levels.{csv,png}`, `$EDGE_SCRATCH/outputs/runs/<run>/` (train.log, sampler_log.jsonl,
edge_speed.csv, ckpt, hf/step100).
