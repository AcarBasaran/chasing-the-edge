# First results: five samplers, seed 0 (2026-11-20)

**ONE SEED PER ARM. These are preliminary.** Seeds 1 and 2 and the Bandit are running (docs/budget.md).

Every arm uses the identical config (`configs/rl_base.yaml`): 100 GRPO steps, 1024 prompts × 6 rollouts per
step, strict process reward, 1 A40. Only `data.sampler` differs. Evaluation uses the paper's protocol: T=0.7,
n=128, strict eval criterion. Test sets: easy op2–10 (225), medium op11–14 (800), hard op15–20 (204).

## 1. How much of the gain is "harder problems at all", and how much is adaptive timing

| arm | levels trained on | hard pass@128 | gain over fixed band | hard newly solved (of 204) |
|---|---|---|---|---|
| fixed band (baseline) | 11–14 | 23.5 | – | 48 |
| uniform random | 2–20 | 37.7 | +14.2 | 77 |
| threshold τ=0.1 | 11 → 17 | 46.1 | +22.6 | 94 |

Most of the improvement over the fixed band comes from **training on harder problems at all**. Uniform random,
which spends the same budget evenly over every level, already gets **63% of τ=0.1's gain** (+14.2 of +22.6pp
pass@128; 29 of 46 extra newly-solved problems). **Adaptive timing is the smaller remaining margin**: +8.4pp
pass@128 and 17 more problems for τ=0.1 over uniform. Section 2 shows where that margin comes from, and why it
is the more interesting part.

Full table:

| arm | levels trained on | easy p@1 | medium p@1 / p@128 | hard p@1 / p@128 | hard newly solved |
|---|---|---|---|---|---|
| base model | – | 85.5 | 17.5 / 38.4 | 0 / 0 | – |
| released RL (paper) | 11–14 | 85.2 | 49.4 / 83.9 | 10.8 / 23.5 | 48 |
| fixed band | 11–14 | 85.5 | 49.2 / 84.8 | 10.2 / 23.5 | 48 |
| uniform random | 2–20 | 86.6 | 47.3 / 82.9 | 16.3 / 37.7 | 77 |
| threshold τ=0.1 | 11 → 17 | 84.0 | 46.9 / 84.8 | 22.9 / 46.1 | 94 |
| threshold τ=0.3 | 11 → 14 | 84.7 | 49.3 / 85.5 | 11.4 / 25.0 | 51 |
| threshold τ=0.5 | 11 only | 85.4 | 33.2 / 58.9 | 0 / 0 | 0 |

## 2. Headline: op18–20, the levels τ=0.1 never trained on

τ=0.1 climbed to op17 and stopped, so **op18–20 are untrained for it**. Uniform random trained on all of 2–20,
**including 18–20**. If τ=0.1 still wins there, the margin is extrapolation from following the edge, not exposure.

**Figure: `results/hard_by_level.png`** (data: `results/hard_by_level.csv`).

| arm | trained at op18–20? (groups / with nonzero gradient) | p@1 | **p@128** | **newly solved** (of 102) |
|---|---|---|---|---|
| **threshold τ=0.1** | **no** (0) | **10.6** | **26.5** | **27** |
| uniform random | yes (16,005 / 487) | 7.1 | 18.6 | 19 |
| fixed band | no (0) | 1.7 | 5.9 | 6 |
| released RL | no | 1.4 | 4.9 | 5 |
| threshold τ=0.3 | no (0) | 1.8 | 4.9 | 5 |
| threshold τ=0.5 | no (0) | 0 | 0 | 0 |

The base model solved 0 of these 102 problems in 128 tries.

* **τ=0.1 beats uniform on levels only uniform trained on.** Newly solved 27 vs 19. Paired by problem, τ=0.1
  solves 9 that uniform does not, and uniform solves 1 that τ=0.1 does not (exact McNemar p = 0.02;
  `results/hard_by_level_mcnemar.csv`). The test is within one seed; seed-to-seed variance is not in it yet.
* **Why uniform's exposure to 18–20 bought little.** Uniform drew 16% of its prompts from 18–20. At the
  model's skill those are almost always all-wrong groups: only 487 of 16,005 groups (3%) had a nonzero gradient.
  τ=0.1 spent its budget at op15–17 when those were at the edge (p ≈ 0.1–0.3). 25% of its groups were
  informative, and that skill carried two to three levels further.
* Per level (pass@128, τ=0.1 / uniform / fixed): op15 85 / 76 / 71, op16 59 / 50 / 32, op17 53 / 44 / 21,
  op18 47 / 38 / 15, op19 26 / 15 / 3, op20 6 / 3 / 0.
* Against the fixed band on the same untrained levels: 27 vs 6 newly solved (22 τ=0.1-only, 1 fixed-only).

## 3. The τ sweep

| arm | path of the active level | mean p in batch | informative groups | hard newly solved |
|---|---|---|---|---|
| τ=0.1 | 11 →12 (step 1) →13 (3) →14 (7) →15 (15) →16 (22) →17 (51) | 0.18 | 25% | 94 |
| τ=0.3 | 11 →12 (1) →13 (4) →14 (24), then stays | 0.35 | 34% | 51 |
| τ=0.5 | 11 throughout | 0.57 | 40% | 0 |

* **Headroom beats gradient throughput.** The ranking is the exact reverse of the informative-group share.
  τ=0.5 has the most useful groups and learns nothing beyond op14. τ=0.1 has the fewest and learns the most.
* **τ=0.5 never moves.** The model's p at op11 sits at 0.55–0.63, never above τ+δ = 0.6 for 2 steps in a
  row. A target at the model's own success rate is a fixed point of the rule (δ = 0.1, patience 2).
  Training only on op11 reaches nothing at op15+.
* **τ=0.3 picked the paper's band on its own.** It reached op14 by step 24 and stayed there, matching the fixed
  band (51 vs 48 newly solved).

`results/sampler_levels.png` shows the level each arm chose, plotted over the measured success rate of every level.

The fixed band's own levels do not stay dead. op13 goes from 1% to 37% informative groups and op14 from 0% to
29% by step 20, by upward generalisation from op11–12 (time series in docs/gate1.md,
`results/informative_over_training.png`). The fixed band's limit is not wasted groups after step 20; it is
that the band never moves past op14.

## 4. Costs and side effects

* **Reliability.** τ=0.1 is slightly lower on medium pass@1 (46.9 vs 49.2) and easy pass@1 (84.0 vs 85.5).
  Medium pass@128 is equal (84.8).
* **Forgetting.** No arm lost a base-solved problem on easy or hard; at most 5 on medium (τ=0.5).
* **Compute.** 7.1–9.3 h per run on one A40 (τ=0.1 is slowest: longer solutions at op16–17). The 18-run plan
  costs about 155 GPU-h.

## 5. What has to be stated in the report

* One seed. The ordering τ=0.1 > uniform > fixed ≈ τ=0.3 > τ=0.5 must survive seeds 1–2.
* τ=0.1 and uniform train on op15–17 and op15–20, the levels of the hard test set (different problems, no
  overlap). So the full hard-set numbers mix in-distribution and extrapolation for these two arms. op18–20 is
  the clean comparison: no arm except uniform ever trained there.

Files: `results/hard_by_level.{csv,png}`, `results/hard_by_level_mcnemar.csv`, `results/method_comparison*.{csv,png}`,
`results/newly_solved.csv`, `results/per_problem/*.csv`, `results/sampler_levels.{csv,png}`,
`$EDGE_SCRATCH/outputs/runs/<run>/` (train.log, sampler_log.jsonl, edge_speed.csv, ckpt, hf/step100).
