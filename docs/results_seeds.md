# Results across seeds (2026-11-20, τ=0.1 at 3 seeds)

Supersedes the [s0] numbers in docs/results_seed0.md and docs/progress_report.md wherever they differ. Seed
policy (docs/budget.md): 3 seeds for τ=0.1, uniform and the op 11–17 control; 2 for fixed band and τ=0.3; the bandit
already had 3; τ=0.5 has 1. **± is the SD over seeds.** Hard test set = 204
problems at op 15–20, op 18–20 = 102 of them; the base model solved none of the op 18–20 problems (0/128 each).

Source: `eval/aggregate_seeds.py` (log `logs/analyse-threshold_tau1_s2-2073588.out`), files
`results/seeds_by_set.csv`, `results/seeds_hard_levels.csv`, `results/seeds_gradient_budget.csv`,
`results/seeds_paired_tests.csv`.

## 1. Main table

| arm | seeds | hard pass@1 | hard pass@128 | hard newly solved (of 204) | **op 18–20 newly solved (of 102)** | informative share |
|---|---|---|---|---|---|---|
| **threshold τ=0.1** | **3** | **22.8 ± 0.2** | **46.6 ± 0.8** | **95.0 ± 1.7** (94, 94, 97) | **27.7 ± 0.6** (27, 28, 28) | 24.7% ± 0.3 |
| uniform random | 3 | 16.8 ± 0.5 | 38.2 ± 0.5 | 78.0 ± 1.0 | 20.0 ± 1.0 | 18.8% ± 0.2 |
| fixed band (paper) | 2 | 9.9 ± 0.4 | 23.5 ± 0.0 | 48.0 ± 0.0 | 6.0 ± 0.0 | 35.4% ± 0.1 |
| threshold τ=0.3 | 2 | 10.5 ± 1.2 | 23.8 ± 1.7 | 48.5 ± 3.5 | 4.0 ± 1.4 | 34.0% ± 0.1 |
| bandit (m = 25) | 3 | 8.5 ± 1.3 | 19.9 ± 1.7 | 40.7 ± 3.5 | 3.3 ± 1.5 | 29.1% ± 0.5 |
| threshold τ=0.5 | 1 | 0.0 | 0.0 | 0 | 0 | 40.1% |
| released RL checkpoint | 1 | 10.8 | 23.5 | 48 | 5 | – |
| *control: uniform op 11–17* | *1 of 3* | *20.3* | *43.1* | *88* | *24* | *28.6%* |

* **The seed-0 result holds.** τ=0.1's three seeds are tight (op 18–20: 27, 28, 28; seed 0 was 27). Its
  ordering against every other trained arm is the same in every seed.
* **τ=0.1 never trained on op 18–20 in any seed** (0 training groups there). All three seeds climbed
  op 11 → op 17 and stayed at op 17 for the rest of the run, with batch p at op 17 ≈ 0.17.
* **Uniform's share of τ=0.1's gain on the hard set:** (38.2 − 23.5) / (46.6 − 23.5) = **64%** (seed 0: 63%).
* **Cost on easier sets (3 seeds):** τ=0.1 easy pass@1 83.9 ± 0.1 vs fixed band 85.4 and uniform 86.4; medium
  pass@1 46.7 ± 0.4 vs 49.2 and 47.3. pass@128 is unchanged on easy (98.5 for both) and higher on medium
  (85.0 vs 84.3 and 83.0). So τ=0.1 is slightly less reliable on what the model can already do, not less able.
* **Gradient in a different place, again:** informative groups at op 15–17, mean over seeds: τ=0.1 20,642,
  uniform 2,300, bandit 161, fixed band / τ=0.3 / τ=0.5 0. τ=0.1's overall informative share (24.7%) is below
  the fixed band's (35.4%) and τ=0.5's (40.1%).

## 2. Paired tests across seeds (hard op 18–20, 102 problems)

**Method (pooled).** The problem is the unit. Per problem: the share of arm A's seeds that newly solve it
minus the share of arm B's seeds. Exact two-sided sign test over problems with a nonzero difference, and a
95% bootstrap CI over problems (10,000 draws) for the difference in newly solved per seed. This accounts for
the problem sample; with 3 seeds it gives only a coarse read on seed-to-seed variance, which the SDs above
show directly.

| comparison (op 18–20) | seeds | problems τ=0.1 better / other better | sign test p | Δ newly solved per seed, τ=0.1 − other [95% CI] |
|---|---|---|---|---|
| **τ=0.1 vs uniform** | **3 vs 3** | **15 / 2** | **0.0023** | **+7.7 [+3.7, +12.3]** |
| τ=0.1 vs fixed band | 3 vs 2 | 26 / 1 | 4 × 10⁻⁷ | +21.7 [+14.5, +29.3] |
| τ=0.1 vs bandit | 3 vs 3 | 29 / 1 | < 10⁻⁶ | +24.3 [+16.3, +33.0] |
| τ=0.1 vs τ=0.3 | 3 vs 2 | 28 / 1 | < 10⁻⁶ | +23.7 [+16.0, +31.8] |

Same comparisons on the whole hard set (op 15–20, 204 problems): τ=0.1 vs uniform 32 / 4 problems,
p ≈ 2 × 10⁻⁶, Δ +17.0 [+10.7, +24.0]; vs fixed band 58 / 5, Δ +47.0 [+35.8, +59.0].

**Within each seed (McNemar, τ=0.1 vs uniform, op 18–20; one seed each):** seed 0: 9 vs 1 problems, p = 0.021;
seed 1: 8 vs 1, p = 0.039; seed 2: 9 vs 1, p = 0.021. Same direction and size in all three seeds.

**Not tested yet: τ=0.1 vs the op 11–17 control.** One control seed so far (seed 0: 24 on op 18–20, 88 on the
hard set, against τ=0.1's 27.7 and 95). Seeds 1–2 are running. No p-value is quoted for this comparison until
they land.

## 3. What this changes in the docs

* Slide footers: "τ=0.1: 3 seeds"; p-values allowed for τ=0.1 vs uniform / fixed band / bandit / τ=0.3.
* The headroom headline (slide 10) is now a 3-seed result: 27.7 ± 0.6 vs uniform 20.0 ± 1.0.
* The range-vs-schedule question is open, and seed 0 of the control is close to τ=0.1. Prepare the
  "the headroom range wins" version of slides 10 and 12 as seriously as the "the schedule wins" one.
