# Results across seeds (final for the Dec 3 presentation; 2026-11-22)

Supersedes the [s0] numbers in docs/results_seed0.md. Seeds per arm (seed policy, docs/budget.md): **3** for
τ=0.1, uniform, the op 11–17 control and the bandit; **2** for fixed band and τ=0.3; **1** for τ=0.5 (it never
leaves op 11). **± is the SD over seeds.** Hard test set = 204 problems at op 15–20; op 18–20 = 102 of them.
The base model solves none of the 204 (0 of 128 samples each), so "newly solved" on the hard set = solved at
least once in 128.

Source: `eval/aggregate_seeds.py` (log `logs/analyse-uniform_11_17_s1-2087045.out`), `eval/easy_regression.py`.
Files: `results/seeds_by_set.csv`, `results/seeds_hard_levels.csv`, `results/seeds_gradient_budget.csv`,
`results/seeds_paired_tests.csv`, `results/easy_regression.csv`, `results/easy_regression_paired.csv`.

## 1. Main table

| arm | seeds | hard pass@1 | hard pass@128 | hard newly solved (of 204) | **op 18–20 newly solved (of 102)** | groups with nonzero gradient |
|---|---|---|---|---|---|---|
| **threshold τ=0.1** | 3 | **22.8 ± 0.2** | **46.6 ± 0.8** | **95.0 ± 1.7** (94, 94, 97) | **27.7 ± 0.6** (27, 28, 28) | 24.7% ± 0.3 |
| control: uniform over op 11–17 | 3 | 20.4 ± 0.2 | 42.3 ± 1.0 | 86.3 ± 2.1 (88, 84, 87) | 24.0 ± 2.0 (24, 22, 26) | 28.6% ± 0.1 |
| uniform over op 2–20 | 3 | 16.8 ± 0.5 | 38.2 ± 0.5 | 78.0 ± 1.0 (77, 78, 79) | 20.0 ± 1.0 (19, 21, 20) | 18.8% ± 0.2 |
| fixed band op 11–14 (paper) | 2 | 9.9 ± 0.4 | 23.5 ± 0.0 | 48.0 ± 0.0 | 6.0 ± 0.0 | 35.4% ± 0.1 |
| threshold τ=0.3 | 2 | 10.5 ± 1.2 | 23.8 ± 1.7 | 48.5 ± 3.5 | 4.0 ± 1.4 | 34.0% ± 0.1 |
| bandit (m = 25) | 3 | 8.5 ± 1.3 | 19.9 ± 1.7 | 40.7 ± 3.5 | 3.3 ± 1.5 | 29.1% ± 0.5 |
| threshold τ=0.5 | 1 | 0.0 | 0.0 | 0 | 0 | 40.1% |
| released RL checkpoint | 1 | 10.8 | 23.5 | 48 | 5 | – |

* **Only uniform over op 2–20 trained on op 18–20 by design** (16,090 groups per run, 517 with a nonzero
  gradient). The bandit reached them only through ε-exploration (5,423 groups, 3 with a nonzero gradient).
  τ=0.1, τ=0.3, τ=0.5, the fixed band and the control never saw those levels.
* **τ=0.1's path is the same in all three seeds:** op 11 → 14 by step 7–8, op 15 at step 16, op 16 at
  step 22–25, op 17 at step 45–54, then op 17 to the end, 46–55% of the run (batch p at op 17 ≈ 0.17, below
  the 0.2 move-up trigger; §6).
* **Where the gradient went** (informative groups per run, mean over seeds): op 15–17: τ=0.1 20,642, control
  8,268, uniform 2,300, bandit 161, fixed band / τ=0.3 / τ=0.5 0. op 11–14: control 21,073, τ=0.1 4,654.

## 2. Paired tests across seeds

**Method (pooled).** The problem is the unit. Per problem: the share of arm A's seeds that newly solve it
minus the share of arm B's seeds. Exact two-sided sign test over problems with a nonzero difference, and a
95% bootstrap CI over problems (10,000 draws) for the difference in newly solved per seed. With 3 seeds this
captures the problem sample well and seed-to-seed variance only coarsely; the SDs in §1 show the latter.

### op 18–20 (102 problems, never trained on by τ=0.1 or the control)

| comparison | problems A better / B better | sign test p | Δ newly solved per seed, A − B [95% CI] |
|---|---|---|---|
| **τ=0.1 vs control (op 11–17)** | **8 / 2** | **0.11** | **+3.7 [+0.7, +7.3]** |
| τ=0.1 vs uniform | 15 / 2 | 0.0023 | +7.7 [+3.7, +12.3] |
| control vs uniform | 10 / 4 | 0.18 | +4.0 [+0.7, +8.0] |
| τ=0.1 vs fixed band | 26 / 1 | 4 × 10⁻⁷ | +21.7 [+14.5, +29.3] |
| control vs fixed band | 25 / 0 | 6 × 10⁻⁸ | +18.0 [+11.5, +25.0] |
| τ=0.1 vs bandit | 29 / 1 | < 10⁻⁶ | +24.3 [+16.3, +33.0] |
| τ=0.1 vs τ=0.3 | 28 / 1 | < 10⁻⁶ | +23.7 [+16.0, +31.8] |

### whole hard set, op 15–20 (204 problems)

| comparison | problems A better / B better | sign test p | Δ newly solved per seed [95% CI] |
|---|---|---|---|
| **τ=0.1 vs control** | **17 / 5** | **0.017** | **+8.7 [+3.3, +14.3]** |
| τ=0.1 vs uniform | 32 / 4 | 2 × 10⁻⁶ | +17.0 [+10.7, +24.0] |
| control vs uniform | 23 / 8 | 0.011 | +8.3 [+3.7, +13.7] |
| τ=0.1 vs fixed band | 58 / 5 | < 10⁻⁶ | +47.0 [+35.8, +59.0] |

**Within each seed (McNemar, one seed each), τ=0.1 vs control:** op 18–20: seed 0 5 vs 2 problems (p = 0.45),
seed 1 6 vs 0 (p = 0.031), seed 2 5 vs 3 (p = 0.73). Whole hard set: 10 vs 4 (p = 0.18), 13 vs 3 (p = 0.021),
14 vs 4 (p = 0.031). Same direction in every seed, small in size.

τ=0.1 vs uniform, op 18–20: 9 vs 1 (p = 0.021), 8 vs 1 (p = 0.039), 9 vs 1 (p = 0.021).

None of these p-values is corrected for multiple comparisons. The τ=0.1 vs uniform result survives any
reasonable correction; τ=0.1 vs control on the whole hard set (p = 0.017) is borderline under one.

## 3. Range or schedule? (the op 11–17 control)

The control samples uniformly over op 11–17, the levels τ=0.1 visited, from step 0. It has the range without
the schedule. **We chose that range by looking at where τ=0.1 went**; nobody had it before the experiment.

**The range is most of the win; the schedule adds a small, real margin.**

| decomposition of τ=0.1's gain over the fixed band | hard newly solved (Δ = 47.0) | op 18–20 newly solved (Δ = 21.7) |
|---|---|---|
| harder problems at all (uniform op 2–20 − fixed band) | +30.0 (64%) | +14.0 (65%) |
| the right range (control − uniform op 2–20) | +8.3 (18%) | +4.0 (18%) |
| the schedule (τ=0.1 − control) | +8.7 (18%) | +3.7 (17%) |

* **The control gets 81% of τ=0.1's gain on the hard set and 83% on op 18–20.** On op 18–20 alone the
  remaining gap is not significant by the sign test (8 vs 2 problems, p = 0.11); the bootstrap CI just
  excludes zero (+0.7 to +7.3). On the whole hard set it is (17 vs 5 problems, p = 0.017).
* **Where the schedule's margin sits, per level** (newly solved, τ=0.1 vs control): op 15 29.0 vs 26.3,
  op 16 20.0 vs 17.7, op 17 18.3 vs 18.3, op 18 15.7 vs 14.3, op 19 9.0 vs 8.0, op 20 3.0 vs 1.7. The gap is at
  the levels τ=0.1 trained while it was still climbing (op 15–16, steps 16–54) and carries one to three
  problems per level beyond. At op 17, where both trained heavily, they are equal.
* **The control gets this with less concentration.** τ=0.1 put 2.5× more informative groups on op 15–17
  (20,642 vs 8,268). The control has the higher overall informative share (28.6% vs 24.7%) because 3/7 of
  its batch is op 11–14, which stays learnable.
* **What the adaptive method contributes is finding the range.** It does not track the edge for the whole
  run: it climbs while levels are learnable and stops at op 17, the first level whose success rate stays below
  its 0.2 trigger (§6). The paper's hand-picked band (op 11–14)
  gets 48 hard problems. The range τ=0.1 found on its own gets 86 when sampled uniformly, and 95 with
  τ=0.1's schedule. Without τ=0.1 we would not have known to try op 11–17: it is the paper's band plus
  three levels the base model cannot solve at all (p = 0 at op 15–17, step 0).

## 4. Reliability cost on easier sets: consistent, not seed noise

τ=0.1's lower easy and medium pass@1 is **consistent across seeds**. It is a tradeoff of climbing, not noise.

| pass@1, T = 0.7 | base | fixed band (2) | uniform (3) | control (3) | **τ=0.1 (3)** |
|---|---|---|---|---|---|
| easy, op 2–10 (225 problems) | 85.5 | 85.4 ± 0.1 | 86.4 ± 0.3 | 85.3 ± 0.1 | **83.9 ± 0.2** (83.98, 84.03, 83.75) |
| medium, op 11–14 (800 problems) | 17.5 | 49.2 ± 0.0 | 47.3 ± 0.1 | 49.1 ± 0.1 | **46.7 ± 0.4** (46.91, 46.88, 46.22) |

Paired by problem (mean over each arm's seeds; `results/easy_regression_paired.csv`):

| τ=0.1 minus … | easy: Δ pass@1 [95% CI] | problems lower / higher | seed pairs with τ=0.1 below | medium: Δ pass@1 [95% CI] | seed pairs below |
|---|---|---|---|---|---|
| fixed band | −1.5 [−2.7, −0.6] | 74 / 34 (p = 0.0002) | 6 / 6 | −2.6 [−3.5, −1.7] | 6 / 6 |
| control | −1.4 [−2.3, −0.7] | 78 / 28 (p = 10⁻⁶) | 9 / 9 | −2.5 [−3.2, −1.8] | 9 / 9 |
| uniform | −2.4 [−4.1, −1.1] | 82 / 22 | 9 / 9 | −0.7 [−1.6, +0.2] | 9 / 9 |
| **base model** | **−1.6 [−3.0, −0.3]** | 66 / 43 (p = 0.035) | 3 / 3 | +29.2 [+26.4, +31.9] | 0 / 3 |

* **The easy gap is 10× the seed SD** (1.5 points vs SD 0.1–0.2), and every one of the 6–9 cross-seed pairs
  has τ=0.1 below. It is spread over op 2–10 (largest at op 6–7), not concentrated in a few problems.
* **τ=0.1 ends slightly below the base model on easy pass@1** (−1.6 points, CI excludes zero). No other arm
  does: the control, which also never trains on op 2–10, stays at the base model's level (85.3 vs 85.5).
  So it is not only "no easy training". A likely cause (not tested): from step ~25 on, every τ=0.1 batch is op 16
  or op 17 at p ≈ 0.15–0.2, so most of its rollouts are failures and most of its gradient pushes
  probability away from failing traces.
* **Reachability is unchanged.** Easy pass@128 is 98.5 for τ=0.1, uniform and the control; base-solved easy
  problems lost: 0.3 per seed for each of them. Medium pass@128 equals the control's (85.0 vs 84.9).
* **Medium:** τ=0.1 is 2.5 points below the two arms that trained heavily on op 11–14 (fixed band, control)
  and level with uniform (CI includes zero). It left op 11–14 by step 8 and spent 16% of its groups there. It
  is still +29 points above the base model.

**Tradeoff, stated plainly:** τ=0.1 buys the most new ability on hard problems (95 vs 86 for the control,
48 for the fixed band) at a cost of 1.4–1.5 points of easy pass@1 and 2.5 points of medium pass@1 relative to
the arms that keep training near the base of the range. The control has no such cost.

## 5. What this means for the claims

1. **Headroom beats gradient throughput** (3 seeds): τ=0.5 has the highest informative share (40.1%) and 0
   new hard problems; τ=0.1 has 24.7% and 95.
2. **Training on the right range is most of the win** (control: 81–83% of τ=0.1's gain).
3. **The schedule adds a small margin** (+8.7 hard problems, p = 0.017; +3.7 on op 18–20, p = 0.11), at a
   small reliability cost on easier sets.
4. **The adaptive method's contribution is finding the range without being told.** The range it found
   (op 11–17) beats the paper's hand-picked band by 38 hard problems even without the schedule.
5. **Stall law** (§6, all 6 threshold runs): a threshold arm climbs while the trained level can reach its
   move-up trigger τ + δ and stops at the first level whose success rate stays below it. τ=0.1 is therefore
   adaptive in the first half of the run only (46–55% of its steps at op 17). The law is the hypothesis for
   the January τ = 0.15 and 0.2 runs.

## 6. How adaptive is τ=0.1? The stall law (2026-11-22)

From existing logs only, no new runs: `eval/stall_law.py` → `results/stall_law.csv`, `stall_law_moves.csv`,
`stall_law_segments.csv`, `stall_law_plateaus.csv`, `stall_law_tracking.csv`. "Batch p" is the success rate in
the training batch at the level being trained (training reward, T = 1). A threshold arm's batch is 100% one
level, 6,144 answers per step.

**Stall law: a threshold arm climbs while the level it trains can reach its move-up trigger h = τ + δ, and
stops at the first level whose success rate stays below h for the rest of the run.** It holds in all six
threshold runs, with one caveat for τ=0.1 (below).

| run | trigger h | stopped at | reached at step | steps there | batch p there, last 20 steps | max batch p there | steps above h there (longest run; a move needs 2) | slope, last 30 steps (Δp per 10 steps) |
|---|---|---|---|---|---|---|---|---|
| τ=0.1 s0 | 0.2 | op 17 | 52 | 48 | 0.178 ± 0.010 | 0.195 | 0 (0) | +0.009 |
| τ=0.1 s1 | 0.2 | op 17 | 45 | 55 | 0.176 ± 0.010 | 0.194 | 0 (0) | +0.007 |
| τ=0.1 s2 | 0.2 | op 17 | 54 | 46 | 0.174 ± 0.010 | 0.206 | 1 (1) | +0.006 |
| τ=0.3 s0 | 0.4 | op 14 | 25 | 75 | 0.369 ± 0.015 | 0.401 | 1 (1) | +0.004 |
| τ=0.3 s1 | 0.4 | op 14 | 23 | 77 | 0.373 ± 0.013 | 0.406 | 1 (1) | +0.002 |
| τ=0.5 s0 | 0.6 | op 11 | 0 | 100 | 0.588 ± 0.017 | 0.629 | 8 (1) | +0.005 |

**Independent check.** Ceilings measured in the arms that train several levels at once (fixed band s0–s1,
control s0–s2; batch p over their last 20 steps): op 11 0.57, op 12 0.54, op 13 0.44, op 14 0.35, op 15 0.31,
op 16 0.21, op 17 0.15 (still rising). The first level below each trigger is op 11 for h = 0.6, op 14 for
h = 0.4 and op 17 for h = 0.2. Given only other arms' data, the law predicts every threshold arm's final level.

* **τ=0.5 and τ=0.3: confirmed.** Their final levels are flat (slope ≤ 0.005 per 10 steps) and below the
  trigger: τ=0.5 by 0.01 (0.59 vs 0.6; above it on 8 single steps, never twice in a row) and τ=0.3 by 0.03
  (0.37 vs 0.4). τ=0.3 only just passed op 13: 0.40–0.42 after 17–20 steps, against a ceiling of 0.44.
* **τ=0.1: confirmed as stated, not as a ceiling.** It stopped at op 17 in all three seeds at 0.17–0.18, and
  was never above 0.2 twice in a row. But op 17 was still rising, by 0.006–0.009 per 10 steps over the last
  30 steps (the control's op 17 rises too). At that rate it would cross 0.2 around step 125. Within 100 steps,
  "op 17's ceiling is below 0.2" and "op 17 rises too slowly to reach 0.2 in time" cannot be told apart; a
  200-step run can. op 16 was a near-stall too: it crossed 0.2 only after 23–29 steps, at 0.20–0.22, against
  an independent ceiling of 0.21.
* **Known-side memory did not cause the stalls.** It refused 0–1 moves per run, none at the final level, and
  the final level never had the two consecutive measurements above h that a move needs.

**How much of the run was adaptive.**

| arm | steps spent changing level (steps before reaching the final level) | steps at the final level | share of the run at the final level |
|---|---|---|---|
| τ=0.1 | 52, 45, 54 | 48, 55, 46 | **46–55%** |
| τ=0.3 | 25, 23 | 75, 77 | **75–77%** |
| τ=0.5 | 0 | 100 | 100% |

The climb also slowed level by level. τ=0.1's batch p rose, per 10 steps of training at that level, by
0.31–0.32 at op 14, 0.12–0.17 at op 15, 0.03–0.05 at op 16 and 0.008–0.012 at op 17
(`results/stall_law_segments.csv`). Levels 11 → 15 took 16 steps, op 16 took 23–29 and op 17 the rest. **τ=0.1
is fully adaptive for about its first sixth, slowly adaptive until step 45–54, and not adaptive at all for
the second half of the run.**

**This qualifies the learning-curve finding** (docs/bandit_analysis.md §6). At op 16–17 τ=0.1 trained at p ≈
0.11–0.20, the range where op 13–15 rose fastest, yet p rose 10–40× more slowly than at op 13–14. The
steepest-point estimate p ≈ 0.10–0.21 comes from op 13–15. At harder levels, a success rate in that range
does not by itself mean fast learning.

**When did τ=0.1 pull ahead of the control? A first look from logs** (tracking set: 32 problems × 8 answers
per level at T = 1, every 10 steps; mean of 3 seeds; τ=0.1 / control):

| step | 20 | 30 | 50 | 80 | 100 |
|---|---|---|---|---|---|
| op 15 | 0.19 / 0.05 | 0.22 / 0.19 | 0.21 / 0.26 | 0.24 / 0.25 | 0.28 / 0.29 |
| op 16 | 0.07 / 0.01 | 0.17 / 0.08 | 0.24 / 0.18 | 0.27 / 0.26 | 0.30 / 0.28 |
| op 17 | 0.01 / 0.00 | 0.04 / 0.01 | 0.07 / 0.05 | 0.13 / 0.09 | 0.18 / 0.11 |
| op 18–20 | 0.000 / 0.000 | 0.011 / 0.002 | 0.019 / 0.012 | 0.039 / 0.018 | 0.044 / 0.032 |

τ=0.1's lead on op 15–16 opens while it climbs (steps 20–50) and is gone in this measure by steps 80–100. Its
lead on op 17 grows during the stall, while it trains op 17 exclusively. On op 18–20 it leads from step 30 on,
by a small margin that does not clearly grow or shrink. The tracking p is a noisy T = 1 success rate on 32
problems per level, not the newly-solved count. The pass@128 evaluation of the step-25/50/75 checkpoints is
proposed and **not yet run** (needs approval).

**Hypothesis for the January τ = 0.15 and 0.2 runs: the trigger h, not τ, sets the final level.** From the
ceilings above, h = 0.20 → op 17 (as τ=0.1), h = 0.25 → op 16, and h = 0.30 → op 15–16 (op 15's ceiling of
0.31 is within noise of 0.30).
* With δ = 0.1, τ = 0.15 (h = 0.25) and τ = 0.2 (h = 0.30) should stop at op 15–16. They would train a
  narrower range than τ=0.1 and, if the range drives the gain, newly solve fewer hard problems.
* With the δ = 0.05 the January plan proposed, τ = 0.15 has h = 0.20, the same trigger as τ=0.1, and should
  reproduce τ=0.1's path.
* The arm predicted to climb past op 17 is one with a trigger below 0.2, e.g. h = 0.15; τ=0.1's own p at op 17
  was 0.17–0.18.

Each prediction is checkable from the sampler log alone (final level and arrival step).
