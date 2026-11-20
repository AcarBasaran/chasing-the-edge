# Chasing the Edge: progress report (draft, 2026-11-20)

COMP550 · Eren Acar Basaran, Turan Berke Yakarer

> **Marking.** Numbers tagged **[s0]** come from **one training seed (seed 0)** per arm and have no
> seed-to-seed error bars yet. Seeds 1 and 2 are running. Untagged numbers are either reproductions of the paper
> (deterministic setup checks) or properties of the base model. Paired significance tests are reported
> **within seed 0 only** and are not quoted in any summary until seeds are aggregated.

## 1. What we set out to do

Zhang, Neubig and Yue (arXiv 2512.07783) show that RL (GRPO) gives a small language model genuinely new
reasoning ability only when training problems sit at its **edge of competence**: hard enough that it often
fails, easy enough that it sometimes succeeds. They pick that band once (11–14 operations) and keep it fixed
for the whole RL run.

GRPO learns only from differences inside a group of G rollouts. A group whose rollouts are all right or all
wrong gives exactly zero gradient. As the model learns, the levels where groups are mixed move upward. Our
question: **does a sampler that tracks the moving edge create more new ability than the fixed band, or only
more reliability on what the model can already do?**

We keep everything from the paper (model, data, GRPO settings, the strict process-verified reward) and change
only which difficulty level each training batch is drawn from. Five samplers:

| arm | rule |
|---|---|
| fixed band | the paper's method: uniform over op 11–14 |
| uniform random | uniform over op 2–20; controls for "harder problems help" vs "timing matters" |
| threshold τ ∈ {0.1, 0.3, 0.5} | one active level; move up when its success rate p stays above τ+δ for 2 steps, down when it stays below τ−δ (δ = 0.1, lower bound clamped at τ/2) |
| bandit | each level is an arm; reward = change in p over m steps; ε-greedy (α = 0.3, ε = 0.1, m = 25). Probe rollouts count against the training budget |

The headline metric is the paper's own: **problems the base model never solved (0 of 128 samples) that the
trained model solves at least once**, on a held-out hard set (op 15–20, 204 problems).

## 2. Setup and reproduction

* **Base model (Gate 1).** The paper's 100M-parameter pretrained checkpoint reproduces its reported base numbers:
  31 of 32 comparisons are within ±6 points. Example: op 11–14 pass@1 18.3 vs ≈19 (Fig. 9), pass@128 41.8 vs ≈42.
  The base model solves no problem at op 15–20 in 128 tries. The 1024-token answer limit is not binding
  (correct answers stay under 440 tokens).
* **Fixed-band baseline (Gate 3).** Our run (strict reward, 100 steps, batch 1024 × 6 rollouts, one A40)
  against the paper's released checkpoint, on the same test problems with the same scoring:

  | test set | base p@1 / p@128 | **ours** p@1 / p@128 | released p@1 / p@128 | paper Fig. 3 (read off) |
  |---|---|---|---|---|
  | easy op 2–10 | 85.5 / 98.7 | 85.5 / 98.7 | 85.2 / 98.7 | 87.0 / 99.0 |
  | medium op 11–14 | 17.5 / 38.4 | **49.2 / 84.8** | 49.4 / 83.9 | 51.0 / 85.5 |
  | hard op 15–20 | 0 / 0 | **10.2 / 23.5** | 10.8 / 23.5 | 9.5 / 26.0 |
  | hard newly solved | – | **48 of 204** | 48 of 204 | – |

  Every number is within 1 point of the released checkpoint. The pipeline reproduces the paper, so the
  comparisons below are against a faithful baseline. (The released checkpoint was probably trained with an
  answer-only reward; we use the strict reward. The match suggests this matters little at 100 steps. We are
  asking the authors to confirm.)

## 3. Results [s0]

### 3.1 Headline: levels τ=0.1 never trained on

Threshold τ=0.1 climbed from op 11 to op 17 and stopped. **op 18–20 are levels it never trained on.**
Uniform random trained on every level, **including 18–20**.

Figure: `results/hard_by_level.png`.

| arm | trained at op 18–20? groups / with nonzero gradient | pass@128 | newly solved (of 102) |
|---|---|---|---|
| threshold τ=0.1 | never | 26.5 [s0] | **27** [s0] |
| uniform random | yes: 16,005 / 487 | 18.6 [s0] | 19 [s0] |
| fixed band | never | 5.9 [s0] | 6 [s0] |
| threshold τ=0.3 | never | 4.9 [s0] | 5 [s0] |
| threshold τ=0.5 | never | 0 [s0] | 0 [s0] |

The base model solved none of these 102 problems.

**Mechanism.** Uniform drew 16% of its prompts from op 18–20, but 97% of those groups had zero gradient
[s0]: at the model's skill, every rollout fails. Exposure to a level does not teach it when nothing is ever
solved. τ=0.1 spent its budget on op 15–17 while those were at the edge (25% of its groups informative
[s0]), and that skill carried two to three levels beyond anything it trained on.

Within seed 0 only, paired by problem: τ=0.1 newly solves 9 problems uniform does not, and uniform 1 that
τ=0.1 does not. Seed-to-seed variance is not included; the pooled test across seeds will replace this.

**Not more gradient, gradient in a different place.** Pooled over the run, τ=0.1 had a *lower* share of
groups with nonzero gradient (25.0% [s0]) than the fixed band (35.4%), τ=0.3 (33.9%) and τ=0.5 (40.1%); only
uniform was lower (18.8%). What differs is where the informative groups were. Beyond op 14, τ=0.1 had 20,739
informative groups, uniform 2,783, every other arm 0 [s0]. That ordering matches op 18–20 newly solved.

### 3.2 How much is "harder problems at all" vs adaptive timing

| arm | hard pass@128 | gain over fixed band | hard newly solved (of 204) |
|---|---|---|---|
| fixed band | 23.5 [s0] | – | 48 [s0] |
| uniform random | 37.7 [s0] | +14.2 [s0] | 77 [s0] |
| threshold τ=0.1 | 46.1 [s0] | +22.6 [s0] | 94 [s0] |

On the whole hard set, uniform random already captures **63% of τ=0.1's gain** over the fixed band [s0].
Most of the improvement is "train on harder problems at all". Adaptive timing is the smaller remaining
margin (+8.4 points, 17 problems [s0]), and it sits on the levels beyond τ=0.1's training range (§3.1).

### 3.3 The τ sweep

| arm | levels visited | groups with nonzero gradient | hard newly solved |
|---|---|---|---|
| τ=0.1 | 11 → 17 by step 51 | 25% [s0] | 94 [s0] |
| τ=0.3 | 11 → 14 by step 24, then stays | 34% [s0] | 51 [s0] |
| τ=0.5 | stays at 11 | 40% [s0] | 0 [s0] |

* **Headroom beats gradient throughput** [s0]. The ranking is the reverse of the informative-group share.
  The arm with the most usable groups (τ=0.5) learns nothing beyond op 14.
* **τ=0.5 is a fixed point.** The model's success rate at op 11 hovers at 0.55–0.63 and never stays above
  the 0.6 bound for two steps. This is a property of the rule (δ = 0.1, patience 2), not a bug.
* **τ=0.3 found the paper's band on its own** and matches it (51 vs 48 newly solved [s0]). If seeds confirm
  this, it is a positive finding about the paper's hand-picked band: it is where a τ=0.3 rule settles.

### 3.4 The bandit: the arm being trained generates its own reward

The bandit (reward = improvement in p between probes) is the weakest trained arm: hard newly solved
40.7 ± 3.5 over 3 seeds, vs 48 for the fixed band. The arm it trains improves because it is trained, so the
greedy choice pays itself and keeps 90% of every batch. op 15 became learnable anyway (probe p = 0.31 by
step 50 in seed 0) and reached second place in Q, but an argmax gives second place nothing. After the first
value update (step 25), op 15+ was sampled only by ε-exploration, at exactly the ε floor of 3.2% of prompts,
in all 3 seeds. Probe noise (SE ≈ 0.09 against Q margins of about 0.01) makes the order among op 12–15
partly chance. The trained arm's head start is not chance. A faster probe interval (m = 10, one seed) does not change it: the
bandit locked onto op 12 even earlier, and op 15+ again got only the ε floor.

Where a level improves fastest: a level's success rate rises in an S, steepest at p ≈ 0.1–0.2 and flat at
a plateau of about 0.35–0.58 (training reward, T = 1.0) [fixed band s0–s1, uniform s0–s1]. τ=0.5's band is the
plateau. τ=0.1's band is the steep part. So τ=0.1 is a principled choice, not a lucky pick: its target band
contains the measured peak. p ≈ 0.5 is the ceiling at this temperature, not the half-solved midpoint, which
corrects the premise in our proposal that p = 0.5 is the sweet spot. The bandit is held behind the steep part
mainly by self-reward. Measurement lag is secondary: m = 10 made it lock on sooner, not later. Details: docs/bandit_analysis.md §4 and §6.
Fixes (softmax selection, optimistic initialisation, removing the self-reward loop) are January work (§7).

### 3.5 Costs

* τ=0.1 is slightly less reliable on easier sets: medium pass@1 46.9 vs 49.2, easy pass@1 84.0 vs 85.5 [s0].
  Medium pass@128 is equal.
* No arm forgets: 0 base-solved problems lost on easy or hard, at most 5 on medium [s0].
* About 8 GPU-hours per run on one A40. The full plan (6 arms × 3 seeds × 100 steps) is about 155 GPU-hours.

## 4. Correcting our own motivation

Our proposal motivated the project with a step-0 measurement: in the base model, op 13 and op 14 give a
gradient in 1.5% and 0.02% of groups. "Half the paper's band is dead weight."

**That is a snapshot, and it does not last.** In the fixed-band run, training on op 11–12 generalises upward.
op 13 reaches 37% and op 14 29% informative groups by step 20, measured on the groups GRPO actually trained
[s0]. After the first ~15 steps the band is not wasteful.

The fixed band's real limitation is the other end. **It never moves past op 14 while op 15–16 become
learnable**: tracking success rate at op 15 goes 0 → 0.24 by step 50 and 0.25 by step 100; op 16 goes
0 → 0.13 [s0]. The samplers that follow the edge train exactly there. The step-0 table stays in the report as
a snapshot, next to the time series (`results/informative_over_training.png`, docs/gate1.md).

## 5. Engineering notes

* verl runs unmodified. Samplers plug into its existing curriculum-sampler hook and log their choice every
  step. A simulated model caught an oscillation bug in the threshold rule before any GPU time was spent.
* About four days of wall-clock were lost to RL jobs that hung silently at Ray startup on the cluster. The fix: a private Ray head
  per job, explicit token auth, and a startup watchdog. Training, evaluation and analysis now run as chained
  SLURM jobs, unattended.

## 6. What is still running

* Seeds 1 and 2 of fixed band, uniform, τ=0.1 and τ=0.3. τ=0.5 stays at one seed: it never leaves op 11,
  so more seeds would buy nothing (budget decision, 2026-11-20).
* Control: uniform over op 11–17, seeds 0–2, queued behind the above.
* Runs are submitted automatically within the cluster's 50-job limit.
* Checkpoint re-evaluation of the fixed band at steps 25/50/75/100 on levels 9–17, for the all-levels version
  of §4.
* `eval/aggregate_seeds.py` turns every new seed into mean ± spread per arm and a pooled problem-level paired
  test. It is already tested on seed 0.

## 7. What is left for January

1. **Seeds.** Confirm or retract §3 with 3 seeds per arm. Replace every [s0] tag with mean ± spread. Report the
   pooled paired test for τ=0.1 vs uniform on op 18–20.
2. **Bandit fixes.** Softmax selection, optimistic initialisation, and removing the self-reward loop (the real fix),
   3 seeds each, ≈ 25 GPU-h per fix (docs/bandit_analysis.md §7).
3. **Longer runs.** 100 steps is half the paper's run. Extend the best arm and the fixed band to ~200 steps:
   does the fixed band catch up, or does the gap widen?
4. **Range vs schedule** (queued, 3 seeds; results likely before Dec 3 if GPUs free up as expected). Uniform over op 11–17 is exactly the range
   τ=0.1 visited, sampled from step 0 without the schedule. If it matches τ=0.1 on op 18–20, the headline gain
   is the range, not the timing. This is the main objection a reviewer could raise.
5. **τ=0.5 fix point.** Rerun with δ = 0.05 or patience 1 to separate "the target is wrong" from "the rule
   cannot move".
6. **Per-problem selection** (future work, noted in Gate 1). Within a level, problems are mostly always or
   never solved. Picking problems by their own measured p could raise the informative share well above ~40%.
7. **Final report**: method, all arms with seeds, the op 18–20 analysis, the motivation correction, compute.
