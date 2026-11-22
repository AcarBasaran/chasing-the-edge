# Chasing the Edge: Adaptive Difficulty for GRPO

**COMP550 progress report** · Eren Acar Basaran, Turan Berke Yakarer · draft of 2026-11-22, due Dec 6

*All numbers are means ± SD over training seeds (3 for τ=0.1, uniform, the op 11–17 control and the bandit; 2
for the fixed band and τ=0.3; 1 for τ=0.5). Paired tests are pooled over seeds with the problem as the unit
and are not corrected for multiple comparisons. Source tables: docs/results_seeds.md.*

## Summary

Zhang, Neubig and Yue (2025) show that reinforcement learning gives a small language model genuinely new
reasoning ability only when it trains at its *edge of competence*, and they fix that edge at one band of
difficulty for the whole run. We replace the fixed band with samplers that choose the difficulty of every
batch as the model learns, keeping everything else identical. Of three adaptive objectives (maximising the
share of informative gradient, maximising measured learning speed, and maximising headroom), headroom wins
clearly. A threshold sampler that moves to the next level as soon as the model's success rate on the current one
exceeds 0.2 (τ=0.1)
newly solves 95 ± 1.7 of 204 hard problems the base model never solved, against 48 for the paper's fixed band.
On the three hardest levels, which it never trained on, it solves 27.7 ± 0.6 of 102, against 20.0 ± 1.0 for
uniform sampling over all levels (p = 0.002). A control that samples τ=0.1's range uniformly from the start
recovers 81% of that gain, so **training on the right range is most of the win, and the schedule adds a small
margin** (+8.7 problems, p = 0.017). The adaptive sampler's main contribution is that it found that range
without being told: the paper's hand-picked band was too low. Two further findings correct our own proposal.
The sampler with the most informative gradient (τ=0.5, 40% of groups) learns nothing new, because a level's
success rate tops out near 0.5 under this setup and learning is fastest near 0.1–0.2. And an improvement-rate
bandit fails for a structural reason: the level it trains improves because it is trained, so its choice pays
for itself and it never moves on.

## 1. Background and question

Group Relative Policy Optimisation (GRPO) samples G completions per problem and computes each completion's
advantage relative to its group. If all G completions are right, or all are wrong, every advantage is zero
and the problem contributes no gradient. With per-problem success rate p, the expected fraction of
informative groups is 1 − pᴳ − (1 − p)ᴳ. With G = 6, this is under 6% at p = 0.01 and above 80% for
p between 0.3 and 0.7.

The paper trains a 100M-parameter model on synthetic arithmetic problems whose difficulty is the number of
operations (op 2–20), with a strict process-verified reward that requires every reasoning step and the final
answer to be correct. Its central finding is that RL produces genuinely new ability only on problems near the
model's edge, and it trains on a fixed band, op 11–14, chosen once.

The edge moves as the model learns: a band that is informative at step 0 becomes easy, and levels above it
become learnable. **Our question is whether a sampler that tracks the moving edge creates more new ability
than the fixed band, or only more reliability on what the model can already do**, and which notion of "the
edge" a sampler should aim for.

## 2. Method

We keep the paper's model, data, GRPO settings and reward. We change only which difficulty level each
training batch is drawn from, through verl's existing curriculum-sampler hook. Every run uses the same budget:
100 steps of 1,024 prompts × 6 rollouts on one A40 GPU (about 8 GPU-hours). Probe rollouts used by the
adaptive samplers to measure success rates count against the budget.

| arm | rule | objective |
|---|---|---|
| fixed band | uniform over op 11–14 | the paper's method |
| uniform | uniform over op 2–20 | control: "harder problems help at all" |
| **uniform op 11–17** | uniform over op 11–17 from step 0 | control: range without schedule (§4.3) |
| threshold τ=0.5 | one active level; move up when its success rate p stays above τ+δ for 2 steps, down when below τ−δ (δ = 0.1, lower bound clamped at τ/2) | gradient efficiency: stay where p ≈ 0.5 |
| threshold τ=0.3 | same, τ = 0.3 | intermediate |
| threshold τ=0.1 | same, τ = 0.1: move up when p > 0.2 | headroom: train where the model mostly fails |
| bandit | each level is an arm; reward = change in p between probes every m = 25 steps; ε-greedy (α = 0.3, ε = 0.1) | learning speed |

A simulated model (success logistic in skill minus level, skill rising over time) was used to test every
sampler before any GPU time was spent. It caught an oscillation in the threshold rule: adjacent levels
straddle any band, so the sampler ping-ponged. We added a memory that never moves into a neighbour freshly
measured on the wrong side.

**Evaluation.** Held-out test sets of 225 easy (op 2–10), 800 medium (op 11–14) and 204 hard (op 15–20)
problems, 128 samples each at T = 0.7, scored with the paper's evaluation criterion. pass@k uses the unbiased
estimator. **The headline metric is the paper's own: problems the base model never solved (0 of 128) that
the trained model solves at least once.** The base model solves none of the 204 hard problems, so on the hard
set every solved problem is new.

## 3. Reproduction

The pretrained checkpoint reproduces the paper's base-model numbers: 31 of 32 comparisons are within six
points, for example op 11–14 pass@128 41.8 against about 42 in the paper's Figure 9. The 1024-token answer
limit is not binding: correct solutions stay under 440 tokens.

Our fixed-band run matches the paper's released RL checkpoint within one point on every test set: medium
pass@1 49.2 vs 49.4, hard pass@128 23.5 vs 23.5, and 48 of 204 hard problems newly solved for both. The
released checkpoint was probably trained with an answer-only reward rather than the strict one we use; the
close match suggests this matters little at 100 steps. All later comparisons are against this baseline.

## 4. Results

### 4.1 Main results

| arm | seeds | hard pass@128 | hard newly solved (of 204) | op 18–20 newly solved (of 102) | groups with nonzero gradient |
|---|---|---|---|---|---|
| **threshold τ=0.1** | 3 | **46.6 ± 0.8** | **95.0 ± 1.7** | **27.7 ± 0.6** | 24.7% |
| uniform op 11–17 (control) | 3 | 42.3 ± 1.0 | 86.3 ± 2.1 | 24.0 ± 2.0 | 28.6% |
| uniform op 2–20 | 3 | 38.2 ± 0.5 | 78.0 ± 1.0 | 20.0 ± 1.0 | 18.8% |
| fixed band (paper) | 2 | 23.5 ± 0.0 | 48.0 ± 0.0 | 6.0 ± 0.0 | 35.4% |
| threshold τ=0.3 | 2 | 23.8 ± 1.7 | 48.5 ± 3.5 | 4.0 ± 1.4 | 34.0% |
| bandit | 3 | 19.9 ± 1.7 | 40.7 ± 3.5 | 3.3 ± 1.5 | 29.1% |
| threshold τ=0.5 | 1 | 0.0 | 0 | 0 | 40.1% |

The samplers behaved the same way in every seed. τ=0.1 climbed from op 11 to op 14 by step 8, reached op 15
at step 16, op 16 at step 22–25 and op 17 at step 45–54, and then stayed at op 17, where its success rate
hovered around 0.17, below the move-up bound. τ=0.3 settled on op 14 by step 25. τ=0.5 never left op 11.
The bandit settled on op 12–14.

### 4.2 Headroom: levels τ=0.1 never trained on

Neither τ=0.1 nor the fixed band ever trained on op 18–20. Uniform sampling did: 16% of its prompts came from
those levels. Yet τ=0.1 newly solves 27.7 ± 0.6 of those 102 problems and uniform 20.0 ± 1.0. Paired by
problem across three seeds each, τ=0.1 solves 15 problems more often than uniform and uniform 2 more often
than τ=0.1 (sign test p = 0.002; +7.7 problems per seed, 95% CI +3.7 to +12.3). The direction is the same in
each seed separately.

The reason is visible in the gradient. 97% of uniform's groups at op 18–20 had zero gradient: at the model's
current skill, every rollout there fails, and exposure to a level teaches nothing when it is never solved.
τ=0.1 instead spent 84% of its groups on op 15–17 while those levels were at the edge, with 20,642
informative groups there per run (uniform: 2,300; the fixed band: 0), and the skill it built carried two to
three levels beyond anything it trained on.

### 4.3 Range or schedule?

τ=0.1 differs from the fixed band in two ways: it trains on a different range of levels (op 11–17 instead of
op 11–14), and it visits them in order, moving up only when the current level reaches p > 0.2. To separate
the two, we trained a control that samples uniformly over op 11–17 from the first step: the same range,
without the schedule. We chose that range by looking at where τ=0.1 went.

**The range is most of the win.** The control newly solves 86.3 ± 2.1 hard problems and 24.0 ± 2.0 on op
18–20: 81% and 83% of τ=0.1's gain over the fixed band. Splitting τ=0.1's gain of 47 hard problems over the
fixed band into steps: training on harder problems at all (uniform over op 2–20) accounts for 30 (64%),
restricting to the right range adds 8.3 (18%), and the schedule adds 8.7 (18%). The split on op 18–20 is
nearly identical (65%, 18%, 17%).

**The schedule adds a small, real margin.** On the whole hard set, τ=0.1 beats the control on 17 problems and
loses on 5 (p = 0.017; +8.7 per seed, 95% CI +3.3 to +14.3), and it is ahead in every seed. On op 18–20 alone
the gap is +3.7 problems per seed (8 vs 2, p = 0.11), which is not significant. Per level, the margin sits at
op 15 (29.0 vs 26.3) and op 16 (20.0 vs 17.7), the levels τ=0.1 trained intensively *while they were at the
edge*. At op 17, where both arms trained heavily, they are equal (18.3 each).

**What the adaptive sampler contributes is the range itself.** The paper's band gets 48 hard problems. The
range τ=0.1 found gets 86 when sampled uniformly and 95 with τ=0.1's schedule. op 11–17 is the paper's band
plus three levels at which the base model has a success rate of exactly zero; nothing in the base model's
measurements points to them, and we would not have tried this range without τ=0.1. A sampler that follows
the edge is a way to find the right range in one run instead of a sweep over ranges. We consider this the
main positive result of the project, and the schedule's margin a secondary one.

### 4.4 More informative gradient did not mean more learning

The τ sweep ranks the arms in the reverse order of their gradient efficiency. τ=0.5 keeps 40.1% of its groups
informative, the highest of any arm, and newly solves 0 hard problems. τ=0.1 keeps 24.7% informative and
newly solves 95. τ=0.3 (34.0%) and the fixed band (35.4%) sit in between on both counts in the opposite
order. τ=0.1 did not receive more gradient than the other arms; it received gradient in a different place.

τ=0.5 is also a fixed point of its own rule. Its success rate at op 11 hovers at 0.55–0.63 and never stays
above the 0.6 move-up bound for two steps, so it trains on op 11 for the whole run.

### 4.5 Why τ=0.1: where a level learns fastest

Per-step success rates of each level during training trace an S-shaped curve. Measured on every level that
starts at p ≈ 0 (fixed band and uniform runs, training reward, T = 1.0), the steepest rise is at
**p ≈ 0.10–0.21**, at 30–50% of the level's final value, and the curve flattens at a ceiling of p ≈ 0.35–0.58
(op 11: 0.57). Improvement at p ≈ 0 is also near zero.

This explains both ends of the τ sweep. τ=0.1's target band, [0.05, 0.2], contains the steepest point: it
trains each level during its fastest improvement and moves on as the curve starts to flatten. τ=0.5's band,
[0.4, 0.6], is the ceiling: it maximises informative groups by sitting where nothing is left to learn, and
because op 11's ceiling (0.57) is below the move-up bound (0.6), it never leaves. Our proposal assumed that
p = 0.5 was the sweet spot because it maximises the informative fraction. Under this reward and temperature
p ≈ 0.5 is the end of a level's learning curve, not its midpoint, and the premise was wrong. τ=0.1 is the
best of the three values we ran and the one the curve predicts. Values between 0.1 and 0.3 are untested.

### 4.6 The bandit: the level it trains generates its own reward

The bandit is the weakest trained arm (40.7 ± 3.5 hard problems, below the fixed band's 48) in all three
seeds. Its reward is the improvement in a level's success rate between probes, and the level it trains is the
level that improves, *because* it is trained. The greedy choice therefore pays for itself: it keeps 90% of
every batch and stays on op 12–14 for the whole run. op 15 became learnable anyway through spill-over (probe
success rate 0.31 by step 50 in seed 0) and reached second place in value, but under an argmax second place
gets nothing. After the first value update, op 15 and above were sampled only by ε-exploration, at exactly
the ε floor of 3.2% of prompts, in all three seeds.

A faster probe interval does not help. With probes every 10 steps instead of 25, the bandit locked on
sooner, onto an even easier level (op 12), and op 15+ again received only the ε floor. Measurement lag
therefore makes the problem worse but is not its cause; self-reward is. Probe noise (standard error about
0.09 against value margins of about 0.01) makes *which* of op 12–15 wins partly chance, but the trained
level's head start is not. We read this as a structural limitation of improvement-rate curricula for GRPO,
not a tuning problem.

### 4.7 Costs: reliability on easier problems

τ=0.1's new ability on hard problems comes with a small, consistent loss of reliability on easier ones.
Its easy pass@1 is 83.9 ± 0.2, against 85.3 ± 0.1 for the control, 85.4 for the fixed band and 85.5 for the
base model; its medium pass@1 is 46.7 ± 0.4, against 49.1 for the control and 49.2 for the fixed band. This
is not seed noise. The easy gap is about ten times the seed standard deviation, every cross-seed pair (6 of 6
against the fixed band, 9 of 9 against the control) has τ=0.1 lower, and paired by problem the deficit
against the base model is −1.6 points (95% CI −3.0 to −0.3). It is spread over op 2–10 rather than
concentrated in a few problems.

Reachability is unaffected. Easy pass@128 is 98.5 for τ=0.1, uniform and the control, each loses on average
0.3 easy problems the base model could solve, and τ=0.1's medium pass@128 equals the control's (85.0 vs 84.9).
τ=0.1 is less reliable on easier problems, not less able. The medium gap follows from where it trained: it
left op 11–14 by step 8 and spent only 16% of its groups there, against 57% for the control. The easy gap
does not follow from that alone, since the control also never trains on op 2–10 and stays at the base
model's level. A plausible but untested explanation is that from step 25 onward every τ=0.1 batch is at op 16
or 17 with a success rate below 0.2, so most of its gradient pushes away from failing solutions.

The practical reading: if the right range is known, sampling it uniformly gets 81% of the gain at no
reliability cost; τ=0.1 finds the range and adds a small margin on the hardest problems at a cost of about
1.5 points of easy and 2.5 points of medium pass@1.

## 5. Correcting our own motivation

Our proposal motivated the project with a step-0 measurement: in the base model, op 13 and op 14 give an
informative group 1.5% and 0.02% of the time, so "half the paper's band is dead weight". That is a snapshot
and it does not last. In the fixed-band run, training on op 11–12 generalises upward: op 13 reaches 37% and
op 14 29% informative groups by step 20. After the first 15 steps the band is not wasteful. Its real
limitation is the other end: it never moves past op 14 while op 15–17 become learnable (op 15 goes from never
solved to 27% informative groups by step 50 without being trained on). That is the gap the adaptive samplers
close, and §4.3 shows that closing it is most of what matters.

## 6. Limitations

* **100 steps.** This is half of the paper's run length. The fixed band might catch up with more steps, or the
  gap might widen; we have not tested it.
* **Seeds.** Three seeds for the main arms, two for the fixed band and τ=0.3, one for τ=0.5. The fixed band's
  two seeds agree exactly on the hard set (48 and 48), and τ=0.5's result (no movement, nothing learned) is a
  property of its rule, so we do not expect more seeds to change those conclusions.
* **The control's range is chosen in hindsight**, from τ=0.1's own trajectory. That is the point of the
  control, but it means the "81%" is an upper bound on what a range chosen in advance would achieve.
* **Multiple comparisons.** The tests are not corrected. The τ=0.1 vs uniform result (p = 0.002) survives any
  reasonable correction; τ=0.1 vs the control on the whole hard set (p = 0.017) is borderline under one.
* **One τ grid.** We ran τ ∈ {0.1, 0.3, 0.5}. The learning-curve analysis predicts a peak between 0.1 and 0.2
  but we have not tested values there.
* **One model and one task family.** All results are for the paper's 100M-parameter model on synthetic
  arithmetic graphs.

## 7. Plan for the final report (January)

1. **τ = 0.15 and τ = 0.2.** Test whether arms that train inside the predicted peak (p ≈ 0.10–0.21) beat τ=0.1.
   With δ = 0.1 their move-up bounds (0.25, 0.30) lie above the peak, so these arms need δ = 0.05 (bands
   [0.10, 0.20] and [0.15, 0.25], checked first in the simulator for oscillation) or a rule that targets the
   band directly. About 38 GPU-hours for two seeds each.
2. **Range-finding as a method.** Run τ=0.1 for a short burn-in, then train uniformly on the range it found.
   If this matches τ=0.1, it combines the range-finding with the control's lack of reliability cost.
3. **200-step runs** of τ=0.1, the control and the fixed band.
4. **Bandit fixes**, three seeds each (about 25 GPU-hours per fix): softmax selection instead of argmax,
   optimistic initialisation, and, the one that targets the mechanism, a reward that excludes the improvement
   an arm receives from its own training.
5. **Reliability cost.** Test whether keeping a small share of op 11–14 in τ=0.1's batches removes the easy
   and medium cost without losing the hard-set gain.
6. **Per-problem selection** within a level: problems are mostly always or never solved, so choosing problems
   by their own success rate could raise the informative share well above 40%.

## 8. Engineering and compute

verl runs unmodified; the samplers plug into its curriculum-sampler hook and log their choice, success-rate
estimates and internal state every step. Training, checkpoint conversion, evaluation and analysis run as
chained SLURM jobs. About four days of wall-clock time and 48 GPU-hours were lost to training jobs that hung
silently at startup; the cause was two jobs on the same node colliding on Ray's default ports and temporary
directory, fixed with a private Ray head and port range per job. Since then every job carries a watchdog that
cancels it if its step counter or log has not advanced for 30 minutes, and logs why. The 17 completed
100-step runs took 139 GPU-hours (8.2 per run on one A40); the project total including evaluation, the
bandit ablation, two cancelled runs and the hang is about 220 GPU-hours.
