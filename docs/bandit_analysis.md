# The bandit's trained arm generates its own reward (2026-11-20)

**Finding.** The bandit's reward is the improvement rate between probes, r_L = p_L(t) − p_L(t−m). The arm it
trains is the arm that improves, *because* it is trained, so the greedy choice pays itself. By step 50,
op 15 had become learnable without the greedy rule ever choosing it: probe p = 0.31 in seed 0 (0.29 and 0.10 in seeds 1
and 2), and its Q was **second** (+0.094 against op 13's +0.102). Under an argmax, second place gets nothing.
op 15 stayed at the ε floor, and op 13, the arm already being trained, kept 90% of every batch. After the
first value update, the greedy choice never went beyond op 14 in any of 3 seeds. The bandit ended below the
fixed band.

**3 seeds (bandit), m = 25, plus one m = 10 ablation (§4): same behaviour.** Figure: `results/bandit_q.png`;
data: `results/bandit_q.csv` (Q, p and probe n per arm per probe), `results/bandit_phases.csv`.
Script: `eval/bandit_q.py`.

## 1. Result

The bandit is the weakest trained arm, below the fixed band, consistently across 3 seeds:

| arm | seeds | hard pass@128 | hard newly solved (of 204) | op 18–20 newly solved | informative groups at op 15–17 |
|---|---|---|---|---|---|
| threshold τ=0.1 | 3 | 46.6 ± 0.8 | 95.0 ± 1.7 | 27.7 ± 0.6 | 20,642 |
| fixed band | 2 | 23.5 ± 0.0 | 48.0 ± 0.0 | 6.0 ± 0.0 | 0 |
| **bandit** | 3 | **19.9 ± 1.7** | **40.7 ± 3.5** | 3.3 ± 1.5 | 161 |

This is a property of its objective, not a bug. We do not tune it away.

## 2. Mechanism: the arm being trained generates its own reward

* **Self-reinforcement (the main effect).** The greedy arm gets 90% of every batch (1 − ε plus its share of
  ε). Training raises its p, the rise is its reward, and the reward keeps it greedy. In seed 0, op 13 went
  from p = 0.28 at step 25 to 0.42 at step 50 under that training and stayed the argmax for the whole run.
  Any other arm can only improve through spill-over from the trained arm, which is smaller by construction.
* **A learnable competitor still loses.** Spill-over did make op 15 learnable: probe p 0.31 / 0.29 / 0.10
  at step 50 (seeds 0/1/2), Q in second or third place (seed 0: op 15 +0.094 vs op 13 +0.102; seed 2 at
  step 75: op 15 +0.090 vs op 12 +0.114). It never won the argmax, so it got no batch share beyond ε.
* **Zero signal at p = 0 (the smaller, earlier effect).** At the first value update (step 25), op 15–17 had
  probe p = 0.00–0.02 at both probes (exactly 0 in seed 0), so Q ≈ 0. That decided who got the first head
  start: op 12–14, which uniform sampling in steps 1–24 had just improved.

**Probe noise.** Each probe measures a level with about 24–70 rollouts, so SE ≈ 0.09 on p ≈ 0.3. The Q
margins at the top are about 0.01. So the *ordering among op 12–15* is partly chance. The trained arm's head
start is not. It comes from the 90% of the batch it receives, not from a lucky probe, and it is present in
all 3 seeds.

## 3. Who sampled op 15+: ε only

Share of training prompts at op 15–20, by phase (ε = 0.1 over 19 levels puts **3.16%** there by itself):

| seed | steps 1–24 (all Q = 0, ties → uniform) | steps 26–99 (Q learned) | steps with the greedy choice at op 15+ | informative op 15+ groups, steps 26–99 |
|---|---|---|---|---|
| 0 | 31.2% | **3.2%** | **0 of 72** | 134 |
| 1 | 31.6% | **3.2%** | **0 of 72** | 137 |
| 2 | 31.9% | **3.2%** | **0 of 72** | 179 |

After the first value update, ε-exploration was the only thing that ever sampled op 15+, in all three seeds:
the op 15+ share equals the ε floor exactly. The bandit's useful signal beyond op 14 (164 informative groups
at op 15–20 per run, mean over 3 seeds) is under 1% of τ=0.1's (20,642, all at op 15–17;
`results/seeds_gradient_budget.csv`).

## 4. Is it just the probe interval? No: ablation m = 10

`configs/rl_bandit_m10.yaml`: identical except probe_every 10. Seed 0, 50 steps (constant lr, so steps 1–50
are the first half of a 100-step run; τ=0.1 had reached op 16 by step 22). Sampler behaviour only, no
test-set eval. 3.7 GPU-hours. Run `bandit_m10_s0`, in `results/bandit_q.{png,csv}` and `results/bandit_phases.csv`.

| | m = 25 (3 seeds) | **m = 10** |
|---|---|---|
| first value update | step 25 | step 10 |
| greedy level after it | op 13 (op 12–14) | **op 12** (steps 10–29), then op 13 |
| training steps with the greedy choice at op 15+ | 0 of 72 per seed | **0 of 36** |
| op 15+ share of prompts after the first update | 3.2% (= ε floor) | **3.2% (= ε floor)** |
| op 15–17 probe p at step 30 / 40 | – | 0.00 / 0.06, 0.00, 0.00 |

Faster probes made the lock-in happen *earlier*, on an *easier* level. With the first update at step 10, the
level that had just risen was op 12 (0.27 → 0.45 in steps 0–10), so the bandit trained op 12, which kept rising under that training (0.52 at step 20). Less training at
op 13–14 also meant less spill-over: op 15 was still at p = 0 at step 30. The failure is not the probe
interval.

## 5. What this says about improvement-rate curricula

Learning-progress curricula (reward = Δ success) are attractive because they need no target success
rate. Two structural properties make them a poor fit for finding new ability with GRPO:

* **Self-reinforcement.** Training on an arm is what makes it improve. Improvement-rate reward therefore
  credits the policy's own choice and locks it in. A level that becomes learnable elsewhere has to beat
  an arm that receives 90% of the training, and with an argmax a close second gets nothing.
* **Zero signal where new ability starts.** A level the model cannot solve yet has p = 0 and r = 0, the
  same as a level it has mastered. This decides the initial head start.

Compare the threshold rule, which reacts to the *level* of p (move up when p > τ + δ). It advances once the
current level is easy enough, whether or not that level is still improving.

## 6. Where on its learning curve does a level improve fastest?

Hypothesis checked (2026-11-20): if a level's S-curve were steepest near p = 0.5, then "train where improvement
is fastest" would be the same target as "train where p ≈ 0.5" (τ=0.5), and the bandit and τ=0.5 would be one
objective reached by different mechanics. **Measured: the curve is steepest at p ≈ 0.1–0.2, not 0.5.**

Figure: `results/scurve.png` (`eval/scurve.py`, data `results/scurve.csv`). It uses the per-step batch p of each
level that starts at p ≈ 0 (~256 groups × 6 rollouts per level per step for the fixed band, training reward, T = 1.0),
with a 5-step mean and the rate measured over ±5 steps. Runs: fixed band seeds 0–1 and uniform seeds 0–1
(`eval/scurve.py` defaults; uniform seed 2 finished later and is not included).

| run | level | steepest rise (Δp per 10 steps) | **p at the steepest point** | plateau p (last 20 steps) |
|---|---|---|---|---|
| fixed band s0 | op 13 | +0.29 (step 5) | **0.19** | 0.46 |
| fixed band s0 | op 14 | +0.20 (step 12) | **0.11** | 0.35 |
| fixed band s1 | op 13 | +0.30 (step 6) | **0.21** | 0.44 |
| fixed band s1 | op 14 | +0.20 (step 12) | **0.11** | 0.35 |
| uniform s0 / s1 | op 14 | +0.15 / +0.19 | **0.15 / 0.18** | 0.33 / 0.32 |
| uniform s0 / s1 | op 15 | +0.13 / +0.11 | **0.10 / 0.12** | 0.26 / 0.28 |

(Higher levels under uniform rise too slowly for a clear peak. op 11–12 start mid-curve.)

What this means:

* **τ=0.1's target band contains the p where op 13–15 learned fastest, but at harder levels the same p
  learns far more slowly.** The steepest rise is at p ≈ 0.10–0.21 for every level that starts at p ≈ 0 and has
  a clear peak here (op 13–15), at 30–50% of the level's plateau. At op 16–17, τ=0.1 itself trained at
  p ≈ 0.11–0.20 and p rose only 0.01–0.05 per 10 steps, 10–40× slower than op 13–14 (docs/results_seeds.md §6).
  That is τ=0.1's target band [0.05, 0.2]. Of the three τ we ran, τ=0.1 is the one whose band contains the
  measured peak, and it is the one that won. (τ between 0.1 and 0.3 was not run, so "best of the three
  tested, and predicted by the curve", not "the optimum". τ = 0.15 and 0.2 are January work.)
* **Levels plateau at p ≈ 0.35–0.58 under the training reward at T = 1.0, not at 1.** The τ=0.5 band
  [0.4, 0.6] is the *ceiling* of these curves, not the half-solved midpoint, and improvement there is ≈ 0.
  τ=0.5 maximises informative groups by sitting on a plateau. It also explains why τ=0.5 never moved: op 11
  levels off at p ≈ 0.57, below the move-up bound τ + δ = 0.6, so the rule's trigger sits above the level's
  ceiling (observed p at op 11: 0.55–0.63, never above 0.6 for 2 steps in a row).
* **The flat bottom holds.** Levels at p ≈ 0 improve at ≈ 0 (fixed band, tracking set: +0.0006 per 10 steps
  over 85 level-intervals at p < 0.02). The improvement signal is weakest exactly where new ability starts.

### Why the bandit misses the steep part: self-reward first, lag second

Two mechanisms could explain it, and the m = 10 ablation separates them.

1. **Self-reward (primary).** The arm being trained improves because it is trained, so its own training
   generates its reward and keeps it greedy (§2). op 15 reached second place in Q and got nothing.
2. **Lag (secondary).** A level's rise is fast (op 13: 0.02 → 0.37 in about 15 steps). Δp over a probe
   interval is measured after much of the rise is over.

**If lag were the main cause, m = 10 should have helped.** It did not (§4). With 2.5× faster probes the bandit
locked on *sooner* (step 10) and onto an *easier* level (op 12). op 15+ again got only the ε floor, and the
greedy choice never went beyond op 14. Faster measurement just let self-reward act earlier, on whichever level
had risen first. So self-reward is the primary mechanism. Lag makes it worse but does not explain it on its own.

τ=0.1 avoids self-reward: it reads the *level* of p, not its change, and moves on as soon as p > 0.2 twice in
a row. It does not avoid stalling. At op 17 its p stayed below 0.2 for the last 46–55 steps of the run, so it
stopped moving (docs/results_seeds.md §6, the stall law).

The three objectives map onto three parts of the same S-curve: **gradient efficiency (τ=0.5) sits on the
plateau, headroom (τ=0.1) climbs through the steep part of op 13–15 and then stalls at op 17, where p sits in
the same range but rises about 0.01 per 10 steps, and learning speed (bandit) is held behind the edge by its own
self-reward, made worse by lag.**

## 7. Fixes (future work, January; not run)

Each needs 3 fresh seeds at about 8 GPU-h each (≈ 25 GPU-h per fix), so none is run before Dec 3.

* **(a) Softmax instead of argmax.** Sample levels in proportion to exp(Q_L / T). op 15 in second place
  (Q +0.094 vs +0.102) would get a batch share close to the leader's instead of nothing. This does not
  remove the self-reward, it only stops the winner from taking everything.
* **(b) Optimistic initialisation of unvisited arms.** Start every Q_L above any reachable reward so each
  level is tried before the bandit settles. This forces exploration of hard levels. But a level at p = 0
  returns r = 0 and drops back, so it does not help with levels that are not learnable yet.
* **(c) Remove the self-reward loop (the real fix: it targets the mechanism).** Either exclude the
  just-trained arm's own rise from its reward (credit an arm only for improvement it did not receive training
  for, e.g. spill-over measured on untrained levels), or mix improvement rate with headroom, e.g.
  r_L = Δp_L + λ · (plateau − p_L) or a bonus for p_L in [0.05, 0.2]. §6 suggests the second version
  points the bandit where τ=0.1 already goes.

