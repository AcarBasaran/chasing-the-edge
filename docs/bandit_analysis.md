# The bandit's trained arm generates its own reward (2026-11-20)

**Finding.** The bandit's reward is the improvement rate between probes, r_L = p_L(t) − p_L(t−m). The arm it
trains is the arm that improves, *because* it is trained, so the greedy choice pays itself. By step 50,
op 15 had become learnable without the greedy rule ever choosing it: probe p = 0.31 in seed 0 (0.29 and 0.10 in seeds 1
and 2), and its Q was **second** (+0.094 against op 13's +0.102). Under an argmax, second place gets nothing.
op 15 stayed at the ε floor, and op 13, the arm already being trained, kept 90% of every batch. After the
first value update, the greedy choice never went beyond op 14 in any of 3 seeds. The bandit ended below the
fixed band.

**3 seeds (bandit), m = 25.** The m = 10 ablation is running (§4). Figure: `results/bandit_q.png`;
data: `results/bandit_q.csv` (Q, p and probe n per arm per probe), `results/bandit_phases.csv`.
Script: `eval/bandit_q.py`.

## 1. Result

The bandit is the weakest trained arm, below the fixed band, consistently across 3 seeds:

| arm | seeds | hard pass@128 | hard newly solved (of 204) | op 18–20 newly solved | informative groups at op 15–17 |
|---|---|---|---|---|---|
| threshold τ=0.1 | 1 | 46.1 | 94 | 27 | 20,739 |
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
the op 15+ share equals the ε floor exactly. The bandit's useful signal beyond op 14 (161 informative groups
on average over the run) is about 1% of τ=0.1's (20,739).

## 4. Is it just the probe interval? Ablation m = 10 (running)

`configs/rl_bandit_m10.yaml`: identical except probe_every 10, seed 0, 50 steps (constant lr, so steps 1–50
are the first half of a 100-step run; τ=0.1 had reached op 16 by step 22). Sampler behaviour only, no
test-set eval. About 4 GPU-hours.

If the limitation is structural, m = 10 puts the first update at step 10, when op 15–17 are still at p = 0.
The argmax goes to op 12–13 even earlier, and op 15+ again gets only the ε floor. If instead the greedy arm
reaches op 15+ by step 50, the m = 25 result is mostly a slow-probe effect, and this section changes.
Optimistic initialisation of unvisited arms is the other candidate. It is a *fix* rather than a test,
so it stays future work.

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
