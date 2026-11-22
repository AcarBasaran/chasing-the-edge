# Dec 3 progress presentation: slide text (final, 2026-11-22)

**Spine: three samplers aim at three different things (the most usable training signal, the fastest
improvement, the most room to improve), and "most room to improve" wins. Training on the right range of
difficulty is most of the win, and the adaptive sampler is what found that range.**
13 slides, 10 minutes (≈ 45 s each; slides 1, 6 and 7 are fast). Spoken lines: docs/speaker_notes.md.

**Audience.** A classmate who has not read the paper. Every term is defined on the slide where it first
appears, and the same plain word is used for the same thing throughout:

| we say on slides | means | avoid on slides |
|---|---|---|
| difficulty / level (op 2–20) | number of arithmetic operations in a problem | "edge of competence" without the gloss |
| try / answer | one sampled solution (a "rollout") | rollout, completion |
| success rate p | share of tries at a level that are fully right | – |
| usable signal / "problem that teaches something" | a problem whose 6 tries are not all the same (nonzero GRPO gradient) | informative group, advantage |
| newly solved | the starting model never solved it in 128 tries; the trained model solves it at least once | pass@128 > 0 |
| reliability (pass@1) | share of single tries that are right | – |
| target success rate τ | the threshold sampler's setting | – |

**Numbers.** All from docs/results_seeds.md: mean ± SD over seeds (τ=0.1, uniform, control, bandit: 3 seeds;
fixed band, τ=0.3: 2; τ=0.5: 1). Footer on every result slide: *"mean ± SD over training seeds (n in
table); paired test over problems; not corrected for multiple comparisons."* Training-dynamics numbers that
come from one run say "(seed 0)". Before the deck, rerun `eval/aggregate_seeds.py`, `eval/untrained_levels.py`,
`eval/compare_methods.py`, `eval/easy_regression.py` and `eval/bandit_q.py`, and regenerate every figure so
the bars show the mean with seed dots.

**Versions.** Slides 12 and 13 come in two versions. **Version A ("the range is most of the win") is primary
and is what the data say**: the control (uniform over op 11–17) gets 81–83% of τ=0.1's gain, and the remaining
gap is small (op 18–20: +3.7, p = 0.11; whole hard set: +8.7, p = 0.017). Version B leads with the schedule's
margin instead. Keep it as a backup only: leading with it would overstate a gap that is not significant on op
18–20.

---

## Act 1: the setup

### Slide 1. Chasing the Edge

* Subtitle: *When an RL-trained model picks its own training difficulty, what should it aim for?*
* Eren Acar Basaran, Turan Berke Yakarer · COMP550 · Dec 3, 2026

### Slide 2. The paper we build on

* **Headline:** RL teaches a small model new problems only when it trains on problems it sometimes solves.
* Zhang, Neubig & Yue (2025) train a small (100M-parameter) language model from scratch on made-up
  arithmetic word problems. **Difficulty = number of operations, from 2 to 20.** The model must write out
  every step; an answer counts only if every step and the final number are right.
* Then they fine-tune it with reinforcement learning (RL): the model tries each problem, gets a reward of 1
  for a fully correct answer and 0 otherwise, and is pushed toward what worked.
* Their finding: RL gives the model *new* ability (problems it could never solve before) only if training
  uses problems at its **"edge"**: hard enough that it often fails, easy enough that it sometimes succeeds.
* They pick that band once, **operations 11–14**, and keep it fixed for the whole run.
* **Figure:** one short 5-operation problem and its step-by-step solution, labelled "difficulty = number
  of operations".

### Slide 3. Why the edge matters

* **Headline:** If all six tries agree, the model learns nothing from that problem.
* The RL method (GRPO) answers each problem **6 times** and learns from the *difference* between its right
  and wrong answers. All 6 right, or all 6 wrong: no difference, **zero learning signal**, compute wasted.
* With success rate p, the share of problems that teach something is **1 − p⁶ − (1 − p)⁶**: near 0 when
  the model always fails or always succeeds, highest in between.
* So what the model can learn from depends on where it is **right now**.
* **Figure:** `results/informative_fraction_by_level.png`, titled **"starting model: share of problems that
  teach something, by difficulty"**: operations 11–12 useful, 13–14 almost nothing, 15–20 exactly zero.

### Slide 4. The edge moves; the fixed band does not

* **Headline:** Training on 11–14 makes 15–17 learnable, but the fixed band never goes there.
* In the paper's fixed-band run, levels it *never trains on* start to teach something (seed 0, step 50):
  operation 15 goes from never solved to **27%** of problems teaching something, 16 to **15%**, 17 to **6%**.
* **Figure:** `results/informative_over_training.png`, left panel.
* *Speaker note:* an adaptive sampler could move up to these levels. That is the idea; slide 7 shows how far
  it actually goes.

## Act 2: three things a sampler could aim for

### Slide 5. Three ways to chase the edge

* **Headline:** We built samplers that pick the difficulty of every training batch, each aiming at
  something different.
* Three columns:

  | most usable signal | fastest improvement | most room to improve |
  |---|---|---|
  | **threshold τ=0.5**: stay at the level the model solves about half the time | **bandit**: keep training the level whose success rate rose most recently | **threshold τ=0.1**: move up as soon as the model solves more than 20% |
  | most problems teach something | chases measured progress | trains where the model mostly fails |
  | risk: nothing left to learn there | risk: rewards its own training | risk: few problems teach something |

* τ = the **target success rate**. The threshold sampler trains one level at a time and moves up when the
  success rate there passes τ + 0.1 (for τ=0.1: 0.2).
* Comparisons: **fixed band** 11–14 (the paper), **uniform** over all levels 2–20, and **uniform over
  11–17** (slide 12).
* Footnote: every arm gets the same compute: 100 training steps × 1,024 problems × 6 tries. The samplers'
  own measurements count against it.

### Slide 6. Our baseline reproduces the paper

* **Headline:** Our version of the paper's method matches their released model within 1 point everywhere.
* Table: easy / medium / hard test sets, reliability (pass@1) and solved-at-least-once (pass@128), ours vs
  released vs the paper's figure (docs/gate3.md). Hard problems newly solved: **48 of 204 for both**.
* *Speaker note:* every comparison that follows is against a faithful baseline.

## Act 3: what each sampler did

### Slide 7. Where each sampler went

* **Headline:** τ=0.5 never leaves level 11. The bandit gets stuck at 12–14. τ=0.3 climbs through the
  paper's band and stops at its top (14). τ=0.1 climbs to 17 and stops there. Same in every seed.
* **Each threshold sampler stops at the first level it cannot push past its move-up point (τ + 0.1):**
  level 11 for τ=0.5 (levels off at 0.59, move-up point 0.6), 14 for τ=0.3 (0.37 vs 0.4), 17 for τ=0.1
  (0.17 vs 0.2). τ=0.1 spends the second half of the run (46–55% of its steps) at level 17.
* **Figure:** `results/sampler_levels.png` (τ=0.1, τ=0.3, τ=0.5 and bandit seed 0), y-axis "difficulty
  being trained", x-axis "training step".
* *Speaker note:* τ=0.1 reaches level 15 at step 16 and level 17 at step 45–54 in all three seeds, then
  stays. It adapts while levels are learnable, then stops.

### Slide 8. More learning signal did not mean more learning

* The whole slide is two lines of large type:

  > **τ=0.5: 40% of problems teach something → 0 new hard problems solved.**
  > **τ=0.1: 25% of problems teach something → 95 new hard problems solved.**

* Small print: *share of training problems whose 6 tries were not all the same, over the whole run; new =
  hard test problems (204, levels 15–20) the starting model never solved in 128 tries and the trained model
  solves at least once. τ=0.1: 3 seeds; τ=0.5: 1 seed.*
* No figure.
* *Speaker note:* the arm with the most usable signal learns nothing new. The next slide says why.

### Slide 9. Why: where a level learns fastest

* **Headline:** Levels 13–15 improve fastest when the model solves them 10–20% of the time, and levels stop
  improving at about 50%.
* τ=0.1 trains levels 13–15 in that fast stretch and moves on. At 16–17 the success rate sits in the same
  range but rises 10–40× more slowly, and at 17 it never reaches the move-up point.
* τ=0.5 trains where the level has already stopped improving. The model's success rate at level 11 levels
  off at 0.57, just below τ=0.5's move-up point (0.6), so it never moves.
* This corrects our own proposal, which assumed "solved half the time" is the sweet spot.
* **Figure:** `results/scurve.png`, right panel: "how fast a level improves" against "current success
  rate", with τ=0.1's and τ=0.5's target ranges shaded.
* *Speaker note:* targets between 0.1 and 0.3 are untested; 0.15 and 0.2 are January's test of this.

### Slide 10. Chasing improvement rewards itself

* **Headline:** The bandit rewards a level for improving, and the level it trains improves *because* it
  trains it.
* So its first choice keeps 90% of every batch and stays at level 12–14 for the whole run.
* Level 15 became learnable anyway (success rate 0.10–0.31 by step 50, depending on the seed) and came a
  close second, but the bandit always takes the top choice: after step 25, levels 15 and up got only the
  3% of problems the bandit picks at random, in all 3 seeds.
* Measuring progress more often does not help (every 10 steps instead of 25: it got stuck sooner, on an
  easier level). Result: **40.7 ± 3.5** new hard problems, *below* the fixed band (48).
* **Figure:** `results/bandit_q.png` (top: each level's score, level 15 in orange comes second and never
  first; bottom: share of each batch).

### Slide 11. Room to improve wins

* **Headline:** On the three hardest levels (18–20), which τ=0.1 never trained on, it newly solves
  **27.7 ± 0.6 of 102** problems.
* Uniform over all levels, which *did* train on 18–20: 20.0 ± 1.0. Fixed band: 6. Bandit: 3.3. τ=0.5: 0.
  The starting model: 0.
* Problem by problem, over 3 seeds each: τ=0.1 solves 15 problems more often than uniform, uniform 2 more
  often than τ=0.1 (p = 0.002). Same direction in every seed.
* **Figure:** `results/hard_by_level.png`, both panels.
* *Speaker note:* uniform spent 16% of its problems on levels 18–20, and 97% of them taught nothing: all 6
  tries failed. Seeing a level you cannot solve at all teaches nothing.

## Act 4: what actually produced the gain

### Slide 12A. The right range is most of the win, and τ=0.1 found it   **(primary)**

* **Headline:** Training uniformly on the levels τ=0.1 visited gets most of its gain. The adaptive
  sampler's contribution is that it *found* those levels.
* **The control.** Uniform over levels 11–17 (where τ=0.1 went) from the first step: same levels, same
  compute, no step-by-step climb. We chose this range after seeing where τ=0.1 went.
* One stacked bar per test set, "share of τ=0.1's gain over the fixed band":

  | | whole hard set (+47 problems) | levels 18–20 (+21.7) |
  |---|---|---|
  | training on harder problems at all (uniform 2–20) | 64% | 65% |
  | the right range (11–17) | 18% | 18% |
  | the climb (τ=0.1 vs the same range) | 18% | 17% |

* **The climb adds a small margin:** +8.7 hard problems per seed (better on 17 problems, worse on 5,
  p = 0.017); on levels 18–20, +3.7 (8 vs 2, p = 0.11, not significant).
* **Bottom line, in the slide's largest type:**
  > The paper's hand-picked band: **48**. The range τ=0.1 found on its own: **86**. With τ=0.1's climb: **95**.
* Small print: *We would not have tried 11–17 without τ=0.1: it is the paper's band plus three levels the
  starting model cannot solve at all.*
* **Figure:** the stacked bars (new; numbers in docs/results_seeds.md §3).

### Slide 13A. What it costs, and what is next   **(primary)**

* **Headline:** τ=0.1 gives up a little reliability on easy problems for the most new ability on hard ones.
* **Cost, same in every seed:** easy problems solved per try 83.9% vs 85.3% for the control and 85.5% for
  the starting model; medium 46.7% vs 49.1%. But the share of easy problems solved at least once is
  unchanged: τ=0.1 is less *reliable*, not less *able*.
* **Practical reading:** if you know the right range, sample it uniformly: 81% of the gain, no cost. If you
  don't, τ=0.1 finds it, and its climb adds a small margin on the hardest problems.
* **Next (January):**
  * Targets 0.15 and 0.2, as a test of the stall law: it predicts they stop earlier (level 15–16), so they
    should do worse if the range is what matters.
  * Find the range with τ=0.1 briefly, then train uniformly on it.
  * 200-step runs: does the fixed band catch up, and does τ=0.1 get past level 17?
  * Fix the bandit so it can't reward itself.

### Slide 12B. The climb matters: τ=0.1 beats its own range sampled uniformly   *(backup)*

* **Headline:** τ=0.1 newly solves more hard problems than uniform sampling over the same levels (11–17).
* Whole hard set: **95.0 ± 1.7 vs 86.3 ± 2.1**; problem by problem, better on 17, worse on 5 (p = 0.017),
  +8.7 per seed [95% CI +3.3, +14.3]. Same direction in every seed.
* Where the gap is: levels 15 (29.0 vs 26.3) and 16 (20.0 vs 17.7), the levels τ=0.1 trained while it
  was still climbing, plus 1–3 problems per level beyond. At level 17, where both trained heavily, they tie
  (18.3 each).
* τ=0.1 put 2.5× more useful training signal on levels 15–17 than the control (20,642 vs 8,268 problems
  that taught something).
* **Caveat on the slide itself:** on levels 18–20 alone the gap is +3.7 (8 vs 2 problems, p = 0.11), not
  significant. Most of τ=0.1's gain over the fixed band (81%) comes from the range.
* **Figure:** per-level bars, τ=0.1 vs control, levels 15–20.

### Slide 13B. What it costs, and what is next   *(backup)*

* **Headline:** The climb's extra ability costs a little reliability on easy problems.
* Cost: easy 83.9% vs 85.3% (control) and 85.5% (starting model); medium 46.7% vs 49.1%. Same in every seed.
  Solved-at-least-once unchanged.
* The control has no such cost, because 57% of its training stays on levels 11–14.
* **Next:** targets 0.15 / 0.2 (a test of the stall law), a climb that keeps some 11–14 problems in every
  batch, 200-step runs, the bandit fix.

---

## Backup slides

| title | content |
|---|---|
| How adaptive is τ=0.1? Stall law, steps climbing vs stalled, rise per level | docs/results_seeds.md §6 |
| Easy/medium reliability: seed by seed | table from docs/results_seeds.md §4 (per-seed pass@1, paired CIs, 6/6 and 9/9 seed pairs) |
| All arms, all test sets | `results/method_comparison.png`; docs/results_seeds.md §1 |
| Paired tests, full table | docs/results_seeds.md §2 (both levels 18–20 and 15–20) |
| Steepest point per level and the fixed band's learning curves | `results/scurve.png` (left panel), docs/bandit_analysis.md §6 |
| Bandit fixes for January | docs/bandit_analysis.md §7 |
| Bandit: measurement noise vs score margins | `results/bandit_q.csv`; docs/bandit_analysis.md §2 |
| Per-level share of problems that teach something, during training | `results/informative_over_training.png` (right panel) |
| Problems within a level are mostly always or never solved | `results/per_problem_p_hist.png` |
| Strict vs answer-only reward on the released checkpoint | docs/gate3.md |
| The step-0 motivation and why it did not last | docs/gate1.md time series |

## Figures to (re)make before the deck

**Projector check (2026-11-22).** Rule: text ≥ 14 pt once the figure is scaled to slide width (13.3 in), i.e. a
figure W inches wide needs fonts ≥ 14 × W / 13.3 pt. When regenerating, relabel axes in the slide vocabulary
above ("difficulty (operations)", "share of problems that teach something", "solved at least once").

| figure | slide | status | fix |
|---|---|---|---|
| `hard_by_level.png` | 11 | **OK** (regenerated: 13 in, 10–12 pt, labels to one decimal, n per arm) | – |
| `sampler_levels.png` | 7 | **illegible**: all 16 runs side by side (61 in wide), 6 pt | rerun `eval/plot_sampler_levels.py` with 4 runs (τ=0.1, τ=0.3, τ=0.5 s0, bandit_s0) and fonts ≥ 12 |
| `bandit_q.png` | 10 | **too small** (4 columns, 17 in, 7 pt ≈ 5.6 pt on screen); "greedy opNN" notes overlap the Q steps; end labels op13/op14 crowd | one column (bandit_s0) + optionally m10; fonts ≥ 12; greedy notes above the plot area |
| `scurve.png` | 9 | **small** (12 in, 8–9 pt ≈ 9.5 pt); τ-band labels light grey on tint; legend sits on the op13/14 traces | right panel alone at slide size, fonts ≥ 12, legend outside, band labels in ink |
| `informative_over_training.png` | 4 | **small** (10.5 in, 6.8–8 pt); left panel's end labels (op9/op11/op12/op10) are stacked away from their lines and read out of order | left panel alone, fonts ≥ 12, label only op11–14 (grouped) and op15–17 |
| `informative_fraction_by_level.png` | 3 | **mostly OK** (8 in, ≈ 12–14 pt); "paper's fixed band" label touches the dashed curves; 5-entry legend is more than slide 3 needs | drop the two dashed level-mean curves and the T=0.7 line for the slide version |
| `method_comparison.png` | 13 thumbnail / backup | **small** (6 panels, 6.5–8 pt); value labels collide with seed dots | for slide 13 use a hard-column-only version; full figure stays backup |
| `per_problem_p_hist.png` | backup | small (≈ 9 pt) but uncluttered | acceptable for backup |
| decomposition bar (12A), per-level bars (12B) | 12 | **do not exist yet** | new; numbers in docs/results_seeds.md §3 |

* Slide 5: three-column graphic (no data).
* Slide 8: no figure; two lines of large type.
* Slide 12A: new stacked bar (decomposition), from docs/results_seeds.md §3. Slide 12B: per-level bars
  τ=0.1 vs control from `results/seeds_hard_levels.csv`.
