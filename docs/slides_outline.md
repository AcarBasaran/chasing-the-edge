# Dec 3 progress presentation: slide text (final draft, 2026-11-22)

**Spine: three samplers optimise three different objectives (gradient efficiency, learning speed, headroom),
and headroom wins. The right range is most of the win, and the adaptive sampler is what found it.**
13 slides, 10 minutes (≈ 45 s each; slides 1, 6 and 7 are fast).

**Numbers.** All from docs/results_seeds.md: mean ± SD over seeds (τ=0.1, uniform, control, bandit: 3 seeds;
fixed band, τ=0.3: 2; τ=0.5: 1). Footer on every result slide: *"mean ± SD over seeds (n in table); pooled
problem-level sign test; no multiple-comparison correction."* Before the deck, rerun `eval/aggregate_seeds.py`,
`eval/untrained_levels.py`, `eval/compare_methods.py`, `eval/easy_regression.py` and `eval/bandit_q.py`, and
regenerate every figure so the bars show the mean with seed dots.

**Versions.** Slides 12 and 13 come in two versions. **Version A ("the range is most of the win") is primary
and is what the data say**: the control (uniform over op 11–17) gets 81–83% of τ=0.1's gain, and the remaining
gap is small (op 18–20: +3.7, p = 0.11; whole hard set: +8.7, p = 0.017). Version B leads with the schedule's
margin instead. Keep it as a backup only: leading with it would overstate a gap that is not significant on op
18–20.

---

## Act 1: the setup

### Slide 1. Chasing the Edge

* Subtitle: *When GRPO picks its own training difficulty, what should it aim for?*
* Eren Acar Basaran, Turan Berke Yakarer · COMP550 · Dec 3, 2026

### Slide 2. The paper's finding

* **Headline:** RL teaches a small LM genuinely new problems only at its edge of competence.
* Zhang, Neubig & Yue (2025): a 100M-parameter model, synthetic arithmetic problems whose difficulty is the
  number of operations (op 2–20). GRPO creates new ability only when trained on problems the model sometimes
  solves.
* They pick that band once (op 11–14) and keep it fixed for the whole run.
* **Figure:** one short op-5 problem and its "Define …" solution, with "difficulty = number of operations".
* *Speaker note:* reward is process-verified: every reasoning step and the answer must be right.

### Slide 3. Why the edge matters for GRPO

* **Headline:** A group whose rollouts are all right or all wrong gives exactly zero gradient.
* Informative fraction of a group of G = 6 rollouts at success rate p: **1 − pᴳ − (1 − p)ᴳ**.
* So where the model is *right now* decides what it can learn from.
* **Figure:** `results/informative_fraction_by_level.png`, labelled **"base model, step 0"**: op 11 and 12
  informative, op 13–14 almost dead, op 15–20 exactly zero.

### Slide 4. The edge moves; the band does not

* **Headline:** Training on op 11–14 makes op 15–17 learnable, but the fixed band never goes there.
* In the fixed-band run, op 15 goes from never solved to 27% informative groups by step 50, op 16 to 17%,
  op 17 to 7%, without ever being trained on.
* **Figure:** `results/informative_over_training.png`, left panel.
* *Speaker note:* this is the gap an adaptive sampler is meant to close.

## Act 2: three objectives

### Slide 5. Three ways to chase the edge

* **Headline:** Each adaptive sampler optimises something different.
* Three columns:

  | gradient efficiency | learning speed | headroom |
  |---|---|---|
  | **threshold τ=0.5**: stay at the level where p ≈ 0.5 | **bandit**: train the level whose p rose most since the last probe | **threshold τ=0.1**: move up as soon as p > 0.2 |
  | maximises informative groups | maximises measured improvement | trains where the model mostly fails |
  | risk: nothing left to learn there | risk: chases its own improvement | risk: too few informative groups |

* Controls: **fixed band** op 11–14 (the paper), **uniform** over op 2–20, **uniform over op 11–17** (slide 12).
* Footnote: same budget for every arm, 100 steps × 1,024 prompts × 6 rollouts; probe rollouts count.

### Slide 6. Our baseline reproduces the paper

* **Headline:** Our fixed band matches the released checkpoint within 1 point on every test set.
* Table: easy / medium / hard pass@1 and pass@128, ours vs released vs paper Fig. 3 (docs/gate3.md).
  Hard newly solved: **48 of 204 for both.**
* *Speaker note:* this is why the comparisons that follow are against a faithful baseline.

## Act 3: what each objective did

### Slide 7. Where each sampler went

* **Headline:** τ=0.5 never leaves op 11, the bandit locks onto op 12–13, τ=0.3 settles on op 14 (the
  paper's band, found on its own), τ=0.1 climbs to op 17, in every seed.
* **Figure:** `results/sampler_levels.png` (τ=0.1, τ=0.3, τ=0.5 and bandit_s0 panels).
* *Speaker note:* τ=0.1 reaches op 15 at step 16 and op 17 at step 45–54 in all three seeds.

### Slide 8. More informative gradient did not mean more learning   *(new)*

* The whole slide is one line with two pairs of numbers, large:

  > **τ=0.5: 40.1% of groups informative → 0 new hard problems.**
  > **τ=0.1: 24.7% of groups informative → 95 new hard problems.**

* Small print underneath: *share of GRPO groups with a nonzero gradient over the whole run; new = hard
  problems (op 15–20, 204) the base model never solved and the trained model solves at least once in 128.
  τ=0.1: 3 seeds; τ=0.5: 1 seed.*
* No figure.
* *Speaker note:* the arm with the most usable gradient learns nothing new; the arm with the least of any
  adaptive arm learns the most. The next slide says why.

### Slide 9. Why: where a level learns fastest

* **Headline:** A level improves fastest at p ≈ 0.10–0.21, and its success rate tops out at p ≈ 0.35–0.58.
* τ=0.1's target band [0.05, 0.2] contains the steepest point: **it targets where learning is fastest**. That
  makes it the best of the three τ we ran and the one the curve predicts.
* τ=0.5's band [0.4, 0.6] is the *ceiling* (training reward, T = 1.0). It maximises informative groups by
  sitting where nothing is left to learn, and since op 11 tops out at 0.57 its move-up bound (0.6) is never
  reached.
* This corrects our own proposal, which assumed p = 0.5 is the sweet spot.
* **Figure:** `results/scurve.png` (right panel: improvement vs p, τ=0.1 and τ=0.5 bands shaded; left panel
  as backup).
* *Speaker note:* τ between 0.1 and 0.3 is untested; τ = 0.15 and 0.2 are January's test of the prediction.

### Slide 10. Learning speed rewards itself

* **Headline:** The bandit's reward is improvement, and the level it trains improves because it is trained.
* Its greedy choice generates its own reward, keeps 90% of every batch, and stays on op 12–14.
* op 15 became learnable anyway (p = 0.31 by step 50) and reached second place in value, but under an argmax
  second place gets nothing: after step 25, op 15+ got only the ε floor (3.2% of prompts) in all 3 seeds.
* Faster probes do not fix it (m = 10: locked on sooner, onto op 12). Result: **40.7 ± 3.5** new hard problems,
  below the fixed band (48).
* **Figure:** `results/bandit_q.png` (top: value per level, op 15 orange reaches second, never first;
  bottom: batch share, greedy level 90%, op 15+ flat at the ε floor).
* *Speaker note:* probe noise (SE ≈ 0.09 against value margins of ≈ 0.01) makes *which* of op 12–15 wins
  partly chance; the trained level's head start is not.

### Slide 11. Headroom wins

* **Headline:** On op 18–20, levels τ=0.1 never trained on, it newly solves **27.7 ± 0.6 of 102** problems.
* Uniform over op 2–20, which *did* train there: 20.0 ± 1.0. Fixed band: 6. Bandit: 3.3 ± 1.5. τ=0.5: 0.
  The base model: 0.
* τ=0.1 vs uniform, paired by problem across 3 seeds each: better on 15 problems, worse on 2 (p = 0.002);
  +7.7 per seed [95% CI +3.7, +12.3]. Same direction in every seed.
* **Figure:** `results/hard_by_level.png`, both panels.
* *Speaker note:* uniform spent 16% of its prompts on op 18–20, and 97% of those groups had zero gradient.
  Exposure does not teach a level the model cannot yet solve.

## Act 4: what actually produced the gain

### Slide 12A. The right range is most of the win, and τ=0.1 found it   **(primary)**

* **Headline:** Sampling τ=0.1's range uniformly from step 0 gets most of τ=0.1's gain. The adaptive
  sampler's contribution is that it *found* that range.
* **The control.** Uniform over op 11–17, the levels τ=0.1 visited, chosen after seeing where τ=0.1 went.
  Same budget, no schedule.
* The numbers, as one stacked bar per test set ("of τ=0.1's gain over the fixed band"):

  | | whole hard set (+47 problems) | op 18–20 (+21.7) |
  |---|---|---|
  | harder problems at all (uniform op 2–20) | 64% | 65% |
  | the right range (op 11–17) | 18% | 18% |
  | the schedule (τ=0.1 vs the same range) | 18% | 17% |

* **The schedule adds a small margin:** +8.7 hard problems per seed (17 vs 5 problems, p = 0.017); on op
  18–20, +3.7 (8 vs 2, p = 0.11, not significant).
* **Bottom line, in the slide's largest type:**
  > The paper's hand-picked band gets **48**. The range τ=0.1 found on its own gets **86**. With τ=0.1's schedule, **95**.
* Small print: *We would not have tried op 11–17 without τ=0.1. It is the paper's band plus three levels the
  base model cannot solve at all.*
* **Figure:** the stacked bars (new; numbers in docs/results_seeds.md §3), or `results/hard_by_level.png` with
  the control added.
* *Speaker note:* say it straight. The schedule is not where most of the gain comes from. Choosing *which*
  levels to train on is, and the paper's choice (op 11–14) was too low. A sampler that tracks the edge is a
  way to find the right range without a sweep. That is the contribution.

### Slide 13A. What it costs, and what is next   **(primary)**

* **Headline:** τ=0.1 trades a little reliability on easy problems for the most new ability on hard ones.
* **Cost, consistent across seeds:** easy pass@1 83.9 ± 0.2 vs 85.3 for the control and 85.5 for the base
  model; medium pass@1 46.7 vs 49.1. Every one of 9 seed pairs against the control is lower. pass@128 is
  unchanged: τ=0.1 is less reliable, not less able.
* **Practical reading:** if you know the right range, sample it uniformly: 81% of the gain, no reliability
  cost. If you do not, τ=0.1 finds it, and its schedule adds a small margin on the hardest problems.
* **Next (January):**
  * τ = 0.15 and 0.2: do arms that train inside the predicted peak (p ≈ 0.10–0.21) do better? Needs a smaller δ.
  * Range-finding as a method: run τ=0.1 briefly, then train uniformly on the range it found.
  * 200-step runs: does the fixed band catch up, or does the gap widen?
  * Fix the bandit's self-reward loop.
* **Figure:** none, or `results/method_comparison.png` (hard column) as a thumbnail.

### Slide 12B. The schedule matters: τ=0.1 beats its own range sampled uniformly   *(backup)*

* **Headline:** τ=0.1 newly solves more hard problems than uniform sampling over the same levels (op 11–17).
* Whole hard set: **95.0 ± 1.7 vs 86.3 ± 2.1**; paired by problem, better on 17, worse on 5 (p = 0.017),
  +8.7 per seed [95% CI +3.3, +14.3]. Same direction in every seed.
* Where the gap is: op 15 (29.0 vs 26.3) and op 16 (20.0 vs 17.7), the levels τ=0.1 trained *while they
  were at the edge*, and 1–3 problems per level beyond. At op 17, where both trained heavily, they are
  equal (18.3 vs 18.3).
* τ=0.1 put 2.5× more informative gradient on op 15–17 than the control (20,642 vs 8,268 groups).
* **Caveat on the slide itself:** on op 18–20 alone the gap is +3.7 (8 vs 2 problems, p = 0.11), not
  significant. Most of τ=0.1's gain over the fixed band (81%) comes from the range.
* **Figure:** per-level bars, τ=0.1 vs control, op 15–20.

### Slide 13B. What it costs, and what is next   *(backup)*

* **Headline:** The schedule's extra ability costs a little reliability on easy problems.
* Cost: easy pass@1 83.9 vs 85.3 (control) and 85.5 (base); medium 46.7 vs 49.1. Consistent across seeds.
  pass@128 unchanged.
* The control has no such cost, because 57% of its batch stays on op 11–14.
* **Next:** τ = 0.15 / 0.2 (train inside the peak), a schedule that keeps a share of op 11–14 to remove the
  cost, 200-step runs, the bandit fix.

---

## Backup slides

| title | content |
|---|---|
| Easy/medium reliability: seed by seed | table from docs/results_seeds.md §4 (per-seed pass@1, paired CIs, 6/6 and 9/9 seed pairs) |
| All arms, all test sets | `results/method_comparison.png`; docs/results_seeds.md §1 |
| Paired tests, full table | docs/results_seeds.md §2 (both op 18–20 and op 15–20) |
| Steepest point per level and the fixed band's S-curves | `results/scurve.png` (left panel), docs/bandit_analysis.md §6 |
| Bandit fixes for January | docs/bandit_analysis.md §7 |
| Bandit: probe noise vs value margins | `results/bandit_q.csv`; docs/bandit_analysis.md §2 |
| Per-level informative share during training | `results/informative_over_training.png` (right panel) |
| Within-level bimodality | `results/per_problem_p_hist.png` |
| Strict vs answer-only reward on the released checkpoint | docs/gate3.md |
| The step-0 motivation and why it did not last | docs/gate1.md time series |

## Figures to (re)make before the deck

* Slide 5: three-column graphic (no data).
* Slide 7: `eval/plot_sampler_levels.py` with τ=0.1, τ=0.3, τ=0.5 and bandit_s0 only.
* Slide 8: no figure; two lines of large type.
* Slide 11: `results/hard_by_level.png` regenerated with seed dots.
* Slide 12A: new stacked bar (decomposition), from docs/results_seeds.md §3. Slide 12B: per-level bars
  τ=0.1 vs control from `results/seeds_hard_levels.csv`.
* Slide 13: optional `results/method_comparison.png` thumbnail.
