# Claims changelog, 2026-11-22: how adaptive is τ=0.1?

Every sentence changed while removing "tracks / follows the edge throughout training" wording and adding the
stall law. Evidence: docs/results_seeds.md §6, `eval/stall_law.py`, `results/stall_law*.csv`.

## docs/results_seeds.md: states how long τ=0.1 sat at its final level

**Before:**

> * **τ=0.1's path is the same in all three seeds:** op 11 → 14 by step 7–8, op 15 at step 16, op 16 at
>   step 22–25, op 17 at step 45–54, then op 17 to the end (batch p at op 17 ≈ 0.17, below the 0.2 move-up bound).

**After:**

> * **τ=0.1's path is the same in all three seeds:** op 11 → 14 by step 7–8, op 15 at step 16, op 16 at
>   step 22–25, op 17 at step 45–54, then op 17 to the end, 46–55% of the run (batch p at op 17 ≈ 0.17, below
>   the 0.2 move-up trigger; §6).

## docs/results_seeds.md: 'when they were at the edge' implied tracking; replaced with what happened

**Before:**

> the levels τ=0.1 trained *when they were at the edge* (op 15–16) and carries one to three problems per
>   level beyond.

**After:**

> the levels τ=0.1 trained while it was still climbing (op 15–16, steps 16–54) and carries one to three
>   problems per level beyond.

## docs/results_seeds.md: adds the honest claim where the contribution is stated

**Before:**

> * **What the adaptive method contributes is finding the range.** The paper's hand-picked band (op 11–14)

**After:**

> * **What the adaptive method contributes is finding the range.** It does not track the edge for the whole
>   run: it climbs while levels are learnable and stops at op 17, the first level whose success rate stays below
>   its 0.2 trigger (§6). The paper's hand-picked band (op 11–14)

## docs/results_seeds.md: adds the stall law to the claims

**Before:**

> 4. **The adaptive method's contribution is finding the range without being told.** The range it found
>    (op 11–17) beats the paper's hand-picked band by 38 hard problems even without the schedule.

**After:**

> 4. **The adaptive method's contribution is finding the range without being told.** The range it found
>    (op 11–17) beats the paper's hand-picked band by 38 hard problems even without the schedule.
> 5. **Stall law** (§6, all 6 threshold runs): a threshold arm climbs while the trained level can reach its
>    move-up trigger τ + δ and stops at the first level whose success rate stays below it. τ=0.1 is therefore
>    adaptive in the first half of the run only (46–55% of its steps at op 17). The law is the hypothesis for
>    the January τ = 0.15 and 0.2 runs.

## docs/results_seeds.md: new §6: stall law, adaptivity share, first look at timing, January hypothesis

**Before:**

> (none: appended)

**After:**

> ## 6. How adaptive is τ=0.1? The stall law (2026-11-22)

## docs/results_seed0.md: 'samplers that follow the edge' overstated; τ=0.1 climbs then stops

**Before:**

> The samplers that follow the edge train exactly
> there.

**After:**

> τ=0.1 trains
> there (op 15–16 while climbing, then op 17 for the rest of the run; results_seeds.md §6).

## docs/bandit_analysis.md: headline overclaimed: τ=0.1 does not sit on the steep part at op 16–17

**Before:**

> * **τ=0.1's target is where learning is fastest, which makes it a principled choice, not a lucky one.** The
>   steepest rise is at p ≈ 0.10–0.21 for every level that starts at p ≈ 0, at 30–50% of the level's plateau.

**After:**

> * **τ=0.1's target band contains the p where op 13–15 learned fastest, but at harder levels the same p
>   learns far more slowly.** The steepest rise is at p ≈ 0.10–0.21 for every level that starts at p ≈ 0 and has
>   a clear peak here (op 13–15), at 30–50% of the level's plateau. At op 16–17, τ=0.1 itself trained at
>   p ≈ 0.11–0.20 and p rose only 0.01–0.05 per 10 steps, 10–40× slower than op 13–14 (docs/results_seeds.md §6).

## docs/bandit_analysis.md: 'avoids both' was too strong; τ=0.1 stalls for half the run

**Before:**

> τ=0.1 avoids both. It reads the *level* of p, not its change, and it moves on as soon as p > 0.2, before the
> level it trained on has had time to reward itself.

**After:**

> τ=0.1 avoids self-reward: it reads the *level* of p, not its change, and moves on as soon as p > 0.2 twice in
> a row. It does not avoid stalling. At op 17 its p stayed below 0.2 for the last 46–55 steps of the run, so it
> stopped moving (docs/results_seeds.md §6, the stall law).

## docs/bandit_analysis.md: 'sits on the steep part' was true for op 13–15 only

**Before:**

> plateau, headroom (τ=0.1) sits on the steep part, and learning speed (bandit) is held behind the edge by its
> own self-reward, made worse by lag.**

**After:**

> plateau, headroom (τ=0.1) climbs through the steep part of op 13–15 and then stalls at op 17, where p sits in
> the same range but rises about 0.01 per 10 steps, and learning speed (bandit) is held behind the edge by its own
> self-reward, made worse by lag.**

## README.md: 'samplers that track the moving edge' claimed what τ=0.1 does only early

**Before:**

> **Our extension** replaces the fixed band with samplers that track the moving edge: a threshold rule,
> a bandit, and a uniform-random control.

**After:**

> **Our extension** replaces the fixed band with samplers that set each batch's difficulty from the model's
> measured success rates (a threshold rule and a bandit), plus uniform-random controls.

## README.md: 'follows the edge' → neutral

**Before:**

> A sampler that follows the edge
> can move there.

**After:**

> An adaptive sampler
> can move there.

## README.md: adds the honest claim and the stall law

**Before:**

>   range. τ=0.5 (40% informative groups) learns nothing new;

**After:**

>   range. τ=0.1 is adaptive mainly early: it climbs op 11 → 17 in the first 45–54 steps and then stays at op 17,
>   the first level whose success rate stays below its 0.2 move-up trigger. This *stall law* predicts the final
>   level of all six threshold runs ([docs/results_seeds.md](docs/results_seeds.md) §6). τ=0.5 (40% informative
>   groups) learns nothing new;

## README.md: fixes a duplicated 'lower' from the earlier edit

**Before:**

> Cost: τ=0.1's easy pass@1 is 1.5 points lower,
>   lower than the fixed band's and the control's,
>   consistently across seeds;

**After:**

> Cost: τ=0.1's easy pass@1 is 1.5 points lower than the
>   fixed band's and the control's, consistently across seeds;

## docs/slides_outline.md: 'follows the edge' → neutral; points to the stall

**Before:**

> * *Speaker note:* a sampler that follows the edge could move up to these levels. That is the idea.

**After:**

> * *Speaker note:* an adaptive sampler could move up to these levels. That is the idea; slide 7 shows how far
>   it actually goes.

## docs/slides_outline.md: adds the stall law as a stated finding

**Before:**

> * **Headline:** τ=0.5 never leaves level 11. The bandit gets stuck at 12–14. τ=0.3 climbs through the
>   paper's band and stops at its top (14). τ=0.1 climbs to 17. Same in every seed.

**After:**

> * **Headline:** τ=0.5 never leaves level 11. The bandit gets stuck at 12–14. τ=0.3 climbs through the
>   paper's band and stops at its top (14). τ=0.1 climbs to 17 and stops there. Same in every seed.
> * **Each threshold sampler stops at the first level it cannot push past its move-up point (τ + 0.1):**
>   level 11 for τ=0.5 (levels off at 0.59, move-up point 0.6), 14 for τ=0.3 (0.37 vs 0.4), 17 for τ=0.1
>   (0.17 vs 0.2). τ=0.1 spends the second half of the run (46–55% of its steps) at level 17.

## docs/slides_outline.md: states that τ=0.1 stops adapting

**Before:**

> * *Speaker note:* τ=0.1 reaches level 15 at step 16 and level 17 at step 45–54 in all three seeds.

**After:**

> * *Speaker note:* τ=0.1 reaches level 15 at step 16 and level 17 at step 45–54 in all three seeds, then
>   stays. It adapts while levels are learnable, then stops.

## docs/slides_outline.md: 'trains each level in that fast stretch' was true for 13–15 only

**Before:**

> * **Headline:** A level improves fastest when the model solves it 10–20% of the time, and stops improving
>   at about 50%.
> * τ=0.1 trains each level exactly in that fast stretch, then moves on.

**After:**

> * **Headline:** Levels 13–15 improve fastest when the model solves them 10–20% of the time, and levels stop
>   improving at about 50%.
> * τ=0.1 trains levels 13–15 in that fast stretch and moves on. At 16–17 the success rate sits in the same
>   range but rises 10–40× more slowly, and at 17 it never reaches the move-up point.

## docs/slides_outline.md: January τ runs framed as a test of the stall law

**Before:**

>   * Targets 0.15 and 0.2: train right inside the fast-improvement stretch.

**After:**

>   * Targets 0.15 and 0.2, as a test of the stall law: it predicts they stop earlier (level 15–16), so they
>     should do worse if the range is what matters.

## docs/slides_outline.md: 200-step runs also resolve ceiling vs slow climb at 17

**Before:**

>   * 200-step runs: does the fixed band catch up?

**After:**

>   * 200-step runs: does the fixed band catch up, and does τ=0.1 get past level 17?

## docs/slides_outline.md: 'while they were at the edge' → what happened

**Before:**

> levels 15 (29.0 vs 26.3) and 16 (20.0 vs 17.7), the levels τ=0.1 trained *while they
>   were at the edge*, plus

**After:**

> levels 15 (29.0 vs 26.3) and 16 (20.0 vs 17.7), the levels τ=0.1 trained while it
>   was still climbing, plus

## docs/slides_outline.md: same, backup version

**Before:**

> * **Next:** targets 0.15 / 0.2, a climb that keeps some 11–14 problems in every batch, 200-step runs, the
>   bandit fix.

**After:**

> * **Next:** targets 0.15 / 0.2 (a test of the stall law), a climb that keeps some 11–14 problems in every
>   batch, 200-step runs, the bandit fix.

## docs/slides_outline.md: new backup slide

**Before:**

> | Easy/medium reliability: seed by seed |

**After:**

> | How adaptive is τ=0.1? Stall law, steps climbing vs stalled, rise per level | docs/results_seeds.md §6 |
> | Easy/medium reliability: seed by seed |

## docs/speaker_notes.md: adds the stall law to slide 7's line

**Before:**

> and τ=0.1 climbs to 17, in every seed." | 0:35 |

**After:**

> and τ=0.1 climbs to 17; each stops at the first level it can't push past its move-up point, so τ=0.1 adapts for the first half of the run and then stays put." | 0:35 |

## docs/speaker_notes.md: 'trains in the fast stretch' → 'while it lasts'

**Before:**

> so τ=0.1 trains in the fast stretch and τ=0.5 trains where nothing is left, which corrects our own proposal." | 0:55 |

**After:**

> so τ=0.1 climbs through that fast stretch while it lasts and τ=0.5 trains where nothing is left, which corrects our own proposal." | 0:55 |

## docs/speaker_notes.md: Q1 answer qualified; new likely question

**Before:**

> p = 0.017), but that is the smaller part, and we say so.
> 

**After:**

> p = 0.017), but that is the smaller part, and we say so. And τ=0.1 does not track the edge all the way: it
> climbs for the first 45–54 steps and then stays at 17, the first level it can't push past 0.2.
> 
> **"Does τ=0.1 really follow the edge?"**
> For the first half of the run. It climbs while the level it trains can pass its move-up point of 0.2, and
> stops at the first level that can't: level 17, for the last 46–55% of the run, in every seed. The same rule
> explains where τ=0.3 (14) and τ=0.5 (11) stop. Its value is the range it finds on the way up. Whether 17 is a
> real ceiling or just slow is what the 200-step runs will tell us.
> 

## docs/progress_report.md: abstract: adds the honest claim

**Before:**

> The adaptive sampler's main contribution is that it found that range
> without being told: the paper's hand-picked band was too low.

**After:**

> τ=0.1 is adaptive mainly early: it climbs while
> levels are learnable and stops at op 17, the first level whose success rate stays below its 0.2 move-up
> trigger, for the last 46–55% of the run. Its main contribution is that it found that range without being told:
> the paper's hand-picked band was too low.

## docs/progress_report.md: scopes the steepest-point claim to the levels it was measured on

**Before:**

>    are steepest at p ≈ 0.10–0.21 and level off at 0.35–0.58,

**After:**

>    are steepest at p ≈ 0.10–0.21 (op 13–15) and level off at 0.35–0.58,

## docs/progress_report.md: contributions: adds the stall law

**Before:**

>    simulated-model harness that caught an oscillation bug before any GPU time was spent, and 17 completed runs
>    with every sampler decision logged.

**After:**

>    simulated-model harness that caught an oscillation bug before any GPU time was spent, and 17 completed runs
>    with every sampler decision logged.
> 6. **A stall law for threshold curricula**: each threshold arm climbs while the level it trains can reach its
>    move-up trigger τ + δ and stops at the first level whose success rate stays below it. Ceilings measured in
>    other arms predict all three final levels. It also says how adaptive τ=0.1 really is (the first half of the
>    run), and it is the hypothesis for the January τ values.

## docs/progress_report.md: 'a sampler that tracks the moving edge' → describes the mechanism without presupposing it works

**Before:**

> become learnable. **Our question is whether a sampler that tracks the moving edge creates more new ability

**After:**

> become learnable. **Our question is whether a sampler that moves the training difficulty up as the model
> learns creates more new ability

## docs/progress_report.md: quantifies the stall

**Before:**

> at step 16, op 16 at step 22–25 and op 17 at step 45–54, and then stayed at op 17, where its success rate
> hovered around 0.17, below the move-up bound.

**After:**

> at step 16, op 16 at step 22–25 and op 17 at step 45–54, and then stayed at op 17 for the remaining 46–55
> steps, where its success rate hovered around 0.17, below the 0.2 move-up trigger (§6.8).

## docs/progress_report.md: 'while those levels were at the edge' → what happened

**Before:**

> τ=0.1 instead spent 84% of its groups on op 15–17 while those levels were at the edge, with 20,642

**After:**

> τ=0.1 instead spent 84% of its groups on op 15–17 at success rates of 0.10–0.21 (op 15–16 while it
> climbed, op 17 for the second half of the run), with 20,642

## docs/progress_report.md: same

**Before:**

> op 15 (29.0 vs 26.3) and op 16 (20.0 vs 17.7), the levels τ=0.1 trained intensively *while they were at the
> edge*.

**After:**

> op 15 (29.0 vs 26.3) and op 16 (20.0 vs 17.7), the levels τ=0.1 trained intensively while it was still
> climbing.

## docs/progress_report.md: 'a sampler that follows the edge' → the honest claim

**Before:**

> A sampler that follows
> the edge is a way to find the right range in one run instead of a sweep over ranges.

**After:**

> A threshold sampler
> is a way to find a good range in one run instead of a sweep over ranges. It does not follow the edge for the
> whole run: it climbs while levels are learnable, stops at the first level it cannot push past its trigger
> (§6.8), and the levels it passed through are the range.

## docs/progress_report.md: 'trains each level during its fastest improvement' was true for op 13–15 only

**Before:**

> This explains both ends of the τ sweep. τ=0.1's target band, [0.05, 0.2], contains the steepest point: it
> trains each level during its fastest improvement and moves on as the curve starts to flatten.

**After:**

> This explains both ends of the τ sweep. τ=0.1's target band, [0.05, 0.2], contains the steepest point
> measured for op 13–15, and on those levels τ=0.1 trained during the fastest improvement and moved on. The
> steepness does not carry over to harder levels: at op 16–17 τ=0.1 trained at p ≈ 0.11–0.20, but p rose only
> 0.01–0.05 per 10 steps there, against 0.31–0.43 at op 13–14 (§6.8).

## docs/progress_report.md: says when the gap is closed

**Before:**

> That is the gap the adaptive samplers
> close, and §6.3 shows that closing it is most of what matters.

**After:**

> That is the gap τ=0.1 closes
> in its first 16–25 steps (op 15 at step 16, op 16 at 22–25), and §6.3 shows that closing it is most of what
> matters.

## docs/progress_report.md: limitations: adds the stall

**Before:**

> * **One τ grid.** We ran τ ∈ {0.1, 0.3, 0.5}. The learning-curve analysis predicts a peak between 0.1 and 0.2
>   but we have not tested values there.

**After:**

> * **One τ grid.** We ran τ ∈ {0.1, 0.3, 0.5}. The learning-curve analysis predicts a peak between 0.1 and 0.2,
>   and the stall law (§6.8) predicts that a higher trigger stops at a lower level; we have not tested values
>   there.
> * **Adaptive mainly early.** τ=0.1 changes level only in its first 45–54 steps, and τ=0.3 in its first
>   23–25; both then train one level to the end. Whether τ=0.1's stall at op 17 is a ceiling or a climb too
>   slow for 100 steps is open (§6.8).

## docs/progress_report.md: January plan: stall law as the hypothesis (original text kept)

**Before:**

> 1. **τ = 0.15 and τ = 0.2.** Test whether arms that train inside the predicted peak (p ≈ 0.10–0.21) beat τ=0.1.

**After:**

> 1. **τ = 0.15 and τ = 0.2, as a test of the stall law (§6.8).** Hypothesis: the trigger h = τ + δ, not τ, sets
>    the final level. Ceilings measured in other arms predict h = 0.25 → op 16 and h = 0.30 → op 15–16, so
>    with δ = 0.1 both arms should stop below op 17 and, if the range drives the gain, newly solve fewer hard
>    problems than τ=0.1. With δ = 0.05, τ = 0.15 has τ=0.1's trigger (0.20) and should repeat its path; a
>    trigger below 0.2 is the arm predicted to climb past op 17. Also test whether arms that train inside the
>    predicted peak (p ≈ 0.10–0.21) beat τ=0.1.

## docs/progress_report.md: 200-step runs: adds the ceiling-vs-slow-climb prediction

**Before:**

> 3. **200-step runs** of τ=0.1, the control and the fixed band.

**After:**

> 3. **200-step runs** of τ=0.1, the control and the fixed band. Prediction: if op 17 keeps rising at its
>    measured ≈ 0.01 per 10 steps, τ=0.1 crosses 0.2 and moves to op 18 around step 125; if op 17 has a
>    ceiling below 0.2, it stays.

## docs/progress_report.md: new §6.8: the stall law, adaptivity share, timing preview

**Before:**

> 
> ## 7. Correcting our own motivation

**After:**

> 
> ### 6.8 How adaptive is τ=0.1? The stall law
> 
> τ=0.1 does not track the edge for the whole run. From the sampler logs of all six threshold runs
> (`eval/stall_law.py`):
> 
> | arm (seeds) | move-up trigger h = τ + δ | stopped at | reached at step | share of the run there | batch p there, last 20 steps | slope, last 30 steps (per 10 steps) |
> |---|---|---|---|---|---|---|
> | τ=0.1 (3) | 0.2 | op 17 | 45–54 | 46–55% | 0.174–0.178 | +0.006 to +0.009 |
> | τ=0.3 (2) | 0.4 | op 14 | 23–25 | 75–77% | 0.369–0.373 | +0.002 to +0.004 |
> | τ=0.5 (1) | 0.6 | op 11 | 0 | 100% | 0.588 | +0.005 |
> 
> **Stall law: a threshold arm climbs while the level it trains can reach its trigger and stops at the first
> level whose success rate stays below it.** No final level ever had the two consecutive measurements above h
> that a move needs. Ceilings measured independently in the fixed band and the control (op 11 0.57, op 12 0.54,
> op 13 0.44, op 14 0.35, op 15 0.31, op 16 0.21, op 17 0.15) predict all three stopping levels. For τ=0.3 and
> τ=0.5 the final level is flat, a true plateau below the trigger. For τ=0.1 it is not demonstrably a plateau:
> op 17 was still rising by about 0.01 per 10 steps at step 100, so a ceiling just under 0.2 and a climb too
> slow for the run are indistinguishable here.
> 
> The climb slowed level by level. τ=0.1's success rate rose by 0.31–0.43 per 10 steps at op 13–14, 0.12–0.17
> at op 15, 0.03–0.05 at op 16 and about 0.01 at op 17. op 11 → 15 took 16 steps, op 16 took 23–29, and op 17
> the remaining 46–55. The honest description is: **τ=0.1 climbs while levels are learnable, stops at the first
> level whose success rate stays below its trigger, and its main contribution is the range it passed through on
> the way** (§6.3). The stall law is the hypothesis for the January τ = 0.15 and 0.2 runs (§9).
> 
> When the gain over the control arose is not settled yet. In the tracking set (32 problems per level, T = 1),
> τ=0.1's lead on op 15–16 opens while it climbs (steps 20–50) and closes by step 80–100. Its lead on op 17 grows
> during the stall. On op 18–20 it leads from step 30 on by a small, roughly steady margin. Evaluating the
> step-25/50/75 checkpoints on the hard set will answer this with the headline metric.
> 
> ## 7. Correcting our own motivation

# Round 2 (2026-11-22): δ = 0.1 for January, 'success rate ≠ learning speed' as its own finding, LaTeX

## docs/results_seeds.md: promoted from a scoping caveat to its own finding, with the implication for success-rate curricula

**Before:**

> **This qualifies the learning-curve finding** (docs/bandit_analysis.md §6). At op 16–17 τ=0.1 trained at p ≈
> 0.11–0.20, the range where op 13–15 rose fastest, yet p rose 10–40× more slowly than at op 13–14. The
> steepest-point estimate p ≈ 0.10–0.21 comes from op 13–15. At harder levels, a success rate in that range
> does not by itself mean fast learning.

**After:**

> **Finding: success rate alone does not predict learning speed.** At the same success rate, harder levels
> learn more slowly. τ=0.1 trained every level from op 13 to op 17 in the same band of success rates (p ≈
> 0.10–0.21), yet the rise per 10 steps fell from 0.31–0.43 at op 13–14 to 0.12–0.17 at op 15, 0.03–0.05 at
> op 16 and about 0.01 at op 17. The same p means 10–40× slower learning two or three levels higher. The
> steepest-point estimate p ≈ 0.10–0.21 (docs/bandit_analysis.md §6) holds for op 13–15, where it was measured;
> it does not transfer upward.
> 
> This matters beyond τ=0.1. **Any curriculum rule that targets a success rate** (our threshold rule, online
> difficulty filtering, dynamic sampling) assumes that a given p means a given amount of learning. Here it does
> not. A target p places the sampler at a level, but how much that level teaches depends on the level too. It is
> also why the climb slowed and stalled: as the levels got harder, the time to push p from about 0.1 to the
> trigger grew from a few steps to more than the rest of the run.

## docs/results_seeds.md: January δ decided = 0.1 (user); hypothesis restated with falsification condition

**Before:**

> **Hypothesis for the January τ = 0.15 and 0.2 runs: the trigger h, not τ, sets the final level.** From the
> ceilings above, h = 0.20 → op 17 (as τ=0.1), h = 0.25 → op 16, and h = 0.30 → op 15–16 (op 15's ceiling of
> 0.31 is within noise of 0.30).
> * With δ = 0.1, τ = 0.15 (h = 0.25) and τ = 0.2 (h = 0.30) should stop at op 15–16. They would train a
>   narrower range than τ=0.1 and, if the range drives the gain, newly solve fewer hard problems.
> * With the δ = 0.05 the January plan proposed, τ = 0.15 has h = 0.20, the same trigger as τ=0.1, and should
>   reproduce τ=0.1's path.
> * The arm predicted to climb past op 17 is one with a trigger below 0.2, e.g. h = 0.15; τ=0.1's own p at op 17
>   was 0.17–0.18.

**After:**

> **Hypothesis for the January τ = 0.15 and 0.2 runs: the trigger h, not τ, sets the final level.**
> DECIDED 2026-11-22: **δ = 0.1**, so the two arms have triggers 0.25 and 0.30, genuinely different from τ=0.1's
> 0.20. Their bands are [0.075, 0.25] and [0.10, 0.30]. From the ceilings above:
> * τ = 0.15 (h = 0.25) should stop at op 16, and τ = 0.2 (h = 0.30) at op 15–16 (op 15's ceiling of 0.31 is
>   within noise of 0.30). Both train a narrower range than τ=0.1; if the range drives the gain, both should
>   newly solve fewer hard problems than τ=0.1's 95.
> * Refuted if either climbs past its predicted level, or stops below it while its final level still has
>   headroom above the trigger in the independent ceilings.
> * The arm predicted to climb past op 17 would need a trigger below 0.2 (e.g. h = 0.15; τ=0.1's own p at op 17
>   was 0.17–0.18). It is not in the January plan.

## docs/results_seeds.md: claims list: δ and the new finding

**Before:**

>    the January τ = 0.15 and 0.2 runs.

**After:**

>    the January τ = 0.15 and 0.2 runs (δ = 0.1, triggers 0.25 and 0.30).
> 6. **Success rate alone does not predict learning speed** (§6): at the same p ≈ 0.10–0.21, op 16–17 learned
>    10–40× more slowly than op 13–14. Any rule that targets a success rate inherits this.

## docs/progress_report.md: January δ decided = 0.1 (user); the δ = 0.05 alternative removed

**Before:**

> 1. **τ = 0.15 and τ = 0.2, as a test of the stall law (§6.8).** Hypothesis: the trigger h = τ + δ, not τ, sets
>    the final level. Ceilings measured in other arms predict h = 0.25 → op 16 and h = 0.30 → op 15–16, so
>    with δ = 0.1 both arms should stop below op 17 and, if the range drives the gain, newly solve fewer hard
>    problems than τ=0.1. With δ = 0.05, τ = 0.15 has τ=0.1's trigger (0.20) and should repeat its path; a
>    trigger below 0.2 is the arm predicted to climb past op 17. Also test whether arms that train inside the
>    predicted peak (p ≈ 0.10–0.21) beat τ=0.1.
>    With δ = 0.1 their move-up bounds (0.25, 0.30) lie above the peak, so these arms need δ = 0.05 (bands
>    [0.10, 0.20] and [0.15, 0.25], checked first in the simulator for oscillation) or a rule that targets the
>    band directly. About 38 GPU-hours for two seeds each.

**After:**

> 1. **τ = 0.15 and τ = 0.2, as a test of the stall law (§6.8).** Hypothesis: the trigger h = τ + δ, not τ, sets
>    the final level. We use **δ = 0.1**, so the triggers are 0.25 and 0.30, genuinely different from τ=0.1's
>    0.20 (bands [0.075, 0.25] and [0.10, 0.30]). Ceilings measured in other arms predict that τ = 0.15 stops at
>    op 16 and τ = 0.2 at op 15–16. Both would then train a narrower range than τ=0.1 and, if the range drives
>    the gain, newly solve fewer hard problems. The law is refuted if either climbs past its predicted level.
>    About 38 GPU-hours for two seeds each.

## docs/progress_report.md: new §6.9: the finding as its own point

**Before:**

> 
> ## 7. Correcting our own motivation

**After:**

> 
> ### 6.9 Success rate alone does not predict learning speed
> 
> At the same success rate, harder levels learn more slowly. τ=0.1 trained op 13 to op 17 in the same band of
> success rates, p ≈ 0.10–0.21, which is where op 13–15 rise fastest. Yet the rise per 10 steps fell from
> 0.31–0.43 at op 13–14 to 0.12–0.17 at op 15, 0.03–0.05 at op 16 and about 0.01 at op 17: 10–40× slower at the
> same p, two or three levels higher. The steepest-point estimate of §6.5 holds for the levels it was measured
> on and does not transfer upward.
> 
> This bears on any curriculum rule that targets a success rate, including our threshold rule and the
> difficulty filters in §2 that keep problems of intermediate accuracy. Such a rule assumes that a given p
> means a given amount of learning. Here a target p chooses the level, but how much that level teaches also
> depends on the level itself. The same effect explains the stall in §6.8: as levels got harder, pushing p from
> about 0.1 to the trigger took longer, from a few steps at op 13–14 to more than the remaining run at op 17.
> 
> ## 7. Correcting our own motivation

## docs/progress_report.md: limitations: the new finding

**Before:**

> * **One model and one task family.** All results are for the paper's 100M-parameter model on synthetic

**After:**

> * **Success rate is not learning speed.** Our rules, and the S-curve analysis behind τ=0.1, read the
>   success rate only. At the same p, op 16–17 learned 10–40× more slowly than op 13–14 (§6.9), so a target
>   success rate does not fix the learning rate, and the steepest-point estimate does not transfer to harder
>   levels.
> * **One model and one task family.** All results are for the paper's 100M-parameter model on synthetic

## docs/progress_report.md: January plan: the new finding

**Before:**

> 6. **Per-problem selection** within a level:

**After:**

> 6. **Learning speed at equal success rate.** Measure the rise per step against p for every level from the
>    existing logs and the January runs. If the slowdown with level holds, test a rule that also reads the
>    measured rise (without the bandit's self-reward, item 4), e.g. move up when the current level's rise has
>    flattened rather than when p crosses a fixed trigger.
> 7. **Per-problem selection** within a level:

## docs/progress_report.md: contributions: the new finding

**Before:**

>    run), and it is the hypothesis for the January τ values.

**After:**

>    run), and it is the hypothesis for the January τ values.
> 7. **Success rate alone does not predict learning speed**: at the same p, harder levels learn 10–40× more
>    slowly, which matters for any curriculum rule that targets a success rate.

## docs/slides_outline.md: slide 9: the new finding as its own bullet

**Before:**

> * τ=0.1 trains levels 13–15 in that fast stretch and moves on. At 16–17 the success rate sits in the same
>   range but rises 10–40× more slowly, and at 17 it never reaches the move-up point.

**After:**

> * τ=0.1 trains levels 13–15 in that fast stretch and moves on. At 16–17 the success rate sits in the same
>   range but rises 10–40× more slowly, and at 17 it never reaches the move-up point.
> * **So success rate alone does not tell you how fast a level will learn**: harder levels learn more slowly
>   at the same rate. That matters for any rule that aims at a target success rate, ours included.

## report/main.tex: LaTeX abstract: honest claim

**Before:**

> the range it discovered uniformly recovers 81\% of the gain: the right range is most of the win, and the
> adaptive sampler is what found it.

**After:**

> the range it discovered uniformly recovers 81\% of the gain: the right range is most of the win, and the
> adaptive sampler is what found it. It does not track the edge for the whole run: it climbs while levels are
> learnable and stops at the first level whose success rate stays below its move-up trigger, for the last
> 46--55\% of the run.

## report/main.tex: LaTeX problem: 'tracks the moving edge' removed

**Before:**

> \textbf{Our question is whether a sampler
> that tracks the moving edge creates more new ability

**After:**

> \textbf{Our question is whether a sampler
> that moves the training difficulty up as the model learns creates more new ability

## report/main.tex: LaTeX results: quantifies the stall

**Before:**

> stayed there, its success rate hovering near 0.17, below the 0.2 move-up bound.

**After:**

> stayed there for the remaining 46--55 steps, its success rate hovering near 0.17, below the 0.2 move-up
> trigger (\S\ref{sec:stall}).

## report/main.tex: LaTeX: 'while those levels were at the edge' removed

**Before:**

> \tauone\ instead put 84\% of its groups on op 15--17 while those levels were at the
> edge (20,642

**After:**

> \tauone\ instead put 84\% of its groups on op 15--17 at success rates of 0.10--0.21
> (20,642

## report/main.tex: LaTeX: 'while they were at the edge' removed

**Before:**

> the levels \tauone\ trained \emph{while they were at the edge};

**After:**

> the levels \tauone\ trained while it was still climbing;

## report/main.tex: LaTeX: 'a sampler that follows the edge' removed; new stall-law subsection

**Before:**

> A sampler that follows
> the edge finds the right range in one run instead of a sweep over ranges.

**After:**

> A threshold sampler
> finds a good range in one run instead of a sweep over ranges: the levels it passes through before it stalls.
> 
> \subsection{How adaptive is \texorpdfstring{\tauone}{tau=0.1}? A stall law}
> \label{sec:stall}
> 
> \begin{table}[t]
> \centering
> \small
> \begin{tabular}{lcccc}
> \toprule
> arm & trigger & stops at & from step & \% of run there \\
> \midrule
> \tauone\ (3 seeds)   & 0.2 & op 17 & 45--54 & 46--55 \\
> \tauthree\ (2 seeds) & 0.4 & op 14 & 23--25 & 75--77 \\
> \taufive\ (1 seed)   & 0.6 & op 11 & 0      & 100 \\
> \bottomrule
> \end{tabular}
> \caption{Where each threshold arm stopped climbing. Its batch success rate there over the last 20 steps:
> 0.17--0.18, 0.37 and 0.59, each below the trigger $\tau+\delta$.}
> \label{tab:stall}
> \end{table}
> 
> \tauone\ does not track the edge for the whole run. \textbf{Each threshold arm climbs while the level it
> trains can reach its move-up trigger $\tau+\delta$, and stops at the first level whose success rate stays
> below it} (Table~\ref{tab:stall}); no final level ever had the two consecutive measurements above the trigger
> that a move needs. Ceilings measured independently in the fixed band and the control (op 11 0.57, op 13
> 0.44, op 14 0.35, op 15 0.31, op 16 0.21, op 17 0.15) predict all three stopping levels. For \tauthree\ and
> \taufive\ the final level is a flat plateau. For \tauone\ op 17 was still rising by about 0.01 per 10 steps,
> so a ceiling just under 0.2 and a climb too slow for 100 steps cannot be told apart. \tauone\ is therefore
> adaptive in the first half of its run, and its main contribution is the range it passes through.

## report/main.tex: LaTeX: 'trains each level during its fastest improvement' scoped to op 13–15

**Before:**

> of the sweep. \tauone's band $[0.05, 0.2]$ contains the steepest point: it trains each level during its
> fastest improvement and moves on as the curve flattens.

**After:**

> of the sweep. \tauone's band $[0.05, 0.2]$ contains the steepest point measured for op 13--15, and on
> those levels it trained during the fastest improvement and moved on.

## report/main.tex: LaTeX: the new finding as its own paragraph

**Before:**

> temperature it is the end of a level's learning curve, not its midpoint. Values of $\tau$ between 0.1 and 0.3
> are untested.

**After:**

> temperature it is the end of a level's learning curve, not its midpoint. Values of $\tau$ between 0.1 and 0.3
> are untested.
> 
> \textbf{Success rate alone does not predict learning speed.} At op 16--17 \tauone\ trained at the same
> $p\approx0.10$--$0.21$, but $p$ rose only 0.01--0.05 per 10 steps, against 0.31--0.43 at op 13--14. At the
> same success rate, harder levels learn 10--40$\times$ more slowly. Any rule that targets a success rate,
> including ours and the difficulty filters of \S\ref{sec:related}, assumes a given $p$ means a given amount of
> learning; here it does not. It is also why the climb slowed and stalled.

## report/main.tex: LaTeX limitations: stall + success-rate finding

**Before:**

> set ($p=0.017$) is borderline. One model and one task family.

**After:**

> set ($p=0.017$) is borderline. Adaptivity is mostly early (\S\ref{sec:stall}), and the samplers read the
> success rate only, which does not fix the learning rate at harder levels. One model and one task family.

## report/main.tex: LaTeX plan: δ = 0.1 and the stall-law hypothesis

**Before:**

> (1) $\tau=0.15$ and $0.2$, which train inside the predicted peak
> (needs $\delta=0.05$);

**After:**

> (1) $\tau=0.15$ and $0.2$ with $\delta=0.1$ as a test of the stall law: their
> triggers (0.25, 0.30) predict stops at op 16 and op 15--16, a narrower range and, if the range drives the gain,
> fewer newly solved problems than \tauone;

## report/main.tex: LaTeX plan: the new finding

**Before:**

> (5) keeping a share of op 11--14 in \tauone's batches to remove the reliability cost.

**After:**

> (5) keeping a share of op 11--14 in \tauone's batches to remove the reliability cost; (6) measuring learning
> speed against success rate per level, and a rule that moves up when a level's rise flattens rather than at a
> fixed success rate.

# Round 3 (2026-11-22): checkpoint evaluation result

## docs/results_seeds.md §6: checkpoint evaluation result replaces "proposed and **not yet run** (needs approval)"

**Before:**

> The pass@128 evaluation of the step-25/50/75 checkpoints is proposed and **not yet run** (needs approval).

**After:** a table of newly solved at steps 25/50/75/100 (τ=0.1 vs control, paired tests), and the paragraph
"**τ=0.1 pulled ahead twice.** … the step-100 advantage over the control comes from concentrating on op 17 late,
not from the climb itself, whose lead had vanished by step 50. That concentration is not adaptive …"

## docs/progress_report.md §6.8: checkpoint evaluation result

**Before:**

> Evaluating the step-25/50/75 checkpoints on the hard set will answer this with the headline metric.

**After:**

> The step-25/50/75 checkpoints on the hard set (newly solved, mean of 3 seeds, pooled paired test) answer it
> with the headline metric. τ=0.1 led by +21 problems at step 25 while climbing (54 vs 33, p < 10⁻⁶). The control
> caught up by step 50 (79 vs 75, p = 0.11; op 18–20 tied, 16.7 vs 16.3). The lead re-formed during the stall at op
> 17: +8 at step 75 (op 18–20: +6, p = 0.023) and +8.7 at step 100. **The final advantage over the control comes
> from concentrated training on op 17 late in the run, not from the climb**, whose lead had disappeared by step
> 50. That late concentration is not itself adaptive.

# Round 4 (2026-11-22): the climb buys compute-efficiency, not final quality

## docs/results_seeds.md: new finding: compute-efficiency of the climb

**Before:**

> 50 coincides with τ=0.1 arriving at op 17.
> 

**After:**

> 50 coincides with τ=0.1 arriving at op 17.
> 
> 
> **The climb buys compute-efficiency, not final quality** (`eval/compute_efficiency.py`,
> `results/compute_efficiency.csv`):
> * **At a quarter of the compute, τ=0.1 matches the paper's final result.** At step 25 τ=0.1 has newly solved
>   54.0 hard problems (56, 57, 49). The fixed band, the paper's method, has 48 after all 100 steps (48, 48).
>   Paired by problem the difference is +6.0 [95% CI +0.2, +12.0], with 17 problems better and 8 worse (sign
>   test p = 0.11), and op 18–20 is tied (6.3 vs 6.0). So "matches, and probably exceeds", not "beats". Its first
>   25 steps took 2.0 GPU-hours against the fixed band's 7.6–8.0 for 100 steps: 25–26% of the GPU time and 25%
>   of the rollouts.
> * **The control wastes its early budget on levels the model cannot yet solve.** At step 25 it has 33, which is
>   15 below the fixed band's final result (3 vs 27 problems, p = 10⁻⁵). In steps 0–24 it put 43% of its groups
>   on op 15–17, and only 1.7–2.4% of those had a nonzero gradient (183–265 groups per run). τ=0.1 had
>   2,167–2,404 informative groups there in the same steps, by reaching op 15–16 only after op 13–14 had made
>   them learnable. Overall, 19–20% of the control's early groups were informative, against 27% for τ=0.1.
> * **The control catches up by step 50** (75 vs 79, p = 0.11; op 18–20 tied), so the climb's early lead does
>   not survive to the end. The final margin (+8.7) forms later, during the stall at op 17 (above). Final quality
>   comes mostly from the range (§3); the climb's own contribution is reaching a given result sooner.
> * Caveat: five comparisons in this paragraph and the table above (four per-step tests, one vs the fixed band),
>   none corrected for multiple comparisons. The fixed band's own step-25 checkpoint was not evaluated.
> 

## docs/results_seeds.md: claims list: compute-efficiency

**Before:**

> 6. **Success rate alone does not predict learning speed** (§6):

**After:**

> 7. **The climb buys compute-efficiency, not final quality** (§6): at step 25 (a quarter of the compute)
>    τ=0.1 newly solves 54 hard problems, matching the fixed band's final 48 (+6 [+0.2, +12.0], p = 0.11),
>    while the control, spending its early budget on op 15–17 at p ≈ 0, has 33. The control catches up by step 50.
> 6. **Success rate alone does not predict learning speed** (§6):

## docs/progress_report.md: report §6.8: compute-efficiency paragraph

**Before:**

> 50. That late concentration is not itself adaptive.
> 

**After:**

> 50. That late concentration is not itself adaptive.
> 
> **The climb buys compute-efficiency, not final quality.** At step 25, a quarter of the training compute (2.0
> GPU-hours against the fixed band's 7.8 for 100 steps), τ=0.1 has already newly solved 54 hard problems. The
> paper's fixed band reaches 48 after all 100 steps. Paired by problem, τ=0.1 at step 25 is +6 problems ahead of the
> fixed band's final result (95% CI +0.2 to +12.0; sign test p = 0.11, 17 vs 8 problems; op 18–20 tied): it matches
> the paper's final result and probably exceeds it. The control at step 25 has only 33. It spent 43% of its first
> 25 steps on op 15–17 while those levels were still at p ≈ 0, where only about 2% of groups had a gradient (183–265
> informative groups per run, against τ=0.1's 2,167–2,404). The control catches up by step 50, so the climb's
> advantage is speed. Final quality comes mostly from the range (§6.3). These five comparisons are not corrected for
> multiple comparisons, and the fixed band's own step-25 checkpoint was not evaluated.
> 

## report/main.tex: LaTeX §stall: compute-efficiency paragraph

**Before:**

> adaptive in the first half of its run, and its main contribution is the range it passes through.

**After:**

> adaptive in the first half of its run, and its main contribution is the range it passes through.
> 
> \textbf{The climb buys compute-efficiency, not final quality.} Evaluating the step-25/50/75 checkpoints,
> \tauone\ at step 25 (a quarter of the compute: 2.0 vs.\ 7.8 GPU-hours) newly solves 54 hard problems, matching the
> fixed band's final 48 (+6, CI $[+0.2,+12.0]$, $p=0.11$). The control at step 25 has 33, because it spends 43\% of its
> early batches on op 15--17 while almost no group there has a gradient. It catches up by step 50 (75 vs.\ 79,
> $p=0.11$), and the final $+8.7$ forms during \tauone's stall at op 17. (Five comparisons, uncorrected.)

## docs/slides_outline.md: slide 12A headline: adds the compute-efficiency point

**Before:**

> ### Slide 12A. The right range is most of the win, and τ=0.1 found it   **(primary)**
> 
> * **Headline:** Training uniformly on the levels τ=0.1 visited gets most of its gain. The adaptive
>   sampler's contribution is that it *found* those levels.

**After:**

> ### Slide 12A. The right range is most of the final win; the climb buys speed   **(primary)**
> 
> * **Headline:** Training uniformly on the levels τ=0.1 found gets most of its final gain. What the climb
>   buys is speed: at a quarter of the compute, τ=0.1 already matches the paper's final result.

## docs/slides_outline.md: slide 12A: compute-efficiency bullets; decomposition moves to backup

**Before:**

> * One stacked bar per test set, "share of τ=0.1's gain over the fixed band":

**After:**

> * **At step 25 (25% of the compute), τ=0.1 has newly solved 54 hard problems. The paper's method, after
>   all 100 steps: 48.** (+6, 95% CI +0.2 to +12, p = 0.11: it matches, and is probably ahead.) That is 2.0
>   against 7.8 GPU-hours.
> * The control at step 25 has only 33. In its first 25 steps it spent 43% of its training on levels 15–17
>   while the model could not solve them yet: only 2% of those problems taught anything.
> * The control catches up by step 50 (75 vs 79). The final gap (+9) forms later, while τ=0.1 sits on level
>   17. **The climb buys compute-efficiency, not final quality.**
> * Decomposition of the final gain (backup figure: one stacked bar per test set, "share of τ=0.1's gain over
>   the fixed band"):

## docs/slides_outline.md: slide 12A bottom line

**Before:**

> * **Bottom line, in the slide's largest type:**
>   > The paper's hand-picked band: **48**. The range τ=0.1 found on its own: **86**. With τ=0.1's climb: **95**.

**After:**

> * **Bottom line, in the slide's largest type:**
>   > At **25%** of the compute, τ=0.1 matches the paper's final result (**54** vs **48**).
>   > At **100%**: the paper's band **48**, the range τ=0.1 found **86**, with the climb **95**.

## docs/slides_outline.md: slide 12A figure = the timeline table

**Before:**

> * Small print: *We would not have tried 11–17 without τ=0.1: it is the paper's band plus three levels the
>   starting model cannot solve at all.*
> * **Figure:** the stacked bars (new; numbers in docs/results_seeds.md §3).

**After:**

> * Small print: *We would not have tried 11–17 without τ=0.1: it is the paper's band plus three levels the
>   starting model cannot solve at all. Five comparisons on this slide, not corrected for multiple comparisons.*
> * **Figure:** `results/timeline_table_slide.png` (`eval/plot_timeline_table.py`): newly solved at steps
>   25/50/75/100, τ=0.1 vs control vs the fixed band's final result, with the paired gaps. The stacked bars of
>   the decomposition move to backup.

## docs/slides_outline.md: backup: decomposition bars

**Before:**

> | How adaptive is τ=0.1? Stall law, steps climbing vs stalled, rise per level | docs/results_seeds.md §6 |

**After:**

> | How adaptive is τ=0.1? Stall law, steps climbing vs stalled, rise per level | docs/results_seeds.md §6 |
> | Range vs climb: decomposition of the final gain (stacked bars, was 12A's figure) | docs/results_seeds.md §3 |

## docs/slides_outline.md: figure checklist

**Before:**

> | decomposition bar (12A), per-level bars (12B) | 12 | **do not exist yet** | new; numbers in docs/results_seeds.md §3 |

**After:**

> | `timeline_table_slide.png` | 12A | **OK** (new, 13.3 in, 12.5–22 pt) | – |
> | decomposition bar (backup), per-level bars (12B) | backup, 12B | **do not exist yet** | new; numbers in docs/results_seeds.md §3 |

## docs/speaker_notes.md: speaker line 12A

**Before:**

> | 12A | The range is most of the win | "Training uniformly on the levels τ=0.1 found gets 81% of its gain: the paper's band gets 48, the range τ=0.1 found gets 86, the climb takes it to 95. The contribution is finding that range." | 1:10 |

**After:**

> | 12A | Range wins; the climb buys speed | "At a quarter of the compute τ=0.1 already matches the paper's final result, 54 against 48, while the control is at 33 because it spends its first steps on levels nobody can learn yet. The control catches up by step 50, and by the end the range τ=0.1 found, 86 of its 95, is most of the win. The climb buys speed, not final quality." | 1:20 |

## docs/speaker_notes.md: timing

**Before:**

> Total ≈ 9:50.

**After:**

> Total ≈ 10:00.
