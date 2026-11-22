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
