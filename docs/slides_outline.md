# Dec 3 progress presentation: slide outline (draft)

**Spine: three samplers optimise three different objectives (gradient efficiency, learning speed and
headroom), and headroom wins.** About 12 slides, 10 minutes.

**Numbers.** Every result slide carries a footer: **"τ=0.1: seed 0 only; seeds 1–2 running"** until they land. The
other arms show mean ± spread (fixed band, uniform, τ=0.3: 2 seeds; bandit: 3). Before the deck, rerun
`eval/aggregate_seeds.py`, `eval/untrained_levels.py`, `eval/compare_methods.py` and `eval/bandit_q.py`. No
p-values on any slide until τ=0.1 has 3 seeds.

## Act 1: the setup (slides 1–4)

| # | title | message (one sentence) | figure / content |
|---|---|---|---|
| 1 | Chasing the Edge | When GRPO picks its own training difficulty, what should it aim for? | title, names, COMP550 |
| 2 | The paper's finding | RL teaches a small LM genuinely new problems only at its edge of competence; the authors fix that band at op 11–14 for the whole run. | one short problem with its "Define …" solution; difficulty = number of operations |
| 3 | Why the edge matters for GRPO | A group whose rollouts are all right or all wrong gives exactly zero gradient, so where the model is *right now* decides what it can learn. | the formula 1 − p^G − (1−p)^G, then `results/informative_fraction_by_level.png`, labelled **step-0 snapshot** |
| 4 | The edge moves; the band does not | In the fixed-band run, op 15–17 go from never solved to learnable while the band stays at op 11–14. | `results/informative_over_training.png`, left panel: op 15 0 → 27% informative by step 50, op 16 0 → 17%, op 17 0 → 7%, without ever being trained on |

## Act 2: three objectives (slides 5–6)

| # | title | message | figure / content |
|---|---|---|---|
| 5 | Three ways to chase the edge | Each adaptive sampler optimises something different: **gradient efficiency** (τ=0.5: stay where p ≈ 0.5, most groups informative), **learning speed** (bandit: go where p is rising fastest), **headroom** (τ=0.1: move on as soon as p > 0.2, so train where the model mostly fails). Uniform and the fixed band are the controls. | a 3-column graphic, one per objective: the rule in one line, what it maximises, and what it costs. Footnote: probe rollouts count against the budget |
| 6 | Gate 3: the baseline reproduces the paper | Our strict-reward fixed band matches the released checkpoint within 1 point on every set: 48 of 204 hard problems newly solved, for both. | table: ours vs released vs Fig. 3 (docs/gate3.md) |

## Act 3: what each objective did (slides 7–10)

| # | title | message | figure / content |
|---|---|---|---|
| 7 | Where each sampler went | τ=0.5 never leaves op 11, the bandit locks onto op 13 after step 25, τ=0.3 settles on op 14 (the paper's band, found on its own), τ=0.1 climbs to op 17. | `results/sampler_levels.png` (threshold panels and one bandit panel) |
| 8 | Gradient efficiency is not the goal | τ=0.5 has the highest share of informative groups of any arm (40%) and learns nothing beyond op 14: 0 hard problems newly solved. | two numbers side by side: informative share 40.1% → hard newly solved 0. Note: p at op 11 hovers at 0.55–0.63, so the rule never fires (fixed point) |
| 9 | Learning speed chases the wrong thing | The bandit's reward is improvement, an unsolvable level shows zero improvement, and the trained level improves *because* it is trained. After step 25 the greedy choice never left op 12–14 in any seed; only ε (3.2% of prompts) ever sampled op 15+. Result: 40.7 ± 3.5 hard newly solved, below the fixed band (48). | **`results/bandit_q.png`**: top row Q per arm (op 15–17 in orange never on top), bottom row batch share (greedy level 90%, op 15+ flat at the ε floor). Add the m = 10 ablation line when it lands (docs/bandit_analysis.md §4) |
| 10 | **Headroom wins** | On op 18–20, levels τ=0.1 never trained on, it newly solves 27 of 102 problems; uniform (which trained there) 20 ± 1, the fixed band 6, the bandit 3.3 ± 1.5, τ=0.5 0. | **`results/hard_by_level.png`**, both panels |

## Act 4: why headroom wins, and the honest caveats (slides 11–12)

| # | title | message | figure / content |
|---|---|---|---|
| 11 | Not more gradient: gradient in a different place | τ=0.1 had a *lower* informative share (25%) than τ=0.5 (40%), the fixed band (35%) and the bandit (29%). What differs is where: 20,739 informative groups beyond op 14, vs 2,778 for uniform (2,288 at op 15–17, 490 at op 18–20; 97% of its op 18–20 groups had zero gradient), 161 for the bandit, 0 for the rest. That count orders the op 18–20 result. | table from `results/seeds_gradient_budget.csv`: informative share and groups beyond op 14, per arm, next to op 18–20 newly solved |
| 12 | Caveats and next steps | Uniform random already captures 63% of τ=0.1's gain on the full hard set, so most of the gain is "harder problems at all". The range-vs-schedule control (uniform over op 11–17, 3 seeds) answers whether the timing matters beyond the range. τ=0.1 seeds 1–2 are running. January: 3 seeds everywhere, 200-step runs. | `results/method_comparison.png` (hard column, pass@128), plus a bullet list from docs/progress_report.md §6–7. Compute: ~8 GPU-h per run |

**If the op 11–17 control matches τ=0.1 on op 18–20** (results likely before Dec 3), slide 10 becomes "the
headroom range wins", and slide 12 says the schedule adds nothing beyond the range. If it falls short, slide 12
says the timing matters. Keep both versions of slides 10 and 12 ready.

## Backup slides

| title | figure |
|---|---|
| Per-level informative share during training, trained levels every step | `results/informative_over_training.png` (right panel) |
| Easy / medium / hard, all arms, pass@1 and pass@128 | `results/method_comparison.png` (full) |
| Bandit: probe noise vs Q margins (why *which* of op 12–15 wins is partly noise) | `results/bandit_q.csv`; docs/bandit_analysis.md §2 |
| Edge speed and the bandit's probe interval | `$EDGE_SCRATCH/outputs/runs/fixed_band_s0/edge_speed.png`; m ≤ 1/speed → m = 25 |
| Within-level bimodality (why level is a coarse label) | `results/per_problem_p_hist.png` |
| Strict vs answer-only reward on the released checkpoint | table from docs/gate3.md |
| The step-0 motivation and why it did not last | docs/gate1.md time series: op 13 informative 1% → 37% by step 20 |

## Figures to (re)make before the deck

* Slide 5: the three-objective graphic (new; plain table-like layout, no data).
* Slide 7: `eval/plot_sampler_levels.py` with τ=0.1, τ=0.3, τ=0.5 and bandit_s0 only.
* Slide 8: optional two-bar figure (informative share vs hard newly solved per arm). The numbers are in
  `results/seeds_gradient_budget.csv` and `results/seeds_by_set.csv`.
* Slide 9: `results/bandit_q.png`. Rerun `eval/bandit_q.py` with the m = 10 run added once it finishes.
* All figures: regenerate after the τ=0.1 seeds land so the bars show the mean with seed dots.
