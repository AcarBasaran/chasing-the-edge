# Dec 3 progress presentation: slide outline (draft)

11 slides, about 10 minutes. Every result slide carries a footer: **"seed 0 only, seeds 1–2 running"**
until the seeds land. When they do, rerun `eval/aggregate_seeds.py`, `eval/untrained_levels.py` and
`eval/compare_methods.py` (the figures then show mean + seed dots) and drop the footer.
No p-values on any slide until seeds are aggregated.

| # | title | message (one sentence) | figure / content |
|---|---|---|---|
| 1 | Chasing the Edge | Does training at a model's *moving* edge of competence create new reasoning ability? | title, names, COMP550 |
| 2 | The paper's finding | RL teaches a small LM genuinely new problems only at its edge of competence; the authors fix that band at op 11–14 for the whole run. | problem format example (one short problem with its "Define …" solution); difficulty = number of operations |
| 3 | Why the edge matters for GRPO | A group whose rollouts are all right or all wrong gives exactly zero gradient, so where the model is *right now* decides what it can learn. | the formula 1 − p^G − (1−p)^G, then `results/informative_fraction_by_level.png` as a **step-0 snapshot** (label it so) |
| 4 | The edge moves | In the fixed-band run, op 15–16 go from never solved to sometimes solved within 50 steps, but the band never moves past op 14. | `results/sampler_levels.png`, fixed-band panel only (crop): measured p per level over steps, with the band drawn on it. Callout: op 13 informative 1% → 37% by step 20 (the step-0 snapshot does not last) |
| 5 | Our samplers | We change only which level each batch is drawn from; everything else is the paper's setup. | table of the arms (fixed band, uniform, threshold τ ∈ {0.1, 0.3, 0.5}, bandit), one line each; "probe rollouts count against the budget" |
| 6 | Gate 3: the baseline reproduces the paper | Our strict-reward fixed band matches the released checkpoint within 1 point on every set: 48 of 204 hard problems newly solved, for both. | table: ours vs released vs Fig. 3 (from docs/gate3.md) |
| 7 | Where each sampler went | τ=0.1 climbs to op 17, τ=0.3 settles on op 14 (the paper's band, found on its own), τ=0.5 never leaves op 11. | `results/sampler_levels.png`, the three threshold panels |
| 8 | **Headline: levels τ=0.1 never trained on** | On op 18–20, τ=0.1 (never trained there) newly solves 27 of 102 problems, uniform (trained there) 19, fixed band 6. | **`results/hard_by_level.png`**, both panels. Speaker note: the right panel's labels carry the mechanism |
| 9 | Why exposure was not enough | 97% of uniform's groups at op 18–20 had zero gradient: every rollout failed, so seeing the level taught nothing; τ=0.1 learned op 15–17 while they were at the edge, and that carried further. | right panel of Fig. 8 enlarged, or a two-bar chart: groups at op 18–20 vs groups with gradient (uniform 16,005 vs 487; τ=0.1 0 vs 0) |
| 10 | Honest accounting | Over the whole hard set, uniform random already captures 63% of τ=0.1's gain over the fixed band. And τ=0.1 did *not* get more gradient: 25% of its groups were informative vs 35% for the fixed band and 40% for τ=0.5. What differs is where: 20,739 informative groups beyond op 14 vs 2,783 for uniform and 0 for the rest. | `results/method_comparison.png` (hard column, pass@128), plus the table from `results/seeds_gradient_budget.csv` (informative share and groups beyond op 14 per arm) |
| 11 | Status and next steps | Seeds 1–2 and the bandit are running; the range-vs-schedule control (uniform over op 11–17, 3 seeds) is queued; in January we confirm with 3 seeds and run longer. | bullet list from docs/progress_report.md §6–7; compute used (~8 GPU-h per run) |

## Backup slides

| title | figure |
|---|---|
| Per-level informative share during training (all levels 9–17) | `results/informative_over_training.png` (checkpoint re-evaluation) |
| Easy / medium / hard, all arms, pass@1 and pass@128 | `results/method_comparison.png` (full) |
| Edge speed and the bandit's probe interval | `$EDGE_SCRATCH/outputs/runs/fixed_band_s0/edge_speed.png`; m ≤ 1/speed → m = 25 |
| Within-level bimodality (why level is a coarse label) | `results/per_problem_p_hist.png` |
| Strict vs answer-only reward on the released checkpoint | table from docs/gate3.md |

## Figures to (re)make before the deck

* Slide 4: a single-panel crop of the fixed-band column of `sampler_levels.png` at slide resolution
  (`eval/plot_sampler_levels.py <run dir>` with one run gives that directly).
* Slide 9: optional small bar chart (exposure vs informative groups at op 18–20); the numbers are in
  `results/hard_by_level.csv` (`train_groups`, `informative_groups`).
* All figures: regenerate after seeds land so bars show the mean with seed dots.
