# Gate 3: fixed-band baseline

Status: **setup done, released checkpoint evaluated, baseline not launched.** Waiting for the agreed scale (docs/budget.md).

## What the baseline is (DECIDED 2026-11-15)

THE baseline is **our own** fixed-band run with the **strict process reward**
(`configs/rl_fixed_band.yaml`): uniform over op 11–14, training reward
`verl/reward_fn.py::compute_score_with_step_process(answer_weight=1, step_weight=0,
zero_on_process_mismatch=True)`. Every other arm uses the identical config (`configs/rl_base.yaml`)
and differs only in `data.sampler`.

The released `extrapolation_rl/id2-10_0.2easy_0.3medium_0.5hard/rl/op11-14_uniform` is a **loose
sanity check only**. It is scored under all three criteria, and `results/gate3_released_rl_comparison.csv`
reports all of them.

## Released checkpoint: what we know

| question | evidence | conclusion |
|---|---|---|
| reward | config named like the checkpoint, `rl-200steps/op11-14_uniform.yaml`, uses `verl/dataset.py::compute_score` = **answer-only** exact match. The strict variant is a separate `op11-14_uniform_process_strict.yaml` (experiment `...-process-strict-200steps`). Paper Sec. 6 introduces process-verified rewards as an *addition* to the outcome reward, and says they "improve pass@1 by 4–5% on op 15–20". That reads as if the Sec. 3 / Fig. 3 runs used the outcome-only reward. | probably **answer-only** |
| length | `total_epochs: 1` over 4 × 50k heldout at batch 1024 ≈ 195 steps (folder `rl-200steps`). The answer-only config's experiment name is `rl-op11-14_uniform-100steps`. App. A.3 says "two training epochs". | **unknown**: 100, ~195 or ~390 steps |
| start point | answer-only yaml: `model.path: .../id2-10_uniform`, `default_local_dir: .../0.4995easy_0.4995medium_0.001hard/...` (template leftovers). Strict script: `.../0.2easy_0.3medium_0.5hard/pt/checkpoint-22157`. | presumably our base; unconfirmed |
| HF card | "Final RL checkpoint for this experiment variant." config.json identical to base except dtype bf16. | no information |

Draft email asking all four questions: `results/email_to_authors.md` (for Eren to send).

## Released checkpoint vs base vs paper Fig. 3

Test sets: easy 25/level, medium **200/level**, hard 34/level. T=0.7, n=128.

Strict criterion (the paper's): base → released, and paper Fig. 3 (read off the figure, ±2pp):

| set | base p@1 / p@8 / p@128 | **released RL** p@1 / p@8 / p@128 | paper Fig. 3 RL op11-14 | newly solved (0/128 → >0) |
|---|---|---|---|---|
| ID op2–10 (225) | 85.5 / 96.5 / 98.7 | 85.2 / 96.4 / 98.7 | 87.0 / 96.3 / 99.0 | 0 of 3 |
| edge op11–14 (800) | 17.5 / 29.6 / 38.4 | **49.4 / 69.6 / 83.9** | 51.0 / 71.0 / 85.5 | 368 of 493 |
| hard op15–20 (204) | 0 / 0 / 0 | **10.8 / 16.2 / 23.5** | 9.5 / 16.3 / 26.0 | **48 of 204** |

**The released checkpoint reproduces Fig. 3 within figure-reading error on every band.** It is a sound
reference: its numbers are what the paper's fixed-band method gives (under whatever reward produced it).
The strict *training* criterion gives the same numbers to 0.2pp on edge and hard, so the RL model does not
pad solutions with extra nodes.

Answer-only scoring (what the release probably optimised) shows the gap the strict reward closes:

| set | base answer-only p@1 / p@128 | released answer-only p@1 / p@128 |
|---|---|---|
| ID | 98.5 / 100 | 97.2 / 99.6 |
| edge | 43.3 / 70.4 | 83.3 / 96.1 |
| hard | **18.5 / 36.8** | 36.8 / 64.7 |

The base model already gets the *answer* right on 18.5% of hard samples with a wrong process (lucky
arithmetic or shortcuts). Under an answer-only reward, op15–20 would not be a zero-gradient zone at all.
That is one more reason the strict reward is the right one for an edge-of-competence study: with
answer-only scoring the "edge" would be smeared by reward hacking. Full table:
`results/gate3_released_rl_comparison.csv`.

**What Gate 3 will compare:** our strict fixed-band run vs this table (strict criterion). Same shape, roughly
same magnitude is the bar. Fig. 7 of the paper says strict reward adds ~4–5pp pass@1 on op15–20 over
outcome-only, so a strict baseline landing at or slightly above the released hard numbers is the expected
outcome. Far below = debug before anything else.

## Run settings (paper vs ours)

| | paper (upstream yaml + App. A.3) | ours (`rl_base.yaml`) |
|---|---|---|
| data | heldout op11–14, 50k/level, RandomSampler | same heldout files (overlong prompts and val/test collisions removed), drawn through our sampler restricted to 11–14 (uniform over equal-size buckets = RandomSampler) |
| batch / mini / micro | 1024 / 256 / 16 per GPU | same |
| G (rollout.n), T | 6, 1.0 | same |
| lr, KL | 1e-6 constant, low_var_kl 1e-3, entropy 0 | same |
| max prompt / response | 1024 / 1024 | same |
| GPUs | 6 | 4 (global batch sizes unchanged, so the optimisation is the same) |
| reward | answer-only (probably; see above) | strict process (DECIDED) |
| steps | 100 / ~195 / ~390 (unclear) | per agreed scale |
| validation | val op2–20 × 200, every 20 steps | tracking set: 32/level × op2–20, n=8, T=1.0, every 10 steps (measurement only) |

## Edge-speed measurement (sets the bandit's m)

The tracking sweep in every run gives p_L(t) for all 19 levels every 10 steps.
`eval/edge_speed.py <run>/train.log` turns that into an edge position (the level where p crosses
τ = 0.5 / 0.3 / 0.1) and its speed in levels per step. The bandit's m must satisfy
m ≤ 1 / speed. To be reported from the baseline run.
