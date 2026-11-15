# Gate 1 — base model checkpoint, test sets, p_L by level

Status: **PASSED** (2026-11-15). Base-model numbers reproduce the paper within
figure-reading error. Raw rollouts: `outputs/gate1_base_eval/` (scratch symlink, not committed).

## How to reproduce

```bash
source slurm/env.sh
$EDGE_PY eval/download_assets.py --with-rl          # login node: model + test/val/heldout splits -> scratch
$EDGE_PY eval/build_testsets.py                     # data/testsets/{easy,medium,hard} + manifest.json
$EDGE_PY eval/check_parser_on_gold.py --out results/gold_solution_stats_by_level.csv
# gold graphs (upstream builder), then the GPU eval array, then the CPU summary:
cd $UPSTREAM && PYTHONPATH=$UPSTREAM $EDGE_PY scripts/build_gold_dependency_graphs.py $EDGE_DATA/val $EDGE_SCRATCH/outputs/gate1_base_eval/gold/val
#   (same for data/testsets/{easy,medium,hard} -> gold/test_{easy,medium,hard})
sbatch slurm/gate1_check_model.sbatch
sbatch --array=0-21%8 slurm/gate1_eval.sbatch
sbatch --dependency=afterok:<array id> slurm/gate1_summarize.sbatch
```

Environment: `$EDGE_SCRATCH/envs/edge` — Python 3.12, torch 2.7.1+cu126, vLLM 0.10.1.1,
**transformers 4.55.4** (pip resolves vLLM 0.10.1.1 to transformers 5.x by default; the
checkpoint was saved with 4.55 — pin it), huggingface_hub 0.36, datasets 3.6. GPUs: partition
`ai`, `--exclude=ai[01-08],ai[11-14]` lands on A40/A6000 (bf16 + vLLM V1 need Ampere).

Compute: 22 tasks, 1.74 GPU-hours of generation, 140M generated tokens
(≈36k output tok/s per A40 for this 100M model).

## Step 1 — upstream repo map (Interplay-LM-Reasoning @ ab728f0)

**Dataset & difficulty.** HF dataset `Interplay-LM-Reasoning/composition`, JSONL. Fields:
`problem, question, solution, op, id, template, mode, length, d`. Difficulty = **`op`** (int,
number of operations); also encoded in the filename (`op12-1k.jsonl`). `d` is graph depth.
Splits: `test/op{2..20}-1k.jsonl` (1000/level), `val/op{2..20}-200.jsonl`,
`heldout/op{2..20}-50k.jsonl` (**the RL training pool**; RL configs list these files),
`train/` (10B-token pretraining shards — never download). `id` is **not unique** (182 unique
ids in 200 rows) — key problems by content hash. Note `op` ≠ number of solution steps: an op-20
problem has median 10 `Define` steps (`results/gold_solution_stats_by_level.csv`).
Loading inside RL: `verl/dataset.py::CustomRLHFDataset` (`datasets.load_dataset("json")`),
selected by `data.custom_cls` in the verl YAML.

**Text format** (not `[Problem]…[Solution] Step 1.` as CLAUDE.md says):
```
<question> {problem} {question} </question> <solution> Define X as t; so t = 4. Define … ; so q = p = 4. </solution> <answer> 4 </answer>
```
Prompt = everything up to and including `<solution>`; generation stops at `</answer>`.

**Eval harness.** `scripts/eval_checkpoints.py` (vLLM generation, answer-only EM,
writes `<ckpt>_{id,ood}_generations.jsonl`) — normally launched by `scripts/meta_run.sh --do-eval`
(env vars `EVAL_DATA_ROOT`, `EVAL_CHECKPOINTS_ROOT`, …). The paper's strict numbers need the
graph pipeline afterwards: `scripts/build_gold_dependency_graphs.py <data_dir> <gold_dir>` →
`scripts/score_dependency_graphs.py <gold_dir> <generations.jsonl> --results-dir …` →
`scripts/aggregate_graph_pass_results.py`. We call `eval_checkpoints.py` and
`score_dependency_graphs.py` directly (`slurm/gate1_eval.sbatch`) and replace the aggregator
(it computes pass@k as "any of the first k samples correct" — biased below k=n) with
`eval/passk.py`.

**Process-verified parser.** `utils/solution_dependency_graph.py::SolutionParser`. Input: the
raw generated string after `<solution>` (tags are stripped; text with or without `</solution>`).
It needs `Define <quantity name> as <single letter>; …` sentences, assignments `X = expr` with
`+ - *` only, and the final answer as `<answer> N` (or `Answer: N`). Output: nodes keyed by
quantity name with value and dependency set. Comparison: `DependencyGraphData.compare`.
Two criteria exist upstream and **they differ**:
- eval (`score_dependency_graphs.py`, paper App. A.4): all gold nodes present with right value and
  parents + right answer; **extra predicted nodes allowed**.
- training reward (`verl/reward_fn.py::compute_score_with_step_process`, `zero_on_process_mismatch=True`
  in `op11-14_uniform_process_strict.yaml`): **extra nodes also zero the reward**.
  At op ≤ 10 this makes train reward up to 0.9pp lower than the eval number (op 5); at op ≥ 11 identical.
  The config whose name matches the released `rl/op11-14_uniform` checkpoint (`op11-14_uniform.yaml`)
  uses **answer-only** reward (`verl/dataset.py::compute_score`); the strict one is
  `op11-14_uniform_process_strict.yaml`. The HF card does not say which produced the release —
  resolve before Gate 3.
Sanity: gold solutions fed back through both criteria score 100% at every level 2–20
(`eval/check_parser_on_gold.py`), and our independent re-score agrees with upstream on 100%
of 566k samples.

**verl dataloader hook point.** `verl/trainer/ppo/ray_trainer.py`: `RayPPOTrainer._create_dataloader`
(line ~357) builds `self.train_dataloader = StatefulDataLoader(..., sampler=train_sampler)`;
batches are consumed in `RayPPOTrainer.fit()` at `for batch_dict in self.train_dataloader:` (line ~992).
The sampler comes from `verl/trainer/main_ppo.py::create_rl_sampler` (line ~376), which already
supports **`data.sampler.class_path` / `data.sampler.class_name`** — a config-loaded
`verl.experimental.dataset.sampler.AbstractCurriculumSampler` whose `update(batch)` is called
after every step (`ray_trainer.py` ~line 1252, with rewards in `batch`). That is a supported
extension point: our `DifficultySampler` can be wrapped in such a sampler with no subclassing
of trainer internals. Requires `data.dataloader_num_workers=0`.

**Vocabulary.** 2,200 embedding rows (`config.vocab_size`); tokenizer has 2,201 entries — the
extra id 2200 is an unused `<special_token_4>`. Tokenizer is byte-identical to upstream
`model_configs/qwen2_100M/tokenizer.json`. Many common words tokenize character-by-character
("number" → n,u,m,b,er) — that is how the model was trained, not a bug.

## Step 2 — checkpoint

`Interplay-LM-Reasoning/extrapolation_rl`, subfolder **`id2-10_0.2easy_0.3medium_0.5hard/base`**.
Why: it is the pretraining mixture of the paper's main extrapolation experiment (Sec. 3 / Fig. 3).
Upstream's Sec. 3 quick-start (`scripts/run_scripts/run_composition_rl.sh`) pretrains on exactly
this config and runs the op7-10/9-12/11-14/17-20 RL sweep from it, and the README names it "the
underlying pre-training config". The other three mixtures are ablations of the pretraining mix.
Loads with transformers 4.55.4: Qwen2ForCausalLM, 12 layers, hidden 768, 12 heads / 2 KV heads,
103,170,816 params, stored fp32 (eval runs it in bf16 under vLLM, as upstream does).

## Step 3 — test sets (`data/testsets/`, `eval/build_testsets.py`)

From `test/op{L}-1k.jsonl`, seed 20261115, after dropping anything whose (problem, question)
also appears in `val/` or `heldout/` (2 collisions found: one at op3, one at op10):

| set | levels | per level | total |
|---|---|---|---|
| easy | 2–10 | 25 | 225 |
| medium | 11–14 | 50 | 200 |
| hard | 15–20 | 34 | 204 |

Equal per-level counts are required: upstream's scorer maps generations to gold graphs by
`__idx % rows_per_level`.

## Step 4 — reproduction vs paper (`results/gate1_paper_comparison.csv`)

The paper gives base numbers only in figures (Fig. 3: pass@k on test; Fig. 9: end-of-pretraining,
val). Ours: val split, 200 problems × 128 samples per level, T=0.7, top-p 1, 1024 new tokens.

| band | metric | ours (val) | paper Fig. 9 | paper Fig. 3 |
|---|---|---|---|---|
| op 2–10 | pass@1 | 87.5 | ≈87 | ≈82.5 |
| op 2–10 | pass@128 | 99.4 | ≈99.5 | ≈99.6 |
| op 11–14 | pass@1 | 18.3 | ≈19 | ≈17 |
| op 11–14 | pass@128 | 41.8 | ≈42 | ≈43.5 |
| op 15–20 | pass@1 / pass@128 | 0 / 0 | 0 / 0 | 0 / 0 |
| op 11 | pass@1 / @128 | 48.9 / 92.0 | ≈49 / 91 | |
| op 12 | pass@1 / @128 | 23.8 / 67.5 | ≈27 / 65 | |
| op 13 | pass@1 / @128 | 0.4 / 7.0 | ≈1 / 7 | |
| op 14 | pass@1 / @128 | 0.0 / 0.5 | ≈0 / 0 | |

31/32 comparisons within ±6pp. The exception is our 200-problem **medium test set**
(pass@128 35.5 vs Fig. 3's 43.5), driven entirely by op 12 — its 50 sampled problems give pass@128 0.46 vs 0.675 on val. A cross-check on
the **full** `test/op12-1k.jsonl` (1000 × 128) gives pass@1 24.2%, pass@128 62.7% — matching val, so the
splits agree and the 50-problem draw was unlucky (~2.4σ). Consequence: 50 problems/level is too noisy for
per-level pass@128 on the medium set; consider 100–200/level there (cheap: ~5 GPU-min per 1000 problems).

## Step 5/6 — see `results/base_model_p_by_level.csv`, `results/base_model_correct_token_lengths_by_level.csv`

Key consequences for the project:
- **The edge is razor-sharp**: p = 0.49 (op11), 0.24 (op12), 0.004 (op13), 0.00004 (op14), 0 for 15–20.
  The paper's "11–14 band" is, at step 0, two informative levels and two dead ones.
- **The 1024-token cap is not binding.** Correct solutions: p95 ≤ 440 tokens at every level;
  gold solutions need ≤ 700 even at op 20; truncation ≤ 0.008% of samples at every level.
  At op ≥ 13 the model under-generates (mean ≈ 285 tokens vs ≈ 470 gold median at op 20) —
  it fails by stopping early, not by running out of budget.
- Prompt length: some prompts exceed 1024 tokens at most levels (max 1192 at op 17); verl's
  `filter_overlong_prompts` with `max_prompt_length: 1024` silently drops them from the RL pool.

## Headline motivation: share of GRPO groups that produce any gradient (added 2026-11-15)

A GRPO group of G rollouts gives zero gradient iff all G rewards are equal. **Our G = 6** (the paper's
`rollout.n`). Reward = strict training reward. Two estimates per level:

* **level-mean formula**: 1 − p^G − (1−p)^G with the level's mean p (the CLAUDE.md back-of-envelope);
* **per-problem**: the same formula averaged over problems, using the unbiased estimator
  C(c,G)/C(n,G) of p^G per problem. This is the real number.

| level | p, T=1.0 | level-mean formula, G=6 | **per-problem, G=6, T=1.0** | per-problem, G=6, T=0.7 | problems never solved (T=1.0, 64 samples) | **measured in real GRPO groups** |
|---|---|---|---|---|---|---|
| 9  | 0.721 | 85.9% | **45.4%** | 34.4% | 0% | |
| 10 | 0.567 | 96.0% | **41.3%** | 34.9% | 2% | 38.7% (256 groups) |
| 11 | 0.483 | 96.8% | **41.9%** | 37.7% | 11% | 46.5% (256 groups) |
| 12 | 0.252 | 82.4% | **36.8%** | 32.6% | 31% | 37.1% (256 groups) |
| 13 | 0.0048 | 2.9% | **2.4%** | 1.5% | 90% | |
| 14 | 0.0001 | 0.05% | **0.05%** | 0.02% | 100% | |
| 15–20 | 0 | 0% | **0%** | 0% | 100% | |

`results/informative_fraction_by_level.csv` (also G=8), plot `results/informative_fraction_by_level.png`.
T=1.0 numbers come from a separate base eval on val op9–15 (64 samples per problem; `results/base_T1.0_per_problem.csv`).
The "measured" column is the actual fraction of groups with nonzero reward variance at steps 0–2 of
the Gate-2 Threshold smoke run, where the model is still effectively the base model
(`results/gate2/threshold_smoke_sampler_log.jsonl`).

**Two findings:**

1. **Half of the paper's band is dead weight at step 0, as predicted.** op13 gives a gradient in 2.4% of
   groups and op14 in 0.05%. Averaged over the band, 20% of the fixed-band groups carry any signal (per-problem, T=1.0).
2. **The level-mean formula overstates the informative share by ~2.3× even at the best levels.** At op11 it
   predicts 97%, but only ~42% of groups are informative; the real GRPO run measured 46.5%. Problems within
   a level are mostly bimodal: always solved or never solved (at op12, 31% of problems are 0/64 while the
   level mean is 0.25). "The edge" is therefore a property of individual problems as much as of levels.
   For the samplers this means the level-mean p is a weak proxy for gradient yield. Logging the measured
   informative fraction per step, which `EdgeCurriculumSampler` already does, is the better signal to plot.

T=0.7 vs T=1.0: the level means barely move (op11 0.489 vs 0.483, op12 0.238 vs 0.252). So the Threshold
start behaviour holds at the RL temperature: τ=0.3 leaves op11 upward, and τ=0.5 holds there.

## Medium test set grown to 200/level (2026-11-15)

`data/testsets/medium/op{11..14}-200.jsonl`. The original 50/level are kept as a subset (`in_v1_50`, and
`data/testsets/medium50/`); 150/level were added with seed 20261116 (`eval/grow_medium.py`). The base model on the
200/level set scores pass@1 17.5 / pass@8 29.6 / pass@128 **38.4**, against paper Fig. 3 at 17 / 31 / 43.5. All 32 Gate-1
checks are now within tolerance; pass@128 is within 5.1pp of Fig. 3 and within 3.3pp of our val-split number (41.8).

## Per-problem success rate within a level (added 2026-11-15)

`results/per_problem_p_hist.png`, `results/per_problem_p_by_level.csv` (`eval/per_problem_p_hist.py`). Base model, strict reward,
T=1.0, 64 samples × 200 val problems per level:

| level | mean p | p < 0.1 | 0.1 ≤ p ≤ 0.9 | p > 0.9 | never solved | always solved |
|---|---|---|---|---|---|---|
| 10 | 0.57 | 20% | 39% | 41% | 2% | 10% |
| 11 | 0.48 | 26% | **38%** | 36% | 11% | 7% |
| 12 | 0.25 | 53% | 40% | 6% | 31% | 0% |
| 13 | 0.005 | 99% | 1% | 0% | 90% | 0% |
| 14 | 0.000 | 100% | 0% | 0% | 100% | 0% |

At op 10–12 roughly 60% of problems are near-never or near-always solved, and these levels overlap in
per-problem difficulty (op 12 has problems above 0.9; op 10 has problems below 0.1). The mass sits *near* 0 and 1
more than exactly at them (op 11: 11% exactly 0, 7% exactly 1), and per-problem estimates from 64 samples carry
binomial noise (SE ≈ 0.06 at p = 0.5) that smooths the histogram slightly but cannot create the two humps.

Implication: the op level is a coarse proxy for difficulty. A sampler that picks *problems* by their own estimated
p (e.g. from previous rollouts of the same problem) could raise the informative share well above the ~40% any
level-based sampler can reach at step 0. **Future work, not built** (decision 2026-11-15); the four
level-based samplers remain the project.

## Audit: prompt-length filtering and token counting across every problem set (2026-11-15)

**Why:** job 2024783 crashed on a 1,037-token prompt in the tracking set. The training pool had been
filtered to ≤ 1024 prompt tokens, but the tracking set was built on a separate path without the filter, and
`rl_base.yaml` turns verl's own filter off (`filter_overlong_prompts: false`, `truncation: error`). This audit
checks every problem set we build or consume, on every row, with `eval/audit_prompt_sets.py` →
`results/prompt_length_audit.csv`.

**What was checked.** A dataset row becomes prompt tokens on three code paths:

| path | where | prompt text | counting |
|---|---|---|---|
| A: verl length filter | upstream `CustomRLHFDataset.maybe_filter_out_long_prompts`. Our replicas in `samplers/build_pool.py`, `eval/build_track_set.py` and `eval/prompt_filter_stats.py` | `verl/dataset.py::compose_prompt` | `tokenizer((prompt, answer), add_special_tokens=True)["input_ids"][0]` |
| B: verl loader | `CustomRLHFDataset.__getitem__` → `postprocess_data` (raises above 1024 with `truncation: error`) | same | `tokenizer(prompt, add_special_tokens=False)` |
| C: eval harness | upstream `scripts/eval_checkpoints.py` → vLLM (`max_model_len` 2048, `max_tokens` 1024) | `compose_text(row).split("<solution>")[0] + "<solution>"` | vLLM default, `add_special_tokens=True` |

Our replicas tokenize the eval-path text (`eval/strict_score.py::compose_prompt`), so they equal A only if the
two prompt builders produce the same string. That is checked too.

**Result: the counting is consistent everywhere.** On all 990,019 rows across nine sets, A = B = C token
counts on every row, and the two prompt builders produce byte-identical text on every row. The tokenizer
adds no special tokens, so `add_special_tokens` makes no difference. One limit and one count hold everywhere.

**Which sets apply the ≤ 1024 filter, and whether they need to:**

| set | rows | max prompt tokens | rows > 1024 | consumer | filter | verdict |
|---|---|---|---|---|---|---|
| `pool_50k` (RL pool, op2–20) | 946,492 | 1024 | **0** | all six arms: training + bandit probes; verl filter off | pre-filtered (`build_pool.py`) | ✅ consistent |
| `track32` (tracking sweep) | 608 | 941 | **0** | in-training validation of all six arms; verl filter off | pre-filtered (`build_track_set.py`, **fixed today**) | ✅ fixed: was 4 overlong prompts |
| `pool_2k` (smoke pool) | 38,000 | 1475 | 139 | Gate-2 smoke configs only | verl filter **on** at load (drops them) | ✅ safe (different path, never used by full runs) |
| `testsets/medium50` | 200 | 1040 | 3 | Gate-2 smoke validation file | verl filter **on** at load | ✅ safe. Its path in `gate2_base.yaml` was stale after the medium set was grown (would crash at load): **fixed today** |
| `testsets/easy` | 225 | 1029 | 1 | eval harness only | none | ⚠️ kept on purpose, see below |
| `testsets/medium` | 800 | 1046 | 5 | eval harness only | none | ⚠️ kept on purpose |
| `testsets/hard` | 204 | 928 | 0 | eval harness only | none | ✅ |
| `val` op2–20 × 200 | 3,800 | 1192 | 17 | eval harness: Gate-1 p_L table, T=1.0 sweep | none | ⚠️ kept on purpose |
| `test/op12-1k` | 1,000 | 1164 | 3 | eval harness: Gate-1 cross-check | none | ⚠️ kept on purpose |

The bandit's probe prompts are drawn from `pool_50k` by the same sampler as training prompts, so they are
covered by the first row. No other code path builds a problem set.

**Evaluation sets are deliberately not length-filtered (6 of 1,229 test problems; 17 of 3,800 val).**
1. The eval harness never truncates or crashes. With `max_model_len` 2048 the longest prompt (1192 tokens) still
   leaves ≥ 856 tokens to answer, more than any correct answer needs (base p95 ≤ 440, gold ≤ 700).
2. Every model is scored on the same problems with the same harness, so method comparisons and the
   newly-solvable count are unaffected.
3. Gate 1 reproduced the paper with these sets as they are.

The one real asymmetry: these few long-story problems can appear at test time but never in RL training.

**DECISION (2026-11-15): keep the long test prompts. The test and val sets stay exactly as built.** Reasons, for the report:
1. **Fidelity to the paper's protocol.** The paper evaluates on its released `test/` split as is. Gate 1 reproduced
   its numbers with these problems included (32/32 checks within tolerance). Removing them would make our test sets a
   filtered subset that no longer matches what Fig. 3 / Fig. 9 report.
2. **No measurement artefact.** The 1024-token limit is a *training* constraint (verl's `max_prompt_length`), not an
   evaluation one. At test time the model gets up to 2048 tokens in total, so even the longest test prompt (1046
   tokens) leaves ≥ 1002 tokens to answer, and the longest val prompt (1192) leaves 856. That is more than any
   correct answer needs (base p95 ≤ 440, gold ≤ 700). Nothing is truncated and nothing fails to parse because of length.
3. **No effect on comparisons.** Every model (base, released checkpoint, all six arms, all seeds) is scored on the
   identical problems with the identical harness. Method differences and the newly-solvable count are paired
   comparisons on the same problems, so a fixed set of 6 problems cannot favour one arm.
4. **Small and disclosed.** 6 of 1,229 test problems (0.5%): 1 easy (op 6, 1029 tokens), 5 medium (2 at op 11, 3 at op 13; 1031–1046 tokens), 0 hard. The headline hard set
   contains none. Being unseen-length problems, they make the test slightly *harder* for every model, never easier.
5. **Easy to check.** `results/prompt_length_audit.csv` lists the counts per set. Any reported number can be
   recomputed without these problems from the per-problem CSVs (`results/per_problem/*.csv` keep one row per
   problem), if a reviewer asks.
