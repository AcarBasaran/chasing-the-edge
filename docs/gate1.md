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
