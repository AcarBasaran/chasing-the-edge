# Overnight log — 2026-11-15

## Summary
(filled in at the end)

## Log

### 04:00–05:10 Gate 1 (before the overnight brief arrived)
- Env: `$EDGE_SCRATCH/envs/edge` (py3.12, torch 2.7.1+cu126, vLLM 0.10.1.1, transformers pinned
  to 4.55.4 — pip had pulled 5.23, which the checkpoint/vLLM 0.10 don't expect).
- Base model `extrapolation_rl/id2-10_0.2easy_0.3medium_0.5hard/base` loads; full eval of
  levels 2–20 (val, 200×128) + easy/medium/hard test sets done: 22 SLURM tasks, 1.74 GPU-h.
  Reproduces paper within figure-reading error (31/32 checks ±6pp). Details: `docs/gate1.md`.
- **Rule deviation (before the "never on login node" rule arrived):** gold-graph building,
  `eval/build_testsets.py`, `eval/check_parser_on_gold.py` and tokenizer inspection ran on the
  login node (CPU, <30 s each, no model weights). Everything from here on goes through SLURM
  (`slurm/run_cpu.sbatch` for CPU work).

### A1 — CLAUDE.md corrected
- Real prompt/solution format; `op` ≠ solution steps; non-unique `id`.
- Reward-vs-eval mismatch documented; rule: samplers use the TRAINING reward, test metrics use EVAL.
- Sampler hook: `data.sampler.class_path`/`class_name` → `AbstractCurriculumSampler` (`__init__(data_source, data_config)`, `__iter__`, `__len__`, `update(batch)`), needs `dataloader_num_workers=0`. Clarified that subclassing this published interface is the intended extension point (not a "verl internals" subclass).
- Chosen mixture + Gate-1 numbers + token-cap finding recorded.

### Gate-1 op12 cross-check (job 2021184)
- Only failed paper check was medium-test pass@128 (35.5 vs ≈43.5), caused by op12 (0.46 on 50 problems vs 0.675 on val).
- Full test/op12-1k × 128: pass@1 24.2%, pass@128 62.7% → consistent with val/paper. Unlucky 50-problem draw. Not a reproduction failure.
- Note for later: medium test set should get more problems per level for per-level pass@128.

### A2 — samplers (`samplers/core.py`, `samplers/verl_adapter.py`)
- Two layers: pure-Python `DifficultySampler`s (FixedBand, UniformRandom, Threshold, Bandit) + one
  `EdgeCurriculumSampler(AbstractCurriculumSampler)` adapter: `__init__(data_source, data_config)`,
  lazy `__iter__` (batch t+1 is drawn only after `update(t)`, needs `dataloader_num_workers=0`),
  `__len__ = total_steps * train_batch_size`, `update(batch)`.
- Success = TRAINING reward: `token_level_scores.sum(-1) >= 1` (strict reward is 0/1); level from
  `extra_info['op']` (fallback `data_source` "difficulty-5B/{op}"). Also logs per-level fraction of
  informative GRPO groups via `uid`. Log: JSONL at `data.sampler.log_path` + stdout every step.
- **Probes are in-batch**: on probe steps k prompts/level replace part of the batch → trained on and
  counted in the same B×n rollout budget as every method. No generation outside the batch.
  Bandit forces probes (default m=25); others default to none.
- **Decision — Threshold tau=0.1**: lower bound clamped to `max(tau-delta, 0.5*tau)` → band [0.05, 0.20].
  Inactive for tau 0.3 ([0.2,0.4]) and 0.5 ([0.4,0.6]), so the sweep differs only in tau.
- Not implemented: sampler `state_dict` for resuming a killed run (not needed for smoke runs).

### A3 — mock harness (`samplers/mock.py`, plots in `results/mock/`)
- Fake model: p_i = sigmoid((skill - L - b_i)/0.7), per-problem offset b_i ~ N(0, 0.5), skill0 = 11;
  skill += 0.012 × (mean informative-group fraction of the batch). B=256, G=8, 500 steps.
  Runs through the REAL verl adapter (iterator → fake rollouts → `update(fake DataProto)`).
- **Bug found & fixed — Threshold oscillation.** tau 0.3/0.5 reversed direction 60/89 times in 500 steps.
  Cause is structural, not noise: the edge is steeper than the band (one level up drops p from > hi to
  < lo), so patience can't help. Fix: known-side memory — never move into a neighbour whose fresh
  estimate (≤ `stale_after`=50 measurements old) puts it on the wrong side. After: 6 moves, 0 reversals,
  skill gain unchanged (4.95 vs 5.13). Regression test added. The `_nomemory` row in the CSV keeps the evidence.
- **Bandit (spec: alpha 0.1, m 25)** lags the edge by ~2.3 levels: Q is an EMA of probe-to-probe progress,
  so it reflects where progress was ~10 probes ago. alpha 0.3 or m=10 cut lag to ~0.8 and add +0.35 skill.
  Kept spec defaults; **recommend alpha=0.3 (or m=10) for the real runs — your call.**
- Caveat for interpreting the mock: learning ∝ informative fraction, i.e. it models gradient throughput
  only, not headroom. It is built to favour tau≈0.5 and **cannot answer the tau question**; it only
  checks the samplers behave (no stalls, no oscillation, tracks skill).
- Tests: `tests/test_samplers.py` (core rules, tau=0.1 back-off, memory/expiry, bandit math, adapter
  laziness/probes/reshuffle/score threshold, mock regressions). 34/34 pass (`pytest -q tests`, CPU job).

### A4 — `eval/passk.py`
- `pass_at_k(n,c,k) = 1 - C(n-c,k)/C(n,k)` (math.comb), plus `informative_fraction(p,G)`.
- `tests/test_passk.py`: brute-force subset enumeration, exact formula, pass@1 = c/n, edges (k=n, n-c<k),
  monotone in k and c, Monte-Carlo unbiasedness vs 1-(1-p)^k, invalid input, CLAUDE.md G=8 table. All pass.
- Note: upstream's `aggregate_graph_pass_results.py` uses "any of the first k samples" — unbiased only at k=n.

### B6 — `results/base_per_problem.csv` (5,429 problems)
- Every evaluated problem: val op2–20 (3,800), test easy/medium/hard (629), full test/op12 (1,000).
- Per problem: n=128, `c_eval` (paper criterion), `c_train` (training reward), `c_answer` (answer-only).
  Keyed by `uid` = sha1(problem‖question). Upstream and our re-score agree on 100% of samples
  (asserted in `eval/export_base_per_problem.py`).
- The headline reference: test_hard has 204/204 problems at c_eval = 0 (all 0/128).

### B7 — per-level base table (`results/base_by_level_summary.csv`) and paper check
| L | p eval | p train | pass@128 | gen p95 tok | correct p95 tok | gold p95 tok | paper p@1 / p@128 |
|---|---|---|---|---|---|---|---|
| 2 | 0.9971 | 0.9960 | 1.000 | 198 | 198 | 196 |  |
| 3 | 0.9963 | 0.9927 | 1.000 | 229 | 229 | 228 |  |
| 4 | 0.9825 | 0.9817 | 1.000 | 247 | 246 | 245 |  |
| 5 | 0.9616 | 0.9528 | 1.000 | 270 | 263 | 265 |  |
| 6 | 0.9732 | 0.9723 | 1.000 | 289 | 287 | 287 |  |
| 7 | 0.8423 | 0.8416 | 1.000 | 320 | 315 | 317 |  |
| 8 | 0.7897 | 0.7851 | 0.995 | 347 | 334 | 341 |  |
| 9 | 0.7459 | 0.7449 | 0.995 | 370 | 358 | 365 |  |
| 10 | 0.5864 | 0.5862 | 0.955 | 392 | 370 | 388 |  |
| 11 | 0.4888 | 0.4888 | 0.920 | 412 | 393 | 414 | 49.0 / 91.0 |
| 12 | 0.2378 | 0.2378 | 0.675 | 406 | 403 | 427 | 27.0 / 65.0 |
| 13 | 0.0036 | 0.0036 | 0.070 | 402 | 439 | 449 | 1.0 / 7.0 |
| 14 | 0.0000 | 0.0000 | 0.005 | 410 | 435 | 492 | 0.0 / 0.0 |
| 15 | 0.0000 | 0.0000 | 0.000 | 399 | — | 507 |  |
| 16 | 0.0000 | 0.0000 | 0.000 | 403 | — | 527 |  |
| 17 | 0.0000 | 0.0000 | 0.000 | 416 | — | 570 |  |
| 18 | 0.0000 | 0.0000 | 0.000 | 411 | — | 586 |  |
| 19 | 0.0000 | 0.0000 | 0.000 | 406 | — | 603 |  |
| 20 | 0.0000 | 0.0000 | 0.000 | 410 | — | 631 |  |

- **Matches the paper** (bands: op2-10 pass@1 87.5 vs ≈87, pass@128 99.4 vs ≈99.5; op11-14 18.3/41.8 vs
  ≈19/42; op15-20 0/0 vs 0/0; per-op 11–14 all within ~3pp). Proceeding to section C.
- p_train = p_eval for op ≥ 11; up to 0.9pp lower at op ≤ 10 (extra-node penalty).
- **Length finding.** The model's own generations saturate at p95 ≈ 400–415 tokens for every op ≥ 11
  (correct-solution max ever seen: 463). That is the length of the longest pretraining-range (op ≤ 10)
  gold solutions (~430). Gold op14+ solutions need 490–630 tokens. So the 1024 cap is not the ceiling;
  a learned length prior is. Worth watching whether RL lengthens outputs — and re-checking the cap then.
