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
