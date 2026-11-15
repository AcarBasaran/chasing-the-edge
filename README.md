# chasing-the-edge

Adaptive difficulty sampling for GRPO: does training at a model's *moving* edge of competence
create new reasoning ability? A COMP550 project by Eren Acar Basaran and Turan Berke Yakarer.

## What the project is

The paper we build on, *On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning
Language Models* (Zhang, Neubig, Yue; [arXiv 2512.07783](https://arxiv.org/abs/2512.07783)), trains a
100M-parameter model on synthetic arithmetic-reasoning problems whose difficulty is the number
of operations (op 2–20). Its key finding: RL (GRPO) teaches the model genuinely new problems only
when training sits at the model's **edge of competence**, meaning problems it sometimes solves and
sometimes fails. The authors pick that band once (op 11–14) and keep it fixed for the whole run.
But the edge moves as the model learns, so a band that is right at step 0 is too easy later.
**Our extension** replaces the fixed band with samplers that track the moving edge: a threshold rule,
a bandit, and a uniform-random control. We ask whether this creates new ability (problems that go from
never solved to sometimes solved) or only makes the model more reliable on what it could already do.

## Why it matters

GRPO samples a group of G answers per problem and learns only from differences *inside* the group.
If all G answers are right, or all are wrong, that problem gives **exactly zero gradient**: compute
spent, nothing learned. Here is the base model we start from (measured in Gate 1, 200 problems ×
128 samples per level, strict process-verified scoring):

| level (op) | base success rate p | **groups with nonzero gradient (measured per problem, G=6)** | naive estimate 1−p⁸−(1−p)⁸ (ignores within-level variance) |
|---|---|---|---|
| 2–6 | 0.96–1.00 | **0.4–9%** | 3–32% |
| 7 | 0.84 | **25%** | 75% |
| 8 | 0.79 | **32%** | 86% |
| 9 | 0.75 | **34%** | 91% |
| 10 | 0.59 | **35%** | 99% |
| **11** | **0.49** | **38%** | ~99% |
| **12** | **0.24** | **33%** | ~89% |
| **13** | **0.004** | **1.5%** | ~3% |
| **14** | **0.00004** | **0.02%** | ~0.03% |
| 15–20 | 0 (0 of 128 on every problem) | **0%** | 0% |

Bold rows are the paper's fixed band. Two things stand out:

* **Half of the paper's band is dead weight at the start of training.** op 13 and op 14 almost never
  produce a gradient, so the run learns from op 11–12 only until the model improves.
* **Even the "good" levels give a gradient in only about a third of groups.** The naive estimate disagrees
  because success is not spread evenly inside a level: each problem has its own p, and those values pile up
  near 0 and near 1, so plugging in the level's average p first overstates the share. The measured column
  averages the formula over individual problems. The real GRPO groups in our smoke run agree with it:
  46.5% at op 11 and 37.1% at op 12. The measured column uses our G = 6 rollouts per group; the naive
  estimate uses G = 8, as in the project brief.
* **The level is a coarse label for difficulty.** At op 11, 62% of problems are either almost never
  (p < 0.1) or almost always (p > 0.9) solved; op 10 and op 12 look the same
  (`results/per_problem_p_hist.png`, `results/per_problem_p_by_level.csv`). Picking problems by their own
  measured p rather than by level could waste far fewer rollouts. *Future work, not built.*

Table: `results/base_model_p_by_level.csv`, `results/informative_fraction_by_level.csv`. Plot:
`results/informative_fraction_by_level.png`. Details: [docs/gate1.md](docs/gate1.md).

## What's done

* **Checkpoint and data.** The paper's pretrained base model (`extrapolation_rl/id2-10_0.2easy_0.3medium_0.5hard/base`,
  the mixture of its main experiment) and the problem dataset are downloaded and load correctly. Held-out test
  sets are built and kept disjoint from everything RL can see: easy op 2–10 (225 problems), medium op 11–14
  (800), hard op 15–20 (204).
* **The base model reproduces the paper** (Gate 1): pass@1 and pass@128 match Fig. 3 / Fig. 9 within
  figure-reading error on every band. The 1024-token answer limit is not what stops the model.
  Its correct answers stay under 440 tokens; it fails on hard problems by stopping too early.
* **The paper's released RL checkpoint also reproduces its Fig. 3**: hard-set pass@128 is 23.5% (paper ≈ 26%),
  and 48 of the 204 hard problems go from 0/128 to solvable. This is our reference for the baseline
  ([docs/gate3.md](docs/gate3.md)).
* **Four samplers** (`samplers/core.py`): fixed band, uniform random, threshold, bandit. They plug into verl's
  existing curriculum hook, so the training code itself is unchanged. They are tested on a simulated model
  first (`samplers/mock.py`, 36 unit tests).
  The simulation caught a real bug: the threshold sampler **oscillated** (60+ reversals in 500 steps), because
  neighbouring levels sit on opposite sides of any target band. Fix: it no longer moves into a level that
  was just measured on the wrong side. Result: 6 moves, 0 reversals, same learning gain.
* **Smoke runs** (Gate 2): one GRPO step, then 20 steps on one A40 with the fixed band (reward 0.21 → 0.38),
  then 20 steps with the threshold sampler choosing levels inside verl (level moved 10 → 14).
* **Environment fixes** needed to get verl running on VALAR: flash-attn built from source (the prebuilt
  wheel needs a newer glibc), uvloop pinned to 0.21, numpy kept at 2.x, `ROCR_VISIBLE_DEVICES` unset in
  jobs, and Ray's temp files kept on node-local `/tmp`. All of this lives in the scripts under `slurm/`.
* **Full-run setup**: one shared config for all six arms (`configs/rl_base.yaml`), so the arms differ only in
  the sampler; a 4-GPU launcher; a 950k-problem training pool; and a tracking set that measures every
  level's success rate every 10 steps.

## What's next

| target date | what |
|---|---|
| Nov 16 | Fixed-band baseline run (4 GPUs, 100 steps = half the paper's ~195) |
| Nov 17 | Measure how fast the edge moves in the baseline; set the bandit's probe interval *m* from it |
| Nov 17 | Email the authors about which reward produced their released checkpoint (draft: `results/email_to_authors.md`) |
| Nov 22 | **Gate 3**: our baseline vs the released checkpoint, same shape and roughly same size |
| Nov 24 | Threshold runs, τ = 0.1, 0.3, 0.5 (τ = 0.3 first: it is the Dec 1 comparison) |
| Nov 24 | Uniform-random baseline run |
| Nov 26 | Bandit run (α = 0.3) |
| Nov 28 | pass@128 on every run, plus the count of hard problems that go from 0/128 to solvable |
| Nov 30 | Extra seeds for the main arms |
| Dec 1 | **Gate 4**: threshold vs fixed band, one τ |
| **Dec 3** | **Progress presentation** |
| **Dec 6** | **Progress report** |

## How to run

Everything runs through SLURM on the `ai` partition, never on the login node. Heavy files live on scratch.

1. **Paths.** `source slurm/env.sh` sets `EDGE_SCRATCH`, `EDGE_REPO`, `UPSTREAM` (the paper's code, cloned
   next to this repo), `HF_HOME` on scratch, and `EDGE_PY` (the Python env). The defaults point at Eren's
   directories. Export your own `EDGE_SCRATCH` / `EDGE_REPO` / `UPSTREAM` before sourcing, and note that the
   `#SBATCH --output` lines and the `source` line in each script use Eren's paths.
2. **Environment.** Exact versions we use:

   | component | version |
   |---|---|
   | paper's code ([Interplay-LM-Reasoning](https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning)) | commit **`ab728f05d81de9af38d0ca155a84166b037e355a`** (2026-06-08), unmodified |
   | verl | **0.5.0.dev**, the copy inside that commit (`verl/`), used through `PYTHONPATH`, not pip-installed |
   | Python | 3.12 (conda env at `$EDGE_SCRATCH/envs/edge`) |
   | torch / CUDA | 2.7.1 / cu126 |
   | vLLM | 0.10.1.1 |
   | transformers | **4.55.4** (pip otherwise picks 5.x, which breaks loading) |
   | flash-attn | 2.8.3, built from source for A40/A6000 (`sm_80`) |
   | ray / tensordict / numpy / uvloop | 2.60.0 / 0.10.0 / 2.2.6 / **0.21.0** |
   | everything else | `requirements-lock.txt` (full `pip freeze`, 223 packages) |

   We never change the paper's code: our samplers plug into verl's existing curriculum-sampler hook
   (`data.sampler.class_path` in the YAML).

   To rebuild from scratch (clones the paper's code at that commit, creates the env, installs verl's
   dependencies, compiles flash-attn, downloads the model and data, then checks the result against the lock file):
   ```bash
   git clone https://github.com/AcarBasaran/chasing-the-edge && cd chasing-the-edge
   export EDGE_SCRATCH=/scratch/$USER/chasing-the-edge EDGE_REPO=$PWD UPSTREAM=$(dirname $PWD)/Interplay-LM-Reasoning
   sbatch --wait --cpus-per-task=32 --mem=160G --time=03:00:00 slurm/run_cpu.sbatch 'bash slurm/setup_env.sh'
   sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY -m pytest -q tests'      # 36 tests should pass
   ```
   `slurm/setup_env.sh` chains the exact steps used on 2026-11-15. Each step ran individually, but the chained
   script has not been run end-to-end, since it refuses to overwrite an existing env. Known traps it handles:
   the prebuilt flash-attn wheel needs a newer glibc than VALAR's; uvloop ≥ 0.22 breaks verl's worker startup;
   verl pins numpy < 2, which breaks pandas and scipy, so we keep numpy 2.2.6. Two traps are handled in the job
   scripts instead: SLURM sets `ROCR_VISIBLE_DEVICES`, which verl rejects, and Ray's sockets can't live on
   scratch's filesystem.
3. **Jobs.**

   | script | what it does |
   |---|---|
   | `slurm/run_cpu.sbatch '<cmd>'` | any CPU-only command from the repo root (tests, tables, plots) |
   | `slurm/gate1_eval.sbatch` | pass@k evaluation with the paper's own scorer. Array tasks 0–18 = val op 2–20, 19–21 = easy/medium/hard. Set `MODEL=`, `TEMP=`, `RUN_DIR=` to evaluate any checkpoint |
   | `slurm/rl_run.sbatch <arm> <seed>` | a full GRPO run on 4 GPUs. Arms: `fixed_band uniform threshold_tau1 threshold_tau3 threshold_tau5 bandit` |
   | `slurm/merge_ckpt.sh <run_id> <step>` | turn a training checkpoint into a model folder the evaluator can load |
   | `slurm/gate2_smoke.sbatch <config>` | 1-GPU smoke runs |

   Tests: `sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY -m pytest -q tests'`.
4. **Where things land** (all under `/scratch/ebasaran22/chasing-the-edge/`):
   * `outputs/runs/<arm>_s<seed>/`: `train.log`, `sampler_log.jsonl` (the sampler's choice every step),
     `resolved_config.yaml`, `git.txt` (code version), `ckpt/global_step_*/`
   * `logs/`: every SLURM job's stdout
   * `outputs/gate1_base_eval/`, `outputs/gate3_released_rl/`: raw generations from the evaluations
   * `hf_snapshots/`: model and dataset downloads; `data/pool_50k/`: the RL training pool

## Where things are

| folder | contents |
|---|---|
| `samplers/` | the four samplers (`core.py`), the verl hook (`verl_adapter.py`), the simulated-model harness (`mock.py`), the pool builder |
| `eval/` | pass@k, strict scoring, test-set builders, every table and plot script |
| `configs/` | one YAML per run type (`rl_*.yaml` for the six arms) |
| `slurm/` | job scripts and environment setup |
| `data/` | test sets (`testsets/`) and the tracking set (`track32/`) |
| `results/` | CSVs and plots: the numbers behind everything above |
| `docs/` | gate reports |
| `tests/` | unit tests |

Reports: [Gate 1 report](docs/gate1.md) · [Gate 3 notes](docs/gate3.md) ·
[overnight log](results/overnight.md) · project brief for Claude Code sessions: [CLAUDE.md](CLAUDE.md).
