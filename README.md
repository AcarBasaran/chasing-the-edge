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
spent, nothing learned. So what matters is where the model is *right now*, and that changes during training.

**The fixed band never moves past op 14, while op 15–16 become learnable.** In our fixed-band baseline
(the paper's method, strict reward, 100 steps), the measured success rate on levels the band never trains on rises:

| level (op) | p at step 0 | p at step 50 | p at step 100 |
|---|---|---|---|
| 13 (in band) | 0.01 | 0.43 | 0.46 |
| 14 (in band) | 0 | 0.28 | 0.31 |
| **15** (never trained) | 0 | 0.23 | **0.25** |
| **16** (never trained) | 0 | 0.09 | **0.12** |

(mean of the two fixed-band seeds; tracking set, 32 problems per level, 8 samples, T=1.0, every 10 steps;
`outputs/runs/fixed_band_s{0,1}/edge_speed.csv`)

By step 50, op 15–16 are where the model sometimes succeeds and sometimes fails, which is exactly the regime
GRPO learns from. A fixed band keeps spending the budget on op 11–14 instead. A sampler that follows the edge
can move there.

**Step 0 is a snapshot.** Here is the base model before any training (Gate 1, 200 problems × 128 samples
per level, strict process-verified scoring):

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

Bold rows are the paper's fixed band. At step 0, op 13 and 14 almost never produce a gradient. **This does not
last.** Training on op 11–12 generalises upward, and in the baseline run (seed 0) op 13 reaches 37% and op 14 29%
informative groups by step 20 (measured on the groups GRPO trained; time series in
[docs/gate1.md](docs/gate1.md), all levels in `results/informative_over_training.png`). The band's problem
is not dead levels; it is that it stays put while the edge moves on.

Two more things the step-0 measurement shows:

* **Even the "good" levels give a gradient in only about a third of groups.** The naive estimate disagrees
  because success is not spread evenly inside a level: each problem has its own p, and those values pile up
  near 0 and near 1, so plugging in the level's average p first overstates the share. The measured column
  averages the formula over individual problems. The real GRPO groups agree with it: 46.5% at op 11 and
  37.1% at op 12. The measured column uses our G = 6 rollouts per group; the naive estimate uses G = 8, as in
  the project brief.
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
  and 48 of the 204 hard problems go from 0/128 to solvable.
* **Gate 3 passed (2026-11-20): our fixed-band baseline reproduces the paper.** Strict reward, 100 steps, one
  A40: hard pass@128 23.5% and 48 problems newly solved in both seeds, the same as the released checkpoint.
  Every test set is within 1 point ([docs/gate3.md](docs/gate3.md)).
* **Results, all arms, 1–3 seeds each** ([docs/results_seeds.md](docs/results_seeds.md); mean ± SD over seeds).
  Threshold τ=0.1 newly solves **95.0 ± 1.7** of the 204 hard problems (fixed band: 48). On op 18–20, which it
  never trained on, it solves **27.7 ± 0.6 of 102**, against 20.0 ± 1.0 for uniform random (which did train
  there, but 97% of those groups had zero gradient; sign test p = 0.002) and 6 for the fixed band. A control
  that samples τ=0.1's range (op 11–17) uniformly from step 0 gets 86.3 ± 2.1, i.e. 81% of τ=0.1's gain: the
  right range is most of the win, the schedule adds +8.7 problems (p = 0.017), and τ=0.1 is what found the
  range. τ=0.5 (40% informative groups) learns nothing new; the bandit (40.7 ± 3.5) ends below the fixed band
  ([docs/bandit_analysis.md](docs/bandit_analysis.md)). Cost: τ=0.1's easy pass@1 is 1.5 points lower,
  lower than the fixed band's and the control's,
  consistently across seeds; pass@128 is unchanged. The seed-0 write-up is kept in
  [docs/results_seed0.md](docs/results_seed0.md) for its figures and per-run detail.
* **Four samplers** (`samplers/core.py`): fixed band, uniform random, threshold, bandit. They plug into verl's
  existing curriculum hook, so the training code itself is unchanged. They are tested on a simulated model
  first (`samplers/mock.py`; 46 unit tests across the repo).
  The simulation caught a real bug: the threshold sampler **oscillated** (60+ reversals in 500 steps), because
  neighbouring levels sit on opposite sides of any target band. Fix: it no longer moves into a level that
  was just measured on the wrong side. Result: 6 moves, 0 reversals, same learning gain.
* **Smoke runs** (Gate 2): one GRPO step, then 20 steps on one A40 with the fixed band (reward 0.21 → 0.38),
  then 20 steps with the threshold sampler choosing levels inside verl (level moved 10 → 14).
* **Environment fixes** needed to get verl running on VALAR: flash-attn built from source (the prebuilt
  wheel needs a newer glibc), uvloop pinned to 0.21, numpy kept at 2.x, `ROCR_VISIBLE_DEVICES` unset in
  jobs, and Ray's temp files kept on node-local `/tmp`. All of this lives in the scripts under `slurm/`.
* **Full-run setup**: one shared config for all seven arms (`configs/rl_base.yaml`), so the arms differ only in
  the sampler; a 4-GPU launcher; a 950k-problem training pool; and a tracking set that measures every
  level's success rate every 10 steps.

## What's next

| when | what |
|---|---|
| done | All 17 runs for Dec 3 (seed counts in [docs/results_seeds.md](docs/results_seeds.md)); no new runs before Dec 3 |
| **Dec 3** | **Progress presentation** ([docs/slides_outline.md](docs/slides_outline.md), [speaker script](docs/speaker_notes.md)) |
| **Dec 6** | **Progress report** (LaTeX: [report/main.tex](report/main.tex); build with `report/build.sbatch`) |
| January | τ = 0.15 / 0.2, range-finding then uniform, 200-step runs, bandit self-reward fix (report §6) |

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
   sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY -m pytest -q tests'      # 46 tests should pass
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
   | `slurm/rl_run.sbatch <arm> <seed>` | a full GRPO run (we use 1 GPU: `--gres=gpu:1 --cpus-per-task=12 --mem=40G`; same effective batch). Arms: `fixed_band uniform uniform_11_17 threshold_tau1 threshold_tau3 threshold_tau5 bandit` |
   | `slurm/merge_ckpt.sh <run_id> <step>` | turn a training checkpoint into a model folder the evaluator can load |
   | `slurm/post_run.sh <run_id> <train_jobid>` | queue merge → pass@128 eval → analysis behind a training job (SLURM dependencies) |
   | `slurm/drain_queue.sh` | submit the runs listed in `$EDGE_SCRATCH/outputs/pending_runs.txt` as the job limit allows |
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
| `configs/` | one YAML per run type (`rl_*.yaml` for the seven arms, plus the bandit m = 10 ablation) |
| `slurm/` | job scripts and environment setup |
| `data/` | test sets (`testsets/`) and the tracking set (`track32/`) |
| `results/` | CSVs and plots: the numbers behind everything above |
| `docs/` | gate reports, results across seeds, bandit analysis, slide text, speaker script, budget |
| `report/` | the progress report in LaTeX (ACL style) |
| `tests/` | unit tests |

Reports: [results across seeds](docs/results_seeds.md) · [bandit analysis](docs/bandit_analysis.md) ·
[Gate 1 report](docs/gate1.md) · [Gate 3 notes](docs/gate3.md) ·
[overnight log](results/overnight.md) · project brief for Claude Code sessions: [CLAUDE.md](CLAUDE.md).
