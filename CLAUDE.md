# Chasing the Edge — project brief

## What this project is

A COMP550 course project extending "On the Interplay of Pre-Training,
Mid-Training, and RL on Reasoning Language Models" (Zhang, Neubig, Yue;
arXiv 2512.07783). Team: Eren Acar Basaran, Turan Berke Yakarer.

The paper's finding: RL (GRPO) gives a small LM genuinely NEW reasoning
ability only when training problems sit at its "edge of competence" —
hard enough that it often fails, easy enough that it sometimes succeeds.
They picked that difficulty band once (11-14 reasoning steps) and kept
it fixed for the whole RL run.

Our extension: the edge MOVES as the model learns. A band that was
informative at step 0 is too easy by step 500. We replace the fixed
sampler with adaptive ones that track the moving edge, and ask whether
this produces new ability or only better reliability.

## The mechanism that justifies all of this

GRPO computes advantage within a group of G rollouts per problem:

    A_i = (r_i - mean(r)) / std(r)

If all G rollouts get the same reward (all right or all wrong), every
A_i = 0 and the gradient contribution is EXACTLY ZERO. Compute spent,
nothing learned.

With success probability p, the fraction of informative groups is:

    1 - p^G - (1-p)^G

At G=8: p=0.01 -> 7.7%, p=0.1 -> 57%, p=0.3 -> 94%, p=0.5 -> 99%,
p=0.9 -> 57%, p=0.99 -> 7.7%.
OUR G IS 6 (the paper's rollout.n; upstream rl-200steps configs). Measured on the
base model: results/informative_fraction_by_level.csv/.png (per-problem unbiased
estimate; the level-mean formula overstates it because p varies across problems).

That curve IS the paper's finding. Everything we build follows from it.

## Resources (verified working)

HuggingFace org: Interplay-LM-Reasoning

- extrapolation_rl — contains base/ (the PRETRAINED checkpoint we start
  from) and rl/ (their finished fixed-band RL runs, for baseline
  comparison). Four pretraining mixtures available:
    id2-10_0.2easy_0.3medium_0.5hard
    id2-10_0.5easy_0.3medium_0.2hard
    id2-10_0.4995easy_0.4995medium_0.001hard
    id2-10_0.475easy_0.475medium_0.05hard
  Load with: AutoModelForCausalLM.from_pretrained(repo_id, subfolder=...)
  We have confirmed the checkpoint downloads and loads.
  CHOSEN MIXTURE: id2-10_0.2easy_0.3medium_0.5hard/base — the pretraining
  mix of the paper's main extrapolation experiment (Sec. 3 / Fig. 3; upstream
  scripts/run_scripts/run_composition_rl.sh). The other three are pretraining
  ablations. Gate 1 reproduced its base numbers (docs/gate1.md).

- extrapolation_midtrain — mid-trained checkpoints. NOT our starting
  point; the model has already seen medium problems.

- context_pretrain / context_pretrain_2 — for the paper's SECOND
  experiment (story contexts). Not relevant to us.

- composition dataset — the synthetic problems, each labeled with its
  number of reasoning steps (= difficulty level).

GitHub: Interplay-LM-Reasoning/Interplay-LM-Reasoning

## The model and data

- 100M-parameter Qwen2.5-style decoder-only transformer.
- Small custom vocabulary (low thousands of tokens).
- Problems are DAGs: nodes are variables, edges are dependencies, ending
  at an answer node. Number of operations = difficulty level.
- Each problem is wrapped in a story context (zoo, school). The
  arithmetic is identical across contexts. WE DO NOT VARY CONTEXT.
- Format (verified against data + upstream code):
    <question> {problem} {question} </question> <solution> Define X as t; so t = 4. Define Y as q; so q = 2 * t = 8. </solution> <answer> 8 </answer>
  The prompt is everything up to and including "<solution>"; generation
  stops at "</answer>". Difficulty = the `op` field (operation count). op is
  NOT the number of solution steps: an op-20 problem has ~10 "Define" steps.
  The dataset `id` field is not unique — key problems by content hash.

## Critical constraints

- PROCESS-VERIFIED SCORING. Reward requires BOTH the reasoning steps and
  the final answer to be correct. A parser reconstructs the predicted
  dependency graph. Rewards are stricter than naive answer accuracy, so
  observed p values run lower than you'd expect.
  If rewards look unexpectedly low, SUSPECT THE PARSER BEFORE THE MODEL.
  (Gate 1 check: gold solutions score 100% through the parser at every level.)

  REWARD vs EVAL MISMATCH. Upstream has two strict criteria:
    eval  (scripts/score_dependency_graphs.py, paper App. A.4): every gold
          node right + answer right; EXTRA predicted nodes ALLOWED.
    train (verl/reward_fn.py::compute_score_with_step_process with
          answer_weight=1, step_weight=0, zero_on_process_mismatch=True):
          extra predicted nodes ALSO zero the reward.
  Samplers MUST use the TRAINING reward (that is what GRPO sees). Report
  test metrics with the EVAL criterion (that is what the paper reports).
  The gap is <=0.9pp at op<=10 and zero at op>=11. eval/strict_score.py
  computes both.

- 1024-TOKEN ANSWER LIMIT. We cap difficulty at 20 steps because of it.
  Danger: if solutions at a level exceed the cap, they truncate, the
  parser fails, reward is 0, and the sampler reads that level as "too
  hard" and backs off. The cap then becomes an invisible ceiling that
  masquerades as the edge. Measure the token-length distribution of
  correct solutions per level on day one.
  MEASURED (Gate 1): NOT binding for the base model. Correct solutions
  p95 <= 440 tokens; gold solutions need <= 700 even at op 20; truncation
  <= 0.008%. Re-check on RL checkpoints (RL may lengthen outputs).
  Separate trap: some PROMPTS exceed max_prompt_length=1024 and verl's
  filter_overlong_prompts silently drops them from the pool.

## What we build

verl (ByteDance RL framework: Ray + vLLM + FSDP, Hydra YAML configs) runs
the training. The loop is: dataloader yields prompts -> vLLM generates G
completions -> reward function scores -> advantages -> actor update ->
weight sync.

WE CHANGE ONLY THE FIRST ARROW. Everything downstream is untouched.

THE HOOK (already in verl, no patching needed): verl/trainer/main_ppo.py::
create_rl_sampler loads `data.sampler.class_path` / `data.sampler.class_name`
from the YAML. The class must subclass
verl.experimental.dataset.sampler.AbstractCurriculumSampler:
    __init__(self, data_source, data_config)   # data_source = the train Dataset
    __iter__ / __len__                          # torch Sampler: yields dataset indices
    update(self, batch: DataProto)              # called by RayPPOTrainer.fit()
                                                # after every step, rewards in batch
Requires data.dataloader_num_workers=0 (else prefetching defeats the sampler).
Batches are consumed at ray_trainer.py `for batch_dict in self.train_dataloader`.

Do NOT subclass or monkey-patch verl internals (subclassing the published
AbstractCurriculumSampler interface above is the intended extension point).
Write a standalone class:

    class DifficultySampler:
        def next_level(self) -> int: ...
        def update(self, level: int, success_rate: float): ...

Four implementations:

1. FixedBand — always sample 11-14. The paper's method, our main baseline.
2. UniformRandom — uniform over all levels. Controls for "harder problems
   help" vs "timing matters".
3. Threshold — keeps one active level L and target success rate tau.
   Move up if p_L > tau + delta, down if p_L < tau - delta.
   Sweep tau in {0.1, 0.3, 0.5}, delta = 0.1.
   NOTE: with tau=0.1 and delta=0.1 the lower bound is p < 0, which is
   unreachable. Needs an asymmetric delta or a clamp. Decide and document.
   DECIDED: lo = max(tau - delta, 0.5*tau) -> [0.05, 0.2] for tau=0.1
   (clamp inactive for 0.3, 0.5). Plus known-side memory: never move into a
   neighbour freshly measured on the wrong side — without it the sampler
   ping-pongs, because adjacent levels straddle any tau+-delta band
   (mock harness; samplers/core.py docstring).
4. Bandit — each level is an arm of a non-stationary bandit.
   reward:  r_L(t) = p_L(t) - p_L(t-m)
   value:   Q_L <- (1-alpha) Q_L + alpha * r_L(t)
   policy:  argmax Q_L with prob 1-epsilon, else uniform random level.
   alpha = 0.1, epsilon = 0.1.
   DECIDED 2026-11-15: alpha = 0.3 (0.1 lags the edge ~2.3 levels in the mock, 0.3 ~0.8);
   m = 25 for now. Re-set m from the edge speed MEASURED in the fixed-band baseline
   (eval/edge_speed.py: m <= 1 / levels-per-step), not from the mock. Report it.

Implementation detail: pre-generate a pool of problems bucketed by op
count; the sampler picks a bucket. Do not generate inside the training loop.

## The tau question is a real experiment, not a hyperparameter sweep

Two forces pull against each other:
- p = 0.5 maximises gradient throughput (most groups informative).
- Low p maximises headroom (levels the model already half-solves have
  little left to teach).
Whichever tau wins is a reportable finding. Treat it as such.

## Probe rollouts and the budget trap

The bandit needs p_L for EVERY arm, not just the one being trained.
Two options:
  (1) periodic evaluation sweep — every m steps, generate a few rollouts
      at every level. Clean estimates, costs compute. PREFERRED, because
      we want these measurements for our plots anyway.
  (2) update only the pulled arm — free, but estimates go stale, and the
      problem is non-stationary so stale means wrong.

VALIDITY TRAP: if probe rollouts don't count against the training budget,
the adaptive methods get more compute than the fixed baseline and any
advantage is confounded. PROBE ROLLOUTS MUST COUNT IN THE BUDGET.
Equalise on total rollouts generated, not problems trained on.

Settings: m (probe interval) = 25-50 steps. At least 8 rollouts per level
per probe, ideally 16 (estimating p=0.3 from 8 samples has SE ~0.16 —
that is a lot of noise feeding a threshold rule).

## Evaluation

pass@1 — how reliably the model does what it can do.
pass@128 — whether a problem is reachable at all.

Use the unbiased estimator, do NOT just generate 128 and check "any correct":

    pass@k = 1 - C(n-c, k) / C(n, k)

with n >= k samples and c correct.

THE HEADLINE NUMBER: save the base model's per-problem pass@128 BEFORE
any RL. Then count problems that scored 0/128 before and >0/128 after.
That is the cleanest possible evidence of new capability — far stronger
than an aggregate percentage.

Test sets: held-out easy, medium (11-14 steps), hard (15-20 steps).
Main result: pass@128 on hard.

Also track:
- which level each sampler chose over training, plotted against measured
  success rates (verifies it follows the edge)
- easy-set accuracy over training (catches forgetting)
- generated tokens and GPU-hours per run

Budget pass@128 evaluation deliberately: 128 samples x N problems x ~1000
tokens x number of runs. A 200-problem hard test set is likely enough and
much cheaper than 1000.

## Build order (gates — do not skip ahead)

MOCK FIRST. Before touching verl, build a fake model whose success rate at
level L is logistic in (skill - L), with skill rising slowly over simulated
steps. Run all four samplers against it. This catches oscillation,
never-moves, and off-by-one EMA bugs in an afternoon instead of a week of
GPU time. HIGHEST VALUE THING WE CAN DO.

Gate 1 (Nov 8): base/ checkpoint loads; reproduce the paper's reported
base-model numbers. Produce a table of p_L vs L. This tells us where the
edge actually starts and sets our tau range and level range.
  DONE 2026-11-15 (results/base_model_p_by_level.csv). Base p_L (T=0.7):
  op10 .59, op11 .49, op12 .24, op13 .004, op14 .00004, op15-20 exactly 0
  (0/128 on every problem). The edge is razor-sharp: at step 0 the 11-14
  band has two informative levels and two dead ones.

Gate 2 (Nov 15): ONE GRPO step runs end-to-end. Not a full run. Expect
version hell: vLLM/torch/CUDA mismatches, Ray not finding GPUs under
SLURM, custom tokenizer loading. Budget days.

THRESHOLD DECIDED 2026-11-15: patience=2, start_level=11 (= fixed band's first level).
Base p(op11) ~0.49, so tau=0.1/0.3 move UP after 2 steps while tau=0.5 HOLDS at 11.
That is a real difference between the arms: log and report it.

Gate 3 (Nov 22): fixed-band baseline reproduced, compared against their
released rl/op11-14_uniform checkpoint. Same shape, roughly same
magnitude — seeds differ so exact match isn't expected.
IF THE BASELINE IS FAR OFF, DO NOT PROCEED. Debug instead. A broken
baseline makes every later comparison meaningless.
  DECIDED 2026-11-15: THE baseline is OUR fixed-band run with the STRICT process
  reward. The released checkpoint was probably trained with the ANSWER-ONLY reward
  (config name + paper Sec. 6), so it is a loose sanity check only: score it under
  eval / train / answer-only criteria, report all three, document the mismatch
  (docs/gate3.md, results/gate3_released_rl_comparison.csv; email draft to the
  authors in results/email_to_authors.md).
  Full-run configs: configs/rl_base.yaml + rl_<arm>.yaml (only data.sampler differs),
  launcher slurm/rl_run.sbatch <arm> <seed>. Scale: docs/budget.md.

Gate 4 (Dec 1): threshold sampler vs baseline, one tau. This is the
Dec 3 progress presentation. A partial or negative result is fine;
having nothing is not.

December: remaining tau values, bandit, uniform random, seeds, full
pass@128 evaluation.

## Environment

- VALAR HPC (Koc University), SLURM. All training goes through sbatch,
  never on a login node.
- `module load git` is required (add to .bashrc).
- QoS `ai` PER USER: 8 running jobs, 8 GPUs, 240 CPUs, 350G memory. Job slots are
  shared with the user's other projects (CPU-only `restock-*` jobs) -- array tasks
  each take a slot, so prefer one 4-GPU job over four 1-GPU jobs.
- GPU nodes: A40 = ai18-ai26, A6000 = ai16 (use --exclude=ai[01-08],ai[11-14]).
- PyTorch only. verl, vLLM, transformers, safetensors are all PyTorch.
  Samplers are plain Python; pass@k uses math.comb.
- Home directory quota is limited. Keep checkpoints and rollout outputs
  on scratch, symlinked in. .gitignore already excludes them.

## Repo conventions

- slurm/   job scripts
- samplers/  the four sampler classes + mock harness
- eval/    pass@k harness, per-problem base pass@128 record, plotting
- configs/ one YAML per run, committed
- results/ one CSV: run_id, method, tau, seed, test_set, metric, value

EVERY RUN GETS A CONFIG FILE AND A DIRECTORY. We will run 20+ jobs; without
this we will not be able to say in January which number came from which
settings.

LOG THE SAMPLER'S CHOICES EVERY STEP: level chosen, p estimates, arm
values. This is our headline figure and it's free to log.

## Division of labour

- Training pipeline (verl, SLURM, the next_level() hook, checkpoints)
- Evaluation and analysis (pass@k harness, base-model pass@128 record,
  plots, mock harness)

The evaluation side can be built and tested with NO working training run,
using the released RL checkpoints as input. Neither person should ever be
fully blocked.

## Expected failure modes

- Adaptive matches but doesn't beat fixed band. MOST LIKELY OUTCOME. Still
  a result: the paper's hand-picked band was near-optimal. Strengthen it
  by showing WHERE the sampler went — if it converged to 11-14 on its own,
  that's a positive finding about the method.
- Sampler oscillates between two levels — probe estimates too noisy, or
  delta too small. More rollouts per probe, bigger delta, or hysteresis
  (require two consecutive measurements past threshold).
- Sampler runs to the top and sticks — probably the token cap. See above.
- pass@1 up but pass@128 flat for every method — real, reportable result
  (RL sharpens rather than teaches here). Check the estimator first.
- Too slow — cut in this order: seeds, test set size, tau sweep (keep 0.1
  and 0.5), training steps. Keep all four methods; the comparison IS the
  project.
