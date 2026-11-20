#!/bin/bash
# Queue everything that follows a training run, chained with SLURM dependencies, so it runs unattended:
#   1. merge   ckpt/global_step_<STEP> -> hf/step<STEP>                      (CPU, after training succeeds)
#   2. eval    pass@128 on test easy / medium / hard, T=0.7 (paper protocol)   (3 x 1 GPU, after merge)
#   3. analyse per-problem CSV, newly-solved count, method comparison, sampler plot, edge speed, timing
#                                                                             (CPU, after all evals)
#   bash slurm/post_run.sh <run_id> <train_jobid> [step=100]
# Run on the login node: it only calls sbatch. Outputs: results/per_problem/<run_id>.csv, results/*.csv|png,
# $EDGE_SCRATCH/outputs/runs/<run_id>/{hf,eval,edge_speed.*,timing.csv}, logs in $EDGE_SCRATCH/logs/.
set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh
RUN_ID="${1:?run id}"; TRAIN_JOB="${2:?training job id}"; STEP="${3:-100}"
R="${EDGE_SCRATCH}/outputs/runs/${RUN_ID}"
cd "${EDGE_REPO}"

merge=$(sbatch --parsable --job-name="merge-${RUN_ID}" --dependency="afterok:${TRAIN_JOB}" \
        slurm/run_cpu.sbatch "bash slurm/merge_ckpt.sh ${RUN_ID} ${STEP}")
ev=$(MODEL="${R}/hf/step${STEP}" RUN_DIR="${R}/eval" \
     sbatch --parsable --job-name="eval-${RUN_ID}" --dependency="afterok:${merge}" --array=19-21 \
     slurm/gate1_eval.sbatch)
an=$(sbatch --parsable --job-name="analyse-${RUN_ID}" --dependency="afterok:${ev}" slurm/run_cpu.sbatch \
     "mkdir -p results/per_problem && \
      \$EDGE_PY eval/export_base_per_problem.py --run-dir ${R}/eval --ckname step${STEP} --val-levels none \
          --out results/per_problem/${RUN_ID}.csv && \
      \$EDGE_PY eval/newly_solved.py && \$EDGE_PY eval/compare_methods.py && \
      `# parse_verl_log exits 1 when the level never changes (a Gate-2 smoke check); for a full run that is a result (tau=0.5), not a failure` \$EDGE_PY eval/parse_verl_log.py ${R}/train.log --csv ${R}/timing.csv --sampler-log ${R}/sampler_log.jsonl || true; \
      \$EDGE_PY eval/edge_speed.py ${R}/train.log --csv ${R}/edge_speed.csv --plot ${R}/edge_speed.png && \
      \$EDGE_PY eval/plot_sampler_levels.py \$(ls -d ${EDGE_SCRATCH}/outputs/runs/*_s[0-9] | grep -v _hung_) \
          --out results/sampler_levels.png; \
      bash slurm/drain_queue.sh || true")
echo "${RUN_ID}: train ${TRAIN_JOB} -> merge ${merge} -> eval ${ev} (array 19-21) -> analyse ${an}" \
  | tee -a "${EDGE_SCRATCH}/outputs/post_run_chains.txt"
