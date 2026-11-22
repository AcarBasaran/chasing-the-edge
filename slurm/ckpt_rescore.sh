#!/bin/bash
# Re-score the step-25/50/75 hard-set generations of slurm/ckpt_eval.sh (CPU only; generations are kept).
# Needed because the three evals of a run shared <run>/eval_ckpt/graph/test_hard and gate1_eval.sbatch clears
# that dir before scoring, so only the last-scored step survived. Scores all three into the same dir, no clearing.
#   sbatch --cpus-per-task=32 slurm/run_cpu.sbatch 'bash slurm/ckpt_rescore.sh'
set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh
GOLD="${EDGE_SCRATCH}/outputs/gate1_base_eval/gold/test_hard"
cd "${UPSTREAM}"; export PYTHONPATH="${UPSTREAM}:${PYTHONPATH:-}"
for r in threshold_tau1_s0 threshold_tau1_s1 threshold_tau1_s2 uniform_11_17_s0 uniform_11_17_s1 uniform_11_17_s2; do
  E="${EDGE_SCRATCH}/outputs/runs/${r}/eval_ckpt"
  for s in 25 50 75; do
    "${EDGE_PY}" -u scripts/score_dependency_graphs.py "${GOLD}" "${E}/gen/test_hard/step${s}_id_generations.jsonl" \
      --results-dir "${E}/graph/test_hard" --workers "${SLURM_CPUS_PER_TASK:-8}" > "${E}/graph/test_hard/score_stdout_step${s}.json"
    echo "scored ${r} step${s}"
  done
done
