#!/bin/bash
# Per-problem success counts for one run's checkpoints at T=1.0 (the RL rollout temperature), to measure how
# the share of informative GRPO groups per level changes during training (eval/informative_over_training.py).
# Same protocol as the step-0 motivation table (results/base_T1.0_per_problem.csv): val set, 200 problems per
# level, 64 samples per problem, T=1.0. Levels 9-17. Runs every level of every checkpoint inside ONE GPU job
# (the QoS allows 50 submitted jobs; an array would take 40 of them).
#   sbatch --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=06:00:00 --job-name=inf-over-train \
#       slurm/run_cpu.sbatch 'bash slurm/informative_over_training.sh fixed_band_s0 25 50 75 100'
set -euo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)
RUN_ID="${1:?run id}"; shift; STEPS=("$@")
R="${EDGE_SCRATCH}/outputs/runs/${RUN_ID}"
LEVELS=(9 10 11 12 13 14 15 16 17)
export TEMP=1.0 SAMPLE_K=64

# Step 0 = the base model. Its op9-15 counts already exist in outputs/base_T1.0; add op16-17.
for L in 16 17; do
  [[ -s "${EDGE_SCRATCH}/outputs/base_T1.0/graph/val_op${L}/.done" ]] && continue
  MODEL="${EDGE_BASE_MODEL}" RUN_DIR="${EDGE_SCRATCH}/outputs/base_T1.0" SLURM_ARRAY_TASK_ID=$((L - 2)) \
    bash "${EDGE_REPO}/slurm/gate1_eval.sbatch"
  date -Is > "${EDGE_SCRATCH}/outputs/base_T1.0/graph/val_op${L}/.done"
done

for S in "${STEPS[@]}"; do
  [[ -d "${R}/hf/step${S}" ]] || bash "${EDGE_REPO}/slurm/merge_ckpt.sh" "${RUN_ID}" "${S}"
  OUT="${R}/eval_T1.0/step${S}"   # one dir per step: gate1_eval clears graph results in its graph dir
  for L in "${LEVELS[@]}"; do
    [[ -s "${OUT}/graph/val_op${L}/.done_step${S}" ]] && continue
    MODEL="${R}/hf/step${S}" RUN_DIR="${OUT}" SLURM_ARRAY_TASK_ID=$((L - 2)) bash "${EDGE_REPO}/slurm/gate1_eval.sbatch"
    date -Is > "${OUT}/graph/val_op${L}/.done_step${S}"
  done
done
echo "[informative_over_training] done: ${RUN_ID} steps 0 ${STEPS[*]}"

# Analysis (table + figure). Failure here does not lose the generations above; rerun the script alone.
cd "${EDGE_REPO}"
"${EDGE_PY}" eval/informative_over_training.py --run "${RUN_ID}" --steps "${STEPS[@]}" \
  || echo "[informative_over_training] analysis FAILED; rerun: \$EDGE_PY eval/informative_over_training.py --run ${RUN_ID} --steps ${STEPS[*]}"
