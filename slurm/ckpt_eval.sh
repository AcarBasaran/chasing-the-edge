#!/bin/bash
# When did tau=0.1 pull ahead of the op 11-17 control? Evaluate their step-25/50/75 checkpoints on the HARD set
# (pass@128, T = 0.7, same protocol as the step-100 evals). Approved 2026-11-22 (~2.1 GPU-h).
#   bash slurm/ckpt_eval.sh            (login node: only calls sbatch)
# Chain: one CPU job merges all 18 checkpoints -> 18 one-GPU hard-set evals (array task 21) -> one CPU job exports
# per-problem CSVs (results/per_problem_ckpt/) and runs eval/ckpt_timeline.py.
# Outputs go to <run>/eval_ckpt_step<S>/ (one dir per step), NOT <run>/eval/: the scorer clears its graph dir first.
# The 2026-11-22 run used one shared <run>/eval_ckpt/ and lost scores; slurm/ckpt_rescore.sh recovered them on CPU.
set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh
cd "${EDGE_REPO}"
RUNS="threshold_tau1_s0 threshold_tau1_s1 threshold_tau1_s2 uniform_11_17_s0 uniform_11_17_s1 uniform_11_17_s2"
STEPS="25 50 75"

merge_cmd=""
for r in ${RUNS}; do for s in ${STEPS}; do
  [[ -f "${EDGE_SCRATCH}/outputs/runs/${r}/hf/step${s}/config.json" ]] || merge_cmd+="bash slurm/merge_ckpt.sh ${r} ${s} && "
done; done
merge_cmd+="true"
merge=$(sbatch --parsable --job-name=merge-ckpt-timeline --time=01:00:00 slurm/run_cpu.sbatch "${merge_cmd}")

evals=()
for r in ${RUNS}; do for s in ${STEPS}; do
  R="${EDGE_SCRATCH}/outputs/runs/${r}"
  # One RUN_DIR per step: gate1_eval.sbatch clears its graph dir before scoring (shared dirs lose earlier steps).
  id=$(MODEL="${R}/hf/step${s}" RUN_DIR="${R}/eval_ckpt_step${s}" \
       sbatch --parsable --job-name="eval-ckpt-${r}-${s}" --dependency="afterok:${merge}" --array=21 \
       --time=01:00:00 slurm/gate1_eval.sbatch)
  evals+=("${id}")
done; done

export_cmd="mkdir -p results/per_problem_ckpt"
for r in ${RUNS}; do for s in ${STEPS}; do
  export_cmd+=" && \$EDGE_PY eval/export_base_per_problem.py --run-dir ${EDGE_SCRATCH}/outputs/runs/${r}/eval_ckpt_step${s} \
--ckname step${s} --val-levels none --sets hard --out results/per_problem_ckpt/${r}_step${s}.csv"
done; done
export_cmd+=" && \$EDGE_PY eval/ckpt_timeline.py"
dep=$(IFS=:; echo "${evals[*]}")
an=$(sbatch --parsable --job-name=analyse-ckpt-timeline --dependency="afterany:${dep}" slurm/run_cpu.sbatch "${export_cmd}")
echo "ckpt timeline: merge ${merge} -> evals ${evals[*]} -> analyse ${an}" \
  | tee -a "${EDGE_SCRATCH}/outputs/post_run_chains.txt"
