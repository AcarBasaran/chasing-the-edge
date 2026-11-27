#!/bin/bash
# Column B analysis (fixed reward AND fixed scorer) after a run's test-set eval. Never touches the column-A
# (paper reward + paper scorer) files: results/per_problem/, base_per_problem.csv, newly_solved.csv, the
# method/seed tables, results/sampler_levels.png.
#   1. per-problem export with the fixed scorer (the default)  -> results/per_problem_fixed/<run_id>.csv
#   2. newly solved for every run in results/per_problem_fixed/ vs the base model graded with the fixed
#      scorer (results/per_problem_fixed/base.csv)               -> results/newly_solved_fixed.csv
#   3. timing and edge-speed logs of the run
#   bash slurm/analyse_fixed.sh <run_id> [step=100]     (via slurm/run_cpu.sbatch; slurm/post_run.sh queues it)
set -euo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)
RUN_ID="${1:?run id}"; STEP="${2:-100}"
R="${EDGE_SCRATCH}/outputs/runs/${RUN_ID}"
D=results/per_problem_fixed
cd "${EDGE_REPO}" && mkdir -p "${D}"
"${EDGE_PY}" eval/export_base_per_problem.py --run-dir "${R}/eval" --ckname "step${STEP}" --val-levels none \
    --out "${D}/${RUN_ID}.csv"
"${EDGE_PY}" eval/parse_verl_log.py "${R}/train.log" --csv "${R}/timing.csv" --sampler-log "${R}/sampler_log.jsonl" || true
"${EDGE_PY}" eval/edge_speed.py "${R}/train.log" --csv "${R}/edge_speed.csv" --plot "${R}/edge_speed.png" || true
if [[ -f "${D}/base.csv" ]]; then
  args=()
  for f in "${D}"/*.csv; do n="$(basename "${f}" .csv)"; [[ "${n}" == base ]] || args+=(--run "${n}=${f}"); done
  "${EDGE_PY}" eval/newly_solved.py "${args[@]}" --base "${D}/base.csv" --out results/newly_solved_fixed.csv
else
  echo "no ${D}/base.csv yet: newly solved skipped (rerun this script once the base re-grade is done)"
fi
bash slurm/drain_queue.sh || true
