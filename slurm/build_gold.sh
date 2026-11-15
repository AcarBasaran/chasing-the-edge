#!/bin/bash
# Pre-build (or refresh) the cached gold dependency graphs for the test sets, with the same
# stamp logic as slurm/gate1_eval.sbatch -- run once before launching parallel evals so that
# array tasks never race on the same gold dir.
#   sbatch --wait slurm/run_cpu.sbatch 'bash slurm/build_gold.sh easy medium hard'
set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh
GOLD_ROOT="${GOLD_ROOT:-${EDGE_SCRATCH}/outputs/gate1_base_eval/gold}"
cd "${UPSTREAM}"; export PYTHONPATH="${UPSTREAM}:${PYTHONPATH:-}"
for SET in "$@"; do
  DATA_ROOT="${EDGE_REPO}/data/testsets/${SET}"; GOLD_DIR="${GOLD_ROOT}/test_${SET}"
  STAMP="$(ls -l --time-style=+ "${DATA_ROOT}"/*.jsonl | awk '{print $5, $NF}' | md5sum | cut -c1-32)"
  if [[ "$(cat "${GOLD_DIR}/.stamp" 2>/dev/null)" == "${STAMP}" ]]; then echo "[gold] ${SET} up to date"; continue; fi
  rm -rf "${GOLD_DIR}"; mkdir -p "${GOLD_DIR}"
  "${EDGE_PY}" -u scripts/build_gold_dependency_graphs.py "${DATA_ROOT}" "${GOLD_DIR}"
  echo "${STAMP}" > "${GOLD_DIR}/.stamp"; ls "${GOLD_DIR}"
done
