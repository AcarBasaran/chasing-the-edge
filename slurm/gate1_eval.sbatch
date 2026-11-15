#!/bin/bash
# Gate 1, steps 4-6: run the UPSTREAM evaluation on the base model.
#
#   generation : Interplay-LM-Reasoning/scripts/eval_checkpoints.py (vLLM)
#   scoring    : Interplay-LM-Reasoning/scripts/score_dependency_graphs.py
#                (process-verified, strict; gold graphs pre-built by
#                 slurm/gate1_prep.sh with scripts/build_gold_dependency_graphs.py)
#
# Sampling follows the paper's offline-eval protocol (App. A.3.3):
#   T=0.7, top-p=1.0, top-k=-1, 1024 new tokens, 128 samples per problem.
#
# Array layout (submit with --array=0-21%8):
#   0..18  -> val/op{2..20}-200.jsonl   (one level per task; the p_L table)
#   19..21 -> data/testsets/{easy,medium,hard} (per-problem base pass@128 record)
#
#SBATCH --job-name=g1-eval
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --exclude=ai[01-08],ai[11-14]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ebasaran22/chasing-the-edge/logs/%x-%A_%a.out

set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh

RUN_DIR="${RUN_DIR:-${EDGE_SCRATCH}/outputs/gate1_base_eval}"
SAMPLE_K="${SAMPLE_K:-128}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"

SETS=(easy medium hard)
if (( TASK <= 18 )); then
  LEVEL=$(( TASK + 2 ))
  NAME="val_op${LEVEL}"
  DATA_ROOT="${EDGE_DATA}/val/op${LEVEL}-200.jsonl"
  GOLD_DIR="${RUN_DIR}/gold/val"
else
  SET="${SETS[$(( TASK - 19 ))]}"
  NAME="test_${SET}"
  DATA_ROOT="${EDGE_REPO}/data/testsets/${SET}"
  GOLD_DIR="${RUN_DIR}/gold/test_${SET}"
fi
# One-off overrides (e.g. a full test/op12-1k cross-check):
#   NAME_OVERRIDE=test_full_op12 DATA_ROOT_OVERRIDE=.../test/op12-1k.jsonl GOLD_DIR_OVERRIDE=... sbatch --array=0 ...
NAME="${NAME_OVERRIDE:-${NAME}}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-${DATA_ROOT}}"
GOLD_DIR="${GOLD_DIR_OVERRIDE:-${GOLD_DIR}}"
GEN_DIR="${RUN_DIR}/gen/${NAME}"
GRAPH_DIR="${RUN_DIR}/graph/${NAME}"
mkdir -p "${GEN_DIR}" "${GRAPH_DIR}"

echo "[task ${TASK}] ${NAME} data=${DATA_ROOT} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
START=$(date +%s)

cd "${UPSTREAM}"
export PYTHONPATH="${UPSTREAM}:${PYTHONPATH:-}"

if [[ ! -s "${GEN_DIR}/base_id_generations.jsonl" ]]; then
  "${EDGE_PY}" -u scripts/eval_checkpoints.py \
    --checkpoints-root "${EDGE_BASE_MODEL}" \
    --checkpoints-pattern none \
    --data-root "${DATA_ROOT}" \
    --id-op-threshold -1 \
    --gen-backend vllm \
    --vllm-data-parallel-size 1 \
    --sample-k "${SAMPLE_K}" \
    --temperature 0.7 --top-p 1.0 --top-k -1 \
    --max-new-tokens 1024 \
    --skip-loss \
    --output-dir "${GEN_DIR}"
fi
GEN_END=$(date +%s)

rm -f "${GRAPH_DIR}"/*.graph_results.jsonl "${GRAPH_DIR}"/*_score_report.jsonl
"${EDGE_PY}" -u scripts/score_dependency_graphs.py \
  "${GOLD_DIR}" "${GEN_DIR}/base_id_generations.jsonl" \
  --results-dir "${GRAPH_DIR}" --workers "${SLURM_CPUS_PER_TASK:-8}" \
  > "${GRAPH_DIR}/score_stdout.json"
tail -n 20 "${GRAPH_DIR}/score_stdout.json"

END=$(date +%s)
printf '{"task": %d, "name": "%s", "host": "%s", "gpu": "%s", "gen_seconds": %d, "total_seconds": %d}\n' \
  "${TASK}" "${NAME}" "$(hostname)" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  $(( GEN_END - START )) $(( END - START )) > "${GEN_DIR}/timing.json"
cat "${GEN_DIR}/timing.json"
