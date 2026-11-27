#!/bin/bash
# Gate 1, steps 4-6: run the UPSTREAM evaluation on the base model.
#
#   generation : Interplay-LM-Reasoning/scripts/eval_checkpoints.py (vLLM)
#   scoring    : Interplay-LM-Reasoning/scripts/score_dependency_graphs.py
#                (process-verified, strict; gold graphs built on first use with
#                 scripts/build_gold_dependency_graphs.py, cached under $GOLD_ROOT)
#
# Sampling follows the paper's offline-eval protocol (App. A.3.3):
#   T=0.7, top-p=1.0, top-k=-1, 1024 new tokens, 128 samples per problem.
#
# Array layout (submit with --array=0-21%8):
#   0..18  -> val/op{2..20}-200.jsonl   (one level per task; the p_L table)
#   19..21 -> data/testsets/{easy,medium,hard} (per-problem base pass@128 record)
#
# Env overrides (also used for every later checkpoint eval):
#   MODEL=<hf dir>   (default $EDGE_BASE_MODEL)    TEMP=0.7   SAMPLE_K=128
#   RUN_DIR=<outputs dir>   GOLD_ROOT=<gold cache> (default: gate1's, gold is model-independent)
#   e.g. released RL ckpt:  MODEL=$EDGE_SNAP/extrapolation_rl/$EDGE_MIXTURE/rl/op11-14_uniform \
#        RUN_DIR=$EDGE_SCRATCH/outputs/gate3_released_rl sbatch --array=19-21 slurm/gate1_eval.sbatch
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
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)

RUN_DIR="${RUN_DIR:-${EDGE_SCRATCH}/outputs/gate1_base_eval}"
SAMPLE_K="${SAMPLE_K:-128}"
TEMP="${TEMP:-0.7}"
MODEL="${MODEL:-${EDGE_BASE_MODEL}}"
CKNAME="$(basename "${MODEL}")"          # eval_checkpoints.py names outputs <dir basename>_id_generations.jsonl
GOLD_ROOT="${GOLD_ROOT:-${EDGE_SCRATCH}/outputs/gate1_base_eval/gold}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"

SETS=(easy medium hard)
if (( TASK <= 18 )); then
  LEVEL=$(( TASK + 2 ))
  NAME="val_op${LEVEL}"
  DATA_ROOT="${EDGE_DATA}/val/op${LEVEL}-200.jsonl"
  GOLD_DIR="${GOLD_ROOT}/val"
else
  SET="${SETS[$(( TASK - 19 ))]}"
  NAME="test_${SET}"
  DATA_ROOT="${EDGE_REPO}/data/testsets/${SET}"
  GOLD_DIR="${GOLD_ROOT}/test_${SET}"
fi
# One-off overrides (e.g. a full test/op12-1k cross-check):
#   NAME_OVERRIDE=test_full_op12 DATA_ROOT_OVERRIDE=.../test/op12-1k.jsonl GOLD_DIR_OVERRIDE=... sbatch --array=0 ...
NAME="${NAME_OVERRIDE:-${NAME}}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-${DATA_ROOT}}"
GOLD_DIR="${GOLD_DIR_OVERRIDE:-${GOLD_DIR}}"
GEN_DIR="${RUN_DIR}/gen/${NAME}"
GRAPH_DIR="${RUN_DIR}/graph/${NAME}"
mkdir -p "${GEN_DIR}" "${GRAPH_DIR}"

echo "[task ${TASK}] ${NAME} model=${MODEL} T=${TEMP} k=${SAMPLE_K} data=${DATA_ROOT} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
START=$(date +%s)
# Stall watchdog (user rule 2026-11-20): cancel this task if its SLURM log has not been written for STALL_MIN
# minutes (default 30). vLLM writes progress to it every few seconds during generation; scoring takes ~2 min.
STALL_MIN="${STALL_MIN:-30}"
OUT_LOG="${EDGE_SCRATCH}/logs/${SLURM_JOB_NAME:-x}-${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-x}.out"
[[ -f "${OUT_LOG}" ]] || OUT_LOG=$(scontrol show job "${SLURM_JOB_ID}" | grep -o 'StdOut=[^ ]*' | cut -d= -f2-)
( while sleep 60; do
    [[ -f "${OUT_LOG}" ]] || continue
    age=$(( $(date +%s) - $(stat -c %Y "${OUT_LOG}") ))
    (( age > STALL_MIN * 60 )) || continue
    echo "[gate1_eval] WATCHDOG: log not written for $(( age / 60 )) min -- cancelling ${SLURM_JOB_ID}"
    echo "$(date -Is) CANCEL(in-job) ${SLURM_JOB_ID} ${SLURM_JOB_NAME:-eval} task ${TASK}: log not written for $(( age / 60 )) min" \
      >> "${EDGE_SCRATCH}/outputs/watchdog.log"
    scancel "${SLURM_JOB_ID}"; exit 0
  done ) &
WATCHDOG_PID=$!
trap 'kill "${WATCHDOG_PID}" 2>/dev/null || true' EXIT   # also when run with `bash` inside another job

cd "${UPSTREAM}"
export PYTHONPATH="${UPSTREAM}:${PYTHONPATH:-}"

# Gold graphs: (re)build when missing or when the data files changed (stamp = file list + sizes).
if [[ -d "${DATA_ROOT}" ]]; then SRC_FILES=("${DATA_ROOT}"/*.jsonl); else SRC_FILES=("${DATA_ROOT}"); fi
STAMP="$(ls -l --time-style=+ "${SRC_FILES[@]}" | awk '{print $5, $NF}' | md5sum | cut -c1-32)"
if [[ "$(cat "${GOLD_DIR}/.stamp" 2>/dev/null)" != "${STAMP}" ]]; then
  if [[ -d "${DATA_ROOT}" || "${GOLD_DIR}" != "${GOLD_ROOT}/val" ]]; then
    echo "[gold] building ${GOLD_DIR}"
    SRC_TMP="$(mktemp -d)"; ln -s "${SRC_FILES[@]}" "${SRC_TMP}/"
    rm -rf "${GOLD_DIR}"; mkdir -p "${GOLD_DIR}"
    "${EDGE_PY}" -u scripts/build_gold_dependency_graphs.py "${SRC_TMP}" "${GOLD_DIR}"
    echo "${STAMP}" > "${GOLD_DIR}/.stamp"; rm -rf "${SRC_TMP}"
  fi   # val/: one shared dir for all 19 levels, built once in Gate 1 -- never rebuilt per task
fi

if [[ ! -s "${GEN_DIR}/${CKNAME}_id_generations.jsonl" ]]; then
  "${EDGE_PY}" -u scripts/eval_checkpoints.py \
    --checkpoints-root "${MODEL}" \
    --checkpoints-pattern none \
    --data-root "${DATA_ROOT}" \
    --id-op-threshold -1 \
    --gen-backend vllm \
    --vllm-data-parallel-size 1 \
    --sample-k "${SAMPLE_K}" \
    --temperature "${TEMP}" --top-p 1.0 --top-k -1 \
    --max-new-tokens 1024 \
    --skip-loss \
    --output-dir "${GEN_DIR}"
fi
GEN_END=$(date +%s)

rm -f "${GRAPH_DIR}"/*.graph_results.jsonl "${GRAPH_DIR}"/*_score_report.jsonl
"${EDGE_PY}" -u scripts/score_dependency_graphs.py \
  "${GOLD_DIR}" "${GEN_DIR}/${CKNAME}_id_generations.jsonl" \
  --results-dir "${GRAPH_DIR}" --workers "${SLURM_CPUS_PER_TASK:-8}" \
  > "${GRAPH_DIR}/score_stdout.json"
tail -n 20 "${GRAPH_DIR}/score_stdout.json"

END=$(date +%s)
printf '{"task": %d, "name": "%s", "model": "%s", "temperature": %s, "sample_k": %d, "host": "%s", "gpu": "%s", "gen_seconds": %d, "total_seconds": %d}\n' \
  "${TASK}" "${NAME}" "${MODEL}" "${TEMP}" "${SAMPLE_K}" "$(hostname)" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  $(( GEN_END - START )) $(( END - START )) > "${GEN_DIR}/timing.json"
cat "${GEN_DIR}/timing.json"
