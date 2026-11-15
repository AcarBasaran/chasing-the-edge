#!/bin/bash
# Convert a verl FSDP actor checkpoint to a HF model dir (for slurm/gate1_eval.sbatch MODEL=...).
#   sbatch --wait slurm/run_cpu.sbatch 'bash slurm/merge_ckpt.sh <run_id> <step>'
# -> $EDGE_SCRATCH/outputs/runs/<run_id>/hf/step<step>/   (tokenizer copied from the base model)
set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh
RUN_ID="${1:?run id}"; STEP="${2:?step}"
RUN_DIR="${EDGE_SCRATCH}/outputs/runs/${RUN_ID}"
SRC="${RUN_DIR}/ckpt/global_step_${STEP}/actor"; DST="${RUN_DIR}/hf/step${STEP}"
[[ -d "${SRC}" ]] || { echo "missing ${SRC}"; exit 2; }
cd "${UPSTREAM}"; export PYTHONPATH="${UPSTREAM}:${PYTHONPATH:-}"
"${EDGE_PY}" -m verl.model_merger merge --backend fsdp --local_dir "${SRC}" --target_dir "${DST}"
# eval_checkpoints.py loads the tokenizer from the model dir; make sure it is the base one, byte-identical.
for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
  [[ -f "${EDGE_BASE_MODEL}/${f}" ]] && cp "${EDGE_BASE_MODEL}/${f}" "${DST}/"
done
ls -la "${DST}"
