#!/bin/bash
# Full RL run, one arm x one seed, on 4 GPUs of one node.
#   sbatch slurm/rl_run.sbatch <arm> <seed> [hydra overrides...]
#   arm in: fixed_band uniform threshold_tau1 threshold_tau3 threshold_tau5 bandit  (configs/rl_<arm>.yaml)
# e.g. sbatch slurm/rl_run.sbatch fixed_band 0 trainer.total_training_steps=195
#      sbatch --gres=gpu:1 --cpus-per-task=12 --mem=64G slurm/rl_run.sbatch fixed_band 0   (1-GPU variant)
# Run directory: $EDGE_SCRATCH/outputs/runs/<arm>_s<seed>[_<RUN_TAG>]/
#   resolved_config.yaml, git.txt, train.log (stdout), sampler_log.jsonl, ckpt/global_step_*/
#SBATCH --job-name=rl
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:4
#SBATCH --exclude=ai[01-08],ai[11-14]
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/ebasaran22/chasing-the-edge/logs/%x-%j.out

set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh
ARM="${1:?arm}"; SEED="${2:?seed}"; shift 2
CONFIG="rl_${ARM}"
[[ -f "${EDGE_REPO}/configs/${CONFIG}.yaml" ]] || { echo "no configs/${CONFIG}.yaml"; exit 2; }
RUN_ID="${ARM}_s${SEED}${RUN_TAG:+_${RUN_TAG}}"
RUN_DIR="${EDGE_SCRATCH}/outputs/runs/${RUN_ID}"
NGPU="$(nvidia-smi -L | wc -l)"
mkdir -p "${RUN_DIR}"

export PYTHONPATH="${UPSTREAM}:${EDGE_REPO}:${PYTHONPATH:-}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0 VLLM_USE_V1=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES          # verl refuses ROCR next to CUDA (Gate 2)
export TMPDIR="/tmp/edge_${SLURM_JOB_ID}" RAY_TMPDIR="/tmp/edge_${SLURM_JOB_ID}"   # sockets can't live on beegfs
mkdir -p "${TMPDIR}"
# On exit keep Ray's own logs (the 2026-11-15 hang left none: they were deleted with TMPDIR).
trap 'cp -r "${RAY_TMPDIR}/session_latest/logs" "${RUN_DIR}/ray_logs_${SLURM_JOB_ID}" 2>/dev/null; rm -rf "${TMPDIR}"' EXIT

# Private Ray head for THIS job. Jobs 2026026/2026027 (two 1-GPU runs on the same node, started the same
# second) both hung forever right after "Started a local Ray instance": 1 CPU-h in 24 h, 1.9 GB RSS, no step.
# Two things were wrong: (1) ray.init() let Ray see all 48 node CPUs inside a 12-CPU cgroup -- our
# "ray_init.num_cpus" override went to a dead key (verl reads ray_kwargs.ray_init, and its own comment says
# "use a fixed number instead of null when using SLURM"); (2) two local Ray clusters on one node share
# default agent/worker ports. Fix: start the head ourselves with --num-cpus/--num-gpus and a job-unique
# port block, then let verl connect via RAY_ADDRESS (ray.init must then get no num_cpus).
PORT_BASE=$(( 20000 + (SLURM_JOB_ID % 200) * 200 ))
"$(dirname "${EDGE_PY}")/ray" start --head --node-ip-address=127.0.0.1 --port="${PORT_BASE}" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${NGPU}" --temp-dir="${RAY_TMPDIR}" \
    --node-manager-port=$((PORT_BASE + 1)) --object-manager-port=$((PORT_BASE + 2)) \
    --dashboard-agent-grpc-port=$((PORT_BASE + 3)) --dashboard-agent-listen-port=$((PORT_BASE + 4)) \
    --metrics-export-port=$((PORT_BASE + 5)) --runtime-env-agent-port=$((PORT_BASE + 6)) \
    --ray-client-server-port=$((PORT_BASE + 7)) --min-worker-port=$((PORT_BASE + 10)) \
    --max-worker-port=$((PORT_BASE + 199)) --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:${PORT_BASE}"

ARGS=(--config-path "${EDGE_REPO}/configs" --config-name "${CONFIG}"
      "trainer.experiment_name=${RUN_ID}" "data.seed=${SEED}" "trainer.n_gpus_per_node=${NGPU}"
      "hydra.run.dir=${RUN_DIR}/hydra" "$@")

cd "${UPSTREAM}"     # upstream configs use cwd-relative paths (searchpath, verl/reward_fn.py, verl/dataset.py)
{ echo "job ${SLURM_JOB_ID} host $(hostname) gpus ${NGPU} start $(date -Is)"
  echo "repo $(git -C "${EDGE_REPO}" rev-parse HEAD 2>/dev/null || echo '?') dirty=$(git -C "${EDGE_REPO}" status --porcelain 2>/dev/null | wc -l)"
  echo "upstream $(git -C "${UPSTREAM}" rev-parse HEAD 2>/dev/null || echo '?')"
  echo "args: ${ARGS[*]}"; } | tee -a "${RUN_DIR}/git.txt"
"${EDGE_PY}" -m verl.trainer.main_ppo "${ARGS[@]}" --cfg job --resolve > "${RUN_DIR}/resolved_config.yaml"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
START=$(date +%s)
# Startup watchdog: if step 1 has not been logged after STARTUP_TIMEOUT_MIN (default 60; step 1 incl. model
# load, vLLM init and the step-0 tracking sweep takes ~15 min on 1 GPU), kill the trainer so the job fails
# in an hour instead of idling to the time limit.
( sleep $(( ${STARTUP_TIMEOUT_MIN:-60} * 60 ))
  grep -q "step:1 - " "${RUN_DIR}/train.log" 2>/dev/null && exit 0
  echo "[rl_run] WATCHDOG: no 'step:1' after ${STARTUP_TIMEOUT_MIN:-60} min -- killing the trainer" | tee -a "${RUN_DIR}/train.log"
  pkill -TERM -u "$(id -u)" -f "verl.trainer.main_ppo .*trainer.experiment_name=${RUN_ID} " ) &
"${EDGE_PY}" -u -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
echo "[rl_run] ${RUN_ID} exit 0 after $(( $(date +%s) - START )) s, ${NGPU} GPUs" | tee -a "${RUN_DIR}/git.txt"
