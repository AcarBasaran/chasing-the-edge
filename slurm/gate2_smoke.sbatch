#!/bin/bash
# Gate 2: GRPO smoke runs with upstream verl on ONE GPU.
#   sbatch slurm/gate2_smoke.sbatch <config name in configs/> [hydra overrides...]
# e.g. one step:  sbatch slurm/gate2_smoke.sbatch gate2_fixed_band_smoke trainer.total_training_steps=1
#SBATCH --job-name=g2-smoke
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --exclude=ai[01-08],ai[11-14]
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ebasaran22/chasing-the-edge/logs/%x-%j.out

set -euo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)
CONFIG="${1:?config name}"; shift || true

export PYTHONPATH="${UPSTREAM}:${EDGE_REPO}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export VLLM_USE_V1=1
# SLURM exports ROCR_VISIBLE_DEVICES next to CUDA_VISIBLE_DEVICES; verl's Worker refuses
# both ("Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set").
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# Ray + multiprocessing create unix sockets under TMPDIR; beegfs (/scratch/tmp) can't hold
# sockets (Errno 95). Use node-local /tmp with a short path (socket paths max 107 chars).
export TMPDIR="/tmp/edge_${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT
OUT="${EDGE_SCRATCH}/outputs/gate2"
mkdir -p "${OUT}"

echo "[gate2] job ${SLURM_JOB_ID} host $(hostname) config ${CONFIG} overrides: $*"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cd "${UPSTREAM}"     # upstream configs use cwd-relative paths (searchpath, verl/reward_fn.py, verl/dataset.py)
START=$(date +%s)
"${EDGE_PY}" -u -m verl.trainer.main_ppo \
  --config-path "${EDGE_REPO}/configs" --config-name "${CONFIG}" "$@"
echo "[gate2] exit 0 after $(( $(date +%s) - START )) s"
