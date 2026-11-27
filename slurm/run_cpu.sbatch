#!/bin/bash
# Run an arbitrary command from the repo root on a compute node (CPU only).
#   sbatch --wait slurm/run_cpu.sbatch <cmd> [args...]
# e.g. sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY -m pytest -q tests'
#SBATCH --job-name=cpu
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ebasaran22/chasing-the-edge/logs/%x-%j.out

set -euo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)
cd "${EDGE_REPO}"
export TOKENIZERS_PARALLELISM=false
echo "[run_cpu] host=$(hostname) cmd: $*"
eval "$@"
