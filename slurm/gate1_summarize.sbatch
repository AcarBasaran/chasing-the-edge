#!/bin/bash
# Gate 1: aggregate rollouts into the committed CSVs (CPU only).
#   sbatch --dependency=afterok:<eval array id> slurm/gate1_summarize.sbatch
#SBATCH --job-name=g1-summ
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ebasaran22/chasing-the-edge/logs/%x-%j.out

set -euo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)
cd "${EDGE_REPO}"
"${EDGE_PY}" -u eval/summarize_gate1.py --workers "${SLURM_CPUS_PER_TASK}"
