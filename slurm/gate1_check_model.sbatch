#!/bin/bash
# Gate 1, step 2: confirm the base checkpoint loads with transformers and
# produces process-verified-correct solutions on easy problems.
#SBATCH --job-name=g1-load
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --exclude=ai[01-08],ai[11-14]
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ebasaran22/chasing-the-edge/logs/%x-%j.out

set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh
cd "${EDGE_REPO}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"${EDGE_PY}" -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
"${EDGE_PY}" -u eval/check_model_load.py "$@"
