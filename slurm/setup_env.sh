#!/bin/bash
# Rebuild the project environment from scratch, in the order it was originally built
# (2026-11-15). Each step below was run as shown; this script chains them.
# Run as a CPU job (flash-attn compiles CUDA kernels for ~15 min):
#   sbatch --wait --cpus-per-task=32 --mem=160G --time=03:00:00 slurm/run_cpu.sbatch 'bash slurm/setup_env.sh'
# Exact versions of every package: requirements-lock.txt (pip freeze of the working env).
set -euo pipefail
source "$(dirname "$0")/env.sh"
UPSTREAM_COMMIT=ab728f05d81de9af38d0ca155a84166b037e355a   # Interplay-LM-Reasoning, "Update README.md", 2026-06-08

if [[ -e "${EDGE_SCRATCH}/envs/edge" ]]; then
  echo "refusing: ${EDGE_SCRATCH}/envs/edge exists (move it away to rebuild)"; exit 2
fi
mkdir -p "${EDGE_SCRATCH}"/{envs,hf_home,data,outputs,checkpoints,pip_cache,conda_pkgs,logs}

# 1. The paper's code, pinned. verl (0.5.0.dev) is used from this checkout via PYTHONPATH, not pip-installed.
if [[ ! -d "${UPSTREAM}/.git" ]]; then
  git clone https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning "${UPSTREAM}"
fi
git -C "${UPSTREAM}" checkout -q "${UPSTREAM_COMMIT}"

# 2. Python 3.12 + vLLM 0.10.1.1 (brings torch 2.7.1+cu126).
export CONDA_PKGS_DIRS="${EDGE_SCRATCH}/conda_pkgs"
conda create -y -p "${EDGE_SCRATCH}/envs/edge" python=3.12
"${EDGE_PY}" -m pip install vllm==0.10.1.1 datasets "huggingface_hub[hf_transfer]" pandas pyyaml

# 3. transformers must be 4.55.x (pip otherwise resolves 5.x, which this checkpoint/vLLM don't expect).
"${EDGE_PY}" -m pip install "transformers==4.55.4" "huggingface_hub[hf_transfer]>=0.34,<1.0" "datasets>=3,<4"

# 4. verl's runtime deps (numpy kept at 2.2.6, uvloop pinned to 0.21.0) and flash-attn built from source
#    (prebuilt wheels need glibc 2.32; VALAR has 2.28).
bash "${EDGE_REPO}/slurm/install_verl_deps.sh"
bash "${EDGE_REPO}/slurm/build_flash_attn.sh"

# 5. Analysis / test tools.
"${EDGE_PY}" -m pip install pytest matplotlib

# 6. Model + data snapshots (base checkpoint, released RL checkpoint, composition dataset).
"${EDGE_PY}" "${EDGE_REPO}/eval/download_assets.py" --with-rl

"${EDGE_PY}" -m pip freeze --exclude-editable | diff - "${EDGE_REPO}/requirements-lock.txt" \
  && echo "env matches requirements-lock.txt" || echo "WARNING: env differs from the lock file (see diff above)"
