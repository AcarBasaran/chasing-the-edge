#!/bin/bash
# Build flash-attn from source against the cluster's glibc 2.28 (the prebuilt
# GitHub wheels need GLIBC_2.32 and fail to load on RHEL 8).
# sm80 SASS runs on sm86 (A40 / A6000) and sm80 (A100), so build only arch 80.
#   sbatch --wait --job-name=c-flash --cpus-per-task=32 --mem=160G --time=03:00:00 \
#          slurm/run_cpu.sbatch 'bash slurm/build_flash_attn.sh'
set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh
module load cuda/12.6.3 gnu12/12.4.0
export CC=gcc CXX=g++
export FLASH_ATTN_CUDA_ARCHS="80"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
export NVCC_THREADS=2
export TMPDIR="${EDGE_SCRATCH}/build_tmp"; mkdir -p "${TMPDIR}"
echo "nvcc: $(nvcc --version | tail -1)  gcc: $(gcc --version | head -1)  MAX_JOBS=${MAX_JOBS}"

# numpy: restore 2.x (verl's numpy<2 pin broke pandas 3.1 / scipy 1.18 / cupy / opencv;
# we run verl with numpy 2 and will fall back only if it actually breaks).
"${EDGE_PY}" -m pip install -c "${EDGE_SCRATCH}/constraints.txt" "numpy==2.2.6" 2>&1 | tail -1

"${EDGE_PY}" -m pip uninstall -y flash-attn 2>&1 | tail -1 || true
START=$(date +%s)
"${EDGE_PY}" -m pip install -v --no-build-isolation --no-deps --no-binary flash-attn "flash-attn==2.8.3" 2>&1 \
  | grep -E "Building|Successfully|error|Error" | tail -20
echo "build took $(( $(date +%s) - START )) s"
"${EDGE_PY}" -c "import flash_attn, flash_attn_2_cuda; from flash_attn.bert_padding import unpad_input; print('flash_attn', flash_attn.__version__, 'loads ok')"
"${EDGE_PY}" -m pip check 2>&1 | tail -6 || true
cd "${UPSTREAM}" && PYTHONPATH="${UPSTREAM}" "${EDGE_PY}" -c "import verl.trainer.main_ppo; from verl.experimental.dataset.sampler import AbstractCurriculumSampler; print('verl import ok')"
