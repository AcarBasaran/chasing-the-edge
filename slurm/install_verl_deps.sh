#!/bin/bash
# Install verl's runtime dependencies into $EDGE_PY's env without disturbing the
# pinned torch / transformers / vLLM. verl itself is imported from $UPSTREAM via
# PYTHONPATH (upstream's setup.py sits inside the package dir; not installed).
#   sbatch --wait --time=01:00:00 slurm/run_cpu.sbatch 'bash slurm/install_verl_deps.sh'
set -euo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)

CONS="${EDGE_SCRATCH}/constraints.txt"
cat > "${CONS}" <<EOF
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
transformers==4.55.4
vllm==0.10.1.1
# uvloop >= 0.22 raises in get_event_loop() with no running loop; verl fsdp_workers.py:592 relies on the old behaviour
uvloop==0.21.0
EOF

# verl/setup.py install_requires (minus transformers/datasets/pandas already present).
# verl pins numpy<2, but that breaks pandas 3.1/scipy 1.18/cupy/opencv; verl runs fine on numpy 2.2.6.
"${EDGE_PY}" -m pip install -c "${CONS}" \
  accelerate codetiming dill hydra-core "numpy==2.2.6" peft "pyarrow>=19.0.0" pybind11 pylatexenc \
  "ray[default]>=2.41.0" torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" wandb "packaging>=20.0" tensorboard \
  2>&1 | tail -3

"${EDGE_PY}" -m pip install -c "${CONS}" "uvloop==0.21.0" 2>&1 | tail -1

# flash-attn: the prebuilt GitHub wheels need GLIBC_2.32 (cluster: 2.28) -> build from source:
#   sbatch --wait --job-name=c-flash --cpus-per-task=32 --mem=160G --time=03:00:00 \
#          slurm/run_cpu.sbatch 'bash slurm/build_flash_attn.sh'

"${EDGE_PY}" -m pip check 2>&1 | tail -5 || true
"${EDGE_PY}" -c "import torch, transformers, vllm, numpy, tensordict, ray; print('torch', torch.__version__, 'tf', transformers.__version__, 'vllm', vllm.__version__, 'numpy', numpy.__version__, 'tensordict', tensordict.__version__, 'ray', ray.__version__)"
cd "${UPSTREAM}" && PYTHONPATH="${UPSTREAM}" "${EDGE_PY}" -c "import verl, verl.trainer.main_ppo; from verl.experimental.dataset.sampler import AbstractCurriculumSampler; print('verl import ok', verl.__version__)"
