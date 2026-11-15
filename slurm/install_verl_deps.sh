#!/bin/bash
# Install verl's runtime dependencies into $EDGE_PY's env without disturbing the
# pinned torch / transformers / vLLM. verl itself is imported from $UPSTREAM via
# PYTHONPATH (upstream's setup.py sits inside the package dir; not installed).
#   sbatch --wait --time=01:00:00 slurm/run_cpu.sbatch 'bash slurm/install_verl_deps.sh'
set -euo pipefail
source /home/ebasaran22/chasing-the-edge/slurm/env.sh

CONS="${EDGE_SCRATCH}/constraints.txt"
cat > "${CONS}" <<EOF
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
transformers==4.55.4
vllm==0.10.1.1
EOF

# verl/setup.py install_requires (minus transformers/datasets/pandas already present).
"${EDGE_PY}" -m pip install -c "${CONS}" \
  accelerate codetiming dill hydra-core "numpy<2.0.0" peft "pyarrow>=19.0.0" pybind11 pylatexenc \
  "ray[default]>=2.41.0" torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" wandb "packaging>=20.0" tensorboard \
  2>&1 | tail -3

# flash-attn: prebuilt wheel matching torch 2.7 / cu12 / py312 / C++ ABI.
ABI=$("${EDGE_PY}" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
WHL="flash_attn-2.8.3+cu12torch2.7cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"
"${EDGE_PY}" -m pip install -c "${CONS}" --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHL}" 2>&1 | tail -2

"${EDGE_PY}" -m pip check 2>&1 | tail -5 || true
"${EDGE_PY}" -c "import torch, transformers, vllm, numpy, tensordict, ray; print('torch', torch.__version__, 'tf', transformers.__version__, 'vllm', vllm.__version__, 'numpy', numpy.__version__, 'tensordict', tensordict.__version__, 'ray', ray.__version__)"
"${EDGE_PY}" -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || echo "flash_attn import failed (needs GPU node to fully load?)"
cd "${UPSTREAM}" && PYTHONPATH="${UPSTREAM}" "${EDGE_PY}" -c "import verl, verl.trainer.main_ppo; from verl.experimental.dataset.sampler import AbstractCurriculumSampler; print('verl import ok', verl.__version__)"
