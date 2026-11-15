# Shared environment for every job and helper script. Source, don't execute:
#   source slurm/env.sh
#
# Everything heavy (env, HF cache, rollouts, checkpoints) lives on scratch;
# home quota is tight.

export EDGE_SCRATCH="${EDGE_SCRATCH:-/scratch/ebasaran22/chasing-the-edge}"
export EDGE_REPO="${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}"
# Upstream code (paper's repo), cloned as a sibling of this repo.
export UPSTREAM="${UPSTREAM:-/home/ebasaran22/Interplay-LM-Reasoning}"

export HF_HOME="${EDGE_SCRATCH}/hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_CACHE_DIR="${EDGE_SCRATCH}/pip_cache"
export VLLM_CACHE_ROOT="${EDGE_SCRATCH}/vllm_cache"
export TRITON_CACHE_DIR="${EDGE_SCRATCH}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${EDGE_SCRATCH}/inductor_cache"
export XDG_CACHE_HOME="${EDGE_SCRATCH}/xdg_cache"

export EDGE_PY="${EDGE_SCRATCH}/envs/edge/bin/python"

# Gate-1 choices (see configs/gate1_base_eval.yaml for the rationale).
export EDGE_HF_MODEL_REPO="Interplay-LM-Reasoning/extrapolation_rl"
export EDGE_MIXTURE="id2-10_0.2easy_0.3medium_0.5hard"
export EDGE_HF_DATA_REPO="Interplay-LM-Reasoning/composition"

# Local snapshots (plain directories, created by eval/download_assets.py).
export EDGE_SNAP="${EDGE_SCRATCH}/hf_snapshots"
export EDGE_BASE_MODEL="${EDGE_SNAP}/extrapolation_rl/${EDGE_MIXTURE}/base"
export EDGE_DATA="${EDGE_SNAP}/composition"
