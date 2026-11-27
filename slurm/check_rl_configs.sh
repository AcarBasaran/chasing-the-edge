#!/bin/bash
# Resolve every arm config through Hydra (CPU only) and print the fields that must differ / must not.
#   sbatch --wait slurm/run_cpu.sbatch 'bash slurm/check_rl_configs.sh'
set -euo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)
export PYTHONPATH="${UPSTREAM}:${EDGE_REPO}:${PYTHONPATH:-}"
cd "${UPSTREAM}"
OUT="${EDGE_SCRATCH}/outputs/cfgcheck"; mkdir -p "${OUT}"
for arm in fixed_band uniform uniform_11_17 threshold_tau1 threshold_tau3 threshold_tau5 bandit; do
  "${EDGE_PY}" -m verl.trainer.main_ppo --config-path "${EDGE_REPO}/configs" --config-name "rl_${arm}" \
     trainer.experiment_name="${arm}_s0" --cfg job --resolve > "${OUT}/${arm}.yaml"
done
cd "${OUT}"
for f in *.yaml; do
  echo "== ${f}: $(grep -A3 '^  sampler:' -m1 "${f}" >/dev/null; "${EDGE_PY}" -c "
import yaml,sys; c=yaml.safe_load(open('${f}')); s=c['data']['sampler']
print(s['method'], s['method_kwargs'], 'probe_every', s['probe_every'], '| B', c['data']['train_batch_size'], 'n', c['actor_rollout_ref']['rollout']['n'], 'steps', c['trainer']['total_training_steps'], 'val n', c['actor_rollout_ref']['rollout']['val_kwargs']['n'], 'log', s['log_path'])")"
done
# Everything except the sampler block and experiment-derived paths must be identical across arms:
for f in *.yaml; do "${EDGE_PY}" -c "
import yaml,json; c=yaml.safe_load(open('${f}')); c['data'].pop('sampler'); c['trainer'].pop('experiment_name'); c['trainer'].pop('default_local_dir')
print(json.dumps(c, sort_keys=True))" | md5sum | sed "s/-/${f}/"; done
