#!/bin/bash
# Submit queued RL runs as SLURM capacity allows, unattended.
# QoS `ai` allows 50 submitted jobs per user (array tasks count one each). One run with its post_run chain
# is 6 jobs (train, merge, 3 eval tasks, analyse), so runs cannot all be submitted at once.
# Queue file: $EDGE_SCRATCH/outputs/pending_runs.txt, one "<arm> <seed>" per line.
# Submits while (submitted jobs + 6) <= 50 - HEADROOM (HEADROOM=3: one slot for this script to resubmit itself, two for ad-hoc analysis jobs), then,
# if lines remain, resubmits itself as a CPU job to try again in 30 minutes.
#   bash slurm/drain_queue.sh            (login node or compute node; only calls sbatch)
set -euo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)
cd "${EDGE_REPO}"
Q="${EDGE_SCRATCH}/outputs/pending_runs.txt"
LIMIT=50; HEADROOM="${HEADROOM:-3}"; PER_RUN=6
# 1-GPU run peaks at ~13 GB RSS (+ <=10 GB Ray object store); 40G lets 8 runs fit the 350G per-user cap.
RUN_OPTS=(--gres=gpu:1 --cpus-per-task=12 --mem=40G --time=16:00:00)

# Keep the stall watchdog alive (it runs outside QoS ai; no-op when already queued or running).
bash slurm/stall_watchdog.sh --submit >/dev/null 2>&1 || true

while [[ -s "${Q}" ]]; do
  n=$(squeue -h -r -u "$(id -un)" | wc -l)
  (( n + PER_RUN <= LIMIT - HEADROOM )) || break
  read -r arm seed < "${Q}"
  j=$(sbatch --parsable --job-name="rl-${arm}" "${RUN_OPTS[@]}" slurm/rl_run.sbatch "${arm}" "${seed}" \
        trainer.total_training_steps=100)
  bash slurm/post_run.sh "${arm}_s${seed}" "${j}" 100
  sed -i '1d' "${Q}"
done

if [[ -s "${Q}" ]]; then
  echo "[drain_queue] $(date -Is): $(wc -l < "${Q}") run(s) still queued; retrying in 30 min"
  sbatch --parsable --job-name=drain-queue --begin=now+30minutes --time=00:10:00 --cpus-per-task=1 --mem=1G \
    slurm/run_cpu.sbatch "bash slurm/drain_queue.sh"
else
  echo "[drain_queue] $(date -Is): queue empty"
fi
