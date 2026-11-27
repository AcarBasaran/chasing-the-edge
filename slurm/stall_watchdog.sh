#!/bin/bash
# Stall watchdog for every RUNNING training and eval job of this user (user rule, 2026-11-20: the Ray hang cost
# 48 GPU-h; never again). A job is cancelled when, after it has run STALL_MIN (30) minutes,
#   * its SLURM log has not been written for STALL_MIN minutes, or
#   * (RL jobs) its step counter ("step:N - ") has not advanced for STALL_MIN minutes, counted from job start
#     until step 1 appears. Healthy numbers: step 1 at ~15 min (model load, vLLM init, step-0 tracking sweep),
#     longest step 415 s over 19 runs; eval jobs write vLLM progress to their log every few seconds.
# Every cancellation is logged with its reason and the log tail in $EDGE_SCRATCH/outputs/watchdog.log.
#
# Runs OUTSIDE QoS `ai` (account comx29, 1 CPU): QoS ai allows only 8 running jobs per user, so a watchdog
# inside it could not start exactly when 8 GPU jobs are running. It loops every EVERY_MIN minutes for 7.5 h (partition t4_ai, 8 h limit; CPU only)
# and queues its own successor (afterany) at start, so a time-out or cancel does not end the chain.
# Training and eval scripts also carry the same check in-job (rl_run.sbatch, gate1_eval.sbatch); this one
# covers jobs submitted before that change and is the second line of defence for the rest.
#
#   bash slurm/stall_watchdog.sh --once        one pass (DRY_RUN=1: report only, cancel nothing;
#                                              ONLY_JOB=<id>: check that job only, for tests with a small STALL_MIN)
#   bash slurm/stall_watchdog.sh --submit      start the chain (no-op if one is already running or queued)
#   scancel -n stall-watchdog                  stop it
set -uo pipefail
source "${EDGE_REPO:-/home/ebasaran22/chasing-the-edge}/slurm/env.sh"   # pinned copy when EDGE_REPO is set (slurm/pin.sh)
cd "${EDGE_REPO}"
STALL_MIN="${STALL_MIN:-30}"
EVERY_MIN="${EVERY_MIN:-5}"
LOG="${EDGE_SCRATCH}/outputs/watchdog.log"
STATE="${EDGE_SCRATCH}/outputs/watchdog_state"
SUBMIT=(--job-name=stall-watchdog --account=comx29 --qos=comx29 --partition=t4_ai --time=08:00:00
        --cpus-per-task=1 --mem=1G --output="${EDGE_SCRATCH}/logs/%x-%j.out")
mkdir -p "${STATE}"

expand_out() {   # StdOut may still hold %A/%a/%j/%x for array tasks
  local p="$1" jid="$2" arr="$3" task="$4" name="$5"
  p="${p//%A/${arr}}"; p="${p//%a/${task}}"; p="${p//%j/${jid}}"; p="${p//%x/${name}}"
  echo "${p}"
}

check_once() {
  local now; now=$(date +%s)
  squeue -h -u "$(id -un)" -t R -o "%i|%A|%F|%K|%j|%S" | while IFS='|' read -r id jid arr task name start; do
    case "${name}" in rl-*|eval-*|g1-eval|inf-over-train) ;; *) continue ;; esac
    [[ -z "${ONLY_JOB:-}" || "${jid}" == "${ONLY_JOB}" ]] || continue     # testing: touch one job only
    local t0 out age reason="" step sf s0 e0
    t0=$(date -d "${start}" +%s) || continue
    (( now - t0 > STALL_MIN * 60 )) || continue
    out=$(scontrol show job "${jid}" 2>/dev/null | grep -o 'StdOut=[^ ]*' | cut -d= -f2-)
    out=$(expand_out "${out}" "${jid}" "${arr}" "${task}" "${name}")
    [[ -f "${out}" ]] || { echo "$(date -Is) WARN ${id} ${name}: log ${out} not found, not checked" >> "${LOG}"; continue; }
    age=$(( now - $(stat -c %Y "${out}") ))
    (( age > STALL_MIN * 60 )) && reason="log not written for $(( age / 60 )) min"
    if [[ -z "${reason}" && "${name}" == rl-* ]]; then
      step=$(grep -aoE 'step:[0-9]+ - ' "${out}" | tail -1 | tr -dc '0-9'); step="${step:-0}"
      sf="${STATE}/${jid}"
      if [[ -f "${sf}" ]] && read -r s0 e0 < "${sf}" && [[ "${s0}" == "${step}" ]]; then
        (( now - e0 > STALL_MIN * 60 )) && reason="step counter stuck at ${step} for $(( (now - e0) / 60 )) min"
      elif [[ -f "${sf}" || "${step}" != 0 ]]; then
        echo "${step} ${now}" > "${sf}"
      else
        echo "0 ${t0}" > "${sf}"          # first look, no step yet: count from job start
      fi
    fi
    [[ -n "${reason}" ]] || continue
    if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "WOULD CANCEL ${id} ${name}: ${reason}"; continue; fi
    { echo "$(date -Is) CANCEL ${id} ${name}: ${reason}"
      echo "  log: ${out}"
      tail -3 "${out}" | tr '\r' '\n' | tail -3 | cut -c1-300 | sed 's/^/  | /'; } >> "${LOG}"
    scancel "${jid}"
  done
}

running_elsewhere() {   # another stall-watchdog already running (not this job)
  squeue -h -u "$(id -un)" -n stall-watchdog -t R -o %A | grep -vqx "${SLURM_JOB_ID:-none}"
}

case "${1:-}" in
  --once) check_once; exit 0 ;;
  --submit)
    if squeue -h -u "$(id -un)" -n stall-watchdog -o %A | grep -q .; then echo "stall-watchdog already queued/running"; exit 0; fi
    sbatch --parsable "${SUBMIT[@]}" slurm/stall_watchdog.sh; exit $? ;;
esac

# Loop mode (inside SLURM).
running_elsewhere && { echo "another stall-watchdog is running; exiting"; exit 0; }
next=$(sbatch --parsable "${SUBMIT[@]}" --dependency="afterany:${SLURM_JOB_ID}" slurm/stall_watchdog.sh) \
  && echo "successor ${next} queued"
echo "$(date -Is) stall-watchdog ${SLURM_JOB_ID} up (stall ${STALL_MIN} min, every ${EVERY_MIN} min)" >> "${LOG}"
end=$(( $(date +%s) + 450 * 60 ))   # t4_ai limit is 8 h
while (( $(date +%s) < end )); do
  check_once
  sleep $(( EVERY_MIN * 60 ))
done
