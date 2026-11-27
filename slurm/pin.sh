#!/bin/bash
# Pin queued jobs to a commit: check the commit out as a detached git worktree on scratch and print its path.
# Jobs submitted with EDGE_REPO set to that path read ALL code, configs and test sets from it (every slurm script
# sources "${EDGE_REPO}/slurm/env.sh"; sbatch exports the submitting environment), so later edits or branch
# switches in the live repo cannot change a pending or running job. Outputs the jobs write under results/ land
# in the pinned copy; copy them back into the repo when the jobs are done.
#   export EDGE_REPO=$(bash slurm/pin.sh [commit=HEAD]) && cd "$EDGE_REPO" && sbatch ... / bash slurm/post_run.sh ...
set -euo pipefail
LIVE=/home/ebasaran22/chasing-the-edge
SCRATCH="${EDGE_SCRATCH:-/scratch/ebasaran22/chasing-the-edge}"
set +u; module load git 2>/dev/null || true; set -u
cd "${LIVE}"
SHA="$(git rev-parse --short "${1:-HEAD}")"
if [[ "${1:-HEAD}" == HEAD && -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "pin.sh: uncommitted changes to tracked files; commit first so the pin matches a commit" >&2; exit 1
fi
DEST="${SCRATCH}/code/${SHA}"
if [[ ! -d "${DEST}" ]]; then
  mkdir -p "${SCRATCH}/code"
  git worktree add --detach "${DEST}" "${SHA}" >/dev/null 2>&1 || { echo "pin.sh: git worktree add failed" >&2; exit 1; }
fi
echo "${DEST}"
