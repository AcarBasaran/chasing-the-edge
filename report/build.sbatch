#!/bin/bash
# Build the progress report PDF on a compute node: sbatch --wait report/build.sbatch
#SBATCH --job-name=report-tex
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ebasaran22/chasing-the-edge/logs/%x-%j.out
set -euo pipefail
module load latex/2025
# The module puts the binaries in 2025/bin (not 2025/bin/<arch>), so kpathsea derives its tree root one level
# too high (.../latex) and finds neither pdflatex.fmt nor article.cls. Point it at the real root.
export TEXMFROOT=/opt/ohpc/pub/apps/latex/2025
cd /home/ebasaran22/chasing-the-edge/report
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || { tail -40 main.log; exit 1; }
bibtex main | tail -5
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "== warnings"; grep -E "Warning|Overfull|undefined" main.log | grep -v "Font shape" | head -30 || true
echo "== pages"; grep -o "Output written on main.pdf ([0-9]* pages" main.log
