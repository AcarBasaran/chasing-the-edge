"""Shared helpers for the cross-run analysis scripts (compare_methods, newly_solved, plot_sampler_levels).

Per-problem CSVs (one per evaluated checkpoint) all have the columns of results/base_per_problem.csv:
    split, uid, op, template, src_line, n, c_eval, c_train, c_answer
Where they live:
    results/base_per_problem.csv                    the base model ("base")
    results/released_rl_op11-14_per_problem.csv     the paper's released fixed-band checkpoint ("released_rl")
    results/per_problem/<arm>_s<seed>.csv           our runs, e.g. fixed_band_s0.csv, threshold_tau3_s1.csv
How a run's CSV is made (after training):
    sbatch --wait slurm/run_cpu.sbatch 'bash slurm/merge_ckpt.sh <run_id> <step>'
    MODEL=$EDGE_SCRATCH/outputs/runs/<run_id>/hf/step<step> RUN_DIR=$EDGE_SCRATCH/outputs/runs/<run_id>/eval \\
        sbatch --array=19-21 slurm/gate1_eval.sbatch
    $EDGE_PY eval/export_base_per_problem.py --run-dir $EDGE_SCRATCH/outputs/runs/<run_id>/eval \\
        --ckname step<step> --val-levels none --out results/per_problem/<run_id>.csv
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_SETS = ("test_easy", "test_medium", "test_hard")
SET_LABEL = {"test_easy": "easy (op 2-10)", "test_medium": "medium (op 11-14)", "test_hard": "hard (op 15-20)"}

# Fixed display order and colours: colour follows the method, never its rank or which runs exist.
METHODS = ["base", "released_rl", "fixed_band", "uniform", "threshold_tau1", "threshold_tau3",
           "threshold_tau5", "bandit"]
METHOD_LABEL = {"base": "base model", "released_rl": "paper's RL\n(released)", "fixed_band": "fixed band\n11-14",
                "uniform": "uniform\nrandom", "threshold_tau1": "threshold\nτ=0.1", "threshold_tau3": "threshold\nτ=0.3",
                "threshold_tau5": "threshold\nτ=0.5", "bandit": "bandit"}
NEUTRAL, NEUTRAL_DARK = "#b5b4ad", "#52514e"
METHOD_COLOR = {"base": NEUTRAL, "released_rl": NEUTRAL_DARK, "fixed_band": "#2a78d6", "uniform": "#eb6834",
                "threshold_tau1": "#1baf7a", "threshold_tau3": "#eda100", "threshold_tau5": "#e87ba4",
                "bandit": "#6250d6"}
RUN_ID = re.compile(r"^(?P<arm>.+?)_s(?P<seed>\d+)(?:_.*)?$")


def parse_run_id(run_id: str) -> tuple[str, int]:
    """'threshold_tau3_s1' -> ('threshold_tau3', 1); 'base' -> ('base', 0)."""
    m = RUN_ID.match(run_id)
    return (m.group("arm"), int(m.group("seed"))) if m else (run_id, 0)


def load_per_problem(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    with Path(path).open() as fh:
        for r in csv.DictReader(fh):
            out[r["split"]][r["uid"]] = r
    return out


def discover(per_problem_dir: Path | None = None, include_released: bool = True) -> list[tuple[str, Path]]:
    """[(run_id, csv path)] for every evaluated checkpoint, base first."""
    runs = [("base", REPO / "results" / "base_per_problem.csv")]
    rel = REPO / "results" / "released_rl_op11-14_per_problem.csv"
    if include_released and rel.exists():
        runs.append(("released_rl", rel))
    d = per_problem_dir or (REPO / "results" / "per_problem")
    if d.exists():
        runs += [(p.stem, p) for p in sorted(d.glob("*.csv"))]
    return runs


def parse_label_paths(items: list[str]) -> list[tuple[str, Path]]:
    """['label=path', ...] -> [(label, Path)]."""
    out = []
    for it in items:
        label, _, path = it.partition("=")
        if not path:
            label, path = Path(it).stem, it
        out.append((label, Path(path)))
    return out


def method_sort_key(arm: str) -> tuple[int, str]:
    return (METHODS.index(arm) if arm in METHODS else len(METHODS), arm)
