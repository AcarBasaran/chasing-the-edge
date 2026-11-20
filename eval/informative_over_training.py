"""How the share of informative GRPO groups per level changes DURING training (fixed-band baseline).

The step-0 motivation table (results/informative_fraction_by_level.csv) is a snapshot of the base model.
During training the edge moves: in fixed_band_s0, op13 went from p=0.01 to 0.44 by step 40 although the
baseline trains on 11-14 uniformly from the start. This script measures the informative share at the
checkpoints saved during the run (steps 0, 25, 50, 75, 100) with the SAME method as the step-0 table:
val set, 200 problems per level, 64 samples per problem, T=1.0 (= RL rollout temperature), TRAINING reward,
per-problem unbiased estimate of 1 - p^G - (1-p)^G with G=6.

Two views:
  checkpoints   per-problem estimate, levels 9-17, steps 0/25/50/75/100   (all levels, sparse in time)
  training      measured directly on the groups GRPO actually trained, every step (sampler_log.jsonl;
                1024 groups per step split over 11-14; levels outside the band are not trained)
The run saved checkpoints every 25 steps, so 20/40/60/80 do not exist; the tracking sweep (every 10 steps)
logs only level means and pass@k, which cannot give a per-problem informative share.

Inputs (made by slurm/informative_over_training.sh):
    $EDGE_SCRATCH/outputs/base_T1.0                    (step 0 = base model)
    $EDGE_SCRATCH/outputs/runs/<run>/eval_T1.0/step<S>
    $EDGE_PY eval/informative_over_training.py --run fixed_band_s0 --steps 25 50 75 100
Outputs: results/informative_over_training.csv, results/informative_over_training.png,
         results/informative_over_training/step<S>.csv (per-problem counts)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
G = 6
LEVELS = list(range(9, 18))
BAND = (11, 12, 13, 14)


def per_problem_frac(n: int, c: int, g: int = G) -> float:
    d = comb(n, g)
    return 1.0 - comb(c, g) / d - comb(n - c, g) / d


def export(run_dir: Path, ckname: str, out: Path) -> None:
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(REPO / "eval" / "export_base_per_problem.py"), "--run-dir", str(run_dir),
                    "--ckname", ckname, "--val-levels", f"{LEVELS[0]}-{LEVELS[-1]}", "--sets", "--out", str(out)],
                   check=True)


def summarise(path: Path) -> dict[int, dict]:
    by = defaultdict(list)
    for r in csv.DictReader(path.open()):
        by[int(r["op"])].append((int(r["n"]), int(r["c_train"])))
    out = {}
    for L, nc in by.items():
        out[L] = {"n_problems": len(nc), "p": sum(c / n for n, c in nc) / len(nc),
                  "informative": sum(per_problem_frac(n, c) for n, c in nc) / len(nc),
                  "never_solved": sum(c == 0 for _, c in nc) / len(nc),
                  "always_solved": sum(c == n for n, c in nc) / len(nc)}
    return out


def training_view(sampler_log: Path) -> dict[int, list[tuple[int, float]]]:
    out = defaultdict(list)
    for line in sampler_log.open():
        d = json.loads(line) if line.strip() else {}
        if d.get("event") == "step":
            for L, f in d.get("informative_frac", {}).items():
                out[int(L)].append((int(d["step"]), float(f)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fixed_band_s0")
    ap.add_argument("--steps", type=int, nargs="+", default=[25, 50, 75, 100])
    ap.add_argument("--cache", type=Path, default=REPO / "results" / "informative_over_training")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "informative_over_training.png")
    a = ap.parse_args()
    scr = Path(os.environ["EDGE_SCRATCH"]) / "outputs"
    R = scr / "runs" / a.run
    cache = a.cache
    export(scr / "base_T1.0", "base", cache / "step0.csv")
    for s in a.steps:
        export(R / "eval_T1.0" / f"step{s}", f"step{s}", cache / f"step{s}.csv")
    steps = [0] + list(a.steps)
    table = {s: summarise(cache / f"step{s}.csv") for s in steps}

    rows = [{"run": a.run, "step": s, "level": L, "in_training_band": L in BAND, **{k: round(v, 4) for k, v in d.items()}}
            for s in steps for L, d in sorted(table[s].items())]
    with a.out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Informative GRPO groups (G={G}), per-problem estimate, T=1.0, {a.run}:")
    print("| level | " + " | ".join(f"step {s}" for s in steps) + " |")
    print("|---|" + "---|" * len(steps))
    for L in LEVELS:
        cells = [f"{100 * table[s][L]['informative']:.1f}% (p={table[s][L]['p']:.3f})" if L in table[s] else "–"
                 for s in steps]
        print(f"| op{L}{' (trained)' if L in BAND else ''} | " + " | ".join(cells) + " |")

    plot(table, steps, training_view(R / "sampler_log.jsonl"), a.run, a.out)


def plot(table, steps, train, run, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    
    INK, INK2, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0", "#b5b4ad"
    # Trained band 11-14: one blue ramp, widely spaced steps (light = easier). Untrained levels: neutral greys.
    col = {11: "#86b6ee", 12: "#2a78d6", 13: "#174a91", 14: "#0a2448",
           9: "#d3d2cc", 10: "#b5b4ad", 15: "#8a8982", 16: "#6b6a64", 17: "#52514e"}

    def label_ends(axis, ends, min_gap=0.03):
        """Direct labels at line ends, nudged apart so they never overlap."""
        placed = []
        for y, text, x in sorted(ends):
            y2 = max([y] + [p + min_gap for p in placed[-1:]])
            placed.append(y2)
            axis.annotate(text, (x, y2), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.8,
                          color=INK)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=150, facecolor=SURF,
                                 gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})
    for axis in (ax, bx):
        axis.set_facecolor(SURF)
        for s in ("top", "right"):
            axis.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            axis.spines[s].set_color(AXIS)
        axis.tick_params(labelsize=7.5, colors=INK2)
        axis.grid(axis="y", color=GRID, lw=0.6)
        axis.set_axisbelow(True)
        axis.set_ylim(0, 0.7)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    ends_a, ends_b = [], []
    for L in LEVELS:
        ys = [table[s].get(L, {}).get("informative", float("nan")) for s in steps]
        trained = L in BAND
        ax.plot(steps, ys, color=col[L], lw=2 if trained else 1.4, ls="-" if trained else (0, (3, 2)),
                marker="o", ms=4.5, mfc=col[L] if trained else SURF, mec=col[L], zorder=3 if trained else 2)
        ends_a.append((ys[-1], f"op{L}" + ("" if trained else " (untrained)"), steps[-1]))
    label_ends(ax, ends_a)
    ax.set_xticks(steps)
    ax.set_xlim(-4, steps[-1] + 22)
    ax.set_xlabel("training step (saved checkpoints)", fontsize=8, color=INK)
    ax.set_ylabel(f"groups with nonzero gradient (G={G})", fontsize=8, color=INK)
    ax.set_title("All levels, checkpoint re-evaluation (val, 200/level, n=64, T=1.0)\n"
                 "solid = trained on (11–14), dashed = never trained on", fontsize=7.8, color=INK, loc="left")

    for L in BAND:
        pts = sorted(train.get(L, []))
        if not pts:
            continue
        xs, ys = zip(*pts)
        k = 5
        sm = [sum(ys[max(0, i - k + 1): i + 1]) / len(ys[max(0, i - k + 1): i + 1]) for i in range(len(ys))]
        bx.plot(xs, ys, color=col[L], lw=0.6, alpha=0.35)
        bx.plot(xs, sm, color=col[L], lw=2)
        ends_b.append((sm[-1], f"op{L}", xs[-1]))
        for s in steps:                                  # checkpoint estimate at the same level, for comparison
            v = table[s].get(L, {}).get("informative")
            if v is not None:
                bx.scatter([s], [v], s=22, facecolor=SURF, edgecolor=col[L], linewidth=1.4, zorder=4)
    label_ends(bx, ends_b)
    bx.set_xlim(-4, 114)
    bx.set_xlabel("training step", fontsize=8, color=INK)
    bx.set_title("Trained levels, measured on the groups GRPO actually trained\n"
                 "(every step, ~256 groups/level; thick = 5-step mean; ○ = checkpoint estimate)",
                 fontsize=7.8, color=INK, loc="left")
    fig.suptitle(f"Share of informative GRPO groups per level during training: {run}", fontsize=8.5, color=INK,
                 x=0.01, ha="left")
    fig.subplots_adjust(left=0.07, right=0.96, top=0.8, bottom=0.14)
    fig.savefig(out, facecolor=SURF, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
