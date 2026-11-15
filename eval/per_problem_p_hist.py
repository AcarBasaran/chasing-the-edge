"""Distribution of PER-PROBLEM success rate p_i within each level (base model), levels 10-14.

Why: the share of GRPO groups with nonzero gradient measured per problem is ~2.3x lower than the
level-mean formula predicts (docs/gate1.md). The reason is that p_i within a level is bimodal:
most problems are always or never solved. If so, the op level is a coarse proxy for
difficulty, and choosing problems by their own p_i (not their level) would yield more
gradient per rollout. FUTURE WORK -- not built.

Reward = strict training reward. Primary: T=1.0 (RL rollout temperature), 64 samples/problem,
200 val problems per level (results/base_T1.0_per_problem.csv). Also T=0.7, 128 samples
(results/base_per_problem.csv) in the CSV.

    sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY eval/per_problem_p_hist.py'
Outputs: results/per_problem_p_by_level.csv, results/per_problem_p_hist.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEVELS = range(10, 15)
# exactly 0 | (0, .1) | [.1,.2) ... [.8,.9) | [.9, 1) | exactly 1
EDGES = [i / 10 for i in range(11)]
LABELS = ["0"] + [f"{a:.1f}-{b:.1f}" for a, b in zip(EDGES[:-1], EDGES[1:])] + ["1"]


def bin_of(c: int, n: int) -> int:
    if c == 0:
        return 0
    if c == n:
        return len(LABELS) - 1
    return 1 + min(int(10 * c / n), 9)


def load(path: Path):
    by = defaultdict(list)
    for r in csv.DictReader(path.open()):
        if r["split"] == "val" and int(r["op"]) in LEVELS:
            by[int(r["op"])].append((int(r["n"]), int(r["c_train"])))
    return by


def summarise(by, temperature):
    rows = []
    for L in LEVELS:
        probs = by.get(L, [])
        if not probs:
            continue
        ps = [c / n for n, c in probs]
        k = len(ps)
        hist = [0] * len(LABELS)
        for n, c in probs:
            hist[bin_of(c, n)] += 1
        rows.append({
            "level": L, "temperature": temperature, "n_problems": k, "samples_per_problem": probs[0][0],
            "mean_p": sum(ps) / k,
            "frac_never_solved": sum(c == 0 for _, c in probs) / k,
            "frac_p_below_0.1": sum(p < 0.1 for p in ps) / k,
            "frac_p_0.1_to_0.9": sum(0.1 <= p <= 0.9 for p in ps) / k,
            "frac_p_above_0.9": sum(p > 0.9 for p in ps) / k,
            "frac_always_solved": sum(c == n for n, c in probs) / k,
            **{f"hist_{lab}": h / k for lab, h in zip(LABELS, hist)},
        })
    return rows


def plot(rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, INK, INK2, SURF, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
    rs = [r for r in rows if r["temperature"] == 1.0]
    fig, axes = plt.subplots(1, len(rs), figsize=(2.3 * len(rs), 3.0), dpi=150, sharey=True, facecolor=SURF)
    x = list(range(len(LABELS)))
    for ax, r in zip(axes, rs):
        ax.set_facecolor(SURF)
        vals = [100 * r[f"hist_{lab}"] for lab in LABELS]
        ax.bar(x, vals, width=0.8, color=BLUE, edgecolor=SURF, linewidth=1)
        ax.set_title(f"op {r['level']}  (mean p {r['mean_p']:.2f})", fontsize=8.5, color=INK, loc="left")
        mid = 100 * r["frac_p_0.1_to_0.9"]
        ax.text(0.5, 0.97, f"{mid:.0f}% of problems\nhave 0.1 ≤ p ≤ 0.9", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=INK2)
        ax.set_xticks([0, 6, 11])
        ax.set_xticklabels(["0", "0.5", "1"], fontsize=7, color=INK2)
        ax.tick_params(axis="y", labelsize=7, colors=INK2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ad")
        ax.set_xlabel("per-problem p", fontsize=7.5, color=INK2)
    axes[0].set_ylabel("% of problems", fontsize=8, color=INK)
    fig.suptitle("Base model: per-problem success rate within each level "
                 "(T=1.0, strict reward, 64 samples × 200 problems per level; outer bars = exactly never / always solved)",
                 fontsize=8, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, facecolor=SURF)


def main():
    rows = summarise(load(REPO / "results" / "base_T1.0_per_problem.csv"), 1.0)
    rows += summarise(load(REPO / "results" / "base_per_problem.csv"), 0.7)
    out = REPO / "results" / "per_problem_p_by_level.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    plot(rows, REPO / "results" / "per_problem_p_hist.png")
    print("| level | T | mean p | never solved | p < 0.1 | 0.1 ≤ p ≤ 0.9 | p > 0.9 | always solved |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['level']} | {r['temperature']} | {r['mean_p']:.3f} | {100*r['frac_never_solved']:.0f}% | "
              f"{100*r['frac_p_below_0.1']:.0f}% | {100*r['frac_p_0.1_to_0.9']:.0f}% | "
              f"{100*r['frac_p_above_0.9']:.0f}% | {100*r['frac_always_solved']:.0f}% |")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
