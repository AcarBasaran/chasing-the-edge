"""Headline motivation: fraction of GRPO groups with NONZERO gradient, per level, base model.

A GRPO group of G rollouts on one problem has zero advantage (zero gradient) iff all G
rewards are equal. For a problem with success probability p:  P(informative) = 1 - p^G - (1-p)^G.

Two estimates per level, both from the TRAINING reward (what GRPO sees):
  from_level_p   plug the level-mean p into the formula (the CLAUDE.md back-of-envelope)
  per_problem    average the formula over problems, using the UNBIASED estimator of p^G
                 from c correct out of n samples: C(c,G)/C(n,G)  (and C(n-c,G)/C(n,G)).
                 This is the real number: within a level some problems are always solved
                 and some never, so the level-mean formula OVERSTATES the informative share.

G = 6 is the paper's (and our) rollout.n (upstream rl-200steps/op11-14_uniform*.yaml); G = 8 shown
for comparison with CLAUDE.md. Temperatures: T=0.7 (Gate 1 eval protocol, val op2-20, n=128) and
T=1.0 (= RL rollout temperature, val op9-15, n=64) when results/base_T1.0_per_problem.csv exists.

    sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY eval/informative_fraction.py'
Outputs: results/informative_fraction_by_level.csv, results/informative_fraction_by_level.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GS = (6, 8)
BAND = (11, 14)


def per_problem_frac(n: int, c: int, G: int) -> float:
    """Unbiased estimate of 1 - p^G - (1-p)^G from c successes in n >= G samples."""
    d = comb(n, G)
    return 1.0 - comb(c, G) / d - comb(n - c, G) / d


def load(path: Path, temperature: float):
    by_level = defaultdict(list)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "val":
                by_level[int(r["op"])].append((int(r["n"]), int(r["c_train"])))
    rows = []
    for L in sorted(by_level):
        probs = by_level[L]
        n_tot = sum(n for n, _ in probs)
        p = sum(c for _, c in probs) / n_tot
        rec = {"level": L, "temperature": temperature, "n_problems": len(probs),
               "samples_per_problem": probs[0][0], "p_train": p,
               "frac_problems_never_solved": sum(c == 0 for _, c in probs) / len(probs),
               "frac_problems_always_solved": sum(c == n for n, c in probs) / len(probs)}
        for G in GS:
            rec[f"G{G}_from_level_p"] = 1 - p ** G - (1 - p) ** G
            rec[f"G{G}_per_problem"] = sum(per_problem_frac(n, c, G) for n, c in probs) / len(probs)
        rows.append(rec)
    return rows


def plot(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, INK2, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.axvspan(BAND[0] - 0.5, BAND[1] + 0.5, color="#e6e5e0", lw=0, zorder=0)
    ax.text(12.5, 1.04, "paper's fixed band (11-14)", ha="center", va="bottom", color=INK2, fontsize=8)
    series = [(0.7, BLUE, "T=0.7 (eval protocol), val op2-20, n=128"),
              (1.0, ORANGE, "T=1.0 (RL rollout temperature), val op9-15, n=64")]
    for T, col, lab in series:
        rs = [r for r in rows if r["temperature"] == T]
        if not rs:
            continue
        x = [r["level"] for r in rs]
        ax.plot(x, [100 * r["G6_per_problem"] for r in rs], color=col, lw=2, marker="o", ms=5,
                label=f"{lab}: per-problem", zorder=3)
        ax.plot(x, [100 * r["G6_from_level_p"] for r in rs], color=col, lw=1.2, ls=(0, (3, 2)),
                label=f"{lab}: level-mean p in formula", zorder=2)
    for r in rows:   # direct labels on the band, T=0.7 per-problem
        if r["temperature"] == 0.7 and BAND[0] <= r["level"] <= BAND[1]:
            v = 100 * r["G6_per_problem"]
            ax.annotate(f"{v:.1f}%" if v >= 1 else f"{v:.2f}%", (r["level"], v), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=INK)
    ax.set_xlim(1.5, 20.5)
    ax.set_ylim(-3, 110)
    ax.set_xticks(range(2, 21))
    ax.set_xlabel("difficulty level (op count)", color=INK)
    ax.set_ylabel("GRPO groups with nonzero gradient (%)", color=INK)
    ax.set_title("Base model, G = 6 rollouts per group: share of groups that produce any gradient",
                 color=INK, fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)


def main() -> None:
    rows = load(REPO / "results" / "base_per_problem.csv", 0.7)
    t1 = REPO / "results" / "base_T1.0_per_problem.csv"
    if t1.exists():
        rows += load(t1, 1.0)
    out_csv = REPO / "results" / "informative_fraction_by_level.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    plot(rows, REPO / "results" / "informative_fraction_by_level.png")
    print("| level | T | p (train reward) | G=6 level-mean | G=6 per-problem | G=8 per-problem | never solved |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['level']} | {r['temperature']} | {r['p_train']:.4f} | {100*r['G6_from_level_p']:.2f}% | "
              f"{100*r['G6_per_problem']:.2f}% | {100*r['G8_per_problem']:.2f}% | "
              f"{100*r['frac_problems_never_solved']:.0f}% |")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
