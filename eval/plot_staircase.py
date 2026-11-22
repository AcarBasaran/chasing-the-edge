"""Slide figure: the sampler staircase (seed 0) + steps tau=0.1 spends at each level (mean over 3 seeds).

Left: level trained at every step for threshold tau=0.1 / 0.3 / 0.5 and the bandit, seed 0. A threshold batch is
one level; for the bandit we plot the level holding most of the batch (its greedy level, ~90%). Before its first
value update (step 25) the bandit spreads the batch over all levels, so no single level is drawn there. The
stretch where tau=0.1 sits on level 17 is shaded. Right: steps tau=0.1 spends at each level, mean of 3 seeds
(bar) with each seed as a dot. Sized for a 13.3-in slide: fonts >= 14 pt.

    $EDGE_PY eval/plot_staircase.py [--runs-root $EDGE_SCRATCH/outputs/runs]
Writes results/staircase_slide.png and .pdf.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "eval"))
from runs_io import METHOD_COLOR  # noqa: E402  (project palette, CVD-checked)

INK, INK2, MUTED = "#1f2328", "#57606a", "#8c959f"
GRID, SURF = "#e6e8eb", "#ffffff"
SERIES = [  # (run, label, colour, small vertical offset so identical early segments stay visible)
    ("threshold_tau1_s0", "τ = 0.1", METHOD_COLOR["threshold_tau1"], +0.10),
    ("threshold_tau3_s0", "τ = 0.3", METHOD_COLOR["threshold_tau3"], -0.10),
    ("threshold_tau5_s0", "τ = 0.5", METHOD_COLOR["threshold_tau5"], 0.0),
    ("bandit_s0", "bandit", METHOD_COLOR["bandit"], 0.0),
]


def levels_per_step(path: Path) -> tuple[list[int | None], list[float]]:
    lv, share = [], []
    for line in path.open():
        r = json.loads(line)
        if r.get("event") != "step":
            continue
        c = Counter({int(k): v for k, v in r["chosen"].items()})
        L, n = c.most_common(1)[0]
        frac = n / sum(c.values())
        lv.append(L if frac >= 0.5 else None)
        share.append(frac)
    return lv, share


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path(os.environ.get("EDGE_SCRATCH", "/scratch")) / "outputs" / "runs")
    a = ap.parse_args()

    plt.rcParams.update({"font.size": 16, "axes.edgecolor": MUTED, "axes.linewidth": 1.0,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13.3, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1], wspace=0.16, left=0.085, right=0.975, top=0.80,
                          bottom=0.13)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)

    # ---- staircase
    data = {run: levels_per_step(a.runs_root / run / "sampler_log.jsonl") for run, *_ in SERIES}
    tau1 = data["threshold_tau1_s0"][0]
    arrive17 = next(t for t in range(len(tau1)) if all(x == 17 for x in tau1[t:]))
    ax.axvspan(arrive17, len(tau1), color=METHOD_COLOR["threshold_tau1"], alpha=0.10, lw=0, zorder=0)
    ax.text((arrive17 + len(tau1)) / 2, 16.55, f"τ = 0.1 stays on level 17:\nsteps {arrive17}–100 (seed 0)",
            ha="center", va="center", fontsize=15, color=INK, linespacing=1.2)

    handles = []
    for run, label, col, off in SERIES:
        lv = data[run][0]
        steps = np.arange(len(lv) + 1)
        y = np.array([np.nan if L is None else L + off for L in lv] + [np.nan], dtype=float)
        y[-1] = y[-2]
        (h,) = ax.step(steps, y, where="post", color=col, lw=3.4, solid_capstyle="round", zorder=3, label=label)
        handles.append(h)
        # direct label in ink, just above the line's right end (the legend carries colour identity too)
        last = next(L for L in reversed(lv) if L is not None)
        ax.text(98.5, last + off + 0.13, label, ha="right", va="bottom", fontsize=16, color=INK, zorder=5)
    b0 = data["bandit_s0"][0]
    first_greedy = next(t for t, L in enumerate(b0) if L is not None)
    ax.annotate(f"bandit: all levels at random\nfor its first {first_greedy} steps",
                xy=(first_greedy, 13), xytext=(29, 15.0), fontsize=14, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=4, shrinkB=6))

    ax.set_xlim(0, 100.5)
    ax.set_ylim(10.4, 17.9)
    ax.set_yticks(range(11, 18))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("training step", fontsize=16, color=INK)
    ax.set_ylabel("level trained\n(number of operations)", fontsize=16, color=INK)
    ax.grid(axis="y", color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=15, length=0, pad=6)
    ax.set_title("Where each sampler trained (seed 0)", fontsize=18, color=INK, loc="left", pad=12,
                 fontweight="bold")

    # ---- steps at each level, tau=0.1, 3 seeds
    per_seed = []
    for s in (0, 1, 2):
        lv = levels_per_step(a.runs_root / f"threshold_tau1_s{s}" / "sampler_log.jsonl")[0]
        c = Counter(lv)
        per_seed.append([c.get(L, 0) for L in range(11, 18)])
    per_seed = np.array(per_seed)
    mean = per_seed.mean(axis=0)
    ys = np.arange(11, 18)
    col = METHOD_COLOR["threshold_tau1"]
    bx.barh(ys, mean, height=0.58, color=col, alpha=0.9, zorder=2)
    for i, L in enumerate(ys):
        bx.scatter(per_seed[:, i], [L] * 3, s=46, color=SURF, edgecolor=INK, linewidth=1.3, zorder=4)
        bx.text(max(mean[i], per_seed[:, i].max()) + 2.2, L, f"{mean[i]:.0f}" if mean[i] >= 9.5 else f"{mean[i]:.1f}",
                ha="left", va="center", fontsize=15, color=INK)
    bx.set_ylim(10.4, 17.9)
    bx.set_yticks(range(11, 18))
    bx.set_xlim(0, 68)
    bx.set_xticks([0, 25, 50])
    bx.set_xlabel("steps at each level", fontsize=16, color=INK)
    bx.grid(axis="x", color=GRID, lw=1.0)
    bx.set_axisbelow(True)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.tick_params(labelsize=15, length=0, pad=6)
    bx.set_title("τ = 0.1: steps per level\n(3 seeds; bar = mean)", fontsize=16, color=INK, loc="left",
                 pad=12)

    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.085, 0.985), ncol=4,
               frameon=False, fontsize=15, handlelength=1.6, columnspacing=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "results" / f"staircase_slide.{ext}", dpi=150, facecolor="white")
    print(f"tau=0.1 seed 0 reaches level 17 at step {arrive17}; mean steps per level 11-17: "
          + ", ".join(f"op{L} {m:.1f}" for L, m in zip(ys, mean)))
    print("wrote results/staircase_slide.png, results/staircase_slide.pdf")


if __name__ == "__main__":
    main()
