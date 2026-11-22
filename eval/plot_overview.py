"""Overview figure for the progress report: the GRPO training loop with the sampler as the one component we replace.

Left (accent, dashed): the sampler. probe -> success rate per level -> level choice -> batch.
Right (grey): verl's GRPO step, identical for every arm.

    $EDGE_PY eval/plot_overview.py            # writes results/overview.png and results/overview.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]

INK, INK2 = "#1f2328", "#57606a"
GREY_FILL, GREY_EDGE = "#f3f4f6", "#8c959f"
ACC, ACC_FILL, ACC_BG = "#0b6e69", "#e3f4f2", "#f4fbfa"
POOL_FILL = "#ffffff"


def box(ax, x, y, w, h, title, body, fill, edge, title_color=INK, fs=12.5, bfs=11):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.6, zorder=2))
    ax.text(x, y + h / 2 - 0.16, title, ha="center", va="top", fontsize=fs, weight="bold", color=title_color,
            zorder=3)
    ax.text(x, y + h / 2 - 0.52, body, ha="center", va="top", fontsize=bfs, color=INK, linespacing=1.35, zorder=3)


def arrow(ax, p, q, color=INK, label=None, lpos=None, rad=0.0, ls="-", lw=1.8, lha="center", lcolor=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=4))
    if label:
        x, y = lpos if lpos else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(x, y, label, ha=lha, va="center", fontsize=10.5, color=lcolor or color, style="italic", zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.5))


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.9, 8.8))
    ax.set_xlim(0, 14.9)
    ax.set_ylim(-0.4, 8.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    xs, Ws = 5.3, 5.0          # sampler column
    xt, Wt = 11.3, 4.5         # training column
    hs = [1.45, 1.45, 2.05, 1.15]
    gap = 0.35
    ys, top = [], 6.95
    for h in hs:
        ys.append(top - h / 2)
        top = top - h - gap

    # Regions.
    ax.add_patch(FancyBboxPatch((2.6, -0.33), 5.4, 7.5, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc=ACC_BG, ec=ACC, lw=2.0, ls=(0, (5, 3)), zorder=0))
    ax.text(xs, 7.95, "THE SAMPLER: the only component we replace", ha="center", va="center", fontsize=14,
            weight="bold", color=ACC)
    ax.text(xs, 7.55, "(verl's data.sampler hook; one rule per arm)", ha="center", va="center", fontsize=11.5,
            color=ACC)
    ax.add_patch(FancyBboxPatch((8.75, -0.33), 5.95, 7.5, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="#fafbfc", ec=GREY_EDGE, lw=1.4, zorder=0))
    ax.text(11.7, 7.95, "verl GRPO step: unchanged", ha="center", va="center", fontsize=14, weight="bold",
            color=INK2)
    ax.text(11.7, 7.55, "(same model, reward, settings and budget in every arm)", ha="center", va="center",
            fontsize=11.5, color=INK2)

    # Sampler column, top to bottom: probe -> success rate per level -> level choice -> batch.
    box(ax, xs, ys[0], Ws, hs[0], "① Probe",
        "every m = 25 steps: 4 problems from every level,\nadded to the training batch (bandit only);\n"
        "trained on, and counted in the budget", ACC_FILL, ACC, ACC)
    box(ax, xs, ys[1], Ws, hs[1], r"② Success rate per level  $\hat{p}_L$",
        "share of answers at level L with reward 1,\nfrom the probe and the trained batch\n"
        "(training reward, T = 1)", ACC_FILL, ACC, ACC)
    box(ax, xs, ys[2], Ws, hs[2], r"③ Level choice  $L_t$",
        r"threshold $\tau$: move up if $\hat{p}_L > \tau+\delta$, down if" "\n"
        r"$\hat{p}_L < \max(\tau-\delta,\,\tau/2)$, 2 steps in a row ($\delta = 0.1$)" "\n"
        r"bandit: $Q_L \leftarrow (1-\alpha)\,Q_L + \alpha\,[\hat{p}_L(t) - \hat{p}_L(t-m)]$" "\n"
        r"and $\varepsilon$-greedy over levels" "\n"
        "fixed band: 11–14  ·  uniform: 2–20 or 11–17",
        ACC_FILL, ACC, ACC, bfs=10.5)
    box(ax, xs, ys[3], Ws, hs[3], "④ Batch",
        "1,024 problems drawn from the pool\nbucket(s) of the chosen level(s)", ACC_FILL, ACC, ACC)

    # Training column, bottom to top.
    box(ax, xt, ys[3], Wt, hs[3], "Generate (vLLM)", "6 answers per problem, T = 1", GREY_FILL, GREY_EDGE)
    box(ax, xt, ys[2], Wt, hs[2], "Strict process reward",
        "r = 1 only if every step and the\nfinal answer are right, else 0", GREY_FILL, GREY_EDGE)
    box(ax, xt, ys[1], Wt, hs[1], "Group advantage",
        r"$A_i = (r_i - \mathrm{mean}\ r)\ /\ \mathrm{std}\ r$" "\n"
        r"all 6 answers agree $\Rightarrow$ $A = 0$: no gradient", GREY_FILL, GREY_EDGE)
    box(ax, xt, ys[0], Wt, hs[0], "Policy update",
        "actor update (FSDP), then weight\nsync to the generator", GREY_FILL, GREY_EDGE)

    # Pool.
    ax.add_patch(FancyBboxPatch((0.15, ys[3] - 0.65), 2.15, 1.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=POOL_FILL, ec=INK2, lw=1.4, zorder=2))
    ax.text(1.22, ys[3] + 0.5, "Problem pool", ha="center", va="top", fontsize=12.5, weight="bold", color=INK)
    ax.text(1.22, ys[3] + 0.12, "946k problems,\nbucketed by op 2–20", ha="center", va="top", fontsize=10.5,
            color=INK, linespacing=1.3)

    # Sampler flow (down).
    for i in range(3):
        arrow(ax, (xs, ys[i] - hs[i] / 2), (xs, ys[i + 1] + hs[i + 1] / 2), color=ACC, lw=2.2)
    arrow(ax, (2.32, ys[3]), (xs - Ws / 2, ys[3]), color=INK2, label="draw", lpos=(2.55, ys[3] + 0.3),
          lcolor=INK2)
    # Batch -> training.
    arrow(ax, (xs + Ws / 2, ys[3]), (xt - Wt / 2, ys[3]), color=INK, label="batch\nof step t",
          lpos=(8.38, ys[3] + 0.42), lcolor=INK)
    # Training flow (up).
    for i in (3, 2, 1):
        arrow(ax, (xt, ys[i] + hs[i] / 2), (xt, ys[i - 1] - hs[i - 1] / 2), color=INK2)
    # Updated weights back to the generator (inside the grey region, right side).
    arrow(ax, (xt + Wt / 2, ys[0] - 0.2), (xt + Wt / 2, ys[3] + 0.2), color=INK2, rad=-0.22)
    ax.text(14.17, (ys[0] + ys[3]) / 2, "updated weights", rotation=90, ha="center", va="center", fontsize=10.5,
            color=INK2, style="italic", bbox=dict(fc="#fafbfc", ec="none", pad=1.5))
    # Rewards feed the per-level success rate.
    arrow(ax, (xt - Wt / 2, ys[2] + 0.35), (xs + Ws / 2, ys[1] - 0.35), color=ACC, lw=2.0)
    ax.text(8.38, (ys[1] + ys[2]) / 2 + 0.05, "rewards\nper level", ha="center", va="center", fontsize=10.5,
            color=ACC, style="italic", bbox=dict(fc="white", ec="none", pad=1.5))

    for ext in ("png", "pdf"):
        fig.savefig(REPO / "results" / f"overview.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote results/overview.png, results/overview.pdf")


if __name__ == "__main__":
    main()
