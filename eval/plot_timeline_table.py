"""Slide 12A figure: hard problems newly solved at steps 25/50/75/100, tau=0.1 vs the op 11-17 control, with the
fixed band's final (step-100) result as the reference row. Rendered as a table at slide size (fonts >= 16 pt).

Inputs: results/ckpt_timeline.csv, results/ckpt_timeline_paired.csv (eval/ckpt_timeline.py) and
results/compute_efficiency.csv (eval/compute_efficiency.py).

    $EDGE_PY eval/plot_timeline_table.py
Writes results/timeline_table_slide.png and .pdf.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from runs_io import METHOD_COLOR  # noqa: E402

INK, INK2, MUTED, RULE = "#1f2328", "#57606a", "#8c959f", "#d0d7de"
TAU, CTRL, FB = METHOD_COLOR["threshold_tau1"], METHOD_COLOR["uniform_11_17"], METHOD_COLOR["fixed_band"]


def read(name: str) -> list[dict]:
    return list(csv.DictReader((REPO / "results" / name).open()))


def main() -> None:
    tl = read("ckpt_timeline.csv")
    pr = read("ckpt_timeline_paired.csv")
    ce = read("compute_efficiency.csv")

    def val(step: int, arm: str) -> str:
        r = next(r for r in tl if int(r["step"]) == step and r["pool"] == "hard" and r["arm"] == arm)
        return f"{float(r['newly_solved_mean']):.0f}"

    def gap(step: int) -> tuple[str, str]:
        r = next(r for r in pr if int(r["step"]) == step and r["pool"] == "hard")
        p = float(r["sign_test_p"])
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"
        return f"{float(r['diff_per_seed']):+.0f}", ptxt

    fb = next(r for r in ce if r["comparison"].startswith("tau=0.1 @ step 25") and r["pool"] == "hard")
    fb_final = f"{float(fb['b_mean']):.0f}"
    hrs = {r["pool"]: float(r["a_mean"]) for r in ce if r["comparison"] == "train_time"}
    tau25_h = sum(hrs[f"threshold_tau1_s{s} steps 1-25"] for s in range(3)) / 3
    fb100_h = sum(hrs[f"fixed_band_s{s} steps 1-100"] for s in range(2)) / 2

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(13.3, 6.9))
    ax.set_xlim(0, 13.3)
    ax.set_ylim(0, 6.9)
    ax.axis("off")

    X_STEP, X_PHASE, X_TAU, X_CTRL, X_FB, X_GAP = 0.35, 2.35, 5.75, 7.45, 9.15, 11.25
    head_y, row_h = 6.25, 0.74
    ax.text(X_STEP, head_y, "step\n(budget)", fontsize=16, color=INK2, ha="left", va="center", linespacing=1.1)
    ax.text(X_PHASE, head_y, "what τ = 0.1\nwas doing", fontsize=16, color=INK2, ha="left", va="center",
            linespacing=1.1)
    for x, h, col in ((X_TAU, "τ = 0.1", TAU), (X_CTRL, "control\n11–17", CTRL), (X_FB, "fixed band\n(paper)", FB)):
        ax.text(x, head_y + 0.05, h, fontsize=17, color=INK, ha="center", va="center", weight="bold", linespacing=1.05)
        ax.plot([x - 0.6, x + 0.6], [head_y - 0.47] * 2, color=col, lw=5, solid_capstyle="round")
    ax.text(X_GAP + 0.45, head_y, "τ = 0.1 − control\n(paired)", fontsize=16, color=INK2, ha="center", va="center",
            linespacing=1.1)
    ax.plot([0.3, 13.0], [head_y - 0.62] * 2, color=RULE, lw=1.2)

    phases = {25: "climbing", 50: "reaches level 17", 75: "stalled at 17", 100: "stalled at 17"}
    ys = {}
    for i, step in enumerate((25, 50, 75, 100)):
        y = head_y - 0.62 - row_h * (i + 0.62)
        ys[step] = y
        ax.text(X_STEP, y, f"{step} ({step}%)", fontsize=17, color=INK, ha="left", va="center")
        ax.text(X_PHASE, y, phases[step], fontsize=15, color=INK2, ha="left", va="center", style="italic")
        ax.text(X_TAU, y, val(step, "threshold_tau1"), fontsize=22, color=INK, ha="center", va="center",
                weight="bold" if step == 25 else "normal")
        ax.text(X_CTRL, y, val(step, "uniform_11_17"), fontsize=22, color=INK, ha="center", va="center")
        ax.text(X_FB, y, fb_final if step == 100 else "–", fontsize=22, color=INK if step == 100 else MUTED,
                ha="center", va="center", weight="bold" if step == 100 else "normal")
        g, p = gap(step)
        ax.text(X_GAP - 0.1, y, g, fontsize=20, color=INK, ha="right", va="center")
        ax.text(X_GAP + 0.1, y, p, fontsize=15, color=INK2, ha="left", va="center")
    for x, step, col in ((X_TAU, 25, TAU), (X_FB, 100, FB)):
        ax.add_patch(FancyBboxPatch((x - 0.62, ys[step] - 0.3), 1.24, 0.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=col, alpha=0.16, ec="none", zorder=0))
    y_bot = ys[100] - 0.45
    ax.plot([0.3, 13.0], [y_bot] * 2, color=RULE, lw=1.2)

    ax.text(X_STEP, y_bot - 0.62,
            f"τ = 0.1 at step 25 ({fb['a_mean'].split('.')[0]}) already matches the paper's final result ({fb_final}),\n"
            f"in a quarter of the compute ({tau25_h:.1f} vs {fb100_h:.1f} GPU-hours).",
            fontsize=17, color=INK, ha="left", va="center", weight="bold", linespacing=1.25)
    ax.text(X_STEP, y_bot - 1.3,
            f"54 vs 48: +{float(fb['diff']):.0f} problems, 95% CI [{float(fb['ci95_lo']):+.1f}, {float(fb['ci95_hi']):+.1f}], "
            f"sign test p = {float(fb['sign_test_p']):.2f}. The control at step 25: 33.",
            fontsize=15, color=INK2, ha="left", va="center")
    ax.text(X_STEP, 0.15, "Hard test problems (204, levels 15–20) newly solved. Mean of 3 seeds (fixed band: 2). "
            "Paired sign test over problems;\nfive comparisons shown, not corrected for multiple comparisons.",
            fontsize=12.5, color=INK2, ha="left", va="bottom", linespacing=1.3)
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "results" / f"timeline_table_slide.{ext}", dpi=150, facecolor="white")
    print("wrote results/timeline_table_slide.png, .pdf")


if __name__ == "__main__":
    main()
