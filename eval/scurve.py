"""Where on its learning curve does a level improve fastest?

Question (2026-11-20): a level's success rate over training is S-shaped. If its steepest point were near
p = 0.5, "train where improvement is fastest" (the bandit) would be the same target as "train where p ~ 0.5"
(threshold tau=0.5). This measures where the steepest point actually is.

Data: the per-step batch success rate p of each level in the GRPO batch (sampler_log.jsonl; ~256 groups x 6
rollouts per level per step for the fixed band, T = 1.0, the training reward), smoothed with a 5-step mean.
Rate = change in p per 10 steps (centred, +-5 steps). Only levels whose curve starts at p ~ 0 (op13+), so the
whole S is visible; op11-12 start mid-curve. Checkpoint re-evaluation (results/informative_over_training.csv)
adds op15-17 for the fixed band, which never trains them.

    $EDGE_PY eval/scurve.py [--runs fixed_band_s0 fixed_band_s1 uniform_s0 uniform_s1]
Writes results/scurve.png and results/scurve.csv (run, level, step, p_smoothed, rate_per_10_steps).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e6e1", "#fcfcfb"
LEVEL_COLOR = {11: "#b7d3f6", 12: "#6da7ec", 13: "#2a78d6", 14: "#104281",
               15: "#f4a37f", 16: "#eb6834", 17: "#9a3b12"}
MIN_N = 60          # rollouts of a level in a step's batch, below that the step is skipped
HALF = 5            # rate window: +-5 steps


def curves(run_dir: Path) -> dict[int, list[tuple[int, float]]]:
    rows = [json.loads(l) for l in (run_dir / "sampler_log.jsonl").open()]
    rows = [r for r in rows if r.get("event") == "step" and not r["probe"]]
    out: dict[int, list[tuple[int, float]]] = {}
    for L in sorted({int(k) for r in rows for k in r["p"]}):
        ser = [(r["step"], float(r["p"][str(L)])) for r in rows
               if str(L) in r["p"] and r["n"].get(str(L), 0) >= MIN_N]
        if len(ser) < 30:
            continue
        p = [x for _, x in ser]
        sm = [sum(p[max(0, i - 2):i + 3]) / len(p[max(0, i - 2):i + 3]) for i in range(len(p))]
        out[L] = [(s, v) for (s, _), v in zip(ser, sm)]
    return out


def rates(ser: list[tuple[int, float]]) -> list[tuple[int, float, float]]:
    out = []
    for i in range(HALF, len(ser) - HALF):
        (s0, p0), (s1, p1) = ser[i - HALF], ser[i + HALF]
        out.append((ser[i][0], ser[i][1], (p1 - p0) / (s1 - s0) * 10))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["fixed_band_s0", "fixed_band_s1", "uniform_s0", "uniform_s1"])
    ap.add_argument("--out", type=Path, default=REPO / "results" / "scurve.png")
    a = ap.parse_args()
    root = Path(os.environ["EDGE_SCRATCH"]) / "outputs" / "runs"
    data = {run: curves(root / run) for run in a.runs}

    rows, peaks = [], []
    for run, cs in data.items():
        for L, ser in cs.items():
            rs = rates(ser)
            for s, p, r in rs:
                rows.append([run, L, s, f"{p:.4f}", f"{r:.4f}"])
            if L >= 13 and ser[0][1] < 0.05 and rs:
                s, p, r = max(rs, key=lambda t: t[2])
                plateau = sum(v for _, v in ser[-20:]) / 20
                peaks.append((run, L, s, p, r, plateau))
    with a.out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "level", "step", "p_smoothed", "rate_per_10_steps"])
        w.writerows(rows)

    print("| run | level | steepest rise (dp per 10 steps) | at step | p there | plateau p (last 20 steps) | p / plateau |")
    print("|---|---|---|---|---|---|---|")
    for run, L, s, p, r, pl in peaks:
        print(f"| {run} | op{L} | {r:+.3f} | {s} | **{p:.2f}** | {pl:.2f} | {p / pl:.0%} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(SURF)
    for ax in (ax0, ax1):
        ax.set_facecolor(SURF)
        ax.grid(axis="y", color=GRID, lw=0.6)

    # Left: the S-curves of the fixed band (first run), trained levels every step + op15-17 from checkpoints.
    run0 = a.runs[0]
    for L, ser in data[run0].items():
        if L not in LEVEL_COLOR:
            continue
        ax0.plot([s for s, _ in ser], [v for _, v in ser], color=LEVEL_COLOR[L], lw=2)
        ax0.text(ser[-1][0] + 1.5, ser[-1][1], f"op{L}", va="center", fontsize=8, color=INK)
    ck = REPO / "results" / "informative_over_training.csv"
    if run0 == "fixed_band_s0" and ck.exists():
        pts: dict[int, list[tuple[int, float]]] = {}
        for r in csv.DictReader(ck.open()):
            if r["run"] == run0 and int(r["level"]) in (15, 16, 17):
                pts.setdefault(int(r["level"]), []).append((int(r["step"]), float(r["p"])))
        for L, ps in pts.items():
            ps.sort()
            ax0.plot([s for s, _ in ps], [v for _, v in ps], color=LEVEL_COLOR[L], lw=1.5, ls=(0, (4, 3)),
                     marker="o", ms=4, mfc=SURF)
            ax0.text(ps[-1][0] + 1.5, ps[-1][1], f"op{L} (not trained)", va="center", fontsize=8, color=INK)
    for run, L, s, p, r, pl in peaks:
        if run == run0:
            ax0.plot([s], [p], "o", ms=8, color=LEVEL_COLOR.get(L, INK2), mec=SURF, mew=2, zorder=5)
    ax0.set_xlim(0, 125)
    ax0.set_ylim(0, 0.7)
    ax0.set_xlabel("training step")
    ax0.set_ylabel("success rate p (training reward, T = 1.0)")
    ax0.set_title(f"{run0}: success rate per level (dot = steepest point)\nsolid = batch p every step (5-step mean); "
                  "dashed = checkpoint re-evaluation", fontsize=9, loc="left", color=INK)

    # Right: rate vs p for every level that starts at p ~ 0, all runs; the two threshold targets shaded.
    ax1.axvspan(0.05, 0.2, color="#1baf7a", alpha=0.12, lw=0)
    ax1.axvspan(0.4, 0.6, color="#e87ba4", alpha=0.12, lw=0)
    ax1.text(0.125, 0.98, "τ=0.1 band\n[0.05, 0.2]", transform=ax1.get_xaxis_transform(), ha="center", va="top",
             fontsize=8, color=INK2)
    ax1.text(0.5, 0.98, "τ=0.5 band\n[0.4, 0.6]", transform=ax1.get_xaxis_transform(), ha="center", va="top",
             fontsize=8, color=INK2)
    ax1.axhline(0, color=INK2, lw=0.8)
    for run, cs in data.items():
        for L, ser in cs.items():
            if L < 13 or ser[0][1] >= 0.05 or L not in LEVEL_COLOR:
                continue
            rs = rates(ser)
            ax1.plot([p for _, p, _ in rs], [r for _, _, r in rs], color=LEVEL_COLOR[L], lw=1.2, alpha=0.8)
    for run, L, s, p, r, pl in peaks:
        if L in LEVEL_COLOR:
            ax1.plot([p], [r], "o", ms=7, color=LEVEL_COLOR[L], mec=SURF, mew=1.5, zorder=5)
    handles = [plt.Line2D([], [], color=LEVEL_COLOR[L], lw=2, label=f"op{L}") for L in (13, 14, 15, 16, 17)]
    ax1.legend(handles=handles, frameon=False, fontsize=8, loc="center right")
    ax1.set_xlim(0, 0.65)
    ax1.set_xlabel("success rate p at that step")
    ax1.set_ylabel("improvement: change in p per 10 steps")
    ax1.set_title("Improvement rate vs current p, levels starting at p ≈ 0\n(" + ", ".join(a.runs) + ")",
                  fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=SURF)
    print(f"wrote {a.out} and {a.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
