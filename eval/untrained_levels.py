"""Hard-set results per level, with what each arm actually trained on: the cleanest comparison in the project.

Threshold tau=0.1 climbed to op17 and stopped, so op18-20 are levels it NEVER trained on. Uniform random
trained on all of 2-20, including 18-20 (but almost none of those groups had a nonzero gradient). If an arm
beats uniform on 18-20 without ever seeing them, that is extrapolation, not exposure.

Per run and hard level (15..20, plus pooled 15-17 and 18-20):
    train_share         share of all training prompts drawn from that level (sampler_log.jsonl, probes incl.)
    train_groups        GRPO groups trained at that level over the whole run
    informative_groups  ... of those, groups with nonzero reward variance (i.e. nonzero gradient)
    pass@1, pass@128    strict eval criterion, unbiased estimator, mean over problems
    newly_solved        problems with 0/128 for the base model and > 0 for the run
Also an exact McNemar test on newly-solved, pooled 18-20, for every pair of arms (paired by problem).

    $EDGE_PY eval/untrained_levels.py [--runs-root $EDGE_SCRATCH/outputs/runs]
Outputs: results/hard_by_level.csv, results/hard_by_level_mcnemar.csv, results/hard_by_level.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from passk import pass_at_k  # noqa: E402
from runs_io import METHOD_COLOR, REPO, discover, load_per_problem, method_sort_key, parse_run_id  # noqa: E402

LEVELS = (15, 16, 17, 18, 19, 20)
POOLS = {"15-17": (15, 16, 17), "18-20": (18, 19, 20)}
LABEL = {"fixed_band": "fixed band 11–14", "uniform": "uniform random 2–20", "uniform_11_17": "uniform op 11–17", "threshold_tau1": "threshold τ=0.1",
         "threshold_tau3": "threshold τ=0.3", "threshold_tau5": "threshold τ=0.5", "bandit": "bandit",
         "released_rl": "paper's RL (released)"}


def exposure(sampler_log: Path) -> tuple[Counter, Counter]:
    groups, informative = Counter(), Counter()
    if not sampler_log.exists():
        return groups, informative
    for line in sampler_log.open():
        d = json.loads(line) if line.strip() else {}
        if d.get("event") != "step":
            continue
        inf = d.get("informative_frac", {})
        for L, c in d["chosen"].items():
            groups[int(L)] += int(c)
            informative[int(L)] += int(c) * float(inf.get(L, 0.0))
    return groups, informative


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path(os.environ.get("EDGE_SCRATCH", "/scratch")) / "outputs" / "runs")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "hard_by_level.png")
    a = ap.parse_args()

    base = load_per_problem(REPO / "results" / "base_per_problem.csv")["test_hard"]
    runs = [(rid, p) for rid, p in discover() if rid != "base"]
    rows, solved = [], {}
    for rid, path in runs:
        arm, seed = parse_run_id(rid)
        hard = load_per_problem(path).get("test_hard", {})
        if not hard:
            continue
        groups, informative = exposure(a.runs_root / rid / "sampler_log.jsonl")
        total = sum(groups.values()) or 1
        solved[rid] = {u: int(r["c_eval"]) > 0 and int(base[u]["c_eval"]) == 0 for u, r in hard.items()}
        for key, levels in [(str(L), (L,)) for L in LEVELS] + list(POOLS.items()):
            probs = [r for r in hard.values() if int(r["op"]) in levels]
            rows.append({
                "run_id": rid, "method": arm, "seed": seed, "level": key, "n_problems": len(probs),
                "train_share": round(sum(groups[L] for L in levels) / total, 4),
                "train_groups": sum(groups[L] for L in levels),
                "informative_groups": round(sum(informative[L] for L in levels)),
                "pass@1": round(100 * sum(pass_at_k(int(r["n"]), int(r["c_eval"]), 1) for r in probs) / len(probs), 3),
                "pass@128": round(100 * sum(pass_at_k(int(r["n"]), int(r["c_eval"]), 128) for r in probs) / len(probs), 3),
                "newly_solved": sum(solved[rid][r["uid"]] for r in probs),
                "trained_here": rid != "released_rl" and sum(groups[L] for L in levels) > 0,
            })
    out_csv = a.out.with_suffix(".csv")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Paired test on pooled 18-20 newly-solved, every pair of runs.
    uids = [u for u, r in base.items() if int(r["op"]) in POOLS["18-20"]]
    ids = sorted(solved, key=lambda r: method_sort_key(parse_run_id(r)[0]))
    mc = []
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            b = sum(solved[x][u] and not solved[y][u] for u in uids)
            c = sum(solved[y][u] and not solved[x][u] for u in uids)
            mc.append({"run_a": x, "run_b": y, "only_a": b, "only_b": c, "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)})
    with a.out.with_name("hard_by_level_mcnemar.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mc[0]))
        w.writeheader()
        w.writerows(mc)

    plot([r for r in rows if r["run_id"] != "released_rl"], a.out)
    print("| run | op18-20 trained groups (informative) | pass@1 | pass@128 | newly solved / 102 |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["level"] == "18-20":
            print(f"| {r['run_id']} | {r['train_groups']} ({r['informative_groups']}) | {r['pass@1']:.1f} | "
                  f"{r['pass@128']:.1f} | {r['newly_solved']} |")
    print(f"wrote {out_csv}, {a.out.with_name('hard_by_level_mcnemar.csv')}, {a.out}")


def plot(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0", "#b5b4ad"
    arms = sorted({r["method"] for r in rows}, key=method_sort_key)

    def mean(arm, level, key):
        v = [r[key] for r in rows if r["method"] == arm and r["level"] == level]
        return sum(v) / len(v) if v else float("nan"), v

    def n_of(arm):
        return len({r["seed"] for r in rows if r["method"] == arm})

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150, facecolor=SURF,
                                 gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.55})
    for axis in (ax, bx):
        axis.set_facecolor(SURF)
        for s in ("top", "right"):
            axis.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            axis.spines[s].set_color(AXIS)
        axis.tick_params(labelsize=11, colors=INK2)

    # Left: pass@128 per hard level. Hollow marker = the arm trained on that level, filled = never trained there.
    ax.axvspan(17.5, 20.5, color=GRID, lw=0, zorder=0)
    ax.text(19, 1.0, "never trained on by τ=0.1,\nτ=0.3, fixed band, uniform 11–17", ha="center", va="top", fontsize=10,
            color=INK2, transform=ax.get_xaxis_transform())
    for arm in arms:
        col = METHOD_COLOR.get(arm, INK2)
        ys = [mean(arm, str(L), "pass@128")[0] for L in LEVELS]
        trained = [any(r["trained_here"] for r in rows if r["method"] == arm and r["level"] == str(L)) for L in LEVELS]
        ax.plot(LEVELS, ys, color=col, lw=2, zorder=2, label=LABEL.get(arm, arm))
        for L, y, t in zip(LEVELS, ys, trained):
            ax.scatter([L], [y], s=60, zorder=3, linewidth=1.8, edgecolor=col,
                       facecolor=SURF if t else col)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f"op{L}" for L in LEVELS])
    ax.set_xlim(14.6, 20.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("pass@128 on hard test set (%)", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("pass@128 per level\n○ = arm trained on this level, ● = never trained on it",
                 fontsize=11, color=INK, loc="left")

    # Right: newly solved on the pooled untrained levels 18-20, one horizontal bar per arm.
    order = sorted(arms, key=lambda m: mean(m, "18-20", "newly_solved")[0])
    for y, arm in enumerate(order):
        m, vals = mean(arm, "18-20", "newly_solved")
        groups = mean(arm, "18-20", "train_groups")[0]
        inf = mean(arm, "18-20", "informative_groups")[0]
        bx.barh(y, m, height=0.62, color=METHOD_COLOR.get(arm, INK2), edgecolor=SURF, linewidth=2, zorder=2)
        if len(vals) > 1:
            bx.hlines(y, min(vals), max(vals), color=INK, lw=1.2, zorder=3)
            bx.scatter(vals, [y] * len(vals), s=12, color=SURF, edgecolor=INK, linewidth=0.8, zorder=4)
        note = (f"trained there: {groups / 1000:.0f}k groups, {inf:.0f} with gradient" if groups
                else "never trained there")
        label = f"{m:.1f}" if len(vals) > 1 else f"{m:.0f}"     # mean over seeds: 27.7, not 28
        bx.annotate(f"{label}   {note}", (max(vals + [m]), y), textcoords="offset points", xytext=(5, 0),
                    va="center", fontsize=10, color=INK)
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels([f"{LABEL.get(a, a)} (n={n_of(a)})" for a in order], fontsize=11, color=INK)
    bx.set_xlim(0, 102 * 0.72)
    bx.set_xlabel("problems newly solved, op18–20\n(of 102; base model solved 0)", fontsize=11, color=INK)
    bx.grid(axis="x", color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    bx.set_title("New capability on levels 18–20\nbar = mean over n seeds, dots = seeds", fontsize=11, color=INK, loc="left")

    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Hard test set (op15–20), strict scoring, T=0.7, 128 samples per problem; mean over seeds",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.2)
    fig.savefig(out, facecolor=SURF, bbox_inches="tight")


if __name__ == "__main__":
    main()
