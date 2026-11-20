"""Compare methods on pass@1 and pass@128 per test set (easy / medium / hard), across seeds.

Inputs: per-problem CSVs (see eval/runs_io.py for where they come from). Metric: the paper's
strict eval criterion, unbiased pass@k = 1 - C(n-c,k)/C(n,k) averaged over problems (eval/passk.py).
Per method: mean over seeds, with every seed shown as a dot and the min-max range as a bar.
Also reports the headline count from eval/newly_solved.py (hard set: base 0/128 -> run > 0).

    $EDGE_PY eval/compare_methods.py                      # every CSV found by runs_io.discover()
    $EDGE_PY eval/compare_methods.py --run fixed_band_s0=... --run threshold_tau3_s0=...
Outputs: results/method_comparison_runs.csv (one row per run x set),
         results/method_comparison.csv (one row per method x set: mean / min / max over seeds),
         results/method_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from newly_solved import newly_solved  # noqa: E402
from passk import mean_pass_at_k  # noqa: E402
from runs_io import (METHOD_COLOR, METHOD_LABEL, NEUTRAL, REPO, SET_LABEL, TEST_SETS, discover,  # noqa: E402
                     load_per_problem, method_sort_key, parse_label_paths, parse_run_id)

KS = (1, 8, 128)
PLOT_KS = (1, 128)


def per_run_rows(runs, base, criterion="c_eval"):
    rows = []
    for run_id, path in runs:
        arm, seed = parse_run_id(run_id)
        data = load_per_problem(path)
        ns = {r["split"]: r for r in newly_solved(base, data, criterion) if r["level"] == "all"} \
            if run_id != "base" else {}
        for split in TEST_SETS:
            if split not in data:
                continue
            counts = [(int(r["n"]), int(r[criterion])) for r in data[split].values()]
            assert min(n for n, _ in counts) >= max(KS), f"{run_id}/{split}: need n >= {max(KS)} samples"
            rec = {"run_id": run_id, "method": arm, "seed": seed, "split": split, "n_problems": len(counts),
                   "criterion": criterion}
            for k in KS:
                rec[f"pass@{k}"] = 100 * mean_pass_at_k(counts, k)
            rec["newly_solved"] = ns[split]["newly_solved"] if split in ns else ""
            rec["base_unsolved"] = ns[split]["base_unsolved"] if split in ns else ""
            rows.append(rec)
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["split"])].append(r)
    out = []
    for (m, split), rs in sorted(groups.items(), key=lambda kv: (method_sort_key(kv[0][0]), TEST_SETS.index(kv[0][1]))):
        rec = {"method": m, "split": split, "n_seeds": len(rs), "seeds": " ".join(str(r["seed"]) for r in rs)}
        for key in [f"pass@{k}" for k in KS] + ["newly_solved"]:
            vals = [float(r[key]) for r in rs if r[key] != ""]
            if vals:
                rec[f"{key}_mean"] = sum(vals) / len(vals)
                rec[f"{key}_min"], rec[f"{key}_max"] = min(vals), max(vals)
        out.append(rec)
    return out


def plot(rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
    methods = sorted({r["method"] for r in rows}, key=method_sort_key)
    # Horizontal bars: method names sit on the y axis, once per row, so they never collide.
    fig, axes = plt.subplots(len(PLOT_KS), len(TEST_SETS),
                             figsize=(3.3 * len(TEST_SETS) + 1.2, (0.3 * len(methods) + 0.9) * len(PLOT_KS)),
                             dpi=150, facecolor=SURF, squeeze=False)
    ys = list(range(len(methods)))[::-1]          # first method on top
    for i, k in enumerate(PLOT_KS):
        for j, split in enumerate(TEST_SETS):
            ax = axes[i][j]
            ax.set_facecolor(SURF)
            top = max([r[f"pass@{k}"] for r in rows if r["split"] == split] + [1.0])
            for y, m in zip(ys, methods):
                vals = [r[f"pass@{k}"] for r in rows if r["method"] == m and r["split"] == split]
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                col = METHOD_COLOR.get(m, INK2)
                ax.barh(y, mean, height=0.7, color=col, edgecolor=SURF, linewidth=2, zorder=2)
                if len(vals) > 1:
                    ax.hlines(y, min(vals), max(vals), color=INK, lw=1.2, zorder=3)
                    ax.scatter(vals, [y] * len(vals), s=12, color=SURF, edgecolor=INK, linewidth=0.8, zorder=4)
                ax.annotate(f"{mean:.1f}", (max(vals), y), textcoords="offset points", xytext=(3, 0),
                            va="center", fontsize=6.5, color=INK)
            ax.set_yticks(ys)
            ax.set_yticklabels([METHOD_LABEL.get(m, m).replace("\n", " ") for m in methods] if j == 0 else [],
                               fontsize=7, color=INK)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=7, colors=INK2)
            ax.set_xlim(0, min(112, top * 1.15 + 2))
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color(NEUTRAL)
            if i == 0:
                ax.set_title(SET_LABEL[split], fontsize=9, color=INK, loc="left")
            if i == len(PLOT_KS) - 1:
                ax.set_xlabel("%", fontsize=7.5, color=INK2)
            if j == len(TEST_SETS) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"pass@{k}", fontsize=8.5, color=INK, rotation=270, labelpad=12)
    n_seeds = max(len({r["seed"] for r in rows if r["method"] == m}) for m in methods)
    fig.suptitle("Test-set accuracy by method (strict process-verified scoring, T=0.7, 128 samples per problem"
                 + (f"; bars = mean over seeds, dots = seeds, line = range)" if n_seeds > 1 else ")"),
                 fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, facecolor=SURF)



def write_csv(rows, path: Path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="label=per_problem.csv (repeatable); label = <arm>_s<seed>")
    ap.add_argument("--base", type=Path, default=REPO / "results" / "base_per_problem.csv")
    ap.add_argument("--criterion", default="c_eval", choices=["c_eval", "c_train", "c_answer"])
    ap.add_argument("--out-prefix", type=Path, default=REPO / "results" / "method_comparison")
    a = ap.parse_args()
    runs = [("base", a.base)] + parse_label_paths(a.run) if a.run else discover()
    base = load_per_problem(a.base)
    rows = per_run_rows(runs, base, a.criterion)
    agg = aggregate(rows)
    write_csv(rows, a.out_prefix.with_name(a.out_prefix.name + "_runs.csv"))
    write_csv(agg, a.out_prefix.with_suffix(".csv"))
    plot(rows, a.out_prefix.with_suffix(".png"))
    print("| method | set | seeds | pass@1 | pass@128 | newly solved (mean) |")
    print("|---|---|---|---|---|---|")
    for r in agg:
        ns = f"{r['newly_solved_mean']:.1f}" if "newly_solved_mean" in r else ""
        print(f"| {r['method']} | {r['split']} | {r['n_seeds']} | {r['pass@1_mean']:.1f} | {r['pass@128_mean']:.1f} | {ns} |")
    print(f"wrote {a.out_prefix}.csv / _runs.csv / .png")


if __name__ == "__main__":
    main()
