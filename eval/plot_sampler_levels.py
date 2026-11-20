"""Which level each sampler chose over training, against measured success rates.

Per run, two panels sharing the step axis:
  top    : levels trained on at each step (dot area = share of the training batch; probe prompts
           excluded), batch-weighted mean level as a line. If the run's train.log has tracking sweeps
           (val-core/difficulty-5B/<L>/.../mean@8, every trainer.test_freq steps), the background is the
           measured success rate p_L(t) of EVERY level (light = solved, dark = unsolved) and the
           lines mark where p crosses 0.5 and 0.1 -- i.e. where the edge is.
  bottom : success rate and share of informative GRPO groups (nonzero gradient) in the training
           batch, measured on the sampler's own batches (sampler_log.jsonl). For Threshold, its
           target band [lo, hi] is shaded.

Inputs: run directories (outputs/runs/<run_id>/ with sampler_log.jsonl [+ train.log]) or explicit
    label=sampler_log.jsonl[,train.log]
    $EDGE_PY eval/plot_sampler_levels.py $EDGE_SCRATCH/outputs/runs/{fixed_band,threshold_tau3}_s0 \\
        --out results/sampler_levels.png
Outputs: the PNG, plus <out>.csv with one row per run x step.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_speed import edge, level_p  # noqa: E402
from parse_verl_log import parse  # noqa: E402
from runs_io import METHOD_COLOR, REPO, parse_run_id  # noqa: E402


def load_sampler_log(path: Path) -> dict:
    init, steps = {}, []
    for line in Path(path).open():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event") == "init":
            init = d
        elif d.get("event") == "step":
            steps.append(d)
    k = int(init.get("probe_prompts_per_level", 0))
    rows = []
    for d in steps:
        chosen = {int(L): int(c) for L, c in d["chosen"].items()}
        if d.get("probe"):           # probe prompts: k per level, on top of the training allocation
            chosen = {L: c - k for L, c in chosen.items() if c - k > 0}
        tot = sum(chosen.values()) or 1
        p, n = d.get("p", {}), d.get("n", {})
        inf = d.get("informative_frac", {})
        w = {L: c / tot for L, c in chosen.items()}
        st = d.get("state", {})
        rows.append({
            "step": int(d["step"]), "probe": bool(d.get("probe")), "share": w,
            "mean_level": sum(L * s for L, s in w.items()),
            "p_trained": sum(s * p[str(L)] for L, s in w.items() if str(L) in p),
            "informative_trained": sum(s * inf[str(L)] for L, s in w.items() if str(L) in inf),
            "lo": st.get("lo"), "hi": st.get("hi"), "active": st.get("active"),
        })
    return {"method": init.get("method", "?"), "kwargs": init.get("kwargs", {}), "rows": rows}


def load_tracking(train_log: Path | None):
    if not train_log or not Path(train_log).exists():
        return []
    return [(r["step"], level_p(r)) for r in parse(str(train_log)) if level_p(r)]


def arm_of(label: str, log: dict) -> str:
    """Colour key: the run id's arm if it is a known method, else derived from the logged method/tau."""
    arm, _ = parse_run_id(label)
    if arm in METHOD_COLOR:
        return arm
    m, tau = log.get("method"), log.get("kwargs", {}).get("tau")
    if m == "threshold" and tau is not None:
        return f"threshold_tau{int(round(10 * float(tau)))}"
    return {"uniform_random": "uniform"}.get(m, m)


def resolve(item: str) -> tuple[str, Path, Path | None]:
    label, _, rest = item.partition("=")
    if not rest:
        d = Path(item)
        if d.is_dir():
            tl = d / "train.log"
            return d.name, d / "sampler_log.jsonl", tl if tl.exists() else None
        label, rest = Path(item).stem, item
    paths = rest.split(",")
    return label, Path(paths[0]), Path(paths[1]) if len(paths) > 1 else None


def plot(runs, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
    seq = LinearSegmentedColormap.from_list("p", ["#0d3a73", "#2a78d6", "#9cc3f0", "#f2f6fc"])  # dark = unsolved
    fig, axes = plt.subplots(2, len(runs), figsize=(4.6 * len(runs), 5.6), dpi=150, facecolor=SURF,
                             sharex="col", gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    mesh = None
    for j, (label, log, track) in enumerate(runs):
        top, bot = axes[0][j], axes[1][j]
        for ax in (top, bot):
            ax.set_facecolor(SURF)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#b5b4ad")
            ax.tick_params(labelsize=7, colors=INK2)
        col = METHOD_COLOR.get(arm_of(label, log), INK)
        rows = log["rows"]
        steps = [r["step"] for r in rows]
        if track:
            ts = [t for t, _ in track]
            levels = sorted({L for _, p in track for L in p})
            edges_t = ts + [ts[-1] + (ts[-1] - ts[-2] if len(ts) > 1 else 10)]
            z = np.array([[p.get(L, np.nan) for _, p in track] for L in levels])
            mesh = top.pcolormesh(edges_t, [L - 0.5 for L in levels] + [levels[-1] + 0.5], z, cmap=seq,
                                  vmin=0, vmax=1, shading="flat", zorder=0)
            for tau, ls in ((0.5, "-"), (0.1, ":")):
                pts = [(t + (edges_t[i + 1] - t) / 2, edge(p, tau)) for i, (t, p) in enumerate(track)]
                pts = [(x, y) for x, y in pts if y is not None]
                if pts:
                    top.plot(*zip(*pts), color=SURF, lw=1.6, ls=ls, zorder=1,
                             label=f"edge: where measured p crosses {tau}")
        xs, ys, ss = [], [], []
        for r in rows:
            for L, s in r["share"].items():
                xs.append(r["step"]), ys.append(L), ss.append(s)
        top.scatter(xs, ys, s=[4 + 40 * s for s in ss], color=col, edgecolor=SURF if track else col,
                    linewidth=0.4, zorder=3, label="level trained on (dot area = share of batch)")
        top.plot(steps, [r["mean_level"] for r in rows], color=INK, lw=1, zorder=4, label="mean level trained on")
        probes = [r["step"] for r in rows if r["probe"]]
        if probes:
            top.scatter(probes, [1.6] * len(probes), marker="|", s=40, color=INK2, zorder=4, label="probe step")
        top.set_ylim(1.2, 20.8)
        top.set_yticks([2, 5, 8, 11, 14, 17, 20])
        top.grid(axis="y", color=GRID, lw=0.5)
        kw = ", ".join(f"{k}={v}" for k, v in log["kwargs"].items() if k != "seed")
        head = f"{log['method']} ({kw})" if kw else log["method"]
        top.set_title(f"{label}\n" + textwrap.fill(head, 46), fontsize=7.5, color=INK, loc="left")
        if j == 0:
            top.set_ylabel("difficulty level (op)", fontsize=8, color=INK)
            bot.set_ylabel("rate in training batch", fontsize=8, color=INK)
        if any(r["lo"] is not None for r in rows):
            bot.fill_between(steps, [r["lo"] for r in rows], [r["hi"] for r in rows], color=GRID, step="mid",
                             lw=0, label="threshold target band", zorder=0)
        bot.plot(steps, [r["p_trained"] for r in rows], color=col, lw=1.6, label="success rate p")
        bot.plot(steps, [r["informative_trained"] for r in rows], color=INK, lw=1, ls="--",
                 label="groups with nonzero gradient")
        bot.set_ylim(0, 1)
        bot.grid(axis="y", color=GRID, lw=0.5)
        bot.set_xlabel("training step", fontsize=8, color=INK)
        bot.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    handles = {}
    for ax in axes.flat:                     # one shared legend, outside the plots
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(4, len(handles)), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.06 if mesh is not None else -0.1))
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
        cb.set_label("measured p (tracking set)", fontsize=7, color=INK)
        cb.ax.tick_params(labelsize=6, colors=INK2)
    else:
        fig.tight_layout()
    fig.savefig(out, facecolor=SURF, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dir | label=sampler_log.jsonl[,train.log]")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "sampler_levels.png")
    a = ap.parse_args()
    runs = []
    for item in a.runs:
        # A run dir without a merged checkpoint is still training: its log is partial (or empty right after
        # launch) and must not enter the shared figure. Skipping it also keeps one fresh run from crashing
        # every other run's analysis job.
        if Path(item).is_dir() and not any((Path(item) / "hf").glob("step*")):
            print(f"skip {Path(item).name}: still training (no hf/step*)")
            continue
        label, log, track = resolve(item)
        loaded = load_sampler_log(log)
        if not loaded["rows"]:
            print(f"skip {label}: no steps logged")
            continue
        runs.append((label, loaded, load_tracking(track)))
    if not runs:
        raise SystemExit("no finished runs to plot")
    plot(runs, a.out)
    with a.out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "method", "step", "probe", "mean_level", "levels", "p_trained", "informative_trained"])
        for label, log, _ in runs:
            for r in log["rows"]:
                w.writerow([label, log["method"], r["step"], int(r["probe"]), f"{r['mean_level']:.3f}",
                            " ".join(f"{L}:{s:.3f}" for L, s in sorted(r["share"].items())),
                            f"{r['p_trained']:.4f}", f"{r['informative_trained']:.4f}"])
    for label, log, track in runs:
        rows = log["rows"]
        print(f"{label}: {log['method']}, {len(rows)} steps, level {rows[0]['mean_level']:.1f} -> "
              f"{rows[-1]['mean_level']:.1f}, tracking sweeps: {len(track)}")
    print(f"wrote {a.out} and {a.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
