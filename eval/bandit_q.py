"""Why the bandit never trains beyond op14: its arm values Q_L over training, and who sampled op15+.

The bandit's reward is the improvement rate r_L = p_L(t) - p_L(t-m) between probes (samplers/core.py::Bandit).
Hypothesis: an arm at p = 0 earns r = 0, and the arm being trained earns the largest r because training on it
is what raises its p. Argmax then sticks to the arm where learning is already happening (rich get richer), and
the levels with the most headroom are sampled only by epsilon-exploration.

For each bandit run this reports, from its sampler_log.jsonl:
  * Q_L after every probe (the value the greedy choice uses until the next probe),
  * the greedy level after every probe,
  * per training phase: share of prompts at op15+, and whether any of it came from the greedy choice.
    With epsilon = 0.1 over 19 levels, epsilon alone puts 0.1 * 6/19 = 3.16% of prompts at op15-20.

    $EDGE_PY eval/bandit_q.py [run dirs...]        (default: $EDGE_SCRATCH/outputs/runs/bandit*_s[0-9]*)
Writes results/bandit_q.png, results/bandit_q.csv (Q per probe) and results/bandit_phases.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HI = 15                                     # "beyond the band": op15 and up
HIGHLIGHT = {12: "#6da7ec", 13: "#2a78d6", 14: "#104281",       # blue ramp: where the greedy choice went
             15: "#f4a37f", 16: "#eb6834", 17: "#9a3b12"}       # orange ramp: headroom it never chose
GREY = "#b5b4ad"
INK, INK2 = "#0b0b0b", "#52514e"


def load(run_dir: Path) -> dict:
    rows = [json.loads(l) for l in (run_dir / "sampler_log.jsonl").open()]
    init = next(r for r in rows if r.get("event") == "init")
    steps = [r for r in rows if r.get("event") == "step"]
    levels = [int(l) for l in init["levels"]]
    return {"name": run_dir.name, "m": int(init["probe_every"]), "eps": float(init["kwargs"].get("epsilon", 0.1)),
            "levels": levels, "total": int(init["total_steps"]), "steps": steps}


def probes(run: dict) -> list[dict]:
    out = []
    for r in run["steps"]:
        if r["probe"]:
            Q = {int(k): float(v) for k, v in r["state"]["Q"].items()}
            out.append({"step": r["step"], "Q": Q, "greedy": int(r["state"]["greedy"]),
                        "p": {int(k): float(v) for k, v in r["p"].items()},
                        "n": {int(k): int(v) for k, v in r["n"].items()}})
    return out


def phases(run: dict) -> list[dict]:
    """Training steps split at the first probe that can update Q (the second probe, step m)."""
    m = run["m"]
    eps_share = run["eps"] * sum(1 for l in run["levels"] if l >= HI) / len(run["levels"])
    groups = {"probe steps": lambda r: r["probe"],
              f"steps 1-{m - 1}: all Q = 0, ties -> uniform": lambda r: not r["probe"] and r["step"] < m,
              f"steps {m + 1}+: Q learned": lambda r: not r["probe"] and r["step"] > m}
    out = []
    for name, sel in groups.items():
        rs = [r for r in run["steps"] if sel(r)]
        if not rs:
            continue
        n = sum(sum(r["chosen"].values()) for r in rs)
        hi = sum(v for r in rs for k, v in r["chosen"].items() if int(k) >= HI)
        greedy_hi = sum(1 for r in rs if int(r["state"]["greedy"]) >= HI)
        inf_hi = sum(r["informative_frac"].get(k, 0.0) * v for r in rs for k, v in r["chosen"].items()
                     if int(k) >= HI)
        out.append({"run": run["name"], "phase": name, "steps": len(rs), "prompts": n, "prompts_op15plus": hi,
                    "share_op15plus": hi / n, "epsilon_only_share": eps_share,
                    "steps_greedy_op15plus": greedy_hi, "informative_groups_op15plus": round(inf_hi)})
    return out


def plot(runs: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    k = len(runs)
    fig, ax = plt.subplots(2, k, figsize=(4.2 * k + 0.6, 6.6), sharey="row", squeeze=False,
                           gridspec_kw={"height_ratios": [1.35, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for j, run in enumerate(runs):
        pr = probes(run)
        T = run["total"]
        a = ax[0][j]
        a.set_facecolor("#fcfcfb")
        a.axhline(0, color=INK2, lw=0.8)
        ends = []
        for L in sorted(run["levels"], key=lambda l: l in HIGHLIGHT):      # grey first, highlights on top
            xs, ys = [], []
            for i, p in enumerate(pr):
                x1 = pr[i + 1]["step"] if i + 1 < len(pr) else T
                xs += [p["step"], x1]
                ys += [p["Q"][L], p["Q"][L]]
            c = HIGHLIGHT.get(L, GREY)
            a.plot(xs, ys, color=c, lw=2 if L in HIGHLIGHT else 1, alpha=1 if L in HIGHLIGHT else 0.7,
                   solid_capstyle="round", zorder=3 if L in HIGHLIGHT else 2)
            if L in HIGHLIGHT:
                ends.append([ys[-1], f"op{L}", c])
        ends.sort()
        for i in range(1, len(ends)):                                      # de-collide end labels
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.012)
        for y, lab, c in ends:
            a.text(T + 1.5, y, lab, va="center", fontsize=8, color=INK)
        for p in pr[1:]:
            a.annotate(f"greedy\nop{p['greedy']}", (p["step"], max(p["Q"].values())), xytext=(3, 6),
                       textcoords="offset points", fontsize=7, color=INK2)
        a.axvspan(0, run["m"], color="#eeeee9", zorder=0)
        a.text(run["m"] / 2, 0.97, "all Q = 0\n(uniform)", transform=a.get_xaxis_transform(), ha="center",
               va="top", fontsize=7, color=INK2)
        a.set_xlim(0, T + 9)
        a.set_title(f"{run['name']}  (m = {run['m']})", fontsize=10, color=INK, loc="left")
        if j == 0:
            a.set_ylabel("arm value Q_L  (EMA of p_L(t) - p_L(t-m))")
        a.grid(axis="y", color="#e6e6e1", lw=0.6)

        b = ax[1][j]
        b.set_facecolor("#fcfcfb")
        tr = [r for r in run["steps"] if not r["probe"]]
        st = [r["step"] for r in tr]
        tot = [sum(r["chosen"].values()) for r in tr]
        g = [r["chosen"].get(str(r["state"]["greedy"]), 0) / n for r, n in zip(tr, tot)]
        hi = [sum(v for kk, v in r["chosen"].items() if int(kk) >= HI) / n for r, n in zip(tr, tot)]
        eps_share = run["eps"] * sum(1 for l in run["levels"] if l >= HI) / len(run["levels"])
        b.plot(st, g, color=HIGHLIGHT[13], lw=2, label="greedy level")
        b.plot(st, hi, color=HIGHLIGHT[16], lw=2, label="op15-20")
        b.axhline(eps_share, color=INK2, lw=1, ls=(0, (4, 3)))
        b.text(T, eps_share + 0.03, f"ε alone: {eps_share:.1%}", ha="right", fontsize=7.5, color=INK2)
        b.axvspan(0, run["m"], color="#eeeee9", zorder=0)
        b.set_xlim(0, T + 9)
        b.set_ylim(0, 1.02)
        b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        b.set_xlabel("training step (probe steps omitted)")
        if j == 0:
            b.set_ylabel("share of the training batch")
            b.legend(frameon=False, fontsize=8, loc="center right")
        b.grid(axis="y", color="#e6e6e1", lw=0.6)
    fig.suptitle("Bandit (reward = improvement rate): arm values and what the batch was drawn from",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "bandit_q.png")
    a = ap.parse_args()
    dirs = a.runs or sorted(Path(os.environ["EDGE_SCRATCH"], "outputs", "runs").glob("bandit*_s[0-9]*"))
    runs = [load(d) for d in dirs if (d / "sampler_log.jsonl").exists()]
    runs = [r for r in runs if len(probes(r)) >= 2]
    if not runs:
        raise SystemExit("no bandit run with two probes yet")
    with a.out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "m", "probe_step", "greedy", "level", "Q", "p_probe", "n_probe"])
        for r in runs:
            for p in probes(r):
                for L in r["levels"]:
                    w.writerow([r["name"], r["m"], p["step"], p["greedy"], L, f"{p['Q'][L]:.4f}",
                                f"{p['p'].get(L, float('nan')):.4f}", p["n"].get(L, 0)])
    ph = [x for r in runs for x in phases(r)]
    pth = a.out.with_name("bandit_phases.csv")
    with pth.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ph[0]))
        w.writeheader()
        w.writerows(ph)
    plot(runs, a.out)
    print("| run | phase | steps | op15+ share of prompts | ε alone | steps with greedy at op15+ | informative op15+ groups |")
    print("|---|---|---|---|---|---|---|")
    for x in ph:
        print(f"| {x['run']} | {x['phase']} | {x['steps']} | {x['share_op15plus']:.1%} | {x['epsilon_only_share']:.1%}"
              f" | {x['steps_greedy_op15plus']} | {x['informative_groups_op15plus']:,} |")
    print()
    for r in runs:
        for p in probes(r)[1:]:
            top = sorted(p["Q"].items(), key=lambda kv: -kv[1])[:3]
            print(f"{r['name']} probe {p['step']}: greedy op{p['greedy']}; top Q "
                  + ", ".join(f"op{L} {q:+.3f}" for L, q in top)
                  + "; op15-17 Q " + " ".join(f"{p['Q'][L]:+.3f}" for L in (15, 16, 17))
                  + "; op15-17 p " + " ".join(f"{p['p'].get(L, 0):.2f}" for L in (15, 16, 17)))
    print(f"wrote {a.out}, {a.out.with_suffix('.csv')}, {pth}")


if __name__ == "__main__":
    main()
