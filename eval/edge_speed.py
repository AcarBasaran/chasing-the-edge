"""How fast does the edge of competence move during a run?  (sets the bandit's probe interval m)

Input: a run's train.log. verl's validation on the tracking set (data/track32, 32 problems x
op2-20, n=8, T=1.0) prints  val-core/difficulty-5B/<L>/<var>/mean@8  every trainer.test_freq steps.
For each validation step t we get p_L(t) for every level, and define the edge position for a
target success rate tau as the (linearly interpolated) level where p_L(t) crosses tau going
up in L. Speed = slope of edge position vs step (levels/step), by least squares over the run
and over the first/second half separately.

Bandit m rule: r_L(t) = p_L(t) - p_L(t-m) must see a level change before the edge has moved
past it, so we want the edge to move <= ~1 level per probe interval:  m <= 1 / speed.

    $EDGE_PY eval/edge_speed.py <run_dir>/train.log [--csv out.csv] [--plot out.png]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_verl_log import parse  # noqa: E402

KEY = re.compile(r"^val-(?:core|aux)/difficulty-5B/(\d+)/([^/]+)/mean@(\d+)$")
TAUS = (0.5, 0.3, 0.1)


def level_p(rec: dict) -> dict[int, float]:
    by_var = defaultdict(dict)
    for k, v in rec.items():
        m = KEY.match(k)
        if m:
            by_var[m.group(2)][int(m.group(1))] = v
    for var in ("acc", "score", "reward"):
        if var in by_var:
            return by_var[var]
    return next(iter(by_var.values()), {})


def edge(p: dict[int, float], tau: float) -> float | None:
    """Largest-level crossing of tau, scanning up from the easiest level (p falls with L)."""
    Ls = sorted(p)
    if not Ls or p[Ls[0]] < tau:
        return None
    for a, b in zip(Ls, Ls[1:]):
        if p[a] >= tau > p[b]:
            return a + (p[a] - tau) / (p[a] - p[b]) * (b - a)
    return float(Ls[-1])          # never drops below tau: edge at/after the top level (censored)


def slope(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--csv")
    ap.add_argument("--plot")
    a = ap.parse_args()
    series = [(r["step"], level_p(r)) for r in parse(a.log)]
    series = [(t, p) for t, p in series if p]
    if not series:
        sys.exit("no val-core/difficulty-5B/<L>/... metrics found (was trainer.test_freq set?)")
    rows = []
    for t, p in series:
        rec = {"step": t, **{f"p_op{L}": p[L] for L in sorted(p)}}
        for tau in TAUS:
            rec[f"edge_tau{tau}"] = edge(p, tau)
        rows.append(rec)
    print(f"{len(rows)} validation points, steps {rows[0]['step']}..{rows[-1]['step']}")
    for tau in TAUS:
        pts = [(r["step"], r[f"edge_tau{tau}"]) for r in rows if r[f"edge_tau{tau}"] is not None]
        if len(pts) < 2:
            print(f"tau={tau}: not enough points")
            continue
        xs, ys = zip(*pts)
        half = xs[len(xs) // 2]
        s_all = slope(xs, ys)
        s1 = slope(*zip(*[q for q in pts if q[0] <= half])) if sum(x <= half for x in xs) > 1 else float("nan")
        s2 = slope(*zip(*[q for q in pts if q[0] >= half])) if sum(x >= half for x in xs) > 1 else float("nan")
        m = 1 / s_all if s_all > 0 else float("inf")
        print(f"tau={tau}: edge {ys[0]:.2f} -> {ys[-1]:.2f} over steps {xs[0]}..{xs[-1]}; "
              f"speed {s_all:.4f} lv/step (1st half {s1:.4f}, 2nd half {s2:.4f}); "
              f"1 level per {m:.0f} steps -> bandit m <= {m:.0f}")
    if a.csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", not k.startswith("edge"), k))
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
        for tau, col in zip(TAUS, ("#2a78d6", "#eb6834", "#1baf7a")):
            pts = [(r["step"], r[f"edge_tau{tau}"]) for r in rows if r[f"edge_tau{tau}"] is not None]
            if pts:
                ax.plot(*zip(*pts), color=col, lw=2, marker="o", ms=4, label=f"level where p = {tau}")
        ax.set_xlabel("training step")
        ax.set_ylabel("edge position (level)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(a.plot)


if __name__ == "__main__":
    main()
