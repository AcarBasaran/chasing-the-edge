"""Is tau=0.1's lower easy/medium pass@1 seed noise or a consistent cost of climbing?

Question (2026-11-21): tau=0.1 easy pass@1 83.9 vs fixed band 85.4. With 3 seeds of tau=0.1 (and 2-3 of the
comparison arms) this reports, for the easy and medium test sets (T = 0.7, eval criterion, n = 128 per problem):

  * pass@1 per arm and seed, overall and per op level, plus the base model;
  * every cross-seed pair (tau=0.1 seed i vs other seed j): how many of them have tau=0.1 below;
  * paired by problem: mean over seeds of A minus mean over seeds of B, 95% bootstrap CI over problems
    (10,000 draws), and an exact sign test over problems whose difference is nonzero;
  * where the drop sits: share of the total difference carried by each op level.

    $EDGE_PY eval/easy_regression.py [--a threshold_tau1] [--b fixed_band uniform uniform_11_17 base]
Writes results/easy_regression.csv (arm, split, level, seed, pass@1) and results/easy_regression_paired.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("test_easy", "test_medium")


def load(path: Path) -> dict[str, dict[str, tuple[int, float]]]:
    """split -> uid -> (op, pass@1 under the eval criterion)."""
    out: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        if r["split"] in SPLITS:
            out[r["split"]][r["uid"]] = (int(r["op"]), int(r["c_eval"]) / int(r["n"]))
    return out


def arms() -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in sorted((REPO / "results" / "per_problem").glob("*_s[0-9].csv")):
        method, seed = f.stem.rsplit("_s", 1)
        out[method][int(seed)] = load(f)
    out["base"][0] = load(REPO / "results" / "base_per_problem.csv")
    return out


def sign_test(pos: int, neg: int) -> float:
    n, k = pos + neg, min(pos, neg)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="threshold_tau1")
    ap.add_argument("--b", nargs="+", default=["fixed_band", "uniform", "uniform_11_17", "base"])
    a = ap.parse_args()
    data = arms()
    rng = np.random.default_rng(0)

    rows = []
    for arm in [a.a] + a.b:
        for seed, d in sorted(data[arm].items()):
            for split in SPLITS:
                probs = d[split]
                rows.append([arm, split, "all", seed, round(100 * np.mean([p for _, p in probs.values()]), 3)])
                for L in sorted({op for op, _ in probs.values()}):
                    rows.append([arm, split, L, seed,
                                 round(100 * np.mean([p for op, p in probs.values() if op == L]), 3)])
    with (REPO / "results" / "easy_regression.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "split", "level", "seed", "pass@1"])
        w.writerows(rows)

    def overall(arm: str, split: str) -> list[float]:
        return [r[4] for r in rows if r[0] == arm and r[1] == split and r[2] == "all"]

    print("pass@1 (eval criterion, T = 0.7), per seed:")
    for split in SPLITS:
        for arm in [a.a] + a.b:
            v = overall(arm, split)
            sd = f" ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else ""
            print(f"  {split:12s} {arm:16s} {np.mean(v):6.2f}{sd}   seeds: " + ", ".join(f"{x:.2f}" for x in v))

    paired = []
    print("\nPaired by problem: A = mean over A's seeds, B = mean over B's seeds; diff = A - B in pass@1 points")
    for split in SPLITS:
        A = data[a.a]
        uids = sorted(next(iter(A.values()))[split])
        ops = np.array([next(iter(A.values()))[split][u][0] for u in uids])
        pa = np.mean([[s[split][u][1] for u in uids] for s in A.values()], axis=0)
        for b in a.b:
            B = data[b]
            if not B:
                continue
            pb = np.mean([[s[split][u][1] for u in uids] for s in B.values()], axis=0)
            diff = 100 * (pa - pb)
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10_000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            pos, neg = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
            va, vb = overall(a.a, split), overall(b, split)
            below = sum(x < y for x in va for y in vb)
            by_level = {int(L): round(float(diff[ops == L].sum() / len(diff)), 3) for L in sorted(set(ops))}
            row = {"split": split, "a": a.a, "b": b, "n_seeds_a": len(va), "n_seeds_b": len(vb),
                   "n_problems": len(diff), "diff_mean": round(float(diff.mean()), 3),
                   "ci95_lo": round(float(lo), 3), "ci95_hi": round(float(hi), 3),
                   "a_higher_problems": pos, "b_higher_problems": neg, "sign_test_p": round(sign_test(pos, neg), 6),
                   "seed_pairs_a_below": f"{below}/{len(va) * len(vb)}",
                   "contribution_by_level": " ".join(f"op{k}:{v:+.2f}" for k, v in by_level.items())}
            paired.append(row)
            print(f"  {split:12s} {a.a} - {b:14s} {row['diff_mean']:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
                  f"problems A higher {pos}, B higher {neg}, sign p={row['sign_test_p']:.2g}; "
                  f"seed pairs with A below: {row['seed_pairs_a_below']}")
            print(f"  {'':12s}   by level (points of the overall diff): {row['contribution_by_level']}")
    with (REPO / "results" / "easy_regression_paired.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(paired[0]))
        w.writeheader()
        w.writerows(paired)
    print("wrote results/easy_regression.csv, results/easy_regression_paired.csv")


if __name__ == "__main__":
    main()
