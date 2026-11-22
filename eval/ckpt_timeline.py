"""When did tau=0.1 pull ahead of the op 11-17 control? Hard-set results at steps 25, 50, 75 (and 100).

Inputs: results/per_problem_ckpt/<run>_step{25,50,75}.csv (slurm/ckpt_eval.sh) and results/per_problem/<run>.csv
(step 100). Base model solves none of the 204 hard problems, so newly solved = solved at least once in 128.
Per step: pass@128 and newly solved (whole hard set, op 15-17, op 18-20), mean +- SD over 3 seeds, and the same
pooled paired test as docs/results_seeds.md section 2 (problem = unit; sign test + 95% bootstrap CI).

    $EDGE_PY eval/ckpt_timeline.py
Writes results/ckpt_timeline.csv and results/ckpt_timeline_paired.csv.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARMS = {"threshold_tau1": "tau=0.1", "uniform_11_17": "control op11-17"}
SEEDS = (0, 1, 2)
STEPS = (25, 50, 75, 100)
POOLS = {"hard": range(15, 21), "op15-17": range(15, 18), "op18-20": range(18, 21)}


def load(path: Path) -> dict[str, tuple[int, bool]]:
    out = {}
    for r in csv.DictReader(path.open()):
        if r["split"] == "test_hard":
            out[r["uid"]] = (int(r["op"]), int(r["c_eval"]) > 0)
    return out


def path_for(arm: str, seed: int, step: int) -> Path:
    if step == 100:
        return REPO / "results" / "per_problem" / f"{arm}_s{seed}.csv"
    return REPO / "results" / "per_problem_ckpt" / f"{arm}_s{seed}_step{step}.csv"


def sign_test(pos: int, neg: int) -> float:
    n, k = pos + neg, min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    data = {(a, s, t): load(path_for(a, s, t)) for a in ARMS for s in SEEDS for t in STEPS if path_for(a, s, t).exists()}
    rows, paired = [], []
    rng = np.random.default_rng(0)
    for t in STEPS:
        for pool, levels in POOLS.items():
            per_arm = {}
            for a in ARMS:
                vals = []
                for s in SEEDS:
                    d = data.get((a, s, t))
                    if d is None:
                        continue
                    vals.append(sum(ok for op, ok in d.values() if op in levels))
                n = sum(1 for op, _ in next(iter(data.values())).values() if op in levels)
                per_arm[a] = vals
                if vals:
                    rows.append({"step": t, "pool": pool, "arm": a, "n_seeds": len(vals), "n_problems": n,
                                 "newly_solved_mean": round(float(np.mean(vals)), 2),
                                 "newly_solved_sd": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else "",
                                 "pass@128_mean": round(100 * float(np.mean(vals)) / n, 2),
                                 "per_seed": " ".join(str(v) for v in vals)})
            a, b = list(ARMS)
            if all((x, s, t) in data for x in ARMS for s in SEEDS):
                uids = sorted(u for u, (op, _) in data[(a, 0, t)].items() if op in levels)
                d = np.array([np.mean([data[(a, s, t)][u][1] for s in SEEDS]) - np.mean([data[(b, s, t)][u][1] for s in SEEDS])
                              for u in uids])
                boots = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(10_000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                pos, neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                paired.append({"step": t, "pool": pool, "diff_per_seed": round(float(d.sum()), 2),
                               "ci95_lo": round(float(lo), 2), "ci95_hi": round(float(hi), 2),
                               "tau1_better": pos, "control_better": neg, "sign_test_p": round(sign_test(pos, neg), 4)})
    for name, data_rows in [("ckpt_timeline.csv", rows), ("ckpt_timeline_paired.csv", paired)]:
        with (REPO / "results" / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data_rows[0]))
            w.writeheader()
            w.writerows(data_rows)

    print("Newly solved hard problems (mean ± SD over seeds; per seed), tau=0.1 vs control")
    for pool in POOLS:
        print(f"\n{pool}")
        for t in STEPS:
            line = [f"  step {t:3d}"]
            for a in ARMS:
                r = next((r for r in rows if r["step"] == t and r["pool"] == pool and r["arm"] == a), None)
                line.append(f"{ARMS[a]}: {r['newly_solved_mean']} ± {r['newly_solved_sd']} ({r['per_seed']})" if r else f"{ARMS[a]}: -")
            p = next((p for p in paired if p["step"] == t and p["pool"] == pool), None)
            if p:
                line.append(f"diff {p['diff_per_seed']:+} [{p['ci95_lo']:+}, {p['ci95_hi']:+}], "
                            f"{p['tau1_better']}/{p['control_better']} problems, p={p['sign_test_p']}")
            print("   ".join(line))
    print("\nwrote results/ckpt_timeline.csv, results/ckpt_timeline_paired.csv")


if __name__ == "__main__":
    main()
