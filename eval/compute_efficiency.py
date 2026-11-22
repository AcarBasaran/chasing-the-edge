"""Does the climb buy compute-efficiency? tau=0.1 at step 25 vs the fixed band's FINAL (step-100) result.

1. Hard-set newly solved: tau=0.1 step 25 (3 seeds) and the op 11-17 control step 25 (3 seeds) vs the fixed band
   at step 100 (2 seeds); pooled paired test as in docs/results_seeds.md section 2 (problem = unit: share of A's
   seeds that solve it minus share of B's; exact sign test over nonzero differences; 95% bootstrap CI).
2. Compute: training rollouts (25 vs 100 steps, identical batch) and wall-clock training time, the sum of verl's
   timing_s/step up to that step (timing.csv; one A40 per run, so seconds = GPU-seconds).
3. Where the first 25 steps went: from sampler_log.jsonl, groups with a nonzero gradient in steps 0-24, overall
   and at op 15-17, for tau=0.1 and the control.

    $EDGE_PY eval/compute_efficiency.py [--runs-root $EDGE_SCRATCH/outputs/runs]
Writes results/compute_efficiency.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
POOLS = {"hard": range(15, 21), "op15-17": range(15, 18), "op18-20": range(18, 21)}


def solved(path: Path) -> dict[str, tuple[int, bool]]:
    return {r["uid"]: (int(r["op"]), int(r["c_eval"]) > 0) for r in csv.DictReader(path.open())
            if r["split"] == "test_hard"}


def sign_test(pos: int, neg: int) -> float:
    n, k = pos + neg, min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(A: list[dict], B: list[dict], levels, rng) -> dict:
    uids = sorted(u for u, (op, _) in A[0].items() if op in levels)
    d = np.array([np.mean([a[u][1] for a in A]) - np.mean([b[u][1] for b in B]) for u in uids])
    boots = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(10_000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pos, neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return {"a_mean": round(float(np.mean([sum(a[u][1] for u in uids) for a in A])), 2),
            "a_per_seed": " ".join(str(sum(a[u][1] for u in uids)) for a in A),
            "b_mean": round(float(np.mean([sum(b[u][1] for u in uids) for b in B])), 2),
            "b_per_seed": " ".join(str(sum(b[u][1] for u in uids)) for b in B),
            "diff": round(float(d.sum()), 2), "ci95_lo": round(float(lo), 2), "ci95_hi": round(float(hi), 2),
            "a_better": pos, "b_better": neg, "sign_test_p": round(sign_test(pos, neg), 5)}


def train_seconds(run_dir: Path, upto: int) -> float:
    total = 0.0
    for r in csv.DictReader((run_dir / "timing.csv").open()):
        step = r.get("training/global_step") or r.get("step")
        t = r.get("timing_s/step")
        if step and t and int(float(step)) <= upto:
            total += float(t)
    return total


def early_budget(run_dir: Path, steps: int = 25) -> dict:
    tot = inf = tot_h = inf_h = 0.0
    for line in (run_dir / "sampler_log.jsonl").open():
        r = json.loads(line)
        if r.get("event") != "step" or r["step"] >= steps:
            continue
        for L, c in r["chosen"].items():
            frac = r.get("informative_frac", {}).get(L, 0.0)
            tot += c
            inf += c * frac
            if 15 <= int(L) <= 17:
                tot_h += c
                inf_h += c * frac
    return {"groups": int(tot), "informative": int(round(inf)), "informative_share": round(inf / tot, 3),
            "groups_op15_17": int(tot_h), "informative_op15_17": int(round(inf_h)),
            "informative_share_op15_17": round(inf_h / tot_h, 3) if tot_h else ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path(os.environ.get("EDGE_SCRATCH", "/scratch")) / "outputs" / "runs")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    ck = REPO / "results" / "per_problem_ckpt"
    pp = REPO / "results" / "per_problem"
    tau25 = [solved(ck / f"threshold_tau1_s{s}_step25.csv") for s in range(3)]
    ctl25 = [solved(ck / f"uniform_11_17_s{s}_step25.csv") for s in range(3)]
    fb100 = [solved(pp / f"fixed_band_s{s}.csv") for s in range(2)]

    rows = []
    print("1. Newly solved: A vs fixed band at step 100 (2 seeds), pooled paired test")
    for name, A in [("tau=0.1 @ step 25", tau25), ("control @ step 25", ctl25)]:
        for pool, levels in POOLS.items():
            r = {"comparison": f"{name} vs fixed band @ step 100", "pool": pool, **paired(A, fb100, levels, rng)}
            rows.append(r)
            print(f"  {name:18s} {pool:8s}: {r['a_mean']} ({r['a_per_seed']}) vs {r['b_mean']} ({r['b_per_seed']}); "
                  f"diff {r['diff']:+} [{r['ci95_lo']:+}, {r['ci95_hi']:+}], {r['a_better']}/{r['b_better']} problems, "
                  f"p={r['sign_test_p']}")

    print("\n2. Training time (sum of timing_s/step; one A40 per run)")
    for run, upto in [(f"threshold_tau1_s{s}", 25) for s in range(3)] + [(f"uniform_11_17_s{s}", 25) for s in range(3)] \
            + [(f"fixed_band_s{s}", 100) for s in range(2)] + [(f"threshold_tau1_s{s}", 100) for s in range(3)]:
        sec = train_seconds(a.runs_root / run, upto)
        rows.append({"comparison": "train_time", "pool": f"{run} steps 1-{upto}", "a_mean": round(sec / 3600, 2)})
        print(f"  {run:20s} steps <= {upto:3d}: {sec / 3600:.2f} h")

    print("\n3. First 25 steps: groups with a nonzero gradient")
    for run in [f"threshold_tau1_s{s}" for s in range(3)] + [f"uniform_11_17_s{s}" for s in range(3)]:
        e = early_budget(a.runs_root / run)
        rows.append({"comparison": "early_budget", "pool": run, **{k: v for k, v in e.items()}})
        print(f"  {run:20s} {e}")

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("comparison", "pool"), k))
    with (REPO / "results" / "compute_efficiency.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote results/compute_efficiency.csv")


if __name__ == "__main__":
    main()
