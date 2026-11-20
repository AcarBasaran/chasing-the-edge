"""Aggregate every arm over its seeds: test-set metrics, the op18-20 breakdown, and paired tests.

Run it whenever new seeds land. It finds every evaluated run (results/per_problem/<arm>_s<seed>.csv, see
eval/runs_io.py) and groups runs by arm. With one seed per arm, spread columns are empty and only the
within-seed tests exist; say so wherever a number is quoted.

Tables (strict eval criterion; pass@k unbiased, mean over problems; newly solved = base 0/128 -> run > 0):
  results/seeds_by_set.csv      arm x test set x metric: n_seeds, mean, sd, min, max, per-seed values
                                metrics: pass@1, pass@128, newly_solved, lost (base-solved -> 0)
  results/seeds_hard_levels.csv arm x hard level (15..20, pooled 15-17 / 18-20): pass@1, pass@128, newly_solved
                                (mean, sd, min, max over seeds) + training exposure at those levels
                                (groups and groups with nonzero gradient, from sampler_log.jsonl, mean over seeds)
  results/seeds_gradient_budget.csv  per arm: share of ALL its GRPO groups over the run (probes included, so
                                every arm has the same 102,400) that had nonzero reward variance, i.e. nonzero
                                gradient; and informative groups by level band (op2-10, 11-14, 15-17, 18-20).
                                Mean, sd, min, max over seeds.
  results/seeds_paired_tests.csv every pair of arms, on hard op18-20 and on the whole hard set:
      within_seed   for each seed index both arms have: problems newly solved only by A / only by B, exact
                    McNemar p. THESE ARE WITHIN ONE SEED: they do not include seed-to-seed variance.
      pooled        the problem is the unit. Per problem, the share of A's seeds that newly solve it minus the
                    share of B's seeds. Exact two-sided sign test over problems with a nonzero difference,
                    and a 95% bootstrap CI (resampling problems, 10,000 draws) for the difference in mean
                    newly-solved count per seed. Seeds count through the per-problem shares, so a problem
                    solved by A in 3/3 seeds and B in 0/3 weighs more than one solved in 1/3 vs 0/3.

    $EDGE_PY eval/aggregate_seeds.py [--runs-root $EDGE_SCRATCH/outputs/runs] [--min-seeds 1]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from passk import pass_at_k  # noqa: E402
from runs_io import REPO, TEST_SETS, discover, load_per_problem, method_sort_key, parse_run_id  # noqa: E402
from untrained_levels import exposure  # noqa: E402

BANDS = {"op2-10": range(2, 11), "op11-14": range(11, 15), "op15-17": range(15, 18), "op18-20": range(18, 21)}
HARD_KEYS = [(str(L), (L,)) for L in range(15, 21)] + [("15-17", (15, 16, 17)), ("18-20", (18, 19, 20))]
SCOPES = {"hard op18-20": ("test_hard", (18, 19, 20)), "hard op15-20": ("test_hard", tuple(range(15, 21)))}


def spread(values: list[float]) -> dict:
    return {"n_seeds": len(values), "mean": round(st.mean(values), 4),
            "sd": round(st.stdev(values), 4) if len(values) > 1 else "",
            "min": round(min(values), 4), "max": round(max(values), 4)}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def sign_test(pos: int, neg: int) -> float:
    return mcnemar_exact(pos, neg)                 # same exact binomial(1/2) two-sided test


def run_metrics(run: dict, base: dict) -> tuple[dict, dict, dict]:
    """(by_set[split][metric], hard_levels[key][metric], newly[split][uid] -> bool)."""
    by_set, newly = {}, {}
    for split in TEST_SETS:
        r, b = run.get(split), base.get(split)
        if not r:
            continue
        assert set(r) == set(b), f"{split}: run and base cover different problems"
        newly[split] = {u: int(b[u]["c_eval"]) == 0 and int(x["c_eval"]) > 0 for u, x in r.items()}
        by_set[split] = {
            "pass@1": 100 * st.mean(pass_at_k(int(x["n"]), int(x["c_eval"]), 1) for x in r.values()),
            "pass@128": 100 * st.mean(pass_at_k(int(x["n"]), int(x["c_eval"]), 128) for x in r.values()),
            "newly_solved": sum(newly[split].values()),
            "lost": sum(int(b[u]["c_eval"]) > 0 and int(x["c_eval"]) == 0 for u, x in r.items()),
        }
    levels = {}
    hard = run.get("test_hard", {})
    for key, ops in HARD_KEYS:
        probs = [x for x in hard.values() if int(x["op"]) in ops]
        if probs:
            levels[key] = {
                "pass@1": 100 * st.mean(pass_at_k(int(x["n"]), int(x["c_eval"]), 1) for x in probs),
                "pass@128": 100 * st.mean(pass_at_k(int(x["n"]), int(x["c_eval"]), 128) for x in probs),
                "newly_solved": sum(newly["test_hard"][x["uid"]] for x in probs)}
    return by_set, levels, newly


def paired(arm_a: str, arm_b: str, runs: dict, uids: list[str], split: str, rng: np.random.Generator) -> list[dict]:
    """Within-seed McNemar per shared seed, and the pooled problem-level test."""
    out = []
    sa, sb = runs[arm_a], runs[arm_b]
    for seed in sorted(set(sa) & set(sb)):
        na, nb = sa[seed]["newly"][split], sb[seed]["newly"][split]
        b = sum(na[u] and not nb[u] for u in uids)
        c = sum(nb[u] and not na[u] for u in uids)
        out.append({"kind": "within_seed", "seed": seed, "only_a": b, "only_b": c,
                    "p_value": round(mcnemar_exact(b, c), 4), "n_seeds_a": 1, "n_seeds_b": 1,
                    "a_better_problems": "", "b_better_problems": "", "diff_mean_newly": b - c,
                    "diff_ci95_lo": "", "diff_ci95_hi": ""})
    ra = np.array([[sa[s]["newly"][split][u] for s in sorted(sa)] for u in uids], float).mean(axis=1)
    rb = np.array([[sb[s]["newly"][split][u] for s in sorted(sb)] for u in uids], float).mean(axis=1)
    d = ra - rb
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    boot = d[rng.integers(0, len(d), size=(10_000, len(d)))].sum(axis=1)
    out.append({"kind": "pooled", "seed": "all", "only_a": "", "only_b": "", "p_value": round(sign_test(pos, neg), 4),
                "n_seeds_a": len(sa), "n_seeds_b": len(sb), "a_better_problems": pos, "b_better_problems": neg,
                "diff_mean_newly": round(float(d.sum()), 3),
                "diff_ci95_lo": round(float(np.percentile(boot, 2.5)), 3),
                "diff_ci95_hi": round(float(np.percentile(boot, 97.5)), 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path(os.environ.get("EDGE_SCRATCH", "/scratch")) / "outputs" / "runs")
    ap.add_argument("--per-problem-dir", type=Path, default=None, help="default: results/per_problem")
    ap.add_argument("--base", type=Path, default=REPO / "results" / "base_per_problem.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--min-seeds", type=int, default=1, help="only arms with at least this many seeds")
    a = ap.parse_args()

    base = load_per_problem(a.base)
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for rid, path in discover(a.per_problem_dir, include_released=True):
        if rid == "base":
            continue
        arm, seed = parse_run_id(rid)
        by_set, levels, newly = run_metrics(load_per_problem(path), base)
        groups, informative = exposure(a.runs_root / rid / "sampler_log.jsonl")
        runs[arm][seed] = {"run_id": rid, "by_set": by_set, "levels": levels, "newly": newly,
                           "groups": groups, "informative": informative}
    arms = sorted((m for m in runs if len(runs[m]) >= a.min_seeds), key=method_sort_key)

    set_rows = []
    for arm in arms:
        for split in TEST_SETS:
            for metric in ("pass@1", "pass@128", "newly_solved", "lost"):
                vals = {s: r["by_set"][split][metric] for s, r in sorted(runs[arm].items()) if split in r["by_set"]}
                if vals:
                    set_rows.append({"method": arm, "split": split, "metric": metric, **spread(list(vals.values())),
                                     "per_seed": " ".join(f"s{s}:{v:.4g}" for s, v in vals.items())})
    level_rows = []
    for arm in arms:
        for key, ops in HARD_KEYS:
            if not any(key in r["levels"] for r in runs[arm].values()):
                continue
            row = {"method": arm, "level": key}
            for metric in ("pass@1", "pass@128", "newly_solved"):
                vals = [r["levels"][key][metric] for r in runs[arm].values() if key in r["levels"]]
                sp = spread(vals)
                row["n_seeds"] = sp["n_seeds"]
                row.update({f"{metric}_{k}": v for k, v in sp.items() if k != "n_seeds"})
            row["train_groups_mean"] = round(st.mean(sum(r["groups"][L] for L in ops) for r in runs[arm].values()))
            row["informative_groups_mean"] = round(st.mean(sum(r["informative"][L] for L in ops)
                                                           for r in runs[arm].values()))
            level_rows.append(row)

    budget_rows = []
    for arm in arms:
        seeds_ = [r for r in runs[arm].values() if sum(r["groups"].values())]
        if not seeds_:
            continue                                   # no sampler log (e.g. the released checkpoint)
        row = {"method": arm, "n_seeds": len(seeds_),
               "groups_per_run": round(st.mean(sum(r["groups"].values()) for r in seeds_))}
        frac = [sum(r["informative"].values()) / sum(r["groups"].values()) for r in seeds_]
        row.update({f"informative_frac_{k}": v for k, v in spread(frac).items() if k != "n_seeds"})
        for band, levels in BANDS.items():
            vals = [sum(r["informative"][L] for L in levels) for r in seeds_]
            row[f"informative_groups_{band}_mean"] = round(st.mean(vals))
            row[f"groups_{band}_mean"] = round(st.mean(sum(r["groups"][L] for L in levels) for r in seeds_))
        budget_rows.append(row)

    rng = np.random.default_rng(20261120)
    test_rows = []
    for scope, (split, ops) in SCOPES.items():
        uids = sorted(u for u, x in base[split].items() if int(x["op"]) in ops)
        if not uids:
            continue
        for i, x in enumerate(arms):
            for y in arms[i + 1:]:
                for r in paired(x, y, runs, uids, split, rng):
                    test_rows.append({"scope": scope, "arm_a": x, "arm_b": y, "n_problems": len(uids), **r})

    for name, rows in (("seeds_by_set.csv", set_rows), ("seeds_hard_levels.csv", level_rows),
                       ("seeds_gradient_budget.csv", budget_rows), ("seeds_paired_tests.csv", test_rows)):
        if not rows:
            continue
        with (a.out_dir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    seeds = {m: len(runs[m]) for m in arms}
    print("seeds per arm: " + ", ".join(f"{m} {n}" for m, n in seeds.items()))
    if max(seeds.values()) == 1:
        print("ONE SEED PER ARM: no spread; paired tests are within seed 0 only.")
    print("\n| arm | seeds | hard pass@1 | hard pass@128 | hard newly solved | op18-20 newly solved (of 102) |")
    print("|---|---|---|---|---|---|")

    def fmt(r, k=""):
        return f"{r[f'{k}mean']:.1f}" + (f" ± {r[f'{k}sd']:.1f}" if r[f"{k}sd"] != "" else "")
    for m in arms:
        get = {(r["split"], r["metric"]): r for r in set_rows if r["method"] == m}
        lv = next((r for r in level_rows if r["method"] == m and r["level"] == "18-20"), None)
        print(f"| {m} | {seeds[m]} | {fmt(get['test_hard', 'pass@1'])} | {fmt(get['test_hard', 'pass@128'])} | "
              f"{fmt(get['test_hard', 'newly_solved'])} | {fmt(lv, 'newly_solved_') if lv else '–'} |")
    print("\nGradient budget: share of all GRPO groups with nonzero gradient, pooled over training; "
          "informative groups by level band (mean over seeds)")
    print("| arm | seeds | informative share | op2-10 | op11-14 | op15-17 | op18-20 |")
    print("|---|---|---|---|---|---|---|")
    for r in budget_rows:
        sd = f" ± {100 * r['informative_frac_sd']:.1f}" if r["informative_frac_sd"] != "" else ""
        print(f"| {r['method']} | {r['n_seeds']} | {100 * r['informative_frac_mean']:.1f}%{sd} | "
              + " | ".join(f"{r[f'informative_groups_{b}_mean']:,}" for b in BANDS) + " |")
    print("\nPaired tests, hard op18-20 (within_seed = one seed only; pooled = problem-level across seeds):")
    for r in test_rows:
        if r["scope"] == "hard op18-20" and {"threshold_tau1", "uniform", "fixed_band"} >= {r["arm_a"], r["arm_b"]}:
            if r["kind"] == "within_seed":
                print(f"  {r['arm_a']} vs {r['arm_b']}, within seed {r['seed']}: only A {r['only_a']}, "
                      f"only B {r['only_b']}, McNemar p={r['p_value']}")
            else:
                print(f"  {r['arm_a']} vs {r['arm_b']}, pooled over {r['n_seeds_a']}/{r['n_seeds_b']} seeds: "
                      f"A better on {r['a_better_problems']} problems, B on {r['b_better_problems']}, sign test "
                      f"p={r['p_value']}; diff in newly solved per seed {r['diff_mean_newly']} "
                      f"[95% CI {r['diff_ci95_lo']}, {r['diff_ci95_hi']}]")
    print(f"\nwrote {', '.join(str(a.out_dir / n) for n in ('seeds_by_set.csv', 'seeds_hard_levels.csv', 'seeds_paired_tests.csv'))}")


if __name__ == "__main__":
    main()
