"""Gate 3 sanity check: the released rl/op11-14_uniform checkpoint vs the base model vs paper Fig. 3.

LOOSE CHECK ONLY. Our baseline is our own fixed-band run with the strict process reward; the
released checkpoint was most likely trained with the ANSWER-ONLY reward (docs/gate3.md), so we
score it under every criterion and report all of them:
  eval    paper's strict criterion (score_dependency_graphs.py `perfect`; extra nodes allowed)
  train   strict training reward (extra nodes also zero it) -- what OUR baseline optimises
  answer  answer-only exact match -- what the released checkpoint (probably) optimised
Test sets: data/testsets/{easy (25/level), medium (200/level), hard (34/level)}, T=0.7, n=128.

    $EDGE_PY eval/compare_released_rl.py
Inputs : results/base_per_problem.csv, results/released_rl_op11-14_per_problem.csv
Outputs: results/gate3_released_rl_comparison.csv (+ markdown table on stdout)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from passk import mean_pass_at_k  # noqa: E402

REPO = HERE.parent
KS = (1, 8, 128)
SETS = {"test_easy": "id_2_10", "test_medium": "edge_11_14", "test_hard": "hard_15_20"}
CRIT = {"eval": "c_eval", "train": "c_train", "answer": "c_answer"}
# Paper Fig. 3 (test split, strict eval criterion), read off the figure, +-2pp.
PAPER = {
    "base": {"id_2_10": {1: 82.5, 8: 96.5, 128: 99.6}, "edge_11_14": {1: 17.0, 8: 31.0, 128: 43.5},
             "hard_15_20": {1: 0.0, 8: 0.0, 128: 0.0}},
    "rl_op11-14": {"id_2_10": {1: 87.0, 8: 96.3, 128: 99.0}, "edge_11_14": {1: 51.0, 8: 71.0, 128: 85.5},
                   "hard_15_20": {1: 9.5, 8: 16.3, 128: 26.0}},
}


def load(path: Path):
    out = defaultdict(dict)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["split"] in SETS:
                out[r["split"]][r["uid"]] = r
    return out


def main() -> None:
    base = load(REPO / "results" / "base_per_problem.csv")
    rel = load(REPO / "results" / "released_rl_op11-14_per_problem.csv")
    rows = []
    for split, band in SETS.items():
        assert set(base[split]) == set(rel[split]), f"{split}: problem sets differ"
        for model, data, paper_key in (("base", base, "base"), ("released_rl", rel, "rl_op11-14")):
            for crit, col in CRIT.items():
                counts = [(int(r["n"]), int(r[col])) for r in data[split].values()]
                rec = {"set": band, "model": model, "criterion": crit, "n_problems": len(counts)}
                for k in KS:
                    rec[f"pass@{k}"] = 100 * mean_pass_at_k(counts, k)
                    rec[f"paper_pass@{k}"] = PAPER[paper_key][band][k] if crit == "eval" else ""
                if model == "released_rl":   # headline-style count: unsolved by base (0/128), solved after
                    rec["newly_solved"] = sum(int(base[split][u][col]) == 0 and int(r[col]) > 0
                                              for u, r in data[split].items())
                    rec["base_unsolved"] = sum(int(base[split][u][col]) == 0 for u in data[split])
                rows.append(rec)
    fields = ["set", "model", "criterion", "n_problems"] + [f"pass@{k}" for k in KS] + \
             [f"paper_pass@{k}" for k in KS] + ["newly_solved", "base_unsolved"]
    out = REPO / "results" / "gate3_released_rl_comparison.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("| set | model | criterion | pass@1 | pass@8 | pass@128 | paper (strict) p@1/p@8/p@128 | newly solved / base-unsolved |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        paper = "/".join(f"{r[f'paper_pass@{k}']}" for k in KS) if r["criterion"] == "eval" else ""
        ns = f"{r['newly_solved']} / {r['base_unsolved']}" if "newly_solved" in r else ""
        print(f"| {r['set']} | {r['model']} | {r['criterion']} | {r['pass@1']:.1f} | {r['pass@8']:.1f} | "
              f"{r['pass@128']:.1f} | {paper} | {ns} |")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
