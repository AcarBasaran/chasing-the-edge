"""Success rate split by problem MODE (normalforward vs forwardreverse) per level and arm.

Each op level mixes two problem types from the GSM-Infinite generator (about half each):
  normalforward  : forward computation (generator drawAll)
  forwardreverse : an unknown x, other quantities written in terms of x, a total given; solve for x
                   (generator drawAllEquan; gold solutions start "we use equations to solve it")
This script reads the per-problem eval CSVs (T=0.7, 128 samples) and the test-set JSONL files
(for the `mode` field) and writes mean strict p, newly-solved share and answer-only accuracy per
(arm, op, mode).

    python eval/mode_split.py --out results/mode_split.csv
    python eval/mode_split.py --per-problem-dir results/per_problem_fixed \
        --base results/per_problem_fixed/base.csv --out results/mode_split_fixed.csv   # fixed parser
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["base", "fixed_band", "uniform", "uniform_11_17", "threshold_tau1", "threshold_tau3",
        "threshold_tau5", "bandit"]


def load_modes() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(ROOT / "data/testsets/*/*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rows.append({"uid": r["uid"], "op": int(r["op"]), "mode": r["mode"]})
    return pd.DataFrame(rows).drop_duplicates("uid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/mode_split.csv")
    ap.add_argument("--per-problem-dir", type=Path, default=ROOT / "results/per_problem")
    ap.add_argument("--base", type=Path, default=ROOT / "results/base_per_problem.csv")
    args = ap.parse_args()
    modes = load_modes()
    out = []
    for arm in ARMS:
        files = ([args.base] if arm == "base"
                 else sorted(glob.glob(str(args.per_problem_dir / f"{arm}_s*.csv"))))
        files = [f for f in files if Path(f).exists()]
        if not files:
            continue
        d = pd.concat([pd.read_csv(f).assign(run=Path(f).stem) for f in files])
        d = d[d.split.isin(["test_easy", "test_medium", "test_hard"])].merge(modes, on=["uid", "op"])
        d["p"] = d.c_eval / d.n
        d["p_train"] = d.c_train / d.n
        d["p_answer"] = d.c_answer / d.n
        d["solved"] = d.c_eval > 0
        g = (d.groupby(["op", "mode"])
               .agg(problems=("uid", "nunique"), runs=("run", "nunique"), p=("p", "mean"),
                    p_train=("p_train", "mean"), p_answer=("p_answer", "mean"), solved=("solved", "mean"))
               .reset_index())
        g.insert(0, "arm", arm)
        out.append(g)
    res = pd.concat(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(args.out, index=False)
    wide = res.pivot_table(index="op", columns=["mode", "arm"], values="p").round(3)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(wide)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
