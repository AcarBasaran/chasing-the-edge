"""THE HEADLINE NUMBER: problems the base model never solved (0/128) that a run solves at least once.

For every evaluated checkpoint and every test set / level:
    base_unsolved   problems with 0 correct out of n for the base model
    newly_solved    ... of those, problems with > 0 correct for the run      (new capability)
    base_solved     problems with > 0 correct for the base model
    lost            ... of those, problems with 0 correct for the run        (forgetting / collapse)
Criterion: the paper's strict eval criterion (c_eval) by default; --criterion c_train | c_answer.
Requires the base and the run to be evaluated on the SAME problems with the SAME n (asserted).

    $EDGE_PY eval/newly_solved.py                         # every CSV found by runs_io.discover()
    $EDGE_PY eval/newly_solved.py --run fixed_band_s0=results/per_problem/fixed_band_s0.csv
Output: results/newly_solved.csv (one row per run x set x level, plus level="all")
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs_io import REPO, TEST_SETS, discover, load_per_problem, parse_label_paths, parse_run_id  # noqa: E402


def newly_solved(base: dict, run: dict, criterion: str = "c_eval", splits=TEST_SETS) -> list[dict]:
    rows = []
    for split in splits:
        if split not in run:
            continue
        b, r = base[split], run[split]
        missing = set(r) ^ set(b)
        assert not missing, f"{split}: base and run cover different problems ({len(missing)} differ)"
        by_level = defaultdict(lambda: dict(n_problems=0, base_unsolved=0, newly_solved=0, base_solved=0, lost=0))
        for uid, rr in r.items():
            br = b[uid]
            assert int(br["n"]) == int(rr["n"]), f"{split}/{uid}: n differs ({br['n']} vs {rr['n']})"
            for key in (int(rr["op"]), "all"):
                d = by_level[key]
                d["n_problems"] += 1
                if int(br[criterion]) == 0:
                    d["base_unsolved"] += 1
                    d["newly_solved"] += int(int(rr[criterion]) > 0)
                else:
                    d["base_solved"] += 1
                    d["lost"] += int(int(rr[criterion]) == 0)
        for level in sorted(by_level, key=lambda k: (k == "all", k if k != "all" else 0)):
            rows.append({"split": split, "level": level, "criterion": criterion, **by_level[level]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="label=per_problem.csv (repeatable)")
    ap.add_argument("--base", type=Path, default=REPO / "results" / "base_per_problem.csv")
    ap.add_argument("--criterion", default="c_eval", choices=["c_eval", "c_train", "c_answer"])
    ap.add_argument("--out", type=Path, default=REPO / "results" / "newly_solved.csv")
    a = ap.parse_args()
    runs = parse_label_paths(a.run) if a.run else [r for r in discover() if r[0] != "base"]
    base = load_per_problem(a.base)
    out = []
    for run_id, path in runs:
        arm, seed = parse_run_id(run_id)
        for row in newly_solved(base, load_per_problem(path), a.criterion):
            out.append({"run_id": run_id, "method": arm, "seed": seed, **row})
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("| run | set | newly solved / base-unsolved | lost / base-solved |")
    print("|---|---|---|---|")
    for r in out:
        if r["level"] == "all":
            print(f"| {r['run_id']} | {r['split']} | {r['newly_solved']} / {r['base_unsolved']} | "
                  f"{r['lost']} / {r['base_solved']} |")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
