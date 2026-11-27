"""Per-problem base-model correct counts (out of 128) -> results/base_per_problem.csv.

This is the project's "newly solvable" reference: after any RL run, a problem with
c_eval == 0 here and > 0 afterwards is new capability.

Columns
  split      val | test_easy | test_medium | test_hard | test_full_op12
  uid        sha1(problem \x1f question)   (dataset `id` is not unique)
  op, template, src_line (line in the source jsonl)
  n          samples (128)
  c_eval     paper's strict criterion (upstream score_dependency_graphs.py `perfect`)
  c_train    training reward (upstream verl/reward_fn.py strict) >= 1
  c_answer   answer-only exact match
Sampling: T=0.7, top-p 1, 1024 new tokens (configs/gate1_base_eval.yaml).

Also exports ANY checkpoint's eval run (same columns) -- used for the released RL
checkpoint, the T=1.0 base sweep and later our own checkpoints:
    $EDGE_PY eval/export_base_per_problem.py --run-dir <RUN_DIR> --ckname <model dir basename> \
        --out results/<name>_per_problem.csv [--val-levels 9-15] [--sets easy medium hard]

Scorer: the variable-letter fix (eval/parser_fix.py) is the DEFAULT since 2026-11-27, so c_eval and c_train
come from the fixed parser (the upstream graph results are ignored; the upstream/ours agreement is only
printed). --upstream-parser scores with the paper's parser instead (c_eval = upstream verdict, asserted to
agree with ours), for comparison with the paper; write that to results/per_problem_upstream/.
--fixed-parser is accepted as a no-op (older commands).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from summarize_gate1 import load_task  # noqa: E402


def uid(row):
    key = (row.get("problem") or "").strip() + "\x1f" + (row.get("question") or "").strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--run-dir", type=Path, help="default: gate1 base run_dir from configs/gate1_base_eval.yaml")
    ap.add_argument("--ckname", default="base")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "base_per_problem.csv")
    ap.add_argument("--val-levels", default="2-20", help="e.g. 2-20, 9-15, or 'none'")
    ap.add_argument("--sets", nargs="*", default=["easy", "medium", "hard"])
    ap.add_argument("--upstream-parser", action="store_true",
                    help="score with the paper's upstream parser (variable-letter bug) instead of the fix")
    ap.add_argument("--fixed-parser", action="store_true", help="no-op: the fix is the default")
    args = ap.parse_args()
    fixed = not args.upstream_parser
    os.environ["EDGE_FIXED_PARSER"] = "1" if fixed else "0"   # for worker processes that re-import strict_score
    import parser_fix
    import strict_score
    parser_fix.apply() if fixed else parser_fix.revert()
    strict_score._gold_graph.cache_clear()
    print(f"scorer: {'fixed parser (canonical)' if fixed else 'upstream parser (paper)'}", flush=True)
    cfg = yaml.safe_load((REPO / "configs" / "gate1_base_eval.yaml").read_text())
    run_dir = args.run_dir or Path(cfg["run_dir"])
    data = Path(os.environ["EDGE_DATA"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["EDGE_BASE_MODEL"])

    levels = [] if args.val_levels == "none" else range(int(args.val_levels.split("-")[0]),
                                                         int(args.val_levels.split("-")[-1]) + 1)
    tasks = [(f"val_op{L}", "val", data / "val" / f"op{L}-200.jsonl") for L in levels]
    tasks += [(f"test_{s}", f"test_{s}", REPO / "data" / "testsets" / s) for s in args.sets]
    if (run_dir / "gen" / "test_full_op12").exists():
        tasks.append(("test_full_op12", "test_full_op12", data / "test" / "op12-1k.jsonl"))

    out = args.out
    fields = ["split", "uid", "op", "template", "src_line", "n", "c_eval", "c_train", "c_answer"]
    total = 0
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, split, root in tasks:
            rows, per_problem, agree = load_task(run_dir, name, root, tok, args.workers, args.ckname)
            if fixed:
                print(f"{name}: fixed parser agrees with upstream on {agree:.4f} of samples", flush=True)
            else:
                assert agree == 1.0, f"{name}: upstream/ours disagreement {agree}"
            for idx in sorted(per_problem):
                p, r = per_problem[idx], rows[idx]
                w.writerow({"split": split, "uid": r.get("uid") or uid(r), "op": r["op"],
                            "template": r.get("template"),
                            "src_line": r.get("src_line", idx if root.is_file() else ""),
                            "n": p["n"], "c_eval": p["c_ours"] if fixed else p["c"], "c_train": p["c_train"], "c_answer": p["c_ans"]})
                total += 1
            print(f"{name}: {len(per_problem)} problems", flush=True)
    print(f"wrote {out} ({total} rows)")


if __name__ == "__main__":
    main()
