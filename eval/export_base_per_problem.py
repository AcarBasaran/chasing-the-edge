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
    args = ap.parse_args()
    cfg = yaml.safe_load((REPO / "configs" / "gate1_base_eval.yaml").read_text())
    run_dir = Path(cfg["run_dir"])
    data = Path(os.environ["EDGE_DATA"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["EDGE_BASE_MODEL"])

    tasks = [(f"val_op{L}", "val", data / "val" / f"op{L}-200.jsonl") for L in range(2, 21)]
    tasks += [(f"test_{s}", f"test_{s}", REPO / "data" / "testsets" / s) for s in ("easy", "medium", "hard")]
    if (run_dir / "gen" / "test_full_op12").exists():
        tasks.append(("test_full_op12", "test_full_op12", data / "test" / "op12-1k.jsonl"))

    out = REPO / "results" / "base_per_problem.csv"
    fields = ["split", "uid", "op", "template", "src_line", "n", "c_eval", "c_train", "c_answer"]
    total = 0
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, split, root in tasks:
            rows, per_problem, agree = load_task(run_dir, name, root, tok, args.workers)
            assert agree == 1.0, f"{name}: upstream/ours disagreement {agree}"
            for idx in sorted(per_problem):
                p, r = per_problem[idx], rows[idx]
                w.writerow({"split": split, "uid": r.get("uid") or uid(r), "op": r["op"],
                            "template": r.get("template"),
                            "src_line": r.get("src_line", idx if root.is_file() else ""),
                            "n": p["n"], "c_eval": p["c"], "c_train": p["c_train"], "c_answer": p["c_ans"]})
                total += 1
            print(f"{name}: {len(per_problem)} problems", flush=True)
    print(f"wrote {out} ({total} rows)")


if __name__ == "__main__":
    main()
