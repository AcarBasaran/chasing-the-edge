"""How much of the heldout RL pool would verl's filter_overlong_prompts drop, per level?

Replicates upstream verl/dataset.py::CustomRLHFDataset.maybe_filter_out_long_prompts exactly:
    doc2len(doc) = len(tokenizer(compose_prompt(doc), add_special_tokens=True)["input_ids"][0])
    keep iff doc2len(doc) <= max_prompt_length            (1024 in every upstream RL config)
Note compose_prompt returns a (prompt, answer) TUPLE, which the tokenizer treats as a
batch of two strings; [0] is the prompt. We pass the same tuple.

    sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY eval/prompt_filter_stats.py'
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strict_score import compose_prompt, gold_answer_str  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=Path(os.environ["EDGE_DATA"]) / "heldout")
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "heldout_prompt_filter_by_level.csv")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(os.environ["EDGE_BASE_MODEL"])

    rows_out = []
    for L in range(2, 21):
        path = args.pool / f"op{L}-50k.jsonl"
        lens = []
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                pair = (compose_prompt(d), gold_answer_str(d.get("solution") or ""))
                lens.append(len(tok(pair, add_special_tokens=True)["input_ids"][0]))
        a = np.array(lens)
        drop = int((a > args.max_prompt_length).sum())
        rec = {"level": L, "n": a.size, "dropped": drop, "drop_frac": drop / a.size,
               "kept": a.size - drop, "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)),
               "p99": float(np.percentile(a, 99)), "max": int(a.max()),
               "frac_over_896": float((a > 896).mean())}
        rows_out.append(rec)
        print(rec, flush=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        for r in rows_out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
