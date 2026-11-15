"""Parser sanity check ("suspect the parser before the model").

For every problem, build the response a perfect policy would emit -- the gold
solution body, `</solution> <answer> X` -- in exactly the shape vLLM returns
(prompt stripped, `</answer>` stop string removed), and score it with both the
eval criterion and the training reward. Anything other than 100% means the
scorer, not the model, caps the measured success rate at that level.

Also reports gold node counts and the gold response token length per level
(the length a correct answer needs -- relevant to the 1024-token cap).

    source slurm/env.sh && $EDGE_PY eval/check_parser_on_gold.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strict_score import _gold_graph, compose_prompt, score_one  # noqa: E402
from utils.text_preprocess import compose_text  # noqa: E402  (upstream, path set by strict_score)


def gold_response(row: dict) -> str:
    full = compose_text(row)
    prompt = compose_prompt(row)
    assert full.startswith(prompt), "prompt is not a prefix of the training text"
    resp = full[len(prompt):]
    assert resp.endswith("</answer>")
    return resp[: -len("</answer>")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(os.environ["EDGE_DATA"]) / "val")
    ap.add_argument("--tokenizer", default=os.environ.get("EDGE_BASE_MODEL"))
    ap.add_argument("--out", type=Path, default=None, help="optional CSV of per-level stats")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows_out = []
    print(f"{'op':>3} {'n':>5} {'perfect':>8} {'train_r':>8} {'nodes(min/med/max)':>20} "
          f"{'resp_tok p50/p95/max':>22} {'prompt_tok p95/max':>19}")
    for path in sorted(args.data.glob("op*.jsonl"), key=lambda p: int(p.stem.split("-")[0][2:])):
        level = int(path.stem.split("-")[0][2:])
        perf = tr = n = 0
        nodes, resp_len, prompt_len = [], [], []
        failures = defaultdict(int)
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            resp = gold_response(row)
            s = score_one(row["solution"], resp)
            perf += s["perfect"]
            tr += s["train_reward"] >= 1.0
            if not s["perfect"]:
                failures[json.dumps({k: s[k] for k in ("missing", "extra", "value_mm", "dep_mm", "answer_ok")})] += 1
            nodes.append(len(_gold_graph(row["solution"]).nodes))
            if tok is not None:
                resp_len.append(len(tok(resp + "</answer>", add_special_tokens=False)["input_ids"]))
                prompt_len.append(len(tok(compose_prompt(row), add_special_tokens=False)["input_ids"]))
        rl = np.array(resp_len) if resp_len else np.zeros(1)
        pl = np.array(prompt_len) if prompt_len else np.zeros(1)
        rec = dict(op=level, n=n, gold_perfect=perf / n, gold_train_reward=tr / n,
                   nodes_min=min(nodes), nodes_median=float(np.median(nodes)), nodes_max=max(nodes),
                   gold_resp_tok_p50=float(np.percentile(rl, 50)), gold_resp_tok_p95=float(np.percentile(rl, 95)),
                   gold_resp_tok_max=int(rl.max()), prompt_tok_p95=float(np.percentile(pl, 95)),
                   prompt_tok_max=int(pl.max()))
        rows_out.append(rec)
        print(f"{level:>3} {n:>5} {perf / n:>8.3f} {tr / n:>8.3f} "
              f"{min(nodes):>6}/{np.median(nodes):>5.0f}/{max(nodes):>5} "
              f"{rec['gold_resp_tok_p50']:>8.0f}/{rec['gold_resp_tok_p95']:>5.0f}/{rec['gold_resp_tok_max']:>5} "
              f"{rec['prompt_tok_p95']:>10.0f}/{rec['prompt_tok_max']:>5}")
        for k, v in failures.items():
            print(f"      failure x{v}: {k}")

    if args.out:
        import csv
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
