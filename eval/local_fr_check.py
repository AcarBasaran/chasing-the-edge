"""Check the parser fix on REAL model outputs, on a laptop (CPU is fine, ~10-20 min).

Generates a few answers with the base model for normalforward (NF) and forwardreverse (FR) test
problems, scores every answer with the upstream parser and with the fixed one (eval/parser_fix.py),
and writes every answer whose verdict changes to a Markdown file so you can read them by hand.

What a correct fix looks like:
  * NF: the two parsers agree on every answer (the fix must not touch forward problems).
  * FR: some answers flip from fail (upstream) to pass (fixed), and when you read them, every flipped
    answer is really correct: same steps and numbers as the gold, only other variable letters.
  * No answer flips from pass to fail.

    python eval/local_fr_check.py --model-dir checkpoints/extrapolation_rl/id2-10_0.2easy_0.3medium_0.5hard/base
    python eval/local_fr_check.py --dry-run      # no model: uses gold solutions with renamed letters

Prompting and decoding follow upstream scripts/eval_checkpoints.py (HF backend): prompt ends with
"<solution>", generation stops at the "</answer>" token, decoded with special tokens kept.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import parser_fix  # noqa: E402
import strict_score  # noqa: E402
from relabel_check import as_generation, relabel  # noqa: E402

SETS = {**{op: "easy" for op in range(2, 11)}, **{op: "medium" for op in range(11, 15)},
        **{op: "hard" for op in range(15, 21)}}


def pick_problems(ops, per_mode, seed):
    rng = random.Random(seed)
    out = []
    for op in ops:
        path = next((REPO / "data" / "testsets" / SETS[op]).glob(f"op{op}-*.jsonl"))
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        for mode in ("normalforward", "forwardreverse"):
            pool = [r for r in rows if r["mode"] == mode]
            out += rng.sample(pool, min(per_mode, len(pool)))
    return out


def generator(args):
    if args.dry_run:
        rng = random.Random(args.seed)
        return lambda row, k: [as_generation(relabel(row["solution"], rng)).split("<solution>")[-1]
                               for _ in range(k)]
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if int(transformers.__version__.split(".")[0]) >= 5:
        # transformers 5.x tokenizes this checkpoint's numbers differently and the model writes nonsense
        # (found 2026-11-27: 4 of 80 normalforward answers passed). The cluster pins 4.55.4.
        raise SystemExit(f"transformers {transformers.__version__} breaks this tokenizer; use 4.x "
                         "(requirements-lock.txt pins 4.55.4)")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(dev).eval()
    eos = tok.convert_tokens_to_ids("</answer>")
    torch.manual_seed(args.seed)

    def gen(row, k):
        enc = tok([strict_score.compose_prompt(row)], return_tensors="pt").to(dev)
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                                 max_new_tokens=1024, num_return_sequences=k, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id,
                                 use_cache=True)   # the checkpoint's config sets use_cache: false -> very slow on CPU
        texts = []
        for seq in out:
            new = seq[enc["input_ids"].shape[1]:]
            hit = (new == eos).nonzero()
            if len(hit):
                new = new[:hit[0].item()]
            texts.append(tok.decode(new, skip_special_tokens=False).strip())
        return texts
    return gen


def score(row, text, fixed):
    (parser_fix.apply if fixed else parser_fix.revert)()
    strict_score._gold_graph.cache_clear()
    s = strict_score.score_one(row["solution"], text)
    rep = None
    if not s["perfect"] and not s["parse_fail"]:
        g = strict_score._gold_graph(row["solution"])
        p = strict_score.SolutionParser()
        rep = g.compare(p.build_graph(p.parse(text.strip())))
    return s, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="checkpoints/extrapolation_rl/id2-10_0.2easy_0.3medium_0.5hard/base")
    ap.add_argument("--ops", type=int, nargs="+", default=[7, 8, 9, 10, 11])
    ap.add_argument("--per-mode", type=int, default=4, help="problems per (op, mode)")
    ap.add_argument("--samples", type=int, default=4, help="answers per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "local_fr_check.md")
    args = ap.parse_args()

    gen = generator(args)
    counts = Counter()
    flips = []
    rows = pick_problems(args.ops, args.per_mode, args.seed)
    for i, row in enumerate(rows):
        for text in gen(row, args.samples):
            up, up_rep = score(row, text, fixed=False)
            fx, _ = score(row, text, fixed=True)
            key = (row["mode"], "pass" if up["perfect"] else "fail", "pass" if fx["perfect"] else "fail")
            counts[key] += 1
            if up["perfect"] != fx["perfect"]:
                flips.append((row, text, up_rep, key))
        print(f"[{i + 1}/{len(rows)}] op {row['op']} {row['mode']}", flush=True)
    parser_fix.revert()

    lines = ["# Parser fix on real model answers", "",
             f"model: `{'DRY RUN (relabelled gold)' if args.dry_run else args.model_dir}`, ops {args.ops}, "
             f"{args.per_mode} problems per (op, mode), {args.samples} answers each, T={args.temperature}", "",
             "| mode | upstream | fixed | answers |", "|---|---|---|---|"]
    lines += [f"| {m} | {u} | {f} | {n} |" for (m, u, f), n in sorted(counts.items())]
    lines += ["", f"## Answers whose verdict changed ({len(flips)})", "",
              "Read each one: is it really correct? (same steps and numbers as the gold, other letters)", ""]
    for row, text, rep, key in flips:
        why = [d["name"] for d in (rep or {}).get("dependency_mismatches", [])]
        lines += [f"### op {row['op']} {row['mode']}: upstream {key[1]} -> fixed {key[2]}",
                  f"upstream complained about the dependencies of: {why}", "",
                  "**gold**", "", row["solution"], "", "**model**", "", text, ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6 + len(counts)]))
    bad_nf = sum(n for (m, u, f), n in counts.items() if m == "normalforward" and u != f)
    bad_flip = sum(n for (m, u, f), n in counts.items() if u == "pass" and f == "fail")
    print(f"\nNF answers whose verdict changed: {bad_nf} (should be 0)")
    print(f"answers that went pass -> fail: {bad_flip} (should be 0)")
    print(f"wrote {args.out}  <- read the flipped answers by hand")


if __name__ == "__main__":
    main()
