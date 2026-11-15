"""Gate-1 smoke test: does the base checkpoint load, and does it produce
parseable, process-verified-correct solutions on easy problems?

Loads with plain transformers (no vLLM), reports architecture / vocab facts,
greedily decodes a handful of validation problems at a few levels, and scores
each with the upstream process-verified parser.

    sbatch slurm/gate1_check_model.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strict_score import compose_prompt, score_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("EDGE_BASE_MODEL"))
    ap.add_argument("--data", default=os.path.join(os.environ.get("EDGE_DATA", ""), "val"))
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 5, 8, 11, 14])
    ap.add_argument("--per-level", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[load] {args.model} on {device} in {time.time() - t0:.1f}s")

    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[arch] {cfg.architectures} layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"max_pos={cfg.max_position_embeddings} tied={cfg.tie_word_embeddings} dtype_on_disk={cfg.torch_dtype}")
    print(f"[arch] parameters={n_params:,}")
    print(f"[vocab] config.vocab_size={cfg.vocab_size} len(tokenizer)={len(tok)} "
          f"bos={tok.bos_token!r}({tok.bos_token_id}) eos={tok.eos_token!r}({tok.eos_token_id}) "
          f"pad={tok.pad_token!r}({tok.pad_token_id})")
    print(f"[vocab] special tokens: {tok.all_special_tokens}")

    stop_id = tok.convert_tokens_to_ids("</answer>")
    eos_ids = [i for i in {tok.eos_token_id, stop_id} if i is not None and i != tok.unk_token_id]

    summary = []
    for level in args.levels:
        path = Path(args.data) / f"op{level}-200.jsonl"
        rows = [json.loads(l) for l in path.open() if l.strip()][: args.per_level]
        n_ok = 0
        for i, row in enumerate(rows):
            prompt = compose_prompt(row)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False,
                      return_token_type_ids=False).to(device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                     eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
            gen_ids = out[0, enc["input_ids"].shape[1]:]
            text = tok.decode(gen_ids, skip_special_tokens=False)
            res = score_one(row["solution"], text)
            n_ok += int(res["perfect"])
            if i == 0:
                print(f"\n----- op={level} example 0 (prompt {enc['input_ids'].shape[1]} tok, "
                      f"gen {len(gen_ids)} tok) -----")
                print("PROMPT TAIL:", prompt[-300:])
                print("GENERATION :", text[:1500])
                print("GOLD       :", row["solution"][:1500])
                print("SCORE      :", res)
        summary.append((level, n_ok, len(rows)))
    print("\n[greedy strict-correct] " + "  ".join(f"op{l}: {k}/{n}" for l, k, n in summary))


if __name__ == "__main__":
    main()
