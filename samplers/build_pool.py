"""Pre-generate the RL problem pool, bucketed by op level (CLAUDE.md: never generate in the loop).

Takes a seeded random subset of each heldout/op{L}-50k.jsonl (the split upstream's RL
configs train on), drops prompts verl's filter_overlong_prompts would drop anyway is NOT
done here (verl does it at load time; <0.8% per level, results/heldout_prompt_filter_by_level.csv),
and asserts disjointness from our test sets.

    sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY samplers/build_pool.py --per-level 2000'
Output: $EDGE_SCRATCH/data/pool_{n}/op{L}-{n}.jsonl + manifest.json

Full-run pool (all 50k per level, overlong prompts and val/test collisions removed, so
every arm loads the identical pool and verl can run with filter_overlong_prompts=false):
    sbatch --wait --time=02:00:00 slurm/run_cpu.sbatch \
        '$EDGE_PY samplers/build_pool.py --per-level 50000 --filter-max-prompt-len 1024 --drop-val'
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def uid(row):
    key = (row.get("problem") or "").strip() + "\x1f" + (row.get("question") or "").strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=2000)
    ap.add_argument("--levels", type=int, nargs="+", default=list(range(2, 21)))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--filter-max-prompt-len", type=int, default=0,
                    help="drop prompts verl's filter_overlong_prompts would drop (exact replica)")
    ap.add_argument("--drop-val", action="store_true", help="drop problems that also appear in val/ (tracking set)")
    args = ap.parse_args()
    doc2len = None
    if args.filter_max_prompt_len:
        import sys
        sys.path.insert(0, str(REPO / "eval"))
        from strict_score import compose_prompt, gold_answer_str  # noqa: E402
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.environ["EDGE_BASE_MODEL"])

        def doc2len(d):  # == eval/prompt_filter_stats.py == upstream maybe_filter_out_long_prompts
            pair = (compose_prompt(d), gold_answer_str(d.get("solution") or ""))
            return len(tok(pair, add_special_tokens=True)["input_ids"][0])
    val_uids = set()
    if args.drop_val:
        for f in (Path(os.environ["EDGE_DATA"]) / "val").glob("op*.jsonl"):
            val_uids.update(uid(json.loads(l)) for l in f.open() if l.strip())
    src = Path(os.environ["EDGE_DATA"]) / "heldout"
    out = Path(os.environ["EDGE_SCRATCH"]) / "data" / f"pool_{args.per_level // 1000}k"
    out.mkdir(parents=True, exist_ok=True)

    test_uids = set()
    for f in (REPO / "data" / "testsets").rglob("*.jsonl"):
        test_uids.update(json.loads(l)["uid"] for l in f.open() if l.strip())

    rng = random.Random(args.seed)
    manifest = {"seed": args.seed, "per_level": args.per_level, "source": str(src),
                "filter_max_prompt_len": args.filter_max_prompt_len, "drop_val": args.drop_val, "levels": {}}
    for L in args.levels:
        rows = [l for l in (src / f"op{L}-50k.jsonl").open() if l.strip()]
        pick = sorted(rng.sample(range(len(rows)), args.per_level))
        dst = out / f"op{L}-{args.per_level // 1000}k.jsonl"
        n_overlap = n_val = n_long = 0
        with dst.open("w") as fh:
            for i in pick:
                d = json.loads(rows[i])
                assert int(d["op"]) == L
                u = uid(d)
                if u in test_uids:
                    n_overlap += 1
                    continue
                if u in val_uids:
                    n_val += 1
                    continue
                if doc2len is not None and doc2len(d) > args.filter_max_prompt_len:
                    n_long += 1
                    continue
                fh.write(rows[i] if rows[i].endswith("\n") else rows[i] + "\n")
        manifest["levels"][str(L)] = {"file": dst.name,
                                      "rows": args.per_level - n_overlap - n_val - n_long,
                                      "test_overlap_dropped": n_overlap, "val_overlap_dropped": n_val,
                                      "overlong_dropped": n_long}
        print(L, manifest["levels"][str(L)], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
