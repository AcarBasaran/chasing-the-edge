"""In-training tracking set: 32 problems per level, op2-20, drawn from val/op{L}-200.jsonl.

verl evaluates it every trainer.test_freq steps (val_kwargs: T=1.0, n=8, same reward as
training), giving p_L(t) for EVERY level in EVERY arm. It is measurement only: samplers
never see it (verl calls sampler.update only on training batches), so it adds no free
information to the adaptive arms. Identical across arms, so it costs every arm the same.
Uses: edge-speed measurement (sets the bandit's m), sampler-vs-edge plots, forgetting.
The full-run pool (pool_50k) has val problems removed, so this set is never trained on.
Prompts longer than max_prompt_length=1024 tokens are excluded BEFORE sampling (same exact replica of
verl's length filter as samplers/build_pool.py). rl_base.yaml sets filter_overlong_prompts=false and
truncation=error, so a single overlong prompt here crashes validation (job 2024783 did, on a
1037-token op8 prompt, in the first version of this set).

    sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY eval/build_track_set.py'
Output: data/track32/op{L}-32.jsonl (committed; ~600 rows)
"""

import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PER_LEVEL, SEED, MAX_PROMPT = 32, 20261117, 1024


def main():
    import sys
    sys.path.insert(0, str(REPO / "eval"))
    from strict_score import compose_prompt, gold_answer_str
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["EDGE_BASE_MODEL"])

    def doc2len(line):  # == samplers/build_pool.py == upstream maybe_filter_out_long_prompts
        d = json.loads(line)
        pair = (compose_prompt(d), gold_answer_str(d.get("solution") or ""))
        return len(tok(pair, add_special_tokens=True)["input_ids"][0])

    src = Path(os.environ["EDGE_DATA"]) / "val"
    out = REPO / "data" / "track32"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for L in range(2, 21):
        rows = [l if l.endswith("\n") else l + "\n" for l in (src / f"op{L}-200.jsonl").open() if l.strip()]
        ok = [i for i in range(len(rows)) if doc2len(rows[i]) <= MAX_PROMPT]
        pick = sorted(rng.sample(ok, PER_LEVEL))
        assert all(doc2len(rows[i]) <= MAX_PROMPT for i in pick)
        print(f"op{L}: {len(rows) - len(ok)} of {len(rows)} val prompts > {MAX_PROMPT} tokens excluded")
        (out / f"op{L}-{PER_LEVEL}.jsonl").write_text("".join(rows[i] for i in pick))
    (out / "README.md").write_text(f"{PER_LEVEL} problems/level from val/op{{L}}-200.jsonl (prompts <= {MAX_PROMPT} tokens), seed {SEED}. "
                                   "Built by eval/build_track_set.py; see its docstring.\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
