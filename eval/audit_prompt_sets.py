"""Audit: does every problem set we build apply the same prompt-length rule, counted the same way?

Three code paths turn a dataset row into prompt tokens:
  A  verl length FILTER   (upstream CustomRLHFDataset.maybe_filter_out_long_prompts; replicated in
                           samplers/build_pool.py, eval/build_track_set.py, eval/prompt_filter_stats.py)
       tokenizer((verl_prompt, answer), add_special_tokens=True)["input_ids"][0]
  B  verl LOADER          (CustomRLHFDataset.__getitem__ -> postprocess_data, raises if > max_prompt_length
                           when truncation=error)
       tokenizer(verl_prompt, add_special_tokens=False)
  C  EVAL harness         (upstream scripts/eval_checkpoints.py -> vLLM generate, default add_special_tokens=True;
                           max_model_len 2048, max_tokens 1024)
       tokenizer(compose_text(row).split("<solution>")[0] + "<solution>", add_special_tokens=True)
verl_prompt = upstream verl/dataset.py::compose_prompt(row)[0].
Our replicas of A (build_pool.py, build_track_set.py, prompt_filter_stats.py) tokenize
(eval/strict_score.py::compose_prompt(row), answer)[0] with add_special_tokens=True, i.e. the EVAL prompt
text counted like C. So replica == A iff the two prompt texts agree (checked: "prompt text differs").

For every set: rows, A==B==C token counts on every row, verl prompt text == eval prompt text on every
row, and how many prompts exceed 1024 tokens. Run over EVERY row, not a sample.

    sbatch --wait --cpus-per-task=32 --mem=64G slurm/run_cpu.sbatch '$EDGE_PY eval/audit_prompt_sets.py'
Output: results/prompt_length_audit.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UP = Path(os.environ["UPSTREAM"])
SCR = Path(os.environ["EDGE_SCRATCH"])
DATA = Path(os.environ["EDGE_DATA"])
LIMIT = 1024

# (set name, files, who consumes it, does that consumer filter by length itself?)
SETS = [
    ("pool_50k", sorted((SCR / "data" / "pool_50k").glob("op*-50k.jsonl")),
     "RL training + bandit probes (configs/rl_*.yaml), verl filter OFF", "pre-filtered by build_pool.py"),
    ("track32", sorted((REPO / "data" / "track32").glob("op*-32.jsonl")),
     "RL validation / tracking sweep (configs/rl_*.yaml), verl filter OFF", "pre-filtered by build_track_set.py"),
    ("pool_2k", sorted((SCR / "data" / "pool_2k").glob("op*-2k.jsonl")),
     "Gate-2 smoke training (configs/gate2_*.yaml)", "verl filter ON at load"),
    ("testsets/medium50", sorted((REPO / "data" / "testsets" / "medium50").glob("*.jsonl")),
     "Gate-2 smoke validation file (gate2_base.yaml)", "verl filter ON at load"),
    ("testsets/easy", sorted((REPO / "data" / "testsets" / "easy").glob("*.jsonl")), "eval harness (pass@k)", "none"),
    ("testsets/medium", sorted((REPO / "data" / "testsets" / "medium").glob("*.jsonl")), "eval harness (pass@k)", "none"),
    ("testsets/hard", sorted((REPO / "data" / "testsets" / "hard").glob("*.jsonl")), "eval harness (pass@k)", "none"),
    ("val (op2-20 x 200)", sorted((DATA / "val").glob("op*-200.jsonl")),
     "eval harness: Gate-1 p_L table, T=1.0 sweep", "none"),
    ("test/op12-1k", [DATA / "test" / "op12-1k.jsonl"], "eval harness: Gate-1 op12 cross-check", "none"),
]

_tok = None


def _init():
    global _tok, compose_prompt_verl, compose_text
    sys.path.insert(0, str(UP))
    from transformers import AutoTokenizer
    from utils.text_preprocess import compose_text as ct
    # verl/dataset.py imports torch/verl at module level; we only need its compose_prompt (and the
    # helper it calls), so exec exactly those two functions from the upstream source text.
    src = (UP / "verl" / "dataset.py").read_text()
    ns: dict = {}
    exec(compile("import re\n" + _extract(src, "extract_answer_flexible")
                 + _extract(src, "compose_prompt"), "verl_dataset_subset", "exec"), ns)
    compose_prompt_verl, compose_text = ns["compose_prompt"], ct
    _tok = AutoTokenizer.from_pretrained(os.environ["EDGE_BASE_MODEL"])


def _extract(src: str, name: str) -> str:
    """Source of a top-level function from a module's text (from 'def name' to the next top-level line)."""
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}("))
    end = next((j for j in range(start + 1, len(lines)) if lines[j] and not lines[j][0].isspace()), len(lines))
    return "\n".join(lines[start:end]) + "\n\n"


def _measure(path: str):
    out = {"rows": 0, "A_max": 0, "B_max": 0, "C_max": 0, "A_over": 0, "B_over": 0, "C_over": 0,
           "A_ne_B": 0, "B_ne_C": 0, "text_ne": 0, "no_solution_marker": 0}
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        vp, ans = compose_prompt_verl(row)
        full = compose_text(row)
        ep = full.split("<solution>")[0] + "<solution>"
        out["no_solution_marker"] += int("<solution>" not in full)
        a = len(_tok((vp, ans), add_special_tokens=True)["input_ids"][0])
        b = len(_tok(vp, add_special_tokens=False)["input_ids"])
        c = len(_tok(ep, add_special_tokens=True)["input_ids"])
        out["rows"] += 1
        for k, v in (("A", a), ("B", b), ("C", c)):
            out[f"{k}_max"] = max(out[f"{k}_max"], v)
            out[f"{k}_over"] += int(v > LIMIT)
        out["A_ne_B"] += int(a != b)
        out["B_ne_C"] += int(b != c)
        out["text_ne"] += int(vp != ep)
    return path, out


def main():
    jobs = [str(f) for _, files, _, _ in SETS for f in files]
    missing = [name for name, files, _, _ in SETS if not files]
    assert not missing, f"no files for {missing}"
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), initializer=_init) as pool:
        res = dict(pool.map(_measure, jobs, chunksize=1))
    rows = []
    for name, files, consumer, filt in SETS:
        agg = {}
        for f in files:
            for k, v in res[str(f)].items():
                agg[k] = max(agg.get(k, 0), v) if k.endswith("_max") else agg.get(k, 0) + v
        rows.append({"set": name, "files": len(files), "consumer": consumer, "length_filter": filt, **agg})
    out = REPO / "results" / "prompt_length_audit.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| set | rows | max tokens (A/B/C) | > 1024 (A/B/C) | A≠B | B≠C | prompt text differs | consumer | filter |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['set']} | {r['rows']:,} | {r['A_max']}/{r['B_max']}/{r['C_max']} | "
              f"{r['A_over']}/{r['B_over']}/{r['C_over']} | {r['A_ne_B']} | {r['B_ne_C']} | {r['text_ne']} | "
              f"{r['consumer']} | {r['length_filter']} |")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
