"""Summarise the Gate-1 base-model evaluation (slurm/gate1_eval.sbatch outputs).

For every generated sample:
  * upstream verdict  -- `perfect` from scripts/score_dependency_graphs.py
  * our re-score      -- eval/strict_score.py (same upstream parser, explicit
                         problem<->generation association by __idx, no id/modulo
                         matching); must agree with upstream 100%
  * train reward      -- upstream verl/reward_fn.py strict reward (what GRPO sees)
  * answer-only       -- upstream exact-match flag

Writes (repo-relative):
  results/base_model_p_by_level.csv
  results/base_model_correct_token_lengths_by_level.csv
  results/gate1_paper_comparison.csv
  eval/base_pass128/{easy,medium,hard}.csv     per-problem base record (headline metric input)
  results/results.csv                          (rows for run_id gate1_base, upserted)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from passk import informative_fraction, pass_at_k  # noqa: E402
from strict_score import compose_prompt, score_one  # noqa: E402

LEVELS = list(range(2, 21))
KS = [1, 2, 4, 8, 16, 32, 64, 128]
CAP = 1024          # max_new_tokens
MAX_MODEL_LEN = 2048


def band_of(level: int) -> str:
    return "id_2_10" if level <= 10 else ("edge_11_14" if level <= 14 else "hard_15_20")


def read_jsonl(path: Path):
    with path.open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def eval_rows(data_root: Path):
    """Rows in exactly the order upstream eval_checkpoints.py assigns __idx."""
    if data_root.is_file():
        return read_jsonl(data_root)
    rows = []
    for f in sorted(data_root.glob("*.jsonl")):   # upstream: sorted() glob, lexicographic
        rows.extend(read_jsonl(f))
    return rows


_ROWS = None


def _init(rows):
    global _ROWS
    _ROWS = rows


def _score(rec):
    row = _ROWS[rec["__idx"]]
    s = score_one(row["solution"], rec.get("gen_solution_answer") or "")
    return s["perfect"], s["train_reward"]


def load_task(run_dir: Path, name: str, data_root: Path, tok, workers: int):
    rows = eval_rows(data_root)
    gens = read_jsonl(run_dir / "gen" / name / "base_id_generations.jsonl")
    graph = read_jsonl(run_dir / "graph" / name / "base_id_generations.jsonl.graph_results.jsonl")

    # Upstream writes one graph result per generation, in order, skipping only
    # empty solutions (missing_solution) -- align on that.
    nonempty = [i for i, g in enumerate(gens) if (g.get("gen_solution_answer") or "").strip()]
    if len(nonempty) != len(graph):
        raise RuntimeError(f"{name}: {len(nonempty)} non-empty generations vs {len(graph)} graph results")
    upstream_perfect = np.zeros(len(gens), dtype=bool)
    for i, gr in zip(nonempty, graph):
        g = gens[i]
        assert str(gr["op"]) == str(g["op"]), (name, i)
        upstream_perfect[i] = bool(gr["perfect"])

    with Pool(workers, initializer=_init, initargs=(rows,)) as pool:
        ours = pool.map(_score, gens, chunksize=256)
    ours_perfect = np.array([o[0] for o in ours], dtype=bool)
    train_ok = np.array([o[1] >= 1.0 for o in ours], dtype=bool)

    prompt_tok = {}
    for idx in {g["__idx"] for g in gens}:
        prompt_tok[idx] = len(tok(compose_prompt(rows[idx]), add_special_tokens=False)["input_ids"])

    per_problem = defaultdict(lambda: dict(n=0, c=0, c_ours=0, c_train=0, c_ans=0, trunc=0, correct_len=[], all_len=[]))
    for i, g in enumerate(gens):
        idx = g["__idx"]
        row = rows[idx]
        assert int(row["op"]) == int(g["op"]), f"{name}: __idx {idx} row op {row['op']} != gen op {g['op']}"
        pp = per_problem[idx]
        pp["n"] += 1
        pp["c"] += int(upstream_perfect[i])
        pp["c_ours"] += int(ours_perfect[i])
        pp["c_train"] += int(train_ok[i])
        pp["c_ans"] += int(g.get("exact_match", 0))
        L = int(g["resp_tokens"])
        budget = min(CAP, MAX_MODEL_LEN - prompt_tok[idx])
        pp["trunc"] += int(L >= budget)
        pp["all_len"].append(L)
        if upstream_perfect[i]:
            pp["correct_len"].append(L)
    agree = float((upstream_perfect == ours_perfect).mean())
    return rows, per_problem, agree


def level_stats(level: int, problems: list[dict], agree: float) -> tuple[dict, dict]:
    n = problems[0]["n"]
    assert all(p["n"] == n for p in problems), "unequal samples per problem"
    ps = np.array([p["c"] / n for p in problems])
    pt = np.array([p["c_train"] / n for p in problems])
    rec = {
        "level": level, "band": band_of(level), "n_problems": len(problems), "samples_per_problem": n,
        "p_L": ps.mean(), "p_L_se": ps.std(ddof=1) / np.sqrt(len(ps)),
    }
    for k in KS[1:]:
        rec[f"pass@{k}"] = float(np.mean([pass_at_k(n, p["c"], k) for p in problems]))
    rec.update({
        "frac_problems_0_of_n": float(np.mean([p["c"] == 0 for p in problems])),
        "frac_problems_n_of_n": float(np.mean([p["c"] == n for p in problems])),
        "p_train_reward": pt.mean(),
        "p_answer_only": float(np.mean([p["c_ans"] / n for p in problems])),
        # Expected fraction of GRPO groups with non-zero advantage, per-problem p (train reward, T=0.7).
        "informative_G6": float(np.mean([informative_fraction(x, 6) for x in pt])),
        "informative_G8": float(np.mean([informative_fraction(x, 8) for x in pt])),
        "trunc_rate": float(np.sum([p["trunc"] for p in problems]) / (n * len(problems))),
        "mean_resp_tokens": float(np.mean([l for p in problems for l in p["all_len"]])),
        "upstream_agreement": agree,
    })
    cl = np.array([l for p in problems for l in p["correct_len"]])
    al = np.array([l for p in problems for l in p["all_len"]])
    lens = {"level": level, "n_correct": int(cl.size)}
    if cl.size:
        lens.update({"mean": cl.mean(), "p50": np.percentile(cl, 50), "p90": np.percentile(cl, 90),
                     "p95": np.percentile(cl, 95), "p99": np.percentile(cl, 99), "max": int(cl.max()),
                     "frac_over_768": float((cl > 768).mean())})
    else:
        lens.update({k: "" for k in ("mean", "p50", "p90", "p95", "p99", "max", "frac_over_768")})
    lens.update({"all_samples_p95": np.percentile(al, 95), "all_samples_trunc_rate": rec["trunc_rate"], "cap": CAP})
    return rec, lens


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {path.relative_to(REPO)}")


def upsert_results(path: Path, run_id: str, new_rows: list[dict]) -> None:
    fields = ["run_id", "method", "tau", "seed", "test_set", "metric", "value"]
    old = []
    if path.exists():
        with path.open() as fh:
            old = [r for r in csv.DictReader(fh) if r["run_id"] != run_id]
    write = old + new_rows
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(write)
    print(f"wrote {path.relative_to(REPO)} (+{len(new_rows)} rows for {run_id})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "gate1_base_eval.yaml")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run_dir = Path(cfg["run_dir"])
    data = Path(os.environ["EDGE_DATA"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["EDGE_BASE_MODEL"])

    # ---- p_L by level (val split) ----
    level_rows, len_rows, band_problems = [], [], defaultdict(list)
    for L in LEVELS:
        _, per_problem, agree = load_task(run_dir, f"val_op{L}", data / "val" / f"op{L}-200.jsonl", tok, args.workers)
        probs = list(per_problem.values())
        rec, lens = level_stats(L, probs, agree)
        level_rows.append(rec)
        len_rows.append(lens)
        band_problems[band_of(L)].extend(probs)
        print(f"op{L:>2}: p={rec['p_L']:.4f} pass@128={rec['pass@128']:.3f} p_train={rec['p_train_reward']:.4f} "
              f"ans={rec['p_answer_only']:.3f} trunc={rec['trunc_rate']:.4f} agree={agree:.5f} "
              f"correct p95={lens['p95']}")

    gold = {}
    gold_csv = REPO / "results" / "gold_solution_stats_by_level.csv"
    if gold_csv.exists():
        with gold_csv.open() as fh:
            gold = {int(r["op"]): r for r in csv.DictReader(fh)}
    for lr in len_rows:
        g = gold.get(lr["level"], {})
        lr["gold_resp_p95"] = g.get("gold_resp_tok_p95", "")
        lr["gold_resp_max"] = g.get("gold_resp_tok_max", "")

    write_csv(REPO / "results" / "base_model_p_by_level.csv", level_rows)
    write_csv(REPO / "results" / "base_model_correct_token_lengths_by_level.csv", len_rows)

    # ---- held-out test sets: per-problem record ----
    results_rows = []
    test_band = {"easy": "id_2_10", "medium": "edge_11_14", "hard": "hard_15_20"}
    test_summ = {}
    for s in ("easy", "medium", "hard"):
        rows, per_problem, agree = load_task(run_dir, f"test_{s}", REPO / "data" / "testsets" / s, tok, args.workers)
        recs = []
        for idx in sorted(per_problem):
            p, r = per_problem[idx], rows[idx]
            recs.append({"uid": r["uid"], "op": r["op"], "src_line": r["src_line"], "template": r["template"],
                         "n": p["n"], "c_perfect": p["c"], "c_train_reward": p["c_train"], "c_answer_only": p["c_ans"],
                         "pass@1": p["c"] / p["n"], "pass@128": pass_at_k(p["n"], p["c"], 128)})
        write_csv(REPO / "eval" / "base_pass128" / f"{s}.csv", recs)
        probs = list(per_problem.values())
        summ = {f"pass@{k}": float(np.mean([pass_at_k(p['n'], p['c'], k) for p in probs])) for k in KS}
        summ["p_train_reward"] = float(np.mean([p["c_train"] / p["n"] for p in probs]))
        summ["frac_0_of_128"] = float(np.mean([p["c"] == 0 for p in probs]))
        summ["agree"] = agree
        test_summ[s] = summ
        print(f"test_{s}: {summ}")
        for m in ("pass@1", "pass@8", "pass@128", "p_train_reward", "frac_0_of_128"):
            results_rows.append({"run_id": cfg["run_id"], "method": "base", "tau": "", "seed": "",
                                 "test_set": s, "metric": m, "value": f"{summ[m]:.6g}"})
    for lr in level_rows:
        for m in ("p_L", "pass@128", "p_train_reward"):
            results_rows.append({"run_id": cfg["run_id"], "method": "base", "tau": "", "seed": "",
                                 "test_set": f"val_op{lr['level']}", "metric": m, "value": f"{lr[m]:.6g}"})

    # ---- compute accounting ----
    gen_tokens, gpu_s = 0, 0
    for tdir in sorted((run_dir / "gen").iterdir()):
        t = tdir / "timing.json"
        if t.exists():
            gpu_s += json.loads(t.read_text())["gen_seconds"]
        g = tdir / "base_id_generations.jsonl"
        with g.open() as fh:
            gen_tokens += sum(json.loads(l)["resp_tokens"] for l in fh if l.strip())
    results_rows.append({"run_id": cfg["run_id"], "method": "base", "tau": "", "seed": "", "test_set": "all",
                         "metric": "generated_tokens", "value": str(gen_tokens)})
    results_rows.append({"run_id": cfg["run_id"], "method": "base", "tau": "", "seed": "", "test_set": "all",
                         "metric": "gpu_hours_generation", "value": f"{gpu_s / 3600:.3f}"})
    print(f"compute: {gen_tokens:,} generated tokens, {gpu_s / 3600:.2f} GPU-hours of generation")
    upsert_results(REPO / "results" / "results.csv", cfg["run_id"], results_rows)

    # ---- comparison with the paper ----
    ref = cfg["paper_reference"]
    tol = ref["gate_tolerance_pp"]
    comp = []

    def add(scope, metric, ours, source, paper):
        diff = 100 * ours - paper
        comp.append({"scope": scope, "metric": metric, "ours_pct": round(100 * ours, 2), "paper_source": source,
                     "paper_pct": paper, "diff_pp": round(diff, 2), "within_tol": abs(diff) <= tol})

    for band, probs in band_problems.items():
        for m in ("pass@1", "pass@8", "pass@128"):
            k = int(m.split("@")[1])
            ours = float(np.mean([pass_at_k(p["n"], p["c"], k) for p in probs]))
            if m in ref["fig9_final_ckpt"][band]:
                add(f"val:{band}", m, ours, "fig9", ref["fig9_final_ckpt"][band][m])
            add(f"val:{band}", m, ours, "fig3", ref["fig3_base"][band][m])
    for s, band in test_band.items():
        for m in ("pass@1", "pass@8", "pass@128"):
            add(f"test:{s}", m, test_summ[s][m], "fig3", ref["fig3_base"][band][m])
    for L in (11, 12, 13, 14):
        lr = next(r for r in level_rows if r["level"] == L)
        add(f"val:op{L}", "pass@1", lr["p_L"], "fig9", ref["fig9_final_ckpt"][f"op{L}"]["pass@1"])
        add(f"val:op{L}", "pass@128", lr["pass@128"], "fig9", ref["fig9_final_ckpt"][f"op{L}"]["pass@128"])
    write_csv(REPO / "results" / "gate1_paper_comparison.csv", comp)
    n_ok = sum(c["within_tol"] for c in comp)
    print(f"GATE CHECK: {n_ok}/{len(comp)} comparisons within +-{tol}pp")
    for c in comp:
        flag = "ok " if c["within_tol"] else "OFF"
        print(f"  [{flag}] {c['scope']:<18} {c['metric']:<9} ours={c['ours_pct']:6.2f}  {c['paper_source']}={c['paper_pct']:6.2f}  diff={c['diff_pp']:+.2f}")


if __name__ == "__main__":
    main()
