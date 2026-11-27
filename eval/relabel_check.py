"""Does the strict scorer depend on which LETTERS a solution uses for its variables? (CPU, ~1 min)

Takes every gold test-set solution, renames its variables consistently to other random letters (the
result is still a fully correct solution), and scores it against the original gold, with the
upstream parser and with the fixed one (eval/parser_fix.py). A letter-independent scorer gives 100%
everywhere. Writes results/relabel_check.csv.

    python eval/relabel_check.py [--copies 4] [--out results/relabel_check.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import parser_fix  # noqa: E402
import strict_score  # noqa: E402

_DEFINE = re.compile(r"Define\s+(.*?)\s+as\s+([A-Za-z]);")
# standalone letter: no (unicode) letter or apostrophe on either side, so the "t" of "don't" is not renamed
_TOKEN = re.compile(r"(?<![^\W\d_])(?<!['’])([A-Za-z])(?![^\W\d_])(?!['’])")
_LETTERS = "abcdefghijklmnopqrstuvwyzABCDEFGHIJKLMNOPQRSTUVWXYZ"   # x stays the unknown


def variables(sol: str) -> set[str]:
    vs = {m.group(2) for m in _DEFINE.finditer(sol)}
    vs |= set(re.findall(r"(?<![^\W\d_])([A-Za-z])\s*=", sol))     # intermediates like "M = x"
    return vs


def relabel(sol: str, rng: random.Random) -> str:
    vs = sorted(variables(sol) - {"x"})
    new = rng.sample([c for c in _LETTERS if c not in vs], len(vs))
    mapping = dict(zip(vs, new))
    return _TOKEN.sub(lambda m: mapping.get(m.group(1), m.group(1)), sol)


def as_generation(sol: str) -> str:
    """What the policy emits after the "<solution>" prompt for this solution."""
    ans = strict_score.gold_answer_str(sol)
    return f"{sol} </solution> <answer> {ans}"


def run(copies: int, seed: int = 0) -> list[dict]:
    files = sorted(glob.glob(str(REPO / "data/testsets/*/*.jsonl")))
    rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
    out = []
    for fixed in (False, True):
        (parser_fix.apply if fixed else parser_fix.revert)()
        strict_score._gold_graph.cache_clear()
        rng = random.Random(seed)
        agg = defaultdict(lambda: [0, 0, 0, 0])                      # n, gold ok, relabel eval ok, train
        for r in rows:
            k = (int(r["op"]), r["mode"])
            gold_ok = strict_score.score_one(r["solution"], as_generation(r["solution"]))["perfect"]
            for _ in range(copies):
                s = strict_score.score_one(r["solution"], as_generation(relabel(r["solution"], rng)))
                a = agg[k]
                a[0] += 1; a[1] += gold_ok; a[2] += s["perfect"]; a[3] += s["train_reward"] >= 1.0
        for (op, mode), (n, g, p, t) in sorted(agg.items()):
            out.append({"parser": "fixed" if fixed else "upstream", "op": op, "mode": mode, "n": n,
                        "gold_ok": g / n, "relabelled_eval_ok": p / n, "relabelled_train_ok": t / n})
    parser_fix.revert()
    strict_score._gold_graph.cache_clear()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "results/relabel_check.csv")
    a = ap.parse_args()
    res = run(a.copies)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader(); w.writerows(res)
    tab = {(r["op"], r["mode"], r["parser"]): r["relabelled_eval_ok"] for r in res}
    print(f"{'op':>3} | {'normalforward up/fixed':>22} | {'forwardreverse up/fixed':>23}")
    for op in sorted({r["op"] for r in res}):
        nf = f"{tab[(op, 'normalforward', 'upstream')]:.3f} / {tab[(op, 'normalforward', 'fixed')]:.3f}"
        fr = f"{tab[(op, 'forwardreverse', 'upstream')]:.3f} / {tab[(op, 'forwardreverse', 'fixed')]:.3f}"
        print(f"{op:>3} | {nf:>22} | {fr:>23}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
