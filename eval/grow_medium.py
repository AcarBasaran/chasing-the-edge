"""Grow the medium test set (op 11-14) from 50 to 200 problems per level.

The first 50 per level (built by eval/build_testsets.py, seed 20261115) are KEPT,
so every Gate 1 number on the 50-problem set is a strict subset of the new set.
150 more per level are drawn (seed 20261116) from the same eligible pool:
test/op{L}-1k.jsonl minus val/heldout collisions, minus duplicates, minus the 50.

Old files move to data/testsets/medium50/ (kept for reproducibility of Gate 1);
new files: data/testsets/medium/op{L}-200.jsonl, with `in_v1_50` marking the old rows.
Easy and hard sets are untouched (rebuilding them with build_testsets.py would
re-draw them, because the three sets share one RNG stream).

    sbatch slurm/run_cpu.sbatch '$EDGE_PY eval/grow_medium.py'
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from build_testsets import file_sha256, read_jsonl, uid

LEVELS = range(11, 15)
OLD_N, NEW_N = 50, 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(os.environ["EDGE_DATA"]))
    ap.add_argument("--testsets", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "testsets")
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args()

    med, old_dir = args.testsets / "medium", args.testsets / "medium50"
    if not old_dir.exists():
        old_dir.mkdir()
        for L in LEVELS:
            shutil.move(str(med / f"op{L}-{OLD_N}.jsonl"), str(old_dir / f"op{L}-{OLD_N}.jsonl"))

    forbidden: set[str] = set()
    for split in ("val", "heldout"):
        for f in sorted((args.data / split).glob("op*.jsonl")):
            forbidden.update(uid(r) for r in read_jsonl(f))

    manifest_path = args.testsets / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rng = random.Random(args.seed)
    for L in LEVELS:
        old_rows = list(read_jsonl(old_dir / f"op{L}-{OLD_N}.jsonl"))
        old_uids = {r["uid"] for r in old_rows}
        src = args.data / "test" / f"op{L}-1k.jsonl"
        rows = list(read_jsonl(src))
        uids = [uid(r) for r in rows]
        seen: set[str] = set()
        eligible = []
        for i, (r, u) in enumerate(zip(rows, uids)):
            if u in forbidden or u in seen or int(r.get("op", -1)) != L:
                continue
            seen.add(u)
            if u not in old_uids:
                eligible.append((i, r, u))
        assert all(u in seen for u in old_uids), f"op{L}: old medium rows no longer eligible"
        extra = rng.sample(eligible, NEW_N - OLD_N)
        picked = [(r["src_line"], {k: v for k, v in r.items()}, r["uid"], True) for r in old_rows]
        picked += [(i, r, u, False) for i, r, u in extra]
        picked.sort(key=lambda t: t[0])
        assert len({u for _, _, u, _ in picked}) == NEW_N
        out = med / f"op{L}-{NEW_N}.jsonl"
        with out.open("w") as fh:
            for i, r, u, was_old in picked:
                rec = dict(r)
                rec["uid"] = u
                rec["src_line"] = i
                rec["in_v1_50"] = was_old
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        st = manifest["levels"][str(L)]
        st.update({"picked": NEW_N, "eligible": len(seen), "v1_picked": OLD_N, "grow_seed": args.seed,
                   "source_sha256": file_sha256(src)})
        print(f"op{L}: kept {len(old_rows)}, added {len(extra)} of {len(eligible)} remaining eligible -> {out}")
    manifest["sets"]["medium"] = {"levels": list(LEVELS), "per_level": NEW_N, "total": NEW_N * len(LEVELS),
                                  "note": f"v1 {OLD_N}/level (seed {manifest['seed']}, now in medium50/) "
                                          f"+ {NEW_N - OLD_N}/level (seed {args.seed}); `in_v1_50` marks v1 rows"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
