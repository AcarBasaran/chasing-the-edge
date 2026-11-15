"""Build the held-out test sets: easy (op 2-10), medium (op 11-14), hard (op 15-20).

Source: the released `test/` split (op{L}-1k.jsonl, 1000 problems per level).
Every candidate is checked for (problem, question) collisions against
  * val/      (used by verl as the validation set during RL), and
  * heldout/  (the RL training pool every sampler draws from),
and against duplicates inside test/ itself. Colliding problems are dropped
before sampling, so the test sets are disjoint from anything RL can see.

Problems are keyed by a content hash -- the dataset's `id` field is NOT unique.

Output layout (one file per level, equal size per level within a set -- the
upstream scorer maps generations to gold graphs by `__idx % rows_per_level`, so
unequal per-level sizes inside one directory would silently mis-score):

  data/testsets/{easy,medium,hard}/op{L}-{n}.jsonl
  data/testsets/manifest.json

    source slurm/env.sh && $EDGE_PY eval/build_testsets.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

SETS = {
    # name: (levels, problems per level)
    "easy": (range(2, 11), 25),     # 225 problems (paper's in-distribution range)
    "medium": (range(11, 15), 50),  # 200 problems (paper's "OOD-edge", the fixed RL band)
    "hard": (range(15, 21), 34),    # 204 problems (paper's "OOD-hard"; headline pass@128)
}


def uid(row: dict) -> str:
    key = (row.get("problem") or "").strip() + "\x1f" + (row.get("question") or "").strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(os.environ["EDGE_DATA"]))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "testsets")
    ap.add_argument("--seed", type=int, default=20261115)
    args = ap.parse_args()

    # Hashes of everything RL can touch.
    forbidden: dict[str, set[str]] = {}
    for split in ("val", "heldout"):
        hs: set[str] = set()
        files = sorted((args.data / split).glob("op*.jsonl"))
        assert files, f"no files under {args.data / split}"
        for f in files:
            hs.update(uid(r) for r in read_jsonl(f))
        forbidden[split] = hs
        print(f"[{split}] {len(files)} files, {len(hs):,} unique problems")

    rng = random.Random(args.seed)
    manifest = {"seed": args.seed, "source": str(args.data / "test"), "sets": {}, "levels": {}}
    for name, (levels, per_level) in SETS.items():
        set_dir = args.out / name
        set_dir.mkdir(parents=True, exist_ok=True)
        for old in set_dir.glob("*.jsonl"):
            old.unlink()
        total = 0
        for L in levels:
            src = args.data / "test" / f"op{L}-1k.jsonl"
            rows = list(read_jsonl(src))
            uids = [uid(r) for r in rows]
            dup_in_test = {u for u, c in Counter(uids).items() if c > 1}
            stats = {
                "source_file": src.name,
                "source_sha256": file_sha256(src),
                "source_rows": len(rows),
                "dup_within_test": sum(u in dup_in_test for u in uids),
                "overlap_val": sum(u in forbidden["val"] for u in uids),
                "overlap_heldout": sum(u in forbidden["heldout"] for u in uids),
                "op_field_mismatch": sum(int(r.get("op", -1)) != L for r in rows),
            }
            seen: set[str] = set()
            eligible = []
            for i, (r, u) in enumerate(zip(rows, uids)):
                if u in forbidden["val"] or u in forbidden["heldout"] or u in seen:
                    continue
                if int(r.get("op", -1)) != L:
                    continue
                seen.add(u)
                eligible.append((i, r, u))
            stats["eligible"] = len(eligible)
            assert len(eligible) >= per_level, f"op{L}: only {len(eligible)} eligible problems"
            picked = sorted(rng.sample(eligible, per_level), key=lambda t: t[0])
            out = set_dir / f"op{L}-{per_level}.jsonl"
            with out.open("w") as fh:
                for i, r, u in picked:
                    rec = dict(r)
                    rec["uid"] = u
                    rec["src_line"] = i
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["picked"] = per_level
            stats["set"] = name
            manifest["levels"][str(L)] = stats
            total += per_level
            print(f"[{name}] op{L}: {stats}")
        manifest["sets"][name] = {"levels": list(levels), "per_level": per_level, "total": total}

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["sets"], indent=2))


if __name__ == "__main__":
    main()
