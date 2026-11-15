"""In-training tracking set: 32 problems per level, op2-20, drawn from val/op{L}-200.jsonl.

verl evaluates it every trainer.test_freq steps (val_kwargs: T=1.0, n=8, same reward as
training), giving p_L(t) for EVERY level in EVERY arm. It is measurement only: samplers
never see it (verl calls sampler.update only on training batches), so it adds no free
information to the adaptive arms. Identical across arms, so it costs every arm the same.
Uses: edge-speed measurement (sets the bandit's m), sampler-vs-edge plots, forgetting.
The full-run pool (pool_50k) has val problems removed, so this set is never trained on.

    sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY eval/build_track_set.py'
Output: data/track32/op{L}-32.jsonl (committed; ~600 rows)
"""

import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PER_LEVEL, SEED = 32, 20261117


def main():
    src = Path(os.environ["EDGE_DATA"]) / "val"
    out = REPO / "data" / "track32"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for L in range(2, 21):
        rows = [l if l.endswith("\n") else l + "\n" for l in (src / f"op{L}-200.jsonl").open() if l.strip()]
        pick = sorted(rng.sample(range(len(rows)), PER_LEVEL))
        (out / f"op{L}-{PER_LEVEL}.jsonl").write_text("".join(rows[i] for i in pick))
    (out / "README.md").write_text(f"{PER_LEVEL} problems/level from val/op{{L}}-200.jsonl, seed {SEED}. "
                                   "Built by eval/build_track_set.py; see its docstring.\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
