"""Download the Gate-1 assets from HuggingFace into scratch.

Downloads only what we need (the 10B-token `train/` pretraining split of the
composition dataset is multiple TB and is never touched):

  * extrapolation_rl/<mixture>/base           -- the pretrained checkpoint we start from
  * extrapolation_rl/<mixture>/rl/op11-14_uniform -- (optional) their fixed-band RL result
  * composition/test/    op{2..20}-1k.jsonl   -- test split, 1000 problems per level
  * composition/val/     op{2..20}-200.jsonl  -- validation split (what the paper's
                                                 pretraining-dynamics eval uses)
  * composition/heldout/ op{2..20}-50k.jsonl  -- the RL training pool (needed later,
                                                 and now to prove test-set disjointness)

Run on the login node (network access; no model is loaded):
    source slurm/env.sh && $EDGE_PY eval/download_assets.py [--with-rl] [--no-heldout]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-rl", action="store_true", help="also fetch rl/op11-14_uniform")
    ap.add_argument("--no-heldout", action="store_true", help="skip the 1.7 GB heldout RL pool")
    args = ap.parse_args()

    snap = Path(os.environ["EDGE_SNAP"])
    mixture = os.environ["EDGE_MIXTURE"]

    model_patterns = [f"{mixture}/base/*"]
    if args.with_rl:
        model_patterns.append(f"{mixture}/rl/op11-14_uniform/*")
    path = snapshot_download(
        repo_id=os.environ["EDGE_HF_MODEL_REPO"],
        allow_patterns=model_patterns,
        local_dir=snap / "extrapolation_rl",
    )
    print(f"model snapshot -> {path}")

    data_patterns = ["README.md", "test/*", "val/*"]
    if not args.no_heldout:
        data_patterns.append("heldout/*")
    path = snapshot_download(
        repo_id=os.environ["EDGE_HF_DATA_REPO"],
        repo_type="dataset",
        allow_patterns=data_patterns,
        local_dir=snap / "composition",
        max_workers=8,
    )
    print(f"dataset snapshot -> {path}")


if __name__ == "__main__":
    main()
