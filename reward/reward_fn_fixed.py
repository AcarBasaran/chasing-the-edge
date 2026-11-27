"""verl custom reward = upstream verl/reward_fn.py with the variable-letter parser fix (eval/parser_fix.py).

Identical to the paper's strict reward except that words in the equation sentence of forward-reverse
solutions no longer create fake dependencies. Use in a run config:

    custom_reward_function:
      path: ${oc.env:EDGE_REPO}/reward/reward_fn_fixed.py
      name: compute_score_with_step_process        # same name and reward_kwargs as rl_base.yaml

verl loads this file by path (verl/trainer/ppo/reward.py::get_custom_reward_fn) in every process that
computes rewards, so the patch below runs wherever the reward runs. Needs $UPSTREAM on sys.path, which
slurm/rl_run.sbatch already sets through PYTHONPATH.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_UPSTREAM = Path(os.environ.get("UPSTREAM", _REPO.parent / "Interplay-LM-Reasoning"))
for p in (_UPSTREAM, _REPO / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import parser_fix  # noqa: E402

parser_fix.apply()

_spec = importlib.util.spec_from_file_location("upstream_reward_fn_for_fixed", _UPSTREAM / "verl" / "reward_fn.py")
_upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_upstream)

compute_score = _upstream.compute_score
compute_score_with_step_process = _upstream.compute_score_with_step_process
compute_score_with_step_process_batched = _upstream.compute_score_with_step_process_batched
compute_score_with_step_process_strict = _upstream.compute_score_with_step_process_strict

PARSER_FIX_APPLIED = parser_fix.is_applied()
