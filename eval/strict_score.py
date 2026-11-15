"""Process-verified scoring, delegating to the upstream code (no re-implementation).

Two criteria, both computed with upstream functions:

  perfect      -- the paper's EVALUATION criterion (scripts/score_dependency_graphs.py,
                  paper App. A.4): every gold node present with correct value and
                  dependency set, and correct final answer. EXTRA predicted nodes
                  are allowed.
  train_reward -- the TRAINING reward used by their fixed-band RL config
                  (verl/reward_fn.py::compute_score_with_step_process with
                  answer_weight=1, step_weight=0, zero_on_process_mismatch=True).
                  Stricter: extra predicted nodes also zero the reward.

The difference matters for us: the sampler will see train_reward, while the
paper's reported numbers use `perfect`.

Upstream repo location comes from $UPSTREAM (see slurm/env.sh).
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

UPSTREAM = Path(os.environ.get("UPSTREAM", Path(__file__).resolve().parents[2] / "Interplay-LM-Reasoning"))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from utils.solution_dependency_graph import DependencyGraphData, SolutionParser  # noqa: E402
from utils.text_preprocess import compose_text  # noqa: E402


def _load_by_path(name: str, path: Path):
    # verl/reward_fn.py only needs `utils` on sys.path; importing it as part of the
    # `verl` package would drag in ray/vllm, so load the file directly.
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_reward_fn = _load_by_path("upstream_reward_fn", UPSTREAM / "verl" / "reward_fn.py")

# Kwargs from scripts/composition/op-difficulty-10B/rl-200steps/op11-14_uniform_process_strict.yaml
TRAIN_REWARD_KWARGS = dict(answer_weight=1.0, step_weight=0.0, value_tolerance=1e-6,
                           zero_on_process_mismatch=True)


def compose_prompt(row: Dict[str, Any]) -> str:
    """Exactly scripts/eval_checkpoints.py::construct_prompts_from_rows (and identical to
    the string verl/dataset.py::compose_prompt feeds the policy during RL)."""
    return compose_text(row).split("<solution>")[0] + "<solution>"


def gold_answer_str(solution: str) -> str:
    """verl/dataset.py::compose_prompt's ground-truth answer (Answer: fallback branch)."""
    m = re.search(r"Answer:\s*(.*)", solution, flags=re.IGNORECASE)
    if not m:
        return ""
    return re.split(r"(?:\n|<)", m.group(1).strip(), maxsplit=1)[0].strip().rstrip(".")


@lru_cache(maxsize=65536)
def _gold_graph(gold_solution: str) -> DependencyGraphData:
    p = SolutionParser()
    return p.build_graph(p.parse(gold_solution))


def score_one(gold_solution: str, generation: str, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Score one generation (text after the `<solution>` prompt, as vLLM returns it with
    the `</answer>` stop string removed) against the gold solution."""
    gold = _gold_graph(gold_solution)
    p = SolutionParser()
    try:
        pred = p.build_graph(p.parse(generation.strip()))
    except Exception:
        return dict(perfect=False, train_reward=0.0, parse_fail=True, answer_ok=False,
                    missing=-1, extra=-1, value_mm=-1, dep_mm=-1)
    rep = gold.compare(pred, value_tolerance=tolerance)
    # Same predicate as score_dependency_graphs.py (extra nodes do not count).
    perfect = not (rep["value_mismatches"] or rep["dependency_mismatches"]
                   or rep["answer_mismatch"] is not None or rep["missing_in_pred"])
    ga = gold_answer_str(gold_solution)
    tr = _reward_fn.compute_score_with_step_process(
        generation, ga, extra_info={"gold_solution": gold_solution, "gold_answer": ga},
        **TRAIN_REWARD_KWARGS)
    return dict(
        perfect=bool(perfect),
        train_reward=float(tr["score"]),
        parse_fail=False,
        answer_ok=bool(tr["outcome_reward"] >= 1.0),
        missing=len(rep["missing_in_pred"]),
        extra=len(rep["extra_in_pred"]),
        value_mm=len(rep["value_mismatches"]),
        dep_mm=len(rep["dependency_mismatches"]),
    )
