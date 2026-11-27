"""Fix for a variable-letter bug in the upstream solution parser (found 2026-11-27). CANONICAL since 2026-11-27:
strict_score.py applies it unless EDGE_FIXED_PARSER=0, and configs/rl_base.yaml trains with it.

THE BUG
upstream utils/solution_dependency_graph.py::SolutionParser._collect_dependencies finds the
variables an expression depends on with re.findall(r"[A-Za-z]", expr), i.e. EVERY letter, also the
letters inside English words. Normal-forward solutions only have pure arithmetic after "=", so
nothing happens there. Forward-reverse (equation) solutions end with a sentence inside the last
step, e.g.

    ... so a = v = 27*x + 28. We know a = 82, so we have 27*x + 28 = 82 and we solve it through
    the following steps: Simplifying the equation: 27*x + 28 = 82. ...

The text after "a = " up to the next "." is treated as an expression, so every earlier variable
whose letter appears in "so we have ... and we solve it through the following steps: Simplifying
the equation" (o, t, v, a, e, h, ...) becomes a fake dependency of the last node. Which fake
dependencies appear depends on which LETTERS the solution happened to use for its variables. The
gold solution picked its letters at random, and the model picks its own, so a fully correct
equation solution usually fails the dependency check on the last node. Both the eval criterion
(scripts/score_dependency_graphs.py) and the strict training reward (verl/reward_fn.py) use this
parser, so it hits both.

Measured with eval/relabel_check.py (gold solutions with variables consistently renamed, i.e.
correct solutions with different letters): forward-reverse scores 100% at op 2-4, 50% at op 8,
2% at op 17. Normal-forward stays at 100%.

THE FIX
Only count standalone single letters (a letter with no other letter on either side) as variable
tokens. Arithmetic like "27*x + 28" is unchanged; words no longer leak letters. Values are not
touched (they never depended on letter choice).

Usage
    from parser_fix import apply; apply()          # patches the class in place, idempotent
    eval/strict_score.py applies it at import (EDGE_FIXED_PARSER=0 keeps the upstream parser)
    reward/reward_fn_fixed.py applies it for verl training
"""

from __future__ import annotations

import re
from typing import Dict, Set

# a single ASCII letter with no (unicode) letter directly before or after it
_STANDALONE = re.compile(r"(?<![^\W\d_])[A-Za-z](?![^\W\d_])")

_ORIGINAL = None


def _collect_dependencies_fixed(self, expr: str, intermediate_deps: Dict[str, Set[str]]) -> Set[str]:
    deps: Set[str] = set()
    for token in _STANDALONE.findall(expr):
        if token in self.var_to_param:
            deps.add(self.var_to_param[token])
        if token in intermediate_deps:
            deps.update(intermediate_deps[token])
    return deps


def _parser_cls():
    from utils.solution_dependency_graph import SolutionParser  # upstream, needs $UPSTREAM on sys.path
    return SolutionParser


def apply() -> None:
    global _ORIGINAL
    cls = _parser_cls()
    if cls._collect_dependencies is _collect_dependencies_fixed:
        return
    _ORIGINAL = cls._collect_dependencies
    cls._collect_dependencies = _collect_dependencies_fixed


def revert() -> None:
    cls = _parser_cls()
    if _ORIGINAL is not None:
        cls._collect_dependencies = _ORIGINAL


def is_applied() -> bool:
    return _parser_cls()._collect_dependencies is _collect_dependencies_fixed
