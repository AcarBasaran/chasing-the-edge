"""The variable-letter bug in the upstream parser and its fix (eval/parser_fix.py).

Needs the upstream repo ($UPSTREAM, default ../Interplay-LM-Reasoning); skipped without it.
"""
import json
import sys
import random
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
strict_score = pytest.importorskip("strict_score")
import parser_fix  # noqa: E402
from relabel_check import as_generation, relabel  # noqa: E402


def _rows(path, mode):
    return [r for r in map(json.loads, open(REPO / path, encoding="utf-8")) if r["mode"] == mode]


@pytest.fixture
def upstream():
    parser_fix.revert(); strict_score._gold_graph.cache_clear()
    yield
    parser_fix.revert(); strict_score._gold_graph.cache_clear()


@pytest.fixture
def fixed():
    parser_fix.apply(); strict_score._gold_graph.cache_clear()
    yield
    parser_fix.revert(); strict_score._gold_graph.cache_clear()


def _ok(sol, gen):
    s = strict_score.score_one(sol, gen)
    return s["perfect"] and s["train_reward"] >= 1.0


FR17 = "data/testsets/hard/op17-34.jsonl"
NF17 = FR17


def test_bug_exists_upstream(upstream):
    """Correct equation solutions with other variable letters mostly FAIL with the upstream parser."""
    rng = random.Random(0)
    rows = _rows(FR17, "forwardreverse")
    ok = [_ok(r["solution"], as_generation(relabel(r["solution"], rng))) for r in rows]
    assert sum(ok) / len(ok) < 0.2
    assert all(_ok(r["solution"], as_generation(r["solution"])) for r in rows)   # gold itself passes


@pytest.mark.parametrize("mode", ["forwardreverse", "normalforward"])
def test_fixed_parser_is_letter_independent(fixed, mode):
    rng = random.Random(0)
    for r in _rows(FR17, mode):
        assert _ok(r["solution"], as_generation(r["solution"]))
        for _ in range(3):
            assert _ok(r["solution"], as_generation(relabel(r["solution"], rng)))


def test_fixed_parser_still_rejects_wrong_answer(fixed):
    r = _rows(FR17, "forwardreverse")[0]
    gen = as_generation(r["solution"])
    wrong = re.sub(r"<answer> (\d+)", lambda m: f"<answer> {int(m.group(1)) + 1}", gen)
    assert not _ok(r["solution"], wrong)


def test_fixed_parser_still_rejects_wrong_step(fixed):
    """Change one constant in an x-expression of an equation solution -> value mismatch."""
    r = _rows(FR17, "forwardreverse")[0]
    sol = r["solution"]
    m = re.search(r"= (\d+)\*x \+ (\d+)\.", sol)
    assert m, "expected an expression like '= 2*x + 2.'"
    bad = sol[:m.start()] + f"= {m.group(1)}*x + {int(m.group(2)) + 1}." + sol[m.end():]
    assert not _ok(sol, as_generation(bad))


def test_fixed_parser_still_rejects_wrong_dependency(fixed):
    """Normal-forward: compute a node from the wrong earlier node -> dependency mismatch."""
    r = _rows(NF17, "normalforward")[0]
    sol = r["solution"]
    # first "so V = A + B = ..." sum of two variables: replace the second variable by the first
    m = re.search(r"so ([A-Za-z]) = ([A-Za-z]) \+ ([A-Za-z]) = ", sol)
    assert m
    bad = sol[:m.start()] + f"so {m.group(1)} = {m.group(2)} + {m.group(2)} = " + sol[m.end():]
    assert not _ok(sol, as_generation(bad))


def test_reward_fn_fixed_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rfixed", REPO / "reward" / "reward_fn_fixed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        assert mod.PARSER_FIX_APPLIED
        r = _rows(FR17, "forwardreverse")[0]
        gen = as_generation(relabel(r["solution"], random.Random(3)))
        ga = strict_score.gold_answer_str(r["solution"])
        out = mod.compute_score_with_step_process(gen, ga, extra_info={"gold_solution": r["solution"], "gold_answer": ga},
                                                  **strict_score.TRAIN_REWARD_KWARGS)
        assert out["score"] == 1.0
    finally:
        parser_fix.revert(); strict_score._gold_graph.cache_clear()


def _graphs(sol):
    p = strict_score.SolutionParser()
    return p.build_graph(p.parse(sol))


def test_fix_changes_only_the_equation_node():
    """On every gold test solution: NF graphs are identical under both parsers; FR graphs differ only
    in the dependencies of the node whose step contains the equation sentence ("We know ...")."""
    import glob
    rows = [json.loads(l) for f in sorted(glob.glob(str(REPO / "data/testsets/*/*.jsonl"))) for l in open(f, encoding="utf-8")]
    try:
        for r in rows:
            parser_fix.revert(); up = _graphs(r["solution"])
            parser_fix.apply(); fx = _graphs(r["solution"])
            assert up.answer == fx.answer and set(up.nodes) == set(fx.nodes)
            changed = [n for n in up.nodes
                       if up.nodes[n].dependencies != fx.nodes[n].dependencies or up.nodes[n].value != fx.nodes[n].value]
            if r["mode"] == "normalforward":
                assert changed == []
            else:
                assert len(changed) <= 1
                for n in changed:
                    assert up.nodes[n].value == fx.nodes[n].value                       # values untouched
                    assert fx.nodes[n].dependencies <= up.nodes[n].dependencies         # only removes deps
                    step = r["solution"].split(f"Define {n} as ")[1]
                    assert "We know" in step                                             # the equation step
    finally:
        parser_fix.revert(); strict_score._gold_graph.cache_clear()


def test_local_fr_check_dry_run(tmp_path):
    import subprocess
    out = tmp_path / "x.md"
    res = subprocess.run([sys.executable, str(REPO / "eval" / "local_fr_check.py"), "--dry-run", "--ops", "8", "17",
                          "--per-mode", "2", "--samples", "2", "--out", str(out)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "NF answers whose verdict changed: 0" in res.stdout
    assert "answers that went pass -> fail: 0" in res.stdout
