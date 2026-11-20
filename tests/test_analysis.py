"""Tests for the cross-run analysis scripts and the arm configs."""

import csv
import json

import numpy as np
import yaml

from compare_methods import aggregate, per_run_rows
from newly_solved import newly_solved
from plot_sampler_levels import load_sampler_log, load_tracking, plot
from runs_io import REPO, load_per_problem, parse_run_id
from samplers.core import Bandit
from samplers.mock import FakePool, MockConfig, make_data_config, simulate
from samplers.verl_adapter import EdgeCurriculumSampler

FIELDS = ["split", "uid", "op", "template", "src_line", "n", "c_eval", "c_train", "c_answer"]


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for split, uid, op, c in rows:
            w.writerow({"split": split, "uid": uid, "op": op, "template": "t", "src_line": 0, "n": 128,
                        "c_eval": c, "c_train": c, "c_answer": c})


def _fixture(tmp_path):
    base = [("test_hard", "a", 15, 0), ("test_hard", "b", 15, 0), ("test_hard", "c", 16, 3),
            ("test_easy", "d", 2, 128)]
    s0 = [("test_hard", "a", 15, 5), ("test_hard", "b", 15, 0), ("test_hard", "c", 16, 0),
          ("test_easy", "d", 2, 128)]
    s1 = [("test_hard", "a", 15, 1), ("test_hard", "b", 15, 2), ("test_hard", "c", 16, 4),
          ("test_easy", "d", 2, 127)]
    paths = {}
    for name, rows in (("base", base), ("fixed_band_s0", s0), ("fixed_band_s1", s1)):
        paths[name] = tmp_path / f"{name}.csv"
        _write(paths[name], rows)
    return paths


def test_parse_run_id():
    assert parse_run_id("threshold_tau3_s1") == ("threshold_tau3", 1)
    assert parse_run_id("fixed_band_s0_smoke4gpu") == ("fixed_band", 0)
    assert parse_run_id("base") == ("base", 0)


def test_newly_solved_counts(tmp_path):
    p = _fixture(tmp_path)
    base = load_per_problem(p["base"])
    rows = newly_solved(base, load_per_problem(p["fixed_band_s0"]))
    hard = next(r for r in rows if r["split"] == "test_hard" and r["level"] == "all")
    assert (hard["base_unsolved"], hard["newly_solved"], hard["base_solved"], hard["lost"]) == (2, 1, 1, 1)
    lvl15 = next(r for r in rows if r["split"] == "test_hard" and r["level"] == 15)
    assert (lvl15["base_unsolved"], lvl15["newly_solved"]) == (2, 1)
    rows1 = newly_solved(base, load_per_problem(p["fixed_band_s1"]))
    assert next(r for r in rows1 if r["split"] == "test_hard" and r["level"] == "all")["newly_solved"] == 2


def test_compare_methods_aggregates_seeds(tmp_path):
    p = _fixture(tmp_path)
    base = load_per_problem(p["base"])
    runs = [("base", p["base"]), ("fixed_band_s0", p["fixed_band_s0"]), ("fixed_band_s1", p["fixed_band_s1"])]
    rows = per_run_rows(runs, base)
    s0 = next(r for r in rows if r["run_id"] == "fixed_band_s0" and r["split"] == "test_hard")
    assert abs(s0["pass@1"] - 100 * (5 / 128) / 3) < 1e-9          # unbiased pass@1 = mean c/n
    assert abs(s0["pass@128"] - 100 / 3) < 1e-9                     # 1 of 3 problems ever solved
    agg = {(r["method"], r["split"]): r for r in aggregate(rows)}
    fb = agg[("fixed_band", "test_hard")]
    assert fb["n_seeds"] == 2 and fb["pass@128_min"] < fb["pass@128_max"] == 100.0
    assert fb["newly_solved_mean"] == 1.5
    assert agg[("base", "test_hard")]["n_seeds"] == 1


def test_sampler_log_excludes_probe_prompts(tmp_path):
    log = tmp_path / "s.jsonl"
    recs = [{"event": "init", "method": "bandit", "kwargs": {}, "probe_prompts_per_level": 4},
            {"event": "step", "step": 0, "probe": True, "chosen": {"11": 104, "12": 4, "13": 4},
             "p": {"11": 0.5, "12": 0.25, "13": 0.0}, "informative_frac": {"11": 0.4, "12": 0.3, "13": 0.0},
             "state": {}},
            {"event": "step", "step": 1, "probe": False, "chosen": {"12": 64, "13": 64},
             "p": {"12": 0.3, "13": 0.1}, "informative_frac": {"12": 0.4, "13": 0.2}, "state": {}}]
    log.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    rows = load_sampler_log(log)["rows"]
    assert rows[0]["share"] == {11: 1.0} and rows[0]["p_trained"] == 0.5
    assert rows[1]["mean_level"] == 12.5 and abs(rows[1]["p_trained"] - 0.2) < 1e-12


def test_plot_on_mock_runs_with_tracking(tmp_path):
    mc = MockConfig(steps=40, batch_size=128, diagnostics=0)
    runs = []
    for name, method, kw, pe in (("fixed_band_s0", "fixed_band", {}, 0),
                                 ("threshold_tau3_s0", "threshold", {"tau": 0.3, "start_level": 11}, 0),
                                 ("bandit_s0", "bandit", {}, 10)):
        log = tmp_path / f"{name}.jsonl"
        simulate(mc, method, kw, pe, log_path=log)
        track = load_tracking(tmp_path / f"{name}.train.log")
        assert len(track) == 4 and set(track[0][1]) == set(range(2, 21))
        runs.append((name, load_sampler_log(log), track))
    assert all(len(r[1]["rows"]) == 40 for r in runs)
    assert [r[1]["method"] for r in runs] == ["fixed_band", "threshold", "bandit"]
    assert any(r["probe"] for r in runs[2][1]["rows"])
    out = tmp_path / "levels.png"
    plot(runs, out)
    assert out.stat().st_size > 10_000


def test_bandit_arm_uses_alpha_0_3():
    cfg = yaml.safe_load((REPO / "configs" / "rl_bandit.yaml").read_text())
    s = cfg["data"]["sampler"]
    assert s["method"] == "bandit" and s["method_kwargs"]["alpha"] == 0.3 and s["probe_every"] == 25
    assert Bandit().alpha == 0.3                                   # core default agrees
    mc = MockConfig(steps=5, batch_size=128, diagnostics=0)
    sampler = EdgeCurriculumSampler(FakePool(10, 0.5, np.random.default_rng(0)),
                                    make_data_config(mc, "bandit", s["method_kwargs"], s["probe_every"]))
    assert sampler.core.alpha == 0.3 and sampler.core.epsilon == 0.1 and sampler.probe_every == 25


def test_aggregate_seeds_spread_and_pooled_test(tmp_path, monkeypatch):
    import sys as _sys

    import aggregate_seeds

    # 4 hard problems at op18 (base never solves any); two arms x two seeds.
    base = [("test_hard", u, 18, 0) for u in "wxyz"]
    runs = {  # problems newly solved per seed
        "threshold_tau1_s0": "wxy", "threshold_tau1_s1": "wx",
        "uniform_s0": "w", "uniform_s1": "z",
    }
    pp = tmp_path / "per_problem"
    pp.mkdir()
    _write(tmp_path / "base.csv", base)
    for rid, solved in runs.items():
        _write(pp / f"{rid}.csv", [("test_hard", u, 18, 3 if u in solved else 0) for u in "wxyz"])
    monkeypatch.setattr(aggregate_seeds, "discover",
                        lambda d=None, include_released=True: [("base", tmp_path / "base.csv")]
                        + [(p.stem, p) for p in sorted(pp.glob("*.csv"))])
    monkeypatch.setattr(_sys, "argv", ["aggregate_seeds.py", "--base", str(tmp_path / "base.csv"),
                                       "--out-dir", str(tmp_path), "--runs-root", str(tmp_path / "none")])
    aggregate_seeds.main()

    rows = list(csv.DictReader(open(tmp_path / "seeds_by_set.csv")))
    tau1 = next(r for r in rows if r["method"] == "threshold_tau1" and r["metric"] == "newly_solved")
    assert tau1["n_seeds"] == "2" and float(tau1["mean"]) == 2.5 and float(tau1["min"]) == 2.0
    assert tau1["per_seed"] == "s0:3 s1:2"

    tests = [r for r in csv.DictReader(open(tmp_path / "seeds_paired_tests.csv")) if r["scope"] == "hard op18-20"]
    # METHODS order puts uniform before threshold_tau1, so A = uniform, B = tau1.
    assert {(r["arm_a"], r["arm_b"]) for r in tests} == {("uniform", "threshold_tau1")}
    within = {r["seed"]: r for r in tests if r["kind"] == "within_seed"}
    assert (within["0"]["only_a"], within["0"]["only_b"]) == ("0", "2")    # uniform {w} vs tau1 {w,x,y}
    assert (within["1"]["only_a"], within["1"]["only_b"]) == ("1", "2")    # {z} vs {w,x}
    pooled = next(r for r in tests if r["kind"] == "pooled")
    # per-problem share of seeds: uniform w .5, x 0, y 0, z .5; tau1 w 1, x 1, y .5, z 0
    assert (pooled["a_better_problems"], pooled["b_better_problems"]) == ("1", "3")
    assert float(pooled["diff_mean_newly"]) == -1.5      # 1.0 - 2.5 newly solved per seed


def test_gradient_budget_pools_all_groups(tmp_path, monkeypatch):
    import sys as _sys

    import aggregate_seeds

    base = [("test_hard", "w", 18, 0)]
    pp, runs_root = tmp_path / "pp", tmp_path / "runs"
    pp.mkdir()
    _write(tmp_path / "base.csv", base)
    _write(pp / "uniform_s0.csv", [("test_hard", "w", 18, 1)])
    (runs_root / "uniform_s0").mkdir(parents=True)
    steps = [  # two steps: 100 groups at op12 (40% informative), 100 at op19 (2%), then 200 at op12 (50%)
        {"event": "step", "step": 0, "chosen": {"12": 100, "19": 100}, "informative_frac": {"12": 0.4, "19": 0.02}},
        {"event": "step", "step": 1, "chosen": {"12": 200}, "informative_frac": {"12": 0.5}},
    ]
    (runs_root / "uniform_s0" / "sampler_log.jsonl").write_text("\n".join(json.dumps(s) for s in steps) + "\n")
    monkeypatch.setattr(aggregate_seeds, "discover", lambda d=None, include_released=True:
                        [("base", tmp_path / "base.csv"), ("uniform_s0", pp / "uniform_s0.csv")])
    monkeypatch.setattr(_sys, "argv", ["aggregate_seeds.py", "--base", str(tmp_path / "base.csv"),
                                       "--out-dir", str(tmp_path), "--runs-root", str(runs_root)])
    aggregate_seeds.main()
    row = next(csv.DictReader(open(tmp_path / "seeds_gradient_budget.csv")))
    assert row["groups_per_run"] == "400"
    assert abs(float(row["informative_frac_mean"]) - (40 + 2 + 100) / 400) < 1e-9
    assert (row["informative_groups_op11-14_mean"], row["informative_groups_op18-20_mean"]) == ("140", "2")


def test_bandit_zero_arm_never_greedy_and_phases(tmp_path):
    """An arm stuck at p = 0 earns r = 0, so it loses argmax to any improving arm; bandit_q counts op15+ prompts
    per phase and attributes none of them to the greedy choice."""
    from bandit_q import load, phases, probes

    b = Bandit(levels=[13, 15], epsilon=0.0, seed=0)
    for p13, p15 in [(0.0, 0.0), (0.3, 0.0), (0.4, 0.0)]:
        b.update(13, p13, probe=True)
        b.update(15, p15, probe=True)
    assert b.Q[15] == 0.0 and b.Q[13] > 0 and b.greedy_level() == 13

    d = tmp_path / "bandit_s0"
    d.mkdir()
    rows = [{"event": "init", "kwargs": {"epsilon": 0.1}, "levels": [13, 15], "probe_every": 2, "total_steps": 4}]
    for step, probe, chosen, greedy in [(0, True, {"13": 5, "15": 5}, 13), (1, False, {"13": 5, "15": 5}, 13),
                                        (2, True, {"13": 5, "15": 5}, 13), (3, False, {"13": 9, "15": 1}, 13)]:
        rows.append({"event": "step", "step": step, "probe": probe, "chosen": chosen,
                     "p": {"13": 0.3, "15": 0.0}, "n": {"13": 30, "15": 30}, "informative_frac": {"13": 0.5},
                     "state": {"Q": {"13": 0.1 * step, "15": 0.0}, "greedy": str(greedy)}})
    (d / "sampler_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    run = load(d)
    assert [p["step"] for p in probes(run)] == [0, 2]
    ph = {x["phase"].split(":")[0]: x for x in phases(run)}
    assert ph["steps 1-1"]["share_op15plus"] == 0.5
    assert ph["steps 3+"]["share_op15plus"] == 0.1 and ph["steps 3+"]["steps_greedy_op15plus"] == 0
    assert abs(ph["steps 3+"]["epsilon_only_share"] - 0.05) < 1e-12


def test_plot_sampler_levels_skips_runs_still_training(tmp_path, capsys, monkeypatch):
    import sys
    import plot_sampler_levels as psl

    d = tmp_path / "threshold_tau1_s9"
    d.mkdir()
    (d / "sampler_log.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", ["plot_sampler_levels.py", str(d), "--out", str(tmp_path / "x.png")])
    try:
        psl.main()
    except SystemExit as e:
        assert "no finished runs" in str(e)
    assert "still training" in capsys.readouterr().out
