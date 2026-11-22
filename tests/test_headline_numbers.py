"""Recompute the report's headline numbers from the raw per-problem CSVs and check them against the summaries.

Every number in the slides and report comes from results/seeds_by_set.csv and results/seeds_hard_levels.csv,
written by eval/aggregate_seeds.py. This test does NOT reuse that script: it reloads the per-problem CSVs with
pandas and recomputes each metric from its definition, so a rerun, a hand edit, a stale summary or a broken
join in the pipeline shows up as a mismatch here before it reaches the report.

Definitions (as in eval/aggregate_seeds.py; strict EVAL criterion = column c_eval, n = 128 samples):
    pass@1        100 * mean over problems of c_eval / n
    pass@128      100 * mean over problems of the unbiased pass@k = 1 - C(n-c, k) / C(n, k)
    newly solved  problems with base c_eval == 0 and run c_eval > 0 (joined on split + uid)
    per arm       mean and SAMPLE SD (ddof=1, statistics.stdev) over seeds; SD is blank with one seed.
                  mean/sd/min/max are written with round(x, 4); the per_seed column with '.4g'.

Imports only stdlib + numpy + pandas (no torch, no samplers/, no eval/ modules).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PER_PROBLEM = RESULTS / "per_problem"
BASE_CSV = RESULTS / "base_per_problem.csv"
BY_SET_CSV = RESULTS / "seeds_by_set.csv"
HARD_LEVELS_CSV = RESULTS / "seeds_hard_levels.csv"

TEST_SETS = ("test_easy", "test_medium", "test_hard")
N_HARD = 204
OP_18_20 = (18, 19, 20)
N_OP_18_20 = 102
TOL = 1e-3                     # summaries are rounded to 4 decimals, so a correct value is off by <= 5e-5
RUN_FILE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)$")


# ---------------------------------------------------------------------------------------------- loading

def _discover() -> dict[str, dict[int, Path]]:
    """{arm: {seed: csv path}} for every results/per_problem/<arm>_s<seed>.csv."""
    arms: dict[str, dict[int, Path]] = {}
    for p in sorted(PER_PROBLEM.glob("*.csv")):
        m = RUN_FILE.match(p.stem)
        assert m, f"{p.name}: file name is not <arm>_s<seed>.csv"
        arms.setdefault(m["arm"], {})[int(m["seed"])] = p
    return arms


RUNS = _discover()
ARMS = sorted(RUNS)


@lru_cache(maxsize=None)
def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"split": str, "uid": str})
    return df[df["split"].isin(TEST_SETS)].reset_index(drop=True)


@lru_cache(maxsize=None)
def _summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"per_seed": str})


def _joined(run_path: Path) -> pd.DataFrame:
    """Run rows joined to the base model's rows on (split, uid); the problem sets must match exactly."""
    run, base = _load(run_path), _load(BASE_CSV)
    for split in TEST_SETS:
        r = set(run.loc[run["split"] == split, "uid"])
        b = set(base.loc[base["split"] == split, "uid"])
        assert r == b, (f"{run_path.name} vs {BASE_CSV.name}, {split}: problem sets differ "
                        f"({len(r - b)} only in run, {len(b - r)} only in base)")
    assert not run.duplicated(["split", "uid"]).any(), f"{run_path.name}: duplicate (split, uid) rows"
    j = run.merge(base[["split", "uid", "op", "c_eval"]], on=["split", "uid"], suffixes=("", "_base"),
                  validate="one_to_one")
    assert (j["op"] == j["op_base"]).all(), f"{run_path.name}: op disagrees with the base file for some uid"
    return j


# ------------------------------------------------------------------------------------------- metrics

def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021), written out here rather than imported from eval/passk.py."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _pass1(df: pd.DataFrame) -> float:
    return 100.0 * float((df["c_eval"] / df["n"]).mean())


def _pass128(df: pd.DataFrame) -> float:
    return 100.0 * float(np.mean([_pass_at_k(int(n), int(c), 128) for n, c in zip(df["n"], df["c_eval"])]))


def _newly_solved(df: pd.DataFrame) -> int:
    return int(((df["c_eval_base"] == 0) & (df["c_eval"] > 0)).sum())


@lru_cache(maxsize=None)
def _seed_metrics(arm: str) -> dict[int, dict[tuple[str, str], float]]:
    """{seed: {(scope, metric): value}} recomputed from the raw CSVs."""
    out = {}
    for seed, path in sorted(RUNS[arm].items()):
        j = _joined(path)
        m = {}
        for split in TEST_SETS:
            s = j[j["split"] == split]
            m[split, "pass@1"] = _pass1(s)
            m[split, "pass@128"] = _pass128(s)
            m[split, "newly_solved"] = _newly_solved(s)
        hard = j[j["split"] == "test_hard"]
        top = hard[hard["op"].isin(OP_18_20)]
        assert len(top) == N_OP_18_20, f"{path.name}: {len(top)} op18-20 hard problems, expected {N_OP_18_20}"
        m["18-20", "newly_solved"] = _newly_solved(top)
        out[seed] = m
    return out


# ------------------------------------------------------------------------------------------ comparison

def _is_blank(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or x == ""


def _check_spread(where: str, values: dict[int, float], row: pd.Series, prefix: str = "",
                  check_per_seed: bool = True) -> list[str]:
    """Compare recomputed per-seed values and their mean/sd/min/max with one summary row.

    Returns human-readable mismatch lines (empty = all good) so a failure lists every disagreement at once.
    """
    bad = []
    vals = np.array(list(values.values()), dtype=float)
    expect = {"mean": vals.mean(), "min": vals.min(), "max": vals.max(),
              "sd": vals.std(ddof=1) if len(vals) > 1 else None}
    if int(row["n_seeds"]) != len(vals):
        bad.append(f"{where}: n_seeds = {row['n_seeds']} in summary, {len(vals)} seed files on disk")
    for stat, want in expect.items():
        got = row[f"{prefix}{stat}"]
        if want is None:
            if not _is_blank(got):
                bad.append(f"{where}: {prefix}{stat} should be blank with one seed, summary has {got}")
        elif _is_blank(got) or abs(float(got) - want) > TOL:
            bad.append(f"{where}: {prefix}{stat} summary={got} recomputed={want:.4f}")
    if check_per_seed:
        listed = dict(tok.split(":") for tok in str(row["per_seed"]).split())
        listed = {int(k.lstrip("s")): v for k, v in listed.items()}
        if set(listed) != set(values):
            bad.append(f"{where}: per_seed lists seeds {sorted(listed)}, files on disk are {sorted(values)}")
        for seed, v in values.items():
            if seed in listed:
                got = float(listed[seed])
                # '.4g' keeps 4 significant figures: allow half a unit in the 4th digit
                tol = 0.0 if v == 0 else 0.51 * 10 ** (math.floor(math.log10(abs(v))) - 3)
                if abs(got - v) > max(tol, 1e-9):
                    bad.append(f"{where}: per_seed s{seed} summary={listed[seed]} recomputed={v:.4g}")
    return bad


def _by_set_row(arm: str, split: str, metric: str) -> pd.Series:
    df = _summary(BY_SET_CSV)
    rows = df[(df["method"] == arm) & (df["split"] == split) & (df["metric"] == metric)]
    assert len(rows) == 1, f"{BY_SET_CSV.name}: expected 1 row for ({arm}, {split}, {metric}), found {len(rows)}"
    return rows.iloc[0]


def _check_by_set(arm: str, split: str, metrics: tuple[str, ...]) -> list[str]:
    per_seed = _seed_metrics(arm)
    bad = []
    for metric in metrics:
        vals = {s: m[split, metric] for s, m in per_seed.items()}
        bad += _check_spread(f"{BY_SET_CSV.name} [{arm}, {split}, {metric}]", vals, _by_set_row(arm, split, metric))
    return bad


# ----------------------------------------------------------------------------------------------- tests

def test_per_problem_runs_found():
    assert ARMS, f"no <arm>_s<seed>.csv files in {PER_PROBLEM}"


def test_base_solves_no_hard_problem():
    """(a) The 'newly solved' headline assumes the hard set starts at 0/128 on every problem."""
    hard = _load(BASE_CSV)
    hard = hard[hard["split"] == "test_hard"]
    assert len(hard) == N_HARD, f"{BASE_CSV.name}: {len(hard)} test_hard rows, expected {N_HARD}"
    assert hard["uid"].is_unique, f"{BASE_CSV.name}: duplicate uids in test_hard"
    assert (hard["n"] == 128).all(), f"{BASE_CSV.name}: test_hard rows with n != 128"
    solved = hard[hard["c_eval"] > 0]
    assert solved.empty, f"{BASE_CSV.name}: base solves {len(solved)} hard problems: {solved['uid'].tolist()[:5]}"


def test_summaries_cover_exactly_the_runs_on_disk():
    """A summary with a missing or extra arm (stale aggregate, deleted run) is caught here."""
    allowed_extra = {"released_rl"}           # aggregate_seeds.py adds the paper's checkpoint from its own CSV
    for path in (BY_SET_CSV, HARD_LEVELS_CSV):
        in_summary = set(_summary(path)["method"])
        missing = set(ARMS) - in_summary
        extra = in_summary - set(ARMS) - allowed_extra
        assert not missing, f"{path.name}: arms with per-problem CSVs but no summary rows: {sorted(missing)}"
        assert not extra, f"{path.name}: arms in the summary with no per-problem CSV: {sorted(extra)}"


@pytest.mark.parametrize("arm", ARMS)
def test_hard_headline(arm):
    """(b) Hard set: newly solved, pass@1, pass@128 per seed and mean/SD/min/max vs seeds_by_set.csv."""
    bad = _check_by_set(arm, "test_hard", ("newly_solved", "pass@1", "pass@128"))
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("arm", ARMS)
def test_op18_20_newly_solved(arm):
    """(c) op 18-20 newly solved (of 102) vs seeds_hard_levels.csv, level '18-20'."""
    df = _summary(HARD_LEVELS_CSV)
    rows = df[(df["method"] == arm) & (df["level"].astype(str) == "18-20")]
    assert len(rows) == 1, f"{HARD_LEVELS_CSV.name}: expected 1 row for ({arm}, 18-20), found {len(rows)}"
    vals = {s: m["18-20", "newly_solved"] for s, m in _seed_metrics(arm).items()}
    bad = _check_spread(f"{HARD_LEVELS_CSV.name} [{arm}, 18-20]", vals, rows.iloc[0],
                        prefix="newly_solved_", check_per_seed=False)
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("arm", ARMS)
def test_easy_medium_pass1(arm):
    """(d) Easy and medium pass@1 vs seeds_by_set.csv (the forgetting / reliability numbers)."""
    bad = _check_by_set(arm, "test_easy", ("pass@1",)) + _check_by_set(arm, "test_medium", ("pass@1",))
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("arm, expected", [("threshold_tau1", {0: 94, 1: 94, 2: 97}),
                                           ("fixed_band", {0: 48, 1: 48})])
def test_anchor_hard_newly_solved(arm, expected):
    """Hard-coded anchor: the per-seed counts quoted in the report, recomputed from the raw CSVs only.

    Catches the case where the per-problem CSVs AND the summaries change together (e.g. a re-evaluation
    overwrote both), which the summary comparisons above cannot see.
    """
    assert arm in RUNS, f"no per-problem CSVs for {arm} in {PER_PROBLEM}"
    got = {s: m["test_hard", "newly_solved"] for s, m in _seed_metrics(arm).items()}
    assert got == expected, f"{arm}: hard newly solved per seed {got}, report quotes {expected}"
