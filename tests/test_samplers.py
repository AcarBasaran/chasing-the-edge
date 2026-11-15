from collections import Counter

import numpy as np
import pytest

from samplers.core import Bandit, FixedBand, Threshold, UniformRandom, make_sampler
from samplers.mock import FakeBatch, FakePool, MockConfig, simulate
from samplers.verl_adapter import EdgeCurriculumSampler


# ---------------- core samplers ----------------

def test_fixed_band_only_11_14():
    s = FixedBand(seed=1)
    c = Counter(s.next_level() for _ in range(4000))
    assert set(c) == {11, 12, 13, 14}
    assert min(c.values()) > 850


def test_uniform_covers_all_levels():
    s = UniformRandom(seed=1)
    c = Counter(s.next_level() for _ in range(19000))
    assert set(c) == set(range(2, 21))
    assert min(c.values()) > 800


def test_threshold_bounds_and_tau01_clamp():
    t = Threshold(tau=0.1, delta=0.1)
    assert t.hi == pytest.approx(0.2) and t.lo == pytest.approx(0.05)   # clamped, reachable
    assert Threshold(tau=0.3).lo == pytest.approx(0.2)
    assert Threshold(tau=0.5).lo == pytest.approx(0.4) and Threshold(tau=0.5).hi == pytest.approx(0.6)


def test_threshold_tau01_can_back_off():
    t = Threshold(tau=0.1, start_level=12, patience=2, min_samples=10)
    for _ in range(2):
        t.update(12, 0.0, n=100)          # p = 0 < 0.05: back off is possible
    assert t.active == 11


def test_threshold_moves_up_with_patience():
    t = Threshold(tau=0.3, start_level=10, patience=2, min_samples=10)
    t.update(10, 0.9, n=100)
    assert t.active == 10                 # one measurement is not enough
    t.update(10, 0.9, n=100)
    assert t.active == 11


def test_threshold_ignores_other_levels_and_pools_small_batches():
    t = Threshold(tau=0.3, start_level=10, patience=1, min_samples=64)
    t.update(15, 1.0, n=1000)             # not the active level
    assert t.active == 10
    t.update(10, 1.0, n=30)               # pooled: 30 < 64
    assert t.active == 10
    t.update(10, 1.0, n=40)               # pooled 70 >= 64 -> decide
    assert t.active == 11


def test_threshold_clamped_at_boundaries():
    t = Threshold(levels=range(2, 21), tau=0.3, start_level=20, patience=1, min_samples=1)
    t.update(20, 1.0, n=10)
    assert t.active == 20
    t = Threshold(levels=range(2, 21), tau=0.3, start_level=2, patience=1, min_samples=1)
    t.update(2, 0.0, n=10)
    assert t.active == 2


def test_threshold_memory_prevents_pingpong():
    t = Threshold(tau=0.3, start_level=11, patience=1, min_samples=1, stale_after=50)
    t.update(11, 0.45, n=10)              # too easy -> up
    assert t.active == 12
    t.update(12, 0.15, n=10)              # too hard, but 11 is KNOWN too easy -> stay
    assert t.active == 12 and t.blocked == 1
    t2 = Threshold(tau=0.3, start_level=11, patience=1, min_samples=1, stale_after=0)
    t2.update(11, 0.45, n=10)
    t2.update(12, 0.15, n=10)             # no memory -> ping-pong
    assert t2.active == 11


def test_threshold_memory_expires():
    t = Threshold(tau=0.3, start_level=11, patience=1, min_samples=1, stale_after=3)
    t.update(11, 0.45, n=10)
    for _ in range(3):
        t.update(12, 0.25, n=10)          # in band: stay, memory of 11 ages
    t.update(12, 0.05, n=10)              # now 11's estimate is stale -> allowed down
    assert t.active == 11


def test_bandit_q_update_math():
    b = Bandit(levels=[11, 12], alpha=0.1, epsilon=0.0, seed=0)
    b.update(11, 0.2, probe=True)
    assert b.Q[11] == 0.0                 # first probe only sets the baseline
    b.update(11, 0.5, probe=True)
    assert b.Q[11] == pytest.approx(0.1 * 0.3)
    b.update(11, 0.6, probe=True)
    assert b.Q[11] == pytest.approx(0.9 * 0.03 + 0.1 * 0.1)
    b.update(12, 0.9, probe=False)        # non-probe feedback never touches Q
    assert b.Q[12] == 0.0
    assert b.next_level() == 11           # greedy with epsilon = 0


def test_bandit_epsilon_explores():
    b = Bandit(alpha=0.1, epsilon=0.1, seed=3)
    b.Q[15] = 1.0
    c = Counter(b.next_level() for _ in range(10000))
    assert c[15] / 10000 == pytest.approx(0.9 + 0.1 / 19, abs=0.015)
    assert len(c) == 19


def test_registry():
    assert isinstance(make_sampler("threshold", tau=0.5), Threshold)
    with pytest.raises(ValueError):
        make_sampler("nope")


# ---------------- verl adapter ----------------

def _adapter(method="threshold", kwargs=None, B=64, steps=6, probe_every=0, k=2):
    pool = FakePool(per_level=50, problem_sd=0.0, rng=np.random.default_rng(0))
    cfg = {"train_batch_size": B, "sampler": {"method": method, "method_kwargs": kwargs or {},
                                              "total_steps": steps, "probe_every": probe_every,
                                              "probe_prompts_per_level": k, "seed": 0}}
    a = EdgeCurriculumSampler(pool, cfg)
    a._log = lambda rec: None
    return a, pool


def test_adapter_len_and_levels():
    a, pool = _adapter("fixed_band")
    assert len(a) == 6 * 64
    idx = list(iter(a))
    assert len(idx) == 6 * 64
    assert set(pool.levels[idx]) <= {11, 12, 13, 14}


def test_adapter_probe_steps_cover_all_levels():
    a, pool = _adapter("bandit", {"epsilon": 0.0}, B=64, steps=4, probe_every=2, k=2)
    it = iter(a)
    for step in range(4):
        idx = [next(it) for _ in range(64)]
        c = Counter(pool.levels[idx])
        if step % 2 == 0:
            assert all(c[L] >= 2 for L in range(2, 21))
        # feed back a batch so the plan/step bookkeeping advances
        lv = pool.levels[idx]
        a.update(FakeBatch(np.zeros(64), lv, np.arange(64).astype(str)))
    assert a.step == 4


def test_adapter_is_lazy_update_changes_next_batch():
    # Threshold starts at 11; feeding p=1 twice must move the NEXT batch to 12.
    a, pool = _adapter("threshold", {"tau": 0.3, "start_level": 11, "patience": 2, "min_samples": 8})
    it = iter(a)
    for _ in range(2):
        idx = [next(it) for _ in range(64)]
        assert set(pool.levels[idx]) == {11}
        a.update(FakeBatch(np.ones(64), pool.levels[idx], np.arange(64).astype(str)))
    idx = [next(it) for _ in range(64)]
    assert set(pool.levels[idx]) == {12}


def test_adapter_reads_levels_from_data_source_when_no_extra_info():
    a, pool = _adapter("threshold", {"tau": 0.3, "start_level": 11, "patience": 1, "min_samples": 8})
    it = iter(a)
    idx = [next(it) for _ in range(64)]
    b = FakeBatch(np.ones(64), pool.levels[idx], np.arange(64).astype(str))
    del b.non_tensor_batch["extra_info"]
    b.non_tensor_batch["data_source"] = np.array([f"difficulty-5B/{l}" for l in pool.levels[idx]], dtype=object)
    a.update(b)
    assert a.core.active == 12


def test_adapter_success_uses_threshold_on_score():
    # scores of 0.5 (e.g. a partial blended reward) must NOT count as success
    a, pool = _adapter("threshold", {"tau": 0.3, "start_level": 11, "patience": 1, "min_samples": 8})
    it = iter(a)
    idx = [next(it) for _ in range(64)]
    a.update(FakeBatch(np.full(64, 0.5), pool.levels[idx], np.arange(64).astype(str)))
    assert a.core.last_p == 0.0


def test_adapter_bucket_reshuffles_when_exhausted():
    a, pool = _adapter("fixed_band", B=64, steps=20)   # 20*64 = 1280 > 4*50 problems
    idx = list(iter(a))
    assert set(pool.levels[idx]) <= {11, 12, 13, 14}
    c = Counter(idx)
    assert max(c.values()) <= 1280 // 200 + 2


# ---------------- mock harness regression ----------------

def test_mock_threshold_no_oscillation_and_tracks_skill():
    mc = MockConfig(steps=200, batch_size=128, diagnostics=0)
    traj, s = simulate(mc, "threshold", {"tau": 0.3, "start_level": 11})
    act = [t["active"] for t in traj]
    moves = [b - a for a, b in zip(act, act[1:]) if b != a]
    assert all(m > 0 for m in moves), "threshold reversed direction"
    assert abs(traj[-1]["skill"] - act[-1]) < 2.5


def test_mock_bandit_runs_and_updates_q():
    mc = MockConfig(steps=120, batch_size=128, diagnostics=0)
    traj, s = simulate(mc, "bandit", {"alpha": 0.1, "epsilon": 0.1}, probe_every=25)
    assert any(abs(q) > 0 for q in s.core.Q.values())
    assert sum(t["probe"] for t in traj) == 5
