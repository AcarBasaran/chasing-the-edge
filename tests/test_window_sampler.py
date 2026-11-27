from collections import Counter

import pytest

from samplers.core import Window, make_sampler


def test_window_weights_25_50_25():
    s = Window(start_level=17, move="none", seed=1)
    c = Counter(s.next_level() for _ in range(40000))
    assert set(c) == {16, 17, 18}
    assert c[17] / 40000 == pytest.approx(0.5, abs=0.01)
    assert c[16] / 40000 == pytest.approx(0.25, abs=0.01)
    assert c[18] / 40000 == pytest.approx(0.25, abs=0.01)


def test_window_renormalises_at_top():
    s = Window(start_level=20, move="none", seed=1)
    lv, w = s.window()
    assert lv == [19, 20] and w == pytest.approx([1 / 3, 2 / 3])


def test_static_window_never_moves():
    s = Window(start_level=12, move="none")
    for _ in range(10):
        s.update(12, 1.0, n=1000)
    assert s.center == 12


def test_center_rule_moves_up_after_patience():
    s = Window(start_level=11, move="center", tau=0.1, delta=0.1, patience=2, min_samples=10)
    s.update(11, 0.5, n=100)
    assert s.center == 11
    s.update(11, 0.5, n=100)
    assert s.center == 12 and s.moves == 1


def test_center_rule_no_down_when_k_minus_1_too_easy():
    s = Window(start_level=17, move="center", tau=0.1, delta=0.1, patience=2, min_samples=10)
    for _ in range(3):
        s.update(16, 0.5, n=100)      # k-1 measured too easy
        s.update(17, 0.0, n=100)      # k below lo
    assert s.center == 17


def test_next_rule_moves_when_level_above_is_learnable():
    s = Window(start_level=17, move="next", next_floor=0.05, patience=2, min_samples=10)
    s.update(17, 0.17, n=100)          # below the 0.2 trigger: center rule would stall here
    s.update(18, 0.06, n=100)
    assert s.center == 17
    s.update(18, 0.07, n=100)
    assert s.center == 18


def test_registry():
    assert isinstance(make_sampler("window", levels=list(range(2, 21)), start_level=17), Window)
