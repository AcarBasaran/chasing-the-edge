import itertools
import random
from math import comb

import pytest

from passk import informative_fraction, mean_pass_at_k, pass_at_k


def brute_force(n, c, k):
    """Exact probability that a uniformly random k-subset of n samples contains a success."""
    samples = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(any(samples[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("n,c,k", [(5, 0, 1), (5, 1, 1), (5, 2, 3), (6, 3, 2), (8, 1, 8), (8, 7, 2), (7, 4, 4)])
def test_matches_brute_force(n, c, k):
    assert pass_at_k(n, c, k) == pytest.approx(brute_force(n, c, k), abs=1e-12)


def test_formula():
    assert pass_at_k(128, 3, 16) == pytest.approx(1 - comb(125, 16) / comb(128, 16))


def test_pass1_is_mean_success():
    for n, c in [(128, 0), (128, 1), (128, 64), (128, 128), (10, 3)]:
        assert pass_at_k(n, c, 1) == pytest.approx(c / n)


def test_edges():
    assert pass_at_k(128, 0, 128) == 0.0
    assert pass_at_k(128, 1, 128) == 1.0      # k == n: "any correct"
    assert pass_at_k(128, 128, 1) == 1.0
    assert pass_at_k(10, 8, 3) == 1.0         # n - c < k


def test_monotone_in_k_and_c():
    for c in range(0, 129, 7):
        vals = [pass_at_k(128, c, k) for k in (1, 2, 4, 8, 16, 32, 64, 128)]
        assert vals == sorted(vals)
    for k in (1, 8, 128):
        vals = [pass_at_k(128, c, k) for c in range(129)]
        assert vals == sorted(vals)


def test_unbiased_vs_monte_carlo():
    # E[pass@k from n samples] equals the true 1-(1-p)^k
    rng = random.Random(0)
    p, n, k, trials = 0.07, 32, 8, 20000
    est = sum(pass_at_k(n, sum(rng.random() < p for _ in range(n)), k) for _ in range(trials)) / trials
    assert est == pytest.approx(1 - (1 - p) ** k, abs=0.01)


def test_invalid():
    with pytest.raises(ValueError):
        pass_at_k(10, 11, 1)
    with pytest.raises(ValueError):
        pass_at_k(10, 1, 11)


def test_mean_and_informative():
    assert mean_pass_at_k([(4, 0), (4, 4)], 1) == pytest.approx(0.5)
    # CLAUDE.md table at G=8
    for p, want in [(0.01, 0.077), (0.1, 0.57), (0.3, 0.94), (0.5, 0.99), (0.9, 0.57)]:
        assert informative_fraction(p, 8) == pytest.approx(want, abs=0.006)
