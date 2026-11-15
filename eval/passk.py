"""Unbiased pass@k (Chen et al., 2021):  pass@k = 1 - C(n-c, k) / C(n, k).

Do NOT estimate pass@k as "any of the first k samples correct" (what upstream's
aggregate_graph_pass_results.py does) -- that is unbiased only at k == n and is
needlessly noisy below it.
"""

from __future__ import annotations

from math import comb
from typing import Iterable, Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got n={n} c={c}")
    if k > n:
        raise ValueError(f"pass@{k} needs n >= k samples, got n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def mean_pass_at_k(counts: Iterable[tuple[int, int]], k: int) -> float:
    vals = [pass_at_k(n, c, k) for n, c in counts]
    return sum(vals) / len(vals) if vals else float("nan")


def informative_fraction(p: float, group_size: int) -> float:
    """Probability a GRPO group of `group_size` rollouts has non-constant reward."""
    return 1.0 - p ** group_size - (1.0 - p) ** group_size


def _self_test(ks: Sequence[int] = (1, 2, 8, 128)) -> None:
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-12           # pass@1 == c/n
    assert abs(pass_at_k(4, 1, 2) - 0.5) < 1e-12            # 1 - C(3,2)/C(4,2)
    assert pass_at_k(128, 1, 128) == 1.0
    assert abs(informative_fraction(0.5, 8) - (1 - 2 * 0.5 ** 8)) < 1e-12
    for k in ks:
        assert pass_at_k(128, 0, k) == 0.0


if __name__ == "__main__":
    _self_test()
    print("passk self-test ok")
