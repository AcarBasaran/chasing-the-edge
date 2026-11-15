"""Difficulty samplers: pure Python, no verl/torch dependency.

Interface (CLAUDE.md):
    next_level() -> int                              level for the next prompt
    update(level, success_rate, n=None, probe=False) one measurement of p_L
                                                     (success_rate from the TRAINING reward,
                                                     n = rollouts behind it, probe = came
                                                     from the scheduled all-level probe)
    state() -> dict                                  JSON-able snapshot for per-step logging

The verl adapter (samplers/verl_adapter.py) calls next_level() once per prompt
and update() once per level present in each finished batch.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

ALL_LEVELS = tuple(range(2, 21))


class DifficultySampler(ABC):
    #: does this sampler need the all-level probe sweep to function?
    needs_probes: bool = False

    def __init__(self, levels: Sequence[int] = ALL_LEVELS, seed: int = 0):
        self.levels = tuple(sorted(int(l) for l in levels))
        if not self.levels:
            raise ValueError("empty level set")
        self.rng = random.Random(seed)

    @abstractmethod
    def next_level(self) -> int: ...

    def update(self, level: int, success_rate: float, n: Optional[int] = None, probe: bool = False) -> None:
        """Default: ignore feedback (non-adaptive samplers)."""

    def state(self) -> dict:
        return {}

    def restrict_levels(self, available: Iterable[int]) -> None:
        """Drop levels that have no problems in the pool."""
        avail = set(int(a) for a in available)
        kept = tuple(l for l in self.levels if l in avail)
        if not kept:
            raise ValueError(f"none of {self.levels} available in pool {sorted(avail)}")
        self.levels = kept


class FixedBand(DifficultySampler):
    """The paper's method: uniform over a fixed band (default op 11-14)."""

    def __init__(self, levels: Sequence[int] = (11, 12, 13, 14), seed: int = 0):
        super().__init__(levels, seed)

    def next_level(self) -> int:
        return self.rng.choice(self.levels)


class UniformRandom(DifficultySampler):
    """Uniform over every level. Controls for 'harder problems help' vs 'timing matters'."""

    def next_level(self) -> int:
        return self.rng.choice(self.levels)


class Threshold(DifficultySampler):
    """One active level L with target success rate tau.

    Move up when p_L > hi, down when p_L < lo, where
        hi = min(tau + delta, 1)
        lo = max(tau - delta, lo_floor_frac * tau)

    DECISION (tau=0.1, delta=0.1): the symmetric lower bound tau - delta = 0 can
    never be crossed (p < 0 is impossible), so the sampler could never back off.
    We clamp the lower bound to lo_floor_frac * tau (default 0.5 * tau), i.e. an
    asymmetric band [0.05, 0.2] for tau = 0.1. For tau = 0.3 and 0.5 the clamp is
    inactive ([0.2, 0.4] and [0.4, 0.6]), so the three runs differ only in tau.
    Rationale: backing off must stay possible (e.g. if a level turns out dead), and a
    multiplicative floor keeps the band's relative width sane at small tau.

    Hysteresis: a move needs `patience` consecutive measurements on the same side
    (CLAUDE.md failure mode "sampler oscillates"). A measurement is pooled until it
    has >= min_samples rollouts, so tiny noisy batches never trigger a move. All
    pooled evidence is discarded after a move (it described the old level).
    Only measurements at the active level are used.

    Known-side memory (fixes the oscillation found in the mock harness): the edge is
    so steep that one level up typically drops p from above `hi` to below `lo`
    (Gate 1: p = .49/.24/.004 at op 11/12/13), so for any band narrower than the
    adjacent-level gap the plain rule ping-pongs every `patience` steps. We remember
    each level's last estimate and refuse to move into a neighbour that is KNOWN to
    be on the wrong side (going down into a level last measured > hi, or up into one
    last measured < lo). Knowledge expires after `stale_after` measurements, since
    the model improves (and can forget). Net effect: the active level is the lowest
    level not yet known to be too easy; p at the active level ranges between the
    post-jump value and `hi`, rather than centring exactly on tau, which no single
    level can do when adjacent levels straddle the band.
    """

    def __init__(self, levels: Sequence[int] = ALL_LEVELS, tau: float = 0.3, delta: float = 0.1,
                 start_level: Optional[int] = None, patience: int = 2, min_samples: int = 64,
                 lo_floor_frac: float = 0.5, stale_after: Optional[int] = 50, seed: int = 0):
        super().__init__(levels, seed)
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        self.tau, self.delta = float(tau), float(delta)
        self.hi = min(tau + delta, 1.0)
        self.lo = max(tau - delta, lo_floor_frac * tau)
        self.patience = int(patience)
        self.min_samples = int(min_samples)
        self.active = int(start_level) if start_level is not None else self.levels[0]
        if self.active not in self.levels:
            raise ValueError(f"start_level {self.active} not in {self.levels}")
        self._succ = 0.0
        self._n = 0
        self.up_streak = 0
        self.down_streak = 0
        self.last_p: Optional[float] = None
        self.moves = 0
        self.stale_after = stale_after          # None/0 disables known-side memory
        self.n_meas = 0
        self.memory: dict[int, tuple[float, int]] = {}   # level -> (p_hat, measurement index)
        self.blocked = 0                                 # moves refused by memory (logged)

    def restrict_levels(self, available: Iterable[int]) -> None:
        super().restrict_levels(available)
        if self.active not in self.levels:
            self.active = min(self.levels, key=lambda l: abs(l - self.active))

    def next_level(self) -> int:
        return self.active

    def update(self, level: int, success_rate: float, n: Optional[int] = None, probe: bool = False) -> None:
        if int(level) != self.active:
            return
        n = self.min_samples if n is None else int(n)
        if n <= 0:
            return
        self._succ += float(success_rate) * n
        self._n += n
        if self._n < self.min_samples:
            return
        p = self._succ / self._n
        self.last_p = p
        self._succ, self._n = 0.0, 0
        self.memory[self.active] = (p, self.n_meas)
        self.n_meas += 1
        if p > self.hi:
            self.up_streak += 1
            self.down_streak = 0
        elif p < self.lo:
            self.down_streak += 1
            self.up_streak = 0
        else:
            self.up_streak = self.down_streak = 0
        idx = self.levels.index(self.active)
        if self.up_streak >= self.patience and idx + 1 < len(self.levels):
            target = self.levels[idx + 1]
            if self._known(target, side="hard"):
                self.blocked += 1
                self.up_streak = 0
            else:
                self._move(target)
        elif self.down_streak >= self.patience and idx > 0:
            target = self.levels[idx - 1]
            if self._known(target, side="easy"):
                self.blocked += 1
                self.down_streak = 0
            else:
                self._move(target)
        elif self.up_streak >= self.patience or self.down_streak >= self.patience:
            # pinned at a boundary: keep counting from zero so the log shows it
            self.up_streak = self.down_streak = 0

    def _known(self, level: int, side: str) -> bool:
        """Is `level` known (fresh estimate) to be too easy / too hard?"""
        if not self.stale_after or level not in self.memory:
            return False
        p, when = self.memory[level]
        if self.n_meas - when > self.stale_after:
            return False
        return p > self.hi if side == "easy" else p < self.lo

    def _move(self, new_level: int) -> None:
        self.active = new_level
        self.up_streak = self.down_streak = 0
        self._succ, self._n = 0.0, 0
        self.moves += 1

    def state(self) -> dict:
        return {"active": self.active, "last_p": self.last_p, "lo": self.lo, "hi": self.hi,
                "up_streak": self.up_streak, "down_streak": self.down_streak, "moves": self.moves,
                "blocked": self.blocked}


class Bandit(DifficultySampler):
    """Non-stationary bandit, one arm per level (CLAUDE.md spec):

        reward  r_L(t) = p_L(t) - p_L(t - m)      (successive PROBE estimates)
        value   Q_L <- (1 - alpha) Q_L + alpha r_L(t)
        policy  argmax_L Q_L w.p. 1 - epsilon, else uniform; ties broken at random

    Only probe measurements update Q, so every arm is measured on the same m-step
    grid and r_L is a true m-step difference. Non-probe (training-batch) measurements
    are kept in `p_train` for logging only.

    alpha defaults to 0.3 (DECIDED 2026-11-15, deviating from the CLAUDE.md spec of 0.1):
    in the mock harness alpha=0.1 lags the edge by ~2.3 levels, alpha=0.3 by ~0.8.
    m (probe interval) stays 25 until the baseline run measures how fast the real edge moves.
    """

    needs_probes = True

    def __init__(self, levels: Sequence[int] = ALL_LEVELS, alpha: float = 0.3, epsilon: float = 0.1,
                 init_q: float = 0.0, seed: int = 0):
        super().__init__(levels, seed)
        self.alpha, self.epsilon = float(alpha), float(epsilon)
        self.Q = {l: float(init_q) for l in self.levels}
        self.p_probe: dict[int, float] = {}
        self.p_train: dict[int, float] = {}
        self.last_r: dict[int, float] = {}

    def restrict_levels(self, available: Iterable[int]) -> None:
        super().restrict_levels(available)
        self.Q = {l: self.Q.get(l, 0.0) for l in self.levels}

    def greedy_level(self) -> int:
        best = max(self.Q.values())
        ties = [l for l in self.levels if math.isclose(self.Q[l], best, rel_tol=0.0, abs_tol=1e-12)]
        return self.rng.choice(ties)

    def next_level(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.levels)
        return self.greedy_level()

    def update(self, level: int, success_rate: float, n: Optional[int] = None, probe: bool = False) -> None:
        level = int(level)
        if level not in self.Q:
            return
        if not probe:
            self.p_train[level] = float(success_rate)
            return
        if level in self.p_probe:
            r = float(success_rate) - self.p_probe[level]
            self.Q[level] = (1.0 - self.alpha) * self.Q[level] + self.alpha * r
            self.last_r[level] = r
        self.p_probe[level] = float(success_rate)

    def state(self) -> dict:
        return {"Q": {str(k): v for k, v in self.Q.items()},
                "p_probe": {str(k): v for k, v in self.p_probe.items()},
                "greedy": max(self.Q, key=self.Q.get)}


REGISTRY = {"fixed_band": FixedBand, "uniform_random": UniformRandom,
            "threshold": Threshold, "bandit": Bandit}


def make_sampler(method: str, **kwargs) -> DifficultySampler:
    try:
        cls = REGISTRY[method]
    except KeyError:
        raise ValueError(f"unknown sampler {method!r}; choose from {sorted(REGISTRY)}") from None
    if "levels" in kwargs and kwargs["levels"] is not None:
        kwargs["levels"] = [int(l) for l in kwargs["levels"]]
    return cls(**kwargs)
