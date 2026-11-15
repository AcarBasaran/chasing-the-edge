"""verl hook: an AbstractCurriculumSampler that delegates level choice to a
samplers.core.DifficultySampler.

verl wiring (no patching): verl/trainer/main_ppo.py::create_rl_sampler does
    sampler = load_extern_type(data.sampler.class_path, data.sampler.class_name)(
                  data_source=train_dataset, data_config=config.data)
and RayPPOTrainer.fit() calls sampler.update(batch=batch) after every step.

YAML (see configs/gate2_threshold_smoke.yaml, configs/rl_*.yaml):
    data:
      dataloader_num_workers: 0          # required: no prefetch, so step t+1 sees update(t)
      sampler:
        class_path: <repo>/samplers/verl_adapter.py
        class_name: EdgeCurriculumSampler
        method: threshold                # fixed_band | uniform_random | threshold | bandit
        method_kwargs: {tau: 0.3, start_level: 11}
        total_steps: 20                  # __len__ = total_steps * train_batch_size
        probe_every: 0                   # m; 0 = no probes (forced >0 for bandit)
        probe_prompts_per_level: 4       # prompts per level in a probe step (x rollout.n rollouts)
        seed: 0
        log_path: /scratch/.../sampler_log.jsonl

Budget: probe prompts are ordinary members of the training batch (same size B every
step), so probe rollouts are trained on and count against the same rollout budget as
every other method. Nothing is generated outside the training batch.

Success signal: the TRAINING reward, i.e. token_level_scores summed per sequence
(verl/reward_fn.py strict reward -> 0/1); success = score >= success_threshold.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from samplers.core import DifficultySampler, make_sampler  # noqa: E402

try:  # the real base class when running inside verl
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler as _Base
except Exception:  # pragma: no cover - unit tests without verl installed
    from torch.utils.data import Sampler as _Base  # type: ignore


def _get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    try:
        val = cfg.get(key, default)
    except AttributeError:
        val = getattr(cfg, key, default)
    return default if val is None else val


def _to_plain(obj):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    return obj


def levels_of_dataset(data_source) -> np.ndarray:
    """Per-row op level of a verl dataset (CustomRLHFDataset keeps an HF `dataframe`)."""
    df = getattr(data_source, "dataframe", None)
    if df is not None:
        return np.asarray(df["op"], dtype=int)
    if hasattr(data_source, "levels"):
        return np.asarray(data_source.levels, dtype=int)
    raise TypeError("dataset exposes neither .dataframe['op'] nor .levels")


def levels_of_batch(batch) -> np.ndarray:
    nt = batch.non_tensor_batch
    if "extra_info" in nt:
        ops = [ei.get("op") if isinstance(ei, dict) else None for ei in nt["extra_info"]]
        if all(o is not None for o in ops):
            return np.asarray(ops, dtype=int)
    if "data_source" in nt:  # CustomRLHFDataset: f"difficulty-5B/{op}"
        return np.asarray([int(str(d).rsplit("/", 1)[-1]) for d in nt["data_source"]], dtype=int)
    raise KeyError("cannot find per-sample op level in batch (need extra_info['op'] or data_source)")


def scores_of_batch(batch) -> np.ndarray:
    t = batch.batch["token_level_scores"]
    return t.sum(-1).float().cpu().numpy()


class EdgeCurriculumSampler(_Base):
    def __init__(self, data_source, data_config):
        cfg = _get(data_config, "sampler")
        self.batch_size = int(_get(data_config, "gen_batch_size", _get(data_config, "train_batch_size")))
        self.total_steps = int(_get(cfg, "total_steps", 0))
        if self.total_steps <= 0:
            raise ValueError("data.sampler.total_steps must be set (it fixes len(dataloader))")
        self.seed = int(_get(cfg, "seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.success_threshold = float(_get(cfg, "success_threshold", 1.0))

        method = str(_get(cfg, "method", "fixed_band"))
        kwargs = dict(_to_plain(_get(cfg, "method_kwargs", {})) or {})
        kwargs.setdefault("seed", self.seed)
        self.core: DifficultySampler = make_sampler(method, **kwargs)
        self.method = method

        # Level buckets over the dataset.
        lv = levels_of_dataset(data_source)
        self.buckets: Dict[int, np.ndarray] = {int(L): np.flatnonzero(lv == L) for L in np.unique(lv)}
        self.core.restrict_levels(self.buckets.keys())
        self._cursor = {L: 0 for L in self.buckets}
        self._perm = {L: self.rng.permutation(idx) for L, idx in self.buckets.items()}

        self.probe_every = int(_get(cfg, "probe_every", 0))
        if self.core.needs_probes and self.probe_every <= 0:
            self.probe_every = 25
        self.probe_k = int(_get(cfg, "probe_prompts_per_level", 4))
        n_probe = self.probe_k * len(self.core.levels)
        if self.probe_every > 0 and n_probe > self.batch_size:
            raise ValueError(f"probe needs {n_probe} prompts > batch size {self.batch_size}")

        self.log_path = _get(cfg, "log_path", None) or os.environ.get("EDGE_SAMPLER_LOG")
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self.step = 0                      # number of update() calls so far
        self._planned: List[dict] = []     # per generated batch: {"levels": Counter, "probe": bool}
        self._log({"event": "init", "method": method, "kwargs": kwargs, "batch_size": self.batch_size,
                   "total_steps": self.total_steps, "probe_every": self.probe_every,
                   "probe_prompts_per_level": self.probe_k,
                   "pool": {str(L): int(len(i)) for L, i in self.buckets.items()},
                   "levels": list(self.core.levels), "state": self.core.state()})

    # ---- torch Sampler API -------------------------------------------------
    def __len__(self) -> int:
        return self.total_steps * self.batch_size

    def _draw(self, level: int) -> int:
        perm, c = self._perm[level], self._cursor[level]
        if c >= len(perm):  # bucket exhausted: reshuffle and start a new pass
            perm = self._perm[level] = self.rng.permutation(self.buckets[level])
            c = 0
        self._cursor[level] = c + 1
        return int(perm[c])

    def is_probe_step(self, step: int) -> bool:
        return self.probe_every > 0 and step % self.probe_every == 0

    def __iter__(self):
        # Lazy generator: with num_workers=0 the DataLoader pulls the indices of batch t+1
        # only after RayPPOTrainer.fit() has called update() for batch t.
        for step in range(self.total_steps):
            levels: List[int] = []
            probe = self.is_probe_step(step)
            if probe:
                for L in self.core.levels:
                    levels.extend([L] * self.probe_k)
            while len(levels) < self.batch_size:
                levels.append(int(self.core.next_level()))
            self.rng.shuffle(levels)
            self._planned.append({"probe": probe, "levels": levels})
            for L in levels:
                yield self._draw(L)

    # ---- curriculum API ----------------------------------------------------
    def update(self, batch) -> None:
        lv = levels_of_batch(batch)
        sc = scores_of_batch(batch)
        ok = sc >= self.success_threshold
        plan = self._planned[self.step] if self.step < len(self._planned) else {"probe": False}
        probe = bool(plan["probe"])

        per_level = {}
        for L in np.unique(lv):
            m = lv == L
            per_level[int(L)] = (float(ok[m].mean()), int(m.sum()))

        # informative groups (GRPO advantage != 0) per level, via the rollout-group uid
        inform = {}
        if "uid" in batch.non_tensor_batch:
            groups = defaultdict(list)
            for u, L, o in zip(batch.non_tensor_batch["uid"], lv, ok):
                groups[(u, int(L))].append(bool(o))
            agg = defaultdict(lambda: [0, 0])
            for (u, L), g in groups.items():
                agg[L][0] += int(0 < sum(g) < len(g))
                agg[L][1] += 1
            inform = {str(L): a / b for L, (a, b) in agg.items()}

        for L, (rate, n) in per_level.items():
            self.core.update(L, rate, n=n, probe=probe)

        self._log({"event": "step", "step": self.step, "probe": probe,
                   "chosen": {str(L): int(c) for L, c in zip(*np.unique(plan.get("levels", []), return_counts=True))},
                   "p": {str(L): r for L, (r, n) in per_level.items()},
                   "n": {str(L): n for L, (r, n) in per_level.items()},
                   "informative_frac": inform,
                   "mean_score": float(sc.mean()), "state": self.core.state()})
        self.step += 1

    # ---- logging -----------------------------------------------------------
    def _log(self, rec: dict) -> None:
        rec["time"] = time.time()
        line = json.dumps(rec, default=float)
        print(f"[EdgeCurriculumSampler] {line}", flush=True)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(line + "\n")
