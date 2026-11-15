"""Mock harness: run every sampler against a fake model, through the REAL verl adapter.

Fake model
    p_i(L, s) = sigmoid((s - L - b_i) / scale)       per problem i at level L
    b_i ~ N(0, problem_sd) is a fixed per-problem difficulty offset, so levels have
    within-level spread like the real data (Gate 1: many problems 0/128 or 128/128).
    Skill s rises with GRPO signal actually received:
        s <- s + eta * mean_over_prompts( 1 - p^G - (1-p)^G )
    i.e. a batch of all-solved or all-failed groups teaches nothing.
    Default s0 = 11.0, scale = 0.7: calibrated so base p is ~0.5 at op11, ~0.2 at op12,
    ~0.06 at op13 (Gate 1 measured .49/.24/.004 - the real edge is even sharper).

Each step: the adapter's iterator yields B pool indices -> G Bernoulli rollouts each ->
a fake DataProto (token_level_scores, extra_info['op'], uid) -> adapter.update(batch).
This exercises the exact code path verl will run, including probe placement and the
step <-> plan bookkeeping.

    sbatch --wait slurm/run_cpu.sbatch '$EDGE_PY -m samplers.mock --out results/mock'
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from samplers.verl_adapter import EdgeCurriculumSampler  # noqa: E402

LEVELS = list(range(2, 21))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FakePool:
    """Stands in for CustomRLHFDataset: exposes per-row levels and per-problem offsets."""

    def __init__(self, per_level: int, problem_sd: float, rng: np.random.Generator):
        self.levels = np.repeat(LEVELS, per_level)
        self.offset = rng.normal(0.0, problem_sd, size=self.levels.size)

    def __len__(self):
        return self.levels.size


class FakeBatch:
    def __init__(self, scores: np.ndarray, levels: np.ndarray, uids: np.ndarray):
        self.batch = {"token_level_scores": torch.tensor(scores, dtype=torch.float32)[:, None]}
        self.non_tensor_batch = {
            "extra_info": np.array([{"op": int(l)} for l in levels], dtype=object),
            "uid": uids.astype(object),
        }


@dataclass
class MockConfig:
    steps: int = 500
    batch_size: int = 256          # prompts per step
    group_size: int = 8            # G rollouts per prompt
    s0: float = 11.0
    scale: float = 0.7
    eta: float = 0.012
    problem_sd: float = 0.5
    per_level: int = 2000
    probe_every: int = 25
    probe_k: int = 4
    seed: int = 0
    diagnostics: int = 1


def make_data_config(mc: MockConfig, method: str, kwargs: dict, probe_every: int):
    return {"train_batch_size": mc.batch_size,
            "sampler": {"method": method, "method_kwargs": kwargs, "total_steps": mc.steps,
                        "probe_every": probe_every, "probe_prompts_per_level": mc.probe_k,
                        "seed": mc.seed, "log_path": None}}


def simulate(mc: MockConfig, method: str, kwargs: dict, probe_every: int = 0, quiet: bool = True):
    rng = np.random.default_rng(mc.seed)
    pool = FakePool(mc.per_level, mc.problem_sd, rng)
    sampler = EdgeCurriculumSampler(pool, make_data_config(mc, method, kwargs, probe_every))
    if quiet:
        sampler._log = lambda rec: None
    it = iter(sampler)
    s = mc.s0
    G = mc.group_size
    traj = []
    for step in range(mc.steps):
        idx = np.array([next(it) for _ in range(mc.batch_size)])
        lv = pool.levels[idx]
        p = sigmoid((s - lv - pool.offset[idx]) / mc.scale)
        succ = rng.random((mc.batch_size, G)) < p[:, None]
        info = 1.0 - p ** G - (1.0 - p) ** G
        realised_inform = float(np.mean([0 < r.sum() < G for r in succ]))
        batch = FakeBatch(succ.reshape(-1).astype(float), np.repeat(lv, G),
                          np.repeat(np.arange(mc.batch_size) + step * mc.batch_size, G).astype(str))
        sampler.update(batch)
        s += mc.eta * float(info.mean())
        st = sampler.core.state()
        traj.append({"step": step, "skill": s, "mean_level": float(lv.mean()),
                     "mode_level": int(np.bincount(lv).argmax()),
                     "p_true_batch": float(p.mean()), "p_measured": float(succ.mean()),
                     "inform_expected": float(info.mean()), "inform_realised": realised_inform,
                     "probe": sampler.is_probe_step(step),
                     "active": st.get("active", st.get("greedy") if any(st.get("Q", {}).values()) else None)})
    return traj, sampler


def run_suite(mc: MockConfig):
    suite = {
        "fixed_band": ("fixed_band", {}, 0),
        "uniform_random": ("uniform_random", {}, 0),
        "threshold_tau0.1": ("threshold", {"tau": 0.1, "start_level": 11}, 0),
        "threshold_tau0.3": ("threshold", {"tau": 0.3, "start_level": 11}, 0),
        "threshold_tau0.5": ("threshold", {"tau": 0.5, "start_level": 11}, 0),
        "bandit": ("bandit", {"alpha": 0.3, "epsilon": 0.1}, mc.probe_every),
    }
    if mc.diagnostics:   # non-default variants, to show what each fix buys
        suite.update({
            "threshold_tau0.3_nomemory": ("threshold", {"tau": 0.3, "start_level": 11, "stale_after": 0}, 0),
            "bandit_alpha0.1_spec": ("bandit", {"alpha": 0.1, "epsilon": 0.1}, mc.probe_every),
            "bandit_m10": ("bandit", {"alpha": 0.3, "epsilon": 0.1}, 10),
        })
    return {name: simulate(mc, m, kw, pe) for name, (m, kw, pe) in suite.items()}


def switches(traj, key="active"):
    vals = [t[key] for t in traj if t[key] is not None]
    return sum(a != b for a, b in zip(vals, vals[1:]))


def reversals(traj, key="active"):
    """Direction changes of the active level (up then down or vice versa) = oscillation."""
    vals = [t[key] for t in traj if t[key] is not None]
    moves = [np.sign(b - a) for a, b in zip(vals, vals[1:]) if b != a]
    return sum(a != b for a, b in zip(moves, moves[1:]))


def summarise(results, mc: MockConfig):
    rows = []
    for name, (traj, sampler) in results.items():
        rows.append({
            "sampler": name,
            "final_skill": round(traj[-1]["skill"], 3),
            "skill_gain": round(traj[-1]["skill"] - mc.s0, 3),
            "mean_inform_realised": round(float(np.mean([t["inform_realised"] for t in traj])), 4),
            "final_mean_level": round(traj[-1]["mean_level"], 2),
            "level_switches": switches(traj) if name not in ("fixed_band", "uniform_random") else "",
            "direction_reversals": int(reversals(traj)) if name not in ("fixed_band", "uniform_random") else "",
            "threshold_blocked": sampler.core.state().get("blocked", ""),
            "lag_final": round(traj[-1]["skill"] - traj[-1]["mean_level"], 2),
        })
    return rows


def plot(results, mc: MockConfig, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.6 * len(names)), sharex=True)
    for row, name in enumerate(names):
        traj, _ = results[name]
        x = [t["step"] for t in traj]
        ax = axes[row, 0]
        ax.plot(x, [t["skill"] for t in traj], color="black", lw=1.5, label="skill (p=0.5 level)")
        ax.plot(x, [t["mean_level"] for t in traj], color="#1f77b4", lw=0.8, alpha=0.8, label="mean chosen level")
        if name not in ("fixed_band", "uniform_random"):
            ax.step(x, [t["active"] if t["active"] is not None else np.nan for t in traj], where="post",
                    color="#d62728", lw=1.2, label="active/greedy level")
        ax.set_ylim(1.5, 20.5)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title("level chosen vs skill")
            ax.legend(fontsize=7, loc="upper left")
        ax = axes[row, 1]
        ax.plot(x, [t["p_measured"] for t in traj], color="#2ca02c", lw=0.8, label="p measured (batch)")
        ax.plot(x, [t["inform_realised"] for t in traj], color="#9467bd", lw=0.8, label="informative groups")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title("success rate and informative-group fraction")
            ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    fig.suptitle(f"Mock: B={mc.batch_size}, G={mc.group_size}, s0={mc.s0}, scale={mc.scale}, "
                 f"eta={mc.eta}, problem_sd={mc.problem_sd}, probes every {mc.probe_every} x{mc.probe_k}/level",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "mock_trajectories.png", dpi=110)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in names:
        traj, _ = results[name]
        ax.plot([t["step"] for t in traj], [t["skill"] for t in traj], label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("skill")
    ax.set_title("Mock: skill growth by sampler (budget-equalised: same B x G every step)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mock_skill.png", dpi=110)

    # bandit internals
    traj, sampler = results["bandit"]
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    q = sampler.core.Q
    ax.bar([str(l) for l in q], list(q.values()))
    ax.set_title("Bandit final Q_L (EMA of probe-to-probe progress)")
    ax.set_xlabel("level")
    fig.tight_layout()
    fig.savefig(out / "mock_bandit_q.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/mock"))
    for f in MockConfig.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    mc = MockConfig(**{k: getattr(args, k) for k in MockConfig.__dataclass_fields__})
    args.out.mkdir(parents=True, exist_ok=True)
    results = run_suite(mc)
    rows = summarise(results, mc)
    with (args.out / "mock_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "mock_config.json").write_text(json.dumps(mc.__dict__, indent=2))
    for r in rows:
        print(r)
    plot(results, mc, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
