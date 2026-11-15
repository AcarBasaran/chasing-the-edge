"""GPU-hour budget for the full experiment and reduced scales.

Every input is a measured number or a stated assumption (ASSUMPTIONS below); rerun with
--sec-per-step-4gpu once the 4-GPU smoke has measured it.

    python eval/budget.py [--sec-per-step-4gpu S] [--markdown]
"""

from __future__ import annotations

import argparse

# ---- measured -------------------------------------------------------------------------
SMOKE_1GPU_SEC_PER_STEP_B256 = 56.6     # results/gate2/fixed_band_smoke_metrics.csv, mean steps 3-20
SMOKE_1GPU_GEN_SEC_B256 = 21.1          #   of which rollout generation
SMOKE_RESP_LEN = 306                    #   mean response length (base model, op11-14)
EVAL_SEC_PER_PROBLEM_128 = 1014 / 800   # base model, medium-200 x 128 samples, 1 A40 (job 2024377)
TEST_PROBLEMS = 225 + 800 + 204         # easy + medium(200/level) + hard
# ---- assumptions (flagged in the report) -----------------------------------------------
RESP_LEN_GROWTH = 1.2                   # mean response grows ~306 -> ~370 (harder levels / RL)
SCALING_EFF_4GPU = 0.8                  # 4-GPU parallel efficiency vs 1 GPU (to be measured)
STARTUP_MIN = 6                         # Ray + vLLM + 950k-row dataset load, per run
SWEEP_EVERY, SWEEP_GENS = 10, 19 * 32 * 8   # tracking sweep: 19 levels x 32 problems x n=8
CKPT_EVALS_PER_RUN = 1                  # final checkpoint pass@128 on easy+medium+hard
EVAL_LEN_GROWTH = 1.2


def per_run(steps: int, sec_per_step_4gpu: float | None):
    b1024_1gpu = 4 * SMOKE_1GPU_SEC_PER_STEP_B256 * RESP_LEN_GROWTH          # GPU-seconds per step
    gpu_sec_step = b1024_1gpu / SCALING_EFF_4GPU if sec_per_step_4gpu is None else 4 * sec_per_step_4gpu
    train = steps * gpu_sec_step / 3600
    sweep_gpu_sec = SWEEP_GENS / 1536 * SMOKE_1GPU_GEN_SEC_B256 * RESP_LEN_GROWTH / SCALING_EFF_4GPU
    sweeps = (steps // SWEEP_EVERY + 1) * sweep_gpu_sec / 3600
    startup = 4 * STARTUP_MIN / 60
    ev = CKPT_EVALS_PER_RUN * TEST_PROBLEMS * EVAL_SEC_PER_PROBLEM_128 * EVAL_LEN_GROWTH / 3600
    wall_h = (train + sweeps + startup) / 4
    return {"train": train, "sweeps": sweeps, "startup": startup, "eval": ev,
            "total": train + sweeps + startup + ev, "wall_h_4gpu": wall_h}


SCALES = [
    # name, arms, seeds, steps, note
    ("A  full: paper steps, 3 seeds", 6, 3, 195, "upstream rl-200steps = 1 epoch of 4x50k"),
    ("B  6 arms x 3 seeds x 100 steps", 6, 3, 100, "paper Sec. 5 uses 100-step RL too"),
    ("C  6 arms x 2 seeds x 100 steps", 6, 2, 100, ""),
    ("D  6 arms x 1 seed  x 100 steps", 6, 1, 100, "no error bars"),
    ("E  2 arms x 3 seeds x 195 steps", 2, 3, 195, "the option the brief rules out"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec-per-step-4gpu", type=float, default=None)
    a = ap.parse_args()
    gate3_extra = per_run(195, a.sec_per_step_4gpu)["total"] - per_run(100, a.sec_per_step_4gpu)["total"]
    print("per run (GPU-h):")
    for s in (100, 195):
        r = per_run(s, a.sec_per_step_4gpu)
        print(f"  {s:3d} steps: " + ", ".join(f"{k} {v:.1f}" for k, v in r.items()))
    print(f"Gate-3 extension of fixed_band s0 from 100 to 195 steps: +{gate3_extra:.1f} GPU-h\n")
    print("| scale | runs | GPU-h | + Gate-3 ext. | wall-clock if 1 x 4-GPU job runs continuously | at 50% duty |")
    print("|---|---|---|---|---|---|")
    for name, arms, seeds, steps, note in SCALES:
        r = per_run(steps, a.sec_per_step_4gpu)
        n = arms * seeds
        tot = n * r["total"] + (gate3_extra if steps < 195 else 0)
        wall_d = tot / 4 / 24
        print(f"| {name} | {n} | {n * r['total']:.0f} | {tot:.0f} | {wall_d:.1f} d | {2 * wall_d:.1f} d |")


if __name__ == "__main__":
    main()
