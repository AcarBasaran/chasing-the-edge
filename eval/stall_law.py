"""How adaptive is the threshold sampler, really? (2026-11-22)

1. Stall law. For every threshold run: the level where it stopped climbing (final active level F), when it got
   there, F's success rate in the training batches from then on (the batch is 100% level F, 6,144 answers per
   step, training reward, T = 1), and the move-up trigger h = tau + delta. Hypothesis: "each arm stalls at the
   first level whose plateau is below its trigger".
   For F we report the batch-p plateau (mean of the last 20 steps), its max, how many steps were above h, the
   longest run of consecutive steps above h (a move needs 2), the slope over the last 30 steps (is it still
   rising?), and moves the known-side memory refused. For every level the arm climbed THROUGH we report the
   batch p of the two measurements that triggered the move (both > h by construction).
2. Climbing vs stalled: steps before the final move (climbing) and steps at F (stalled).
3. Preview of "when did the gain happen", from logs only: the tracking set (32 problems/level, 8 answers at
   T = 1, every 10 steps; edge_speed.csv) for tau=0.1 vs the op 11-17 control on op 15-20.

    $EDGE_PY eval/stall_law.py [--runs-root $EDGE_SCRATCH/outputs/runs]
Writes results/stall_law.csv, results/stall_law_moves.csv, results/stall_law_segments.csv,
results/stall_law_plateaus.csv, results/stall_law_tracking.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
THRESHOLD_RUNS = ["threshold_tau1_s0", "threshold_tau1_s1", "threshold_tau1_s2",
                  "threshold_tau3_s0", "threshold_tau3_s1", "threshold_tau5_s0"]
TRACK_ARMS = {"threshold_tau1": ["threshold_tau1_s0", "threshold_tau1_s1", "threshold_tau1_s2"],
              "uniform_11_17": ["uniform_11_17_s0", "uniform_11_17_s1", "uniform_11_17_s2"]}


def steps_of(path: Path) -> tuple[dict, list[dict]]:
    init, steps = {}, []
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") == "init":
            init = rec
        elif rec.get("event") == "step":
            steps.append(rec)
    return init, steps


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path(os.environ.get("EDGE_SCRATCH", "/scratch")) / "outputs" / "runs")
    a = ap.parse_args()

    rows, moves_rows, segs = [], [], []
    for rid in THRESHOLD_RUNS:
        init, steps = steps_of(a.runs_root / rid / "sampler_log.jsonl")
        tau = float(init["kwargs"]["tau"])
        delta = float(init["kwargs"].get("delta", 0.1))
        h = min(tau + delta, 1.0)
        # level trained at step t = the one key of "chosen" (state.active is already the NEXT step's level)
        active = [int(max(s["chosen"], key=s["chosen"].get)) for s in steps]
        p = [float(s["p"][str(L)]) for s, L in zip(steps, active)]      # batch p at the trained level
        n_steps = len(steps)
        final = active[-1]
        arrive = next(t for t in range(n_steps) if all(x == final for x in active[t:]))
        # moves: the level trained at step t differs from step t-1
        prev = active[0]
        for t in range(1, n_steps):
            if active[t] != prev:
                trig = [round(p[t - 2], 3), round(p[t - 1], 3)]
                moves_rows.append({"run": rid, "tau": tau, "trigger": h, "step": t, "from": prev, "to": active[t],
                                   "p_two_steps_before_move": f"{trig[0]} {trig[1]}",
                                   "steps_at_from_level": t - next(s for s in range(t) if all(x == prev for x in active[s:t]))})
                prev = active[t]
        # per-level segments: how fast did p rise while the arm trained there?
        t0 = 0
        for t in range(1, n_steps + 1):
            if t == n_steps or active[t] != active[t0]:
                seg = np.array(p[t0:t])
                k = min(3, len(seg))
                rise = (seg[-k:].mean() - seg[:k].mean())
                segs.append({"run": rid, "tau": tau, "level": active[t0], "first_step": t0, "steps": t - t0,
                             "p_first3": round(float(seg[:k].mean()), 3), "p_last3": round(float(seg[-k:].mean()), 3),
                             "rise_per_10_steps": round(float(rise) / max(t - t0 - k, 1) * 10, 4),
                             "final": t == n_steps})
                t0 = t
        pf = np.array(p[arrive:])
        tail = np.array(p[-20:])
        last30 = np.array(p[-30:])
        slope = float(np.polyfit(np.arange(len(last30)), last30, 1)[0] * 10)
        above = pf > h
        rows.append({
            "run": rid, "tau": tau, "trigger_up": h, "start_level": active[0], "final_level": final,
            "arrival_step": arrive, "steps_climbing": arrive, "steps_at_final": n_steps - arrive,
            "share_at_final": round((n_steps - arrive) / n_steps, 3),
            "n_moves": int(steps[-1]["state"]["moves"]), "moves_refused_by_memory": int(steps[-1]["state"]["blocked"]),
            "plateau_p_last20": round(float(tail.mean()), 3), "sd_last20": round(float(tail.std(ddof=1)), 3),
            "max_p_at_final": round(float(pf.max()), 3),
            "steps_above_trigger_at_final": int(above.sum()), "longest_streak_above_trigger": longest_run(above),
            "slope_last30_per10steps": round(slope, 4),
            "plateau_minus_trigger": round(float(tail.mean()) - h, 3),
        })

    out = REPO / "results"
    for name, data in [("stall_law.csv", rows), ("stall_law_moves.csv", moves_rows), ("stall_law_segments.csv", segs)]:
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print("STALL LAW (batch p at the trained level, training reward, T = 1)")
    keys = ["run", "trigger_up", "final_level", "arrival_step", "steps_at_final", "plateau_p_last20", "sd_last20",
            "max_p_at_final", "steps_above_trigger_at_final", "longest_streak_above_trigger",
            "slope_last30_per10steps", "moves_refused_by_memory"]
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))
    print("\nMOVES")
    for m in moves_rows:
        print(m)
    print("\nSEGMENTS (batch p at the trained level while there)")
    for g in segs:
        print(g)

    # Independent plateau per level: arms that train many levels at once (fixed band, op 11-17 control), batch p
    # over their last 20 steps. Then the law's prediction for any trigger h: the first level >= 11 whose plateau < h.
    plat = []
    for rid in ["fixed_band_s0", "fixed_band_s1", "uniform_11_17_s0", "uniform_11_17_s1", "uniform_11_17_s2"]:
        _, st = steps_of(a.runs_root / rid / "sampler_log.jsonl")
        per = {}
        for s in st:
            for L, v in s["p"].items():
                per.setdefault(int(L), []).append(float(v))
        for L, v in sorted(per.items()):
            v = np.array(v)
            plat.append({"run": rid, "level": L, "plateau_p_last20": round(float(v[-20:].mean()), 3),
                         "slope_last30_per10steps": round(float(np.polyfit(np.arange(30), v[-30:], 1)[0] * 10), 4)})
    with (out / "stall_law_plateaus.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(plat[0]))
        w.writeheader()
        w.writerows(plat)
    pl = {L: float(np.mean([r["plateau_p_last20"] for r in plat if r["level"] == L]))
          for L in sorted({r["level"] for r in plat})}
    print("\nINDEPENDENT PLATEAUS (fixed band + control, last 20 steps, mean over runs): "
          + "  ".join(f"op{L} {v:.3f}" for L, v in pl.items()))
    for h in (0.6, 0.4, 0.3, 0.25, 0.2, 0.15):
        stop = next((L for L in sorted(pl) if L >= 11 and pl[L] < h), None)
        print(f"  trigger {h:.2f}: predicted final level op{stop}" if stop else f"  trigger {h:.2f}: beyond op17")

    # Tracking-set preview: op 15-20 success rate over training, tau=0.1 vs control.
    track = []
    for arm, runs in TRACK_ARMS.items():
        for rid in runs:
            for r in csv.DictReader((a.runs_root / rid / "edge_speed.csv").open()):
                for L in range(15, 21):
                    track.append({"arm": arm, "run": rid, "step": int(r["step"]), "level": L,
                                  "p": float(r[f"p_op{L}"])})
    with (out / "stall_law_tracking.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(track[0]))
        w.writeheader()
        w.writerows(track)
    print("\nTRACKING SET p (32 problems x 8 answers, T = 1), mean over 3 seeds: tau=0.1 / control")
    steps = sorted({t["step"] for t in track})
    for grp, levels in [("op15-17", range(15, 18)), ("op18-20", range(18, 21))]:
        print(f"  {grp}: " + "  ".join(
            f"s{s}: " + "/".join(
                f"{np.mean([t['p'] for t in track if t['arm'] == arm and t['step'] == s and t['level'] in levels]):.3f}"
                for arm in TRACK_ARMS) for s in steps))
    for L in range(15, 21):
        print(f"  op{L}: " + "  ".join(
            f"s{s}: " + "/".join(
                f"{np.mean([t['p'] for t in track if t['arm'] == arm and t['step'] == s and t['level'] == L]):.3f}"
                for arm in TRACK_ARMS) for s in steps))


if __name__ == "__main__":
    main()
