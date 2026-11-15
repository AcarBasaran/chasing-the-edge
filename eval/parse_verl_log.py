"""Extract per-step metrics from a verl console log and check smoke-run health.

verl's console logger prints one line per step:  "step:3 - actor/pg_loss:0.01 - critic/score/mean:0.2 ..."
Usage:
    $EDGE_PY eval/parse_verl_log.py <slurm log> [--csv out.csv] [--sampler-log sampler_log.jsonl]
Exit code 0 iff: >=1 step parsed, every loss finite, mean score > 0 on at least one step,
and (if a sampler log is given) the chosen level changed at least once.
"""

import argparse
import csv
import json
import math
import re
import sys

KEYS = ["critic/score/mean", "critic/rewards/mean", "actor/pg_loss", "actor/kl_loss", "actor/entropy",
        "actor/grad_norm", "response_length/mean", "response_length/clip_ratio", "timing_s/step", "timing_s/gen"]
STEP_RE = re.compile(r"step:(\d+) - (.*)$")


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = STEP_RE.search(line)
        if not m:
            continue
        rec = {"step": int(m.group(1))}
        for part in m.group(2).split(" - "):
            if ":" in part:
                k, v = part.split(":", 1)
                v = re.sub(r"^np\.\w+\((.*)\)$", r"\1", v.strip())   # verl prints np.float64(x)
                try:
                    rec[k.strip()] = float(v)
                except ValueError:
                    pass
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--csv")
    ap.add_argument("--sampler-log")
    a = ap.parse_args()
    rows = parse(a.log)
    ok = bool(rows)
    print(f"{len(rows)} steps parsed")
    for r in rows:
        print("  " + "  ".join(f"{k.split('/')[-1] if k != 'step' else 'step'}={r.get(k, float('nan')):.4g}"
                               for k in ["step"] + KEYS[:7]))
    losses = [r.get(k) for r in rows for k in ("actor/pg_loss", "actor/kl_loss") if k in r]
    finite = all(math.isfinite(x) for x in losses) and bool(losses)
    nonzero = any(r.get("critic/score/mean", 0) > 0 for r in rows)
    print(f"losses finite: {finite} ({len(losses)} values)   reward nonzero on some step: {nonzero}")
    ok = ok and finite and nonzero
    if a.sampler_log:
        steps = [json.loads(l) for l in open(a.sampler_log) if '"event": "step"' in l]
        levels = []
        for s in steps:
            ch = s["chosen"]
            top = max(ch, key=ch.get) if ch else None
            levels.append(top)
            print(f"  sampler step {s['step']}: chosen={ch} p={ {k: round(v, 3) for k, v in s['p'].items()} } "
                  f"state={ {k: s['state'].get(k) for k in ('active', 'last_p', 'moves', 'blocked')} }")
        changed = len(set(levels)) > 1
        print(f"sampler steps logged: {len(steps)}   level changed: {changed} ({levels})")
        ok = ok and changed
    if a.csv and rows:
        keys = ["step"] + sorted({k for r in rows for k in r if k != "step"})
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print("SMOKE OK" if ok else "SMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
