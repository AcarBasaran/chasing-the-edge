import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from edge_speed import edge  # noqa: E402


def test_edge_interpolates_crossing():
    p = {10: 0.9, 11: 0.5, 12: 0.1, 13: 0.0}
    assert edge(p, 0.5) == 11.0
    assert math.isclose(edge(p, 0.3), 11.5)
    assert edge({2: 0.2, 3: 0.1}, 0.5) is None          # below tau everywhere
    assert edge({19: 0.9, 20: 0.8}, 0.5) == 20.0         # censored at the top


def test_edge_speed_on_synthetic_log(tmp_path):
    # logistic p_L(t) = 1/(1+exp(L - s(t))), s = 11.5 + t/50  -> edge moves 0.02 lv/step
    lines = []
    for t in range(0, 101, 10):
        s = 11.5 + t / 50
        kv = " - ".join(f"val-core/difficulty-5B/{L}/acc/mean@8:{1 / (1 + math.exp(L - s)):.6f}" for L in range(2, 21))
        lines.append(f"step:{t} - {kv}")
    log = tmp_path / "train.log"
    log.write_text("\n".join(lines) + "\n")
    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "eval" / "edge_speed.py"), str(log)],
                         capture_output=True, text=True, check=True).stdout
    import re
    speed = float(re.search(r"tau=0.5: .*?speed ([0-9.]+)", out).group(1))
    assert abs(speed - 0.02) < 0.002, out
