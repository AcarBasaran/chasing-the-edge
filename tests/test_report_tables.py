"""Tests for eval/report_tables.py: the report's tables come from the committed CSVs, with known values.

Imports only report_tables (stdlib + runs_io), never samplers/ (samplers/mock.py imports torch).
"""

import subprocess
import sys

import pytest

import report_tables
from runs_io import REPO


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    out = tmp_path_factory.mktemp("tables")
    subprocess.run([sys.executable, str(REPO / "eval" / "report_tables.py"), "--out-dir", str(out)], check=True)
    return {name: (out / f"{name}.tex").read_text(encoding="utf-8") for name in ("main_results", "decomposition")}


def _row(text, label):
    """Cells of the row starting with `label`, whitespace-stripped."""
    line = next(l for l in text.splitlines() if l.startswith(label))
    return [c.strip() for c in line.removesuffix(r"\\").split("&")]


def test_main_known_values(tables):
    t = tables["main_results"]
    assert t.lstrip("%").count(r"\begin{tabular}") == 1 and r"\caption" not in t and r"\label" not in t
    tau1 = _row(t, r"threshold \tauone")
    assert tau1[1] == "3" and tau1[4] == r"$\mathbf{95.0\pm1.7}$" and tau1[6] == r"24.7\%"
    fixed = _row(t, "fixed band")
    assert fixed[1:6] == ["2", r"$9.9\pm0.4$", r"$23.5\pm0.0$", r"$48.0\pm0.0$", r"$6.0\pm0.0$"]
    assert _row(t, r"threshold \taufive")[1:] == ["1", "0.0", "0.0", "0", "0", r"40.1\%"]
    # 0.2865 sits on a rounding boundary; the report says 28.6 (Python float formatting).
    assert _row(t, "control: uniform op 11--17")[6] == r"28.6\%"


def test_main_row_order(tables):
    labels = [l.split("&")[0].strip() for l in tables["main_results"].splitlines()
              if l.count("&") == 6 and not l.startswith(("&", "arm"))]
    assert labels == [label for _, label, _ in report_tables.ARMS]


def test_decomposition(tables):
    t = tables["decomposition"]
    assert r"hard ($+47.0$) & op 18--20 ($+21.7$)" in t
    assert _row(t, "harder problems")[1:] == [r"$+30.0$ (64\%)", r"$+14.0$ (65\%)"]
    assert _row(t, "right range")[1:] == [r"$+8.3$ (18\%)", r"$+4.0$ (18\%)"]
    assert _row(t, "schedule")[1:] == [r"$+8.7$ (18\%)", r"$+3.7$ (17\%)"]


def test_steps_telescope_to_total():
    data = report_tables.load(REPO / "results")
    _, parts = report_tables.decomposition(data)
    for col in ("hard", "18-20"):
        assert sum(s for (c, _), (_, s) in parts.items() if c == col) == pytest.approx(100.0)


def test_cell_formatting():
    assert report_tables.cell(22.7839, 0.2008, 3, bold=True) == r"$\mathbf{22.8\pm0.2}$"
    assert report_tables.cell(0.0, None, 1, count=True) == "0"
    assert report_tables.cell(0.0, None, 1) == "0.0"


def test_missing_arm_raises(tmp_path):
    for name in ("seeds_by_set.csv", "seeds_hard_levels.csv", "seeds_gradient_budget.csv"):
        lines = (REPO / "results" / name).read_text().splitlines(keepends=True)
        (tmp_path / name).write_text("".join(l for l in lines if not l.startswith("bandit,")))
    with pytest.raises(KeyError, match="bandit"):
        report_tables.load(tmp_path)


def test_no_torch_imported():
    # Fresh interpreter: in a full test run other modules (samplers.mock) have already imported torch.
    code = (f"import sys; sys.path.insert(0, {str(REPO / 'eval')!r}); import report_tables; "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
