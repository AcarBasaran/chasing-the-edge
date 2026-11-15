"""Join Gate-1 per-level outputs into results/base_by_level_summary.csv and print markdown.

Columns: level, p_eval (paper criterion), p_train (training reward), pass@128,
gen_tok_p95 (all of the model's own generations), correct_tok_p95, gold_tok_p95,
paper_pass@1 / paper_pass@128 (Fig. 9 per-op panel, where the paper shows that level).
"""

import csv
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"


def read(name):
    with (R / name).open() as fh:
        return {int(r["level"] if "level" in r else r["op"]): r for r in csv.DictReader(fh)}


def main():
    p = read("base_model_p_by_level.csv")
    t = read("base_model_correct_token_lengths_by_level.csv")
    ref = yaml.safe_load((REPO / "configs" / "gate1_base_eval.yaml").read_text())["paper_reference"]["fig9_final_ckpt"]
    rows = []
    for L in range(2, 21):
        r = ref.get(f"op{L}", {})
        rows.append({
            "level": L,
            "p_eval": float(p[L]["p_L"]),
            "p_train": float(p[L]["p_train_reward"]),
            "pass@128": float(p[L]["pass@128"]),
            "gen_tok_p95": float(t[L]["all_samples_p95"]),
            "correct_tok_p95": t[L]["p95"] and float(t[L]["p95"]),
            "gold_tok_p95": float(t[L]["gold_resp_p95"]),
            "trunc_rate": float(p[L]["trunc_rate"]),
            "paper_pass@1": r.get("pass@1", ""),
            "paper_pass@128": r.get("pass@128", ""),
        })
    with (R / "base_by_level_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("| L | p eval | p train | pass@128 | gen p95 tok | correct p95 tok | gold p95 tok | paper p@1 / p@128 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        paper = f"{r['paper_pass@1']} / {r['paper_pass@128']}" if r["paper_pass@1"] != "" else ""
        ct = f"{r['correct_tok_p95']:.0f}" if r["correct_tok_p95"] != "" else "—"
        print(f"| {r['level']} | {r['p_eval']:.4f} | {r['p_train']:.4f} | {r['pass@128']:.3f} | "
              f"{r['gen_tok_p95']:.0f} | {ct} | {r['gold_tok_p95']:.0f} | {paper} |")


if __name__ == "__main__":
    main()
