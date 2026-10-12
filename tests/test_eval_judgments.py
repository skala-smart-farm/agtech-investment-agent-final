"""eval/eval_judgments.py 채점 검사: 비운 gold 칸은 빼고, 정확도·YES 정밀도·재현율을 계산한다."""
from __future__ import annotations

import csv
import json

from eval.eval_judgments import FIELDS, score


def test_score_skips_blank_and_counts(tmp_path):
    gold = tmp_path / "gold.csv"
    with open(gold, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for qid, g in [("M3", "UNKNOWN"), ("M4", "YES"), ("C1", "YES"), ("D3", "")]:
            w.writerow({k: "" for k in FIELDS} | {"candidate": "A", "qid": qid, "gold": g})
    log = tmp_path / "run_log.json"
    rows = [{"qid": q, "answer": a} for q, a in [("M3", "YES"), ("M4", "YES"), ("C1", "UNKNOWN"), ("D3", "YES")]]
    log.write_text(json.dumps({"evaluations": [{"name": "A", "rows": rows}]}), encoding="utf-8")
    s = score(str(log), str(gold), str(tmp_path / "out.json"))
    assert s["n"] == 3 and s["accuracy"] == round(1 / 3, 3)          # D3 는 gold 가 비어 빠짐
    assert s["yes_precision"] == 0.5 and s["yes_recall"] == 0.5      # YES 2개 중 M4 만 맞음, 정답 YES 2개 중 M4 만 찾음
    assert s["wrong_by_qid"] == {"M3": 1, "C1": 1}
