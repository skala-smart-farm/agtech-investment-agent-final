"""실행 기록(run_log.json)에서 동종 기준 집단 파일(계약 C10 형식)을 만든다.

본 실행용 data/reference_class.json 은 app.py --calibrate 가 만든다. 이 스크립트는 이미 있는 실행 기록으로
같은 형식의 파일을 다시 만들 때 쓴다(예: v1 판정으로 만든 테스트 픽스처 tests/fixtures/reference_v1.json).
- v1 run_log: evaluations[].rows[].answer 를 신호(YES 1 · NO −1 · UNKNOWN 0 · N/A 없음)로 바꾼다. v1 기록의 평가 행에는
  지역·단계·세부 분야가 없어 screened(지역·단계)와 '[평가 n] 이름 (지역, 단계, 분야)' 로그에서 채운다.
- v2 run_log: evaluations[] 의 rows(또는 scorecard.rows)에서 x(없으면 answer)를 쓴다.
--rows-out 을 주면 회사별 판정 행(dim·qid·answer·x)도 따로 쓴다(배수 재계산 테스트용).
API 를 부르지 않는다.

    uv run python -m eval.build_reference_class --from-runlog outputs/run_log.json \\
        --out tests/fixtures/reference_v1.json --rows-out tests/fixtures/rows_v1.json
"""
from __future__ import annotations

import argparse
import json
import re

from agents.decision import _reference_data, _write_json, _x
from core.config import ROOT
from core.judge import load_rubric

LOG_EVAL = re.compile(r"^\[평가 \d+\] (.+) \(([^,]+), ([^,]+), ([^)]+)\)$")


def evaluations_from_runlog(run_log: dict) -> list[dict]:
    """run_log 의 평가 기록 → {name, region, stage, segment_id, rows[{dim, qid, answer, x}]} 목록 (평가 순서 유지)."""
    dim_of = {q["id"]: d["id"] for d in load_rubric()["dimensions"] for q in d["questions"]}
    screened = {s["name"]: s for s in run_log.get("screened", [])}
    logged = {m.group(1): m.groups()[1:] for line in run_log.get("log", []) if (m := LOG_EVAL.match(line))}
    out = []
    for e in run_log.get("evaluations", []):
        rows = e.get("rows") or (e.get("scorecard") or {}).get("rows") or []
        region, stage, seg = logged.get(e["name"], (None, None, None))
        s = screened.get(e["name"], {})
        out.append({"name": e["name"], "region": e.get("region") or region or s.get("region"),
                    "stage": e.get("stage") or stage or s.get("stage"), "segment_id": e.get("segment_id") or seg,
                    "rows": [{"dim": r.get("dim") or dim_of[r["qid"]], "qid": r["qid"], "answer": r["answer"], "x": _x(r)}
                             for r in rows]})
    return out


def main(src: str, out: str, rows_out: str | None) -> None:
    path = ROOT / src if not src.startswith("/") else src
    run_log = json.loads(open(path, encoding="utf-8").read())
    evals = evaluations_from_runlog(run_log)
    data = _reference_data(evals, run_log.get("run_date", ""), f"eval/build_reference_class.py --from-runlog {src}")
    _write_json(out, data)
    print(f"기준 집단 {data['n']}곳 → {out}")
    if rows_out:
        _write_json(rows_out, {"run_date": run_log.get("run_date", ""), "source": src, "companies": evals})
        print(f"판정 행 {len(evals)}곳 → {rows_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-runlog", required=True, help="run_log.json 경로 (프로젝트 루트 기준)")
    ap.add_argument("--out", required=True, help="기준 집단 파일 경로 (계약 C10 형식)")
    ap.add_argument("--rows-out", help="회사별 판정 행을 쓸 경로 (선택)")
    a = ap.parse_args()
    main(a.from_runlog, a.out, a.rows_out)
