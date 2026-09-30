"""OpenAI API 사용량 집계 (반 공용 키 비용 확인용).

LLM 응답 캐시(SQLite)에는 호출마다 모델명·토큰 수가 남는다. 캐시를 지우거나 --fresh 로 새 캐시를 쓰지 않았다면
캐시가 곧 사용 기록이다(같은 프롬프트를 다시 부르면 캐시가 답해 과금되지 않는다).
용도는 캐시 키에 남은 응답 스키마 이름으로 나눈다. 결과: outputs/eval/api_usage.md

    python -m eval.api_usage
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict

from core.config import ROOT, path
from core.cost import model_price

PURPOSE = {
    "Answers": "투자 판단 평가표 판정", "Eligibility": "적격성 판정", "Draft": "보고서 작성", "TechAnalysis": "기술·팀 분석",
    "CompetitionAnalysis": "경쟁사 분석", "Grades": "RAG 관련성 판정", "FoundList": "후보 추출(발굴)",
    "MarketAnalysis": "시장성 분석", "Rewrite": "RAG 질의 재작성", "QA": "평가 질문 생성(build_qa)",
    "Check": "평가 질문 검수(build_qa)", "Verdict": "RAG 답변 판정(eval_judge)", "기타": "RAG 답변 생성(eval_judge)",
}


def _purpose(llm_key: str) -> str:
    m = re.search(r"response_format', <class '[\w.]+\.(\w+)'>", llm_key)
    name = m.group(1) if m else "기타"
    return PURPOSE.get(name, name)


def collect() -> dict:
    rows = defaultdict(lambda: {"calls": 0, "in": 0, "cached_in": 0, "out": 0, "models": set()})
    for db in sorted(ROOT.glob("**/llm_cache*.sqlite")):
        if ".venv" in db.parts:
            continue
        con = sqlite3.connect(db)
        for llm, resp in con.execute("select llm, response from full_llm_cache"):
            msg = json.loads(resp)["kwargs"]["message"]["kwargs"]
            meta = msg.get("response_metadata", {})
            usage = meta.get("token_usage") or {}
            r = rows[_purpose(llm)]
            r["calls"] += 1
            r["in"] += usage.get("prompt_tokens", 0)
            r["cached_in"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            r["out"] += usage.get("completion_tokens", 0)
            r["models"].add(meta.get("model_name", "?"))
    return rows


def main() -> None:
    rows = collect()
    lines = ["# OpenAI API 사용량 (LLM 응답 캐시 기준)", "",
             "| 용도 | 호출 | 입력 토큰 | 출력 토큰 | 비용(USD) | 모델 |", "|---|---|---|---|---|---|"]
    tot = {"calls": 0, "in": 0, "out": 0, "usd": 0.0}
    for k, r in sorted(rows.items(), key=lambda x: -x[1]["in"]):
        model = sorted(r["models"])[0]
        price = model_price(model)  # 가격표: config.yaml models.price_per_mtok
        usd = ((r["in"] - r["cached_in"]) * price[0] + r["cached_in"] * price[0] / 4 + r["out"] * price[1]) / 1e6
        tot["calls"] += r["calls"]; tot["in"] += r["in"]; tot["out"] += r["out"]; tot["usd"] += usd
        lines.append(f"| {k} | {r['calls']} | {r['in']:,} | {r['out']:,} | {usd:.3f} | {', '.join(sorted(r['models']))} |")
    lines.append(f"| **합계** | **{tot['calls']}** | {tot['in']:,} | {tot['out']:,} | **{tot['usd']:.2f}** | |")
    lines += ["", "- 저장소에 들어 있는 캐시(replay/)와 로컬 캐시(cache/, 저장소 제외)를 모두 센다.",
              "- 캐시에 저장되기 전에 실패한 호출은 빠진다. 실제로 API 를 부른 실행의 비용은 outputs/cost_history.jsonl 에도 남는다."]
    out = path("outputs/eval/api_usage.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
