"""LLM-as-a-Judge 로 Agentic RAG 최종 답변 평가 (O/X 이진 판정).

평가 대상은 v2 서브그래프(rag.agentic_rag.answer_question)가 생성·점검까지 마친 최종 답변이다.
라우팅 시연을 위해 이 평가에서만 직접 답(allow_direct=True)을 허용한다(분석 에이전트 호출은 끔).
- Relevance: 답변이 질문에 맞게 답했는가
- Faithfulness: 답변의 내용이 검색된 근거로 뒷받침되는가
- Correctness: 답변이 정답과 같은 사실을 말하는가
1~5점 척도 대신 이진 판정을 써서 판정 기준을 분명히 한다.
행마다 경로(route)·재작성·재생성 횟수·상태(status)를 남기고, 요약에 경로 분포와 재작성·웹 보완·재생성·not_found 비율을 더한다.

    python -m eval.eval_judge --n 20
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from pydantic import BaseModel, Field

from core.config import get_config, path
from core.llm import structured
from rag.agentic_rag import answer_question
from tools.sources import SourceRegistry


class Verdict(BaseModel):
    relevance: bool = Field(description="답변이 질문이 묻는 것에 직접 답하면 true")
    faithfulness: bool = Field(description="답변의 모든 사실이 근거에 있으면 true (근거 밖 내용이 있으면 false)")
    correctness: bool = Field(description="답변이 정답과 같은 사실을 말하면 true")
    reason: str


JUDGE = """너는 RAG 시스템 평가자다. 각 기준을 O/X(true/false)로 판정하라.
질문: {q}
정답: {gold}
시스템 답변: {ans}
시스템이 사용한 근거:
{ctx}"""

ROUTES = ("docs", "web", "both", "direct")


def summarize_rows(rows: list[dict]) -> dict:
    """judge_eval.json 요약 (계약 키: relevance, faithfulness, correctness, n, route_dist, rewrite_rate,
    web_fallback_rate, regen_rate, not_found_rate). 비율은 문항 기준(해당 일이 한 번이라도 있었던 문항 수 ÷ n)."""
    k = len(rows)
    if not k:
        return {"n": 0}

    def rate(pred) -> float:
        return round(sum(1 for r in rows if pred(r)) / k, 3)

    routes = Counter(r["route"] for r in rows)
    return {"relevance": rate(lambda r: r["relevance"]), "faithfulness": rate(lambda r: r["faithfulness"]),
            "correctness": rate(lambda r: r["correctness"]), "n": k,
            "route_dist": {x: routes.get(x, 0) for x in ROUTES},        # agent 가 고른 경로별 문항 수
            "rewrite_rate": rate(lambda r: r["rewrites"] > 0),           # 질의 재작성이 있었던 문항
            "web_fallback_rate": rate(lambda r: r["web_fallback"]),      # 문서가 부족해 교정형 웹 보완을 한 문항
            "regen_rate": rate(lambda r: r["regenerations"] > 0),        # 점검(not_grounded) 뒤 답변을 다시 쓴 문항
            "not_found_rate": rate(lambda r: r["status"] == "not_found")}


def main(n: int) -> None:
    get_config()
    qa = [json.loads(l) for l in path("data/eval/qa_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    judge = structured(Verdict, "judge")
    rows = []
    for q in qa:
        reg = SourceRegistry()
        out = answer_question(q["question"], "평가 질문", reg, "eval", allow_direct=True)
        ctx = reg.brief(out["evidence_ids"], 900) or "(근거 없음)"
        v: Verdict = judge.invoke(JUDGE.format(q=q["question"], gold=q["answer"], ans=out["answer"], ctx=ctx))
        rows.append({"id": q["id"], "question": q["question"], "answer": out["answer"], "gold": q["answer"],
                     "route": out["route"], "status": out["status"], "rewrites": out["rewrites"],
                     "regenerations": out["regenerations"],
                     "web_fallback": any("web_fallback" in t for t in out["trace"]),
                     "cited_ids": out["cited_ids"], "evidence_ids": out["evidence_ids"], **v.model_dump()})
        r = rows[-1]
        print(f"{q['id']}: R={v.relevance} F={v.faithfulness} C={v.correctness} "
              f"(경로 {r['route']}, 재작성 {r['rewrites']}회, 재생성 {r['regenerations']}회, {r['status']})")
    summary = summarize_rows(rows)
    path("outputs/eval/judge_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                                                               indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    main(ap.parse_args().n)
