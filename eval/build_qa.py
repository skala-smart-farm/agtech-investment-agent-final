"""검색 평가용 질문 세트 만들기.

코퍼스 조각을 문서별로 고르게 뽑아, 그 조각만 보고 답할 수 있는 구체적인 질문을 만든 뒤
Judge LLM 이 "질문이 구체적이고 이 조각으로 답이 되는가"를 다시 검사해 통과한 것만 남긴다.
정답 위치는 (문서, 페이지) 단위로 기록한다 → 조각 경계가 달라도 공정하게 비교된다.

    python -m eval.build_qa --n 40
"""
from __future__ import annotations

import argparse
import json
import random

from pydantic import BaseModel, Field

from core.config import get_config, path
from core.llm import structured
from rag.index import get_chunks


class QA(BaseModel):
    question: str = Field(description="조각의 구체적 사실(수치·고유명사·정책명)을 묻는 질문. 조각 문장을 그대로 베끼지 말 것")
    answer: str = Field(description="짧은 정답")


class Check(BaseModel):
    specific: bool = Field(description="질문이 이 조각을 특정할 만큼 구체적이면 true (일반론 질문이면 false)")
    answerable: bool = Field(description="조각만으로 정답을 확인할 수 있으면 true")


GEN = """아래는 애그테크·스마트농업 보고서의 한 조각이다. 투자 분석가가 실제로 검색할 법한 질문 1개와 짧은 정답을 만들어라.
- 조각에 있는 구체적 사실(수치, 연도, 기관·정책·기술 이름)을 묻는다.
- 질문만 보고도 어느 주제인지 알 수 있게 쓰되, 조각의 문장을 그대로 베끼지 않는다.
- 조각 언어와 상관없이 질문은 한국어로 쓴다(영문 고유명사는 그대로).
- 보고서 이름·발행기관·"~보고서에 따르면" 같은 출처 표현은 질문에 넣지 않는다 (실제 사용자는 출처를 모르고 묻는다).

문서: {title} ({publisher}, {year}) p.{page}
조각:
{text}"""

GEN_KEYWORD = """아래는 애그테크·스마트농업 보고서의 한 조각이다. 에이전트가 검색창에 넣을 법한 **키워드형 질의** 1개와 짧은 정답을 만들어라.
- 조각에 나오는 고유명사·정책명·기술명·수치 표현을 그대로 2~4개 넣은 짧은 질의 (예: "스마트농업법 시행 2024 육성지구 지정")
- 조각 언어와 상관없이 한국어로 쓰되 영문 고유명사는 그대로 둔다.
- 보고서 이름·발행기관은 넣지 않는다.

문서: {title} ({publisher}, {year}) p.{page}
조각:
{text}"""

CHK = """질문과 조각을 보고 판정하라.
질문: {q}
정답: {a}
조각:
{text}"""


def main(n: int, seed: int = 7, style: str = "natural") -> None:
    get_config()
    rng = random.Random(seed)
    chunks = [c for c in get_chunks() if c.metadata.get("kind") == "text" and len(c.page_content) > 350]
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.metadata["doc_id"], []).append(c)
    docs = sorted(by_doc)
    picks = []
    i = 0
    while len(picks) < n * 2 and any(by_doc.values()):  # 문서별로 번갈아 뽑아 편중 방지 (탈락분 대비 2배)
        d = docs[i % len(docs)]
        if by_doc[d]:
            picks.append(by_doc[d].pop(rng.randrange(len(by_doc[d]))))
        i += 1
    gen, chk = structured(QA), structured(Check, "judge")
    out = []
    for c in picks:
        m = c.metadata
        tmpl = GEN_KEYWORD if style == "keyword" else GEN
        qa: QA = gen.invoke(tmpl.format(title=m["title"], publisher=m["publisher"], year=m["year"], page=m["page"],
                                       text=c.page_content))
        ok: Check = chk.invoke(CHK.format(q=qa.question, a=qa.answer, text=c.page_content))
        if ok.specific and ok.answerable:
            out.append({"id": f"q{len(out) + 1:02d}", "question": qa.question, "answer": qa.answer,
                        "doc_id": m["doc_id"], "page": m["page"], "lang": m.get("lang", "ko")})
        if len(out) >= n:
            break
    f = path("data/eval/qa_set.jsonl" if style == "natural" else "data/eval/qa_keyword.jsonl")
    f.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    print(f"{len(out)}문항 저장 → {f} (후보 {len(picks)}개 중)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--style", choices=["natural", "keyword"], default="natural",
                    help="natural: 자연어 질문형(표현을 바꿔 물음) / keyword: 키워드형(고유명사·수치를 그대로 씀)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.n, a.seed, a.style)
