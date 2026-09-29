"""📊 시장성 평가 에이전트 (Scorecard '시장성 25%', 문항 M1~M4, Agentic RAG 의 주 사용처).

  1) 질의 분해: LLM 이 세부 분야의 시장성 질문을 검색 가능한 하위 질문 3~6개로 나눈다 (v1 프롬프트·기본 질문 그대로).
     투자 동향 하위 질문이 없으면 기본 투자 동향 질문 1개를 더한다 (투자 사이클 확인).
  2) 하위 질문마다 Agentic RAG(answer_question): 문서 검색·웹 검색 중 어디서 찾을지는 RAG 서브그래프의 agent 노드가 고르고,
     근거를 채점·재작성·웹 보완한 뒤 인용 달린 답을 만든다. 검색 없이 바로 답하는 경로는 끈다(수치 환각 방지).
  3) 종합: 하위 답변과 근거로 시장 분석을 쓰고, 시장 규모 대표 수치를 구조화한다(size). 달러 환산(value_usd_m)은 코드가 한다.
하위 질문 수(최대 6 + 투자 동향 1)와 서브그래프의 재작성·재생성 상한으로 반복 횟수를 제한한다.
분해·하위 답변 기록은 rag_traces 에 남긴다. 같은 세부 분야는 분석을 한 번만 하고(market_cache) 시장성 판정만 후보마다 다시 한다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agents.tech import CITE_ID, candidate_line, evidence_blocks, judge_own
from core.config import get_config, get_segment, run_date
from core.llm import structured
from core.llm_bounded import bounded
from core.prompts import render
from rag import agentic_rag as rag  # answer_question·corpus_catalog 는 호출 시점에 찾는다
from tools.sources import SourceRegistry

AGENT = "market"
MIN_SUBQ, MAX_SUBQ = 3, 6
# LLM 이 하위 질문을 3개 미만으로 내면 채우는 기본 질문 (분해 실패 대비)
DEFAULT_SUBQ = [("국내 스마트농업·스마트팜 시장 규모 (기관별 추정치, 기준 연도)", "시장 규모"),
                ("{seg} 글로벌 시장 규모와 연평균 성장률 (2024년 이후 발표)", "성장률"),
                ("애그테크 벤처 투자 동향 ({prev}년 투자액 증감, 국가별 증감)", "투자 동향")]


class SubQuestion(BaseModel):
    question: str = Field(description="검색할 수 있는 구체 하위 질문 (지표·지역·연도 포함)")
    purpose: str = Field(description="분석 항목: 시장 규모 / 성장률 / 수요 / 지불 의향 / 정책·규제 / 도입 장벽 / 투자 동향")


class Decomposition(BaseModel):
    sub_questions: list[SubQuestion] = Field(description="하위 질문 3~6개")


class MarketSize(BaseModel):
    value: float | None = Field(description="대표 시장 규모 수치 (unit 단위로 환산한 숫자). 근거에 수치가 없으면 null")
    unit: Literal["USD_M", "KRW_EOK", ""] = Field(
        description="USD_M = 백만 달러(10억 달러 = 1000), KRW_EOK = 억 원(1조 원 = 10000). 수치가 없으면 빈 문자열")
    year: int | None = Field(description="수치의 기준 연도 (전망치면 발표 연도), 모르면 null")
    scope: str = Field(description="수치가 가리키는 시장 범위 (예: '국내 스마트팜 시장', '글로벌 농업 로봇 시장')")
    source_id: str = Field(description="이 수치가 적힌 근거 id 1개, 없으면 빈 문자열")
    raw: str = Field(description="근거에 적힌 수치 표현 그대로 (예: '5조 9,000억 원'), 없으면 빈 문자열")


class MarketAnalysis(BaseModel):
    market_size: str = Field(description="국내·글로벌 시장 규모와 기준 연도·추정 기관 (수치마다 근거 id)")
    size: MarketSize = Field(description="세부 분야에 가장 가까운 시장의 대표 규모 수치 1개 (최신 기준 연도 우선)")
    growth: str = Field(description="성장률·전망과 발표 연도 (근거 id), 지난 기간 전망은 '(YYYY년 발표 전망)'")
    growth_rate_pct: float | None = Field(description="대표 연평균 성장률(CAGR) % 숫자, 근거에 없으면 null")
    growth_year: int | None = Field(description="그 성장률을 발표한 연도, 모르면 null")
    growth_source_id: str = Field(description="그 성장률이 적힌 근거 id, 없으면 빈 문자열")
    investment_trend: str = Field(description="최근 애그테크 투자 동향: 연도·증감률·국가별 차이 (근거 id), 없으면 '확인 불가'")
    demand_drivers: list[str] = Field(description="수요 요인과 고객의 실제 문제 (근거 id)")
    willingness_to_pay: str = Field(description="농가·기업의 지불 의향·도입 사례 (근거 id), 없으면 '확인 불가'")
    policy_regulation: str = Field(description="정책 지원과 규제 (근거 id)")
    adoption_barriers: list[str] = Field(description="도입 장벽·시장 리스크 (근거 id)")
    evidence_ids: list[str]


def plan(seg: dict, date: str, catalog: str) -> tuple[str, list[SubQuestion], bool, bool]:
    """큰 질문 → 하위 질문(최대 6개). 분해가 3개 미만이면 기본 질문으로 채우고, 투자 동향 질문이 없으면 1개 더한다.
    반환 (큰 질문, 하위 질문, 기본 질문으로 채웠는지, 투자 동향 질문을 더했는지)."""
    question = (f"{seg['name']} 분야 스타트업의 시장성: 국내·글로벌 시장 규모와 성장률, 수요 요인, 지불 의향, "
                f"정책·규제, 도입 장벽, 최근 투자 동향")
    dec: Decomposition = structured(Decomposition).invoke(
        render("market_decompose", segment=seg["name"], question=question, run_date=date, catalog=catalog))
    subs = [s for s in dec.sub_questions if s.question.strip()][:MAX_SUBQ]
    padded = len(subs) < MIN_SUBQ
    for q, purpose in DEFAULT_SUBQ:
        if len(subs) >= MIN_SUBQ:
            break
        subs.append(SubQuestion(question=q.format(seg=seg["name"], prev=int(date[:4]) - 1), purpose=purpose))
    added = not any("투자" in s.purpose for s in subs)
    if added:  # 투자 사이클(최근 투자액 증감)은 시장 성장성 판단에 꼭 필요하다
        q, purpose = DEFAULT_SUBQ[2]
        subs.append(SubQuestion(question=q.format(prev=int(date[:4]) - 1), purpose=purpose))
    return question, subs, padded, added


def to_usd_m(value: float | None, unit: str, fx_krw_per_usd: float) -> float | None:
    """시장 규모 수치를 백만 달러로 환산한다. KRW_EOK(억 원) → value × 1억 ÷ 환율 ÷ 100만."""
    if value is None:
        return None
    if unit == "USD_M":
        return round(value, 1)
    if unit == "KRW_EOK":
        return round(value * 1e8 / fx_krw_per_usd / 1e6, 1)
    return None


def size_record(s: MarketSize, valid: set[str]) -> dict:
    """구조화한 시장 규모 + 달러 환산값. 근거 id 가 저장소에 없으면 source_id·value_usd_m 을 None 으로 둔다(근거 없는 수치 환산 안 함)."""
    d = s.model_dump()
    d["source_id"] = s.source_id if s.source_id in valid else None
    fx = get_config().roi.fx_krw_per_usd
    d["value_usd_m"] = to_usd_m(s.value, s.unit, fx) if d["source_id"] else None
    return d


def _analysis_text(c: dict, m: dict, tech: dict) -> str:
    """시장성 판정에 넘기는 요약문. M3(경제적 효과)·M4(수익 모델)는 회사 문항이라 기술 요약의 제품·장점을 함께 넣는다."""
    s = m.get("size") or {}
    size = (f"{s.get('raw')} ({s.get('scope')}, {s.get('year')}) [{s.get('source_id')}]" if s.get("source_id")
            else "근거 있는 대표 수치 없음")
    return "\n".join([
        candidate_line(c),
        f"[기술 요약] 제품: {tech.get('product') or '확인 불가'} / 장점: {'; '.join(tech.get('pros') or []) or '확인 불가'}",
        f"[시장] 규모: {m['market_size']} / 대표 수치: {size} / 성장: {m['growth']}",
        f"[투자 동향] {m['investment_trend']}",
        f"[수요] {'; '.join(m['demand_drivers']) or '확인 불가'} / 지불 의향: {m['willingness_to_pay']}",
        f"[정책·규제] {m['policy_regulation']} / 도입 장벽: {'; '.join(m['adoption_barriers']) or '확인 불가'}",
    ])


def market_node(state: dict) -> dict:
    c = state["current"]
    seg = get_segment(c["segment_id"])
    reg = SourceRegistry(state.get("registry"))
    cache = state.get("market_cache", {})
    if seg["id"] in cache:  # 같은 세부 분야: 분석은 재사용하고, 시장성 판정은 이 후보 기준으로 다시 한다
        out = dict(cache[seg["id"]])
        out["criterion"] = judge_own("market", state, reg, out["pool_ids"], _analysis_text(c, out, state.get("tech", {})))
        msg = (f"[시장성] {seg['name']}: 분석 캐시 재사용 → 시장성 YES {out['criterion']['yes']}/{out['criterion']['n']}")
        print(msg)
        return {"market": out, "registry": reg.data, "log": [msg]}

    date = state.get("run_date") or run_date()
    question, subs, padded, added = plan(seg, date, rag.corpus_catalog())
    traces = [{"agent": AGENT, "question": question, "step": "decompose", "trace": [],
               "sub_questions": [s.question for s in subs], "padded_with_defaults": padded,
               "added_investment_question": added}]
    ids: list[str] = []
    answers = []
    for s in subs:
        r = rag.answer_question(s.question, s.purpose, reg, AGENT, allow_direct=False)
        answers.append({"question": s.question, "purpose": s.purpose, "answer": r.get("answer") or "",
                        "route": r.get("route"), "status": r.get("status"), "evidence_ids": list(r["evidence_ids"])})
        traces.append({"agent": AGENT, "question": s.question, "purpose": s.purpose, "step": "answer",
                       "route": r.get("route"), "status": r.get("status"), "rewrites": r.get("rewrites", 0),
                       "regenerations": r.get("regenerations", 0), "trace": r.get("trace", [])})
        ids += r["evidence_ids"]
    ids = list(dict.fromkeys(ids))
    listing = "\n\n".join(f"[{n}] {a['question']} (용도: {a['purpose']}, 경로 {a['route']}, 상태 {a['status']})\n"
                          f"{a['answer'] or '(생성 답변 없음 — 아래 근거를 직접 읽을 것)'}"
                          for n, a in enumerate(answers, 1))
    evidence = "\n\n".join(evidence_blocks(reg, ids, keys=[], max_chars=700).values())
    res: MarketAnalysis = bounded(MarketAnalysis).invoke(
        render("market", segment=seg["name"], run_date=date, answers=listing, evidence=evidence))
    valid = set(ids)
    out = res.model_dump()
    out["size"] = size_record(res.size, valid)
    out["growth_source_id"] = res.growth_source_id if res.growth_source_id in valid else None
    refs = list(res.evidence_ids) + [i for a in answers for i in CITE_ID.findall(a["answer"])]
    out["evidence_ids"] = [i for i in dict.fromkeys(refs) if i in valid]
    out["pool_ids"] = ids
    out["segment"], out["segment_id"] = seg["name"], seg["id"]
    out["answers"] = answers
    crit = judge_own("market", state, reg, ids, _analysis_text(c, out, state.get("tech", {})))
    routes = {k: sum(a["route"] == k for a in answers) for k in ("docs", "web", "direct")}
    msg = (f"[시장성] {seg['name']}: 하위 질문 {len(subs)}개{' (투자 동향 추가)' if added else ''} "
           f"(문서 {routes['docs']}·웹 {routes['web']}·직접 {routes['direct']}, "
           f"grounded {sum(a['status'] == 'grounded' for a in answers)}개), 대표 규모 "
           f"{out['size']['value_usd_m'] if out['size']['value_usd_m'] is not None else '없음'}(USD_M), "
           f"근거 {len(out['evidence_ids'])}건 (문서 {sum(i.startswith('D') for i in out['evidence_ids'])}건) "
           f"→ 시장성 YES {crit['yes']}/{crit['n']}")
    print(msg)
    return {"registry": reg.data, "market": {**out, "criterion": crit}, "market_cache": {seg["id"]: out},
            "log": [msg], "rag_traces": traces}
