"""P4 분석 에이전트(창업자·기술 요약·시장성·경쟁사) 검사. 키·네트워크 없이 돈다.

web_search·enrich·structured(LLM)·answer_question·judge_dimension·make_tools·tips_agtech 는 가짜로 바꿔 끼운다.
- 검색 쿼리: 창업자 + 기술 요약 쿼리 = v1 tech_node 쿼리 (글자·deep 까지 같아야 검색 캐시가 재사용된다)
- 노드마다 반환 키(계약 C2·C8), 근거 없는 인물·날짜 없는 마일스톤 제거, t0 우선순위, 판정 차원, 근거 저장소 전체 반환
- 통합 1건: 실제 answer_question_v1(재현용 캐시)로 기술 요약의 기준선 단계를 돌린다
"""
from __future__ import annotations

import ast
import hashlib
import subprocess

import pytest

import agents.competition as comp
import agents.founder as fd
import agents.market as mk
import agents.tech as tc
import core.judge
import rag.agentic_rag
import tools.agent_tools
import tools.channels
from core.config import ROOT
from core.prompts import render
from tools.sources import SourceRegistry

RUN_DATE = "2026-09-29"
# v1 tech_node 의 검색 쿼리 (v1-safe agents/tech.py 그대로, {name}=공식명, {q}=영문명)
V1_KR = [("{name} 대표 창업자 이력 인터뷰", True), ("{name} CTO 연구소장 기술 개발", False),
         ("{name} 수상 선정 혁신상 우수기업 출시", False),
         ("{name} 특허 등록 기술", False), ("{name} 실증 농가 효과 수확량 절감", True),
         ("{name} 매출 고객 농가 수 설치", True), ("{name} 협약 계약 공급 농협 지자체 수출", False)]
V1_GLOBAL = [("{q} founder CEO background interview", True), ("{q} CTO technology team", False),
             ("{q} patent", False), ("{q} field trial results yield savings", True),
             ("{q} revenue customers farms deployed", True), ("{q} partnership contract distribution", False)]
# v1 시장성 질의 분해의 큰 질문 (market_decompose 프롬프트 입력 = LLM 캐시 키)
V1_MARKET_Q = ("{seg} 분야 스타트업의 시장성: 국내·글로벌 시장 규모와 성장률, 수요 요인, 지불 의향, "
               "정책·규제, 도입 장벽, 최근 투자 동향")

OLD = {"Wold01": {"id": "Wold01", "key": "web:https://old.example.com/a", "kind": "web", "url": "https://old.example.com/a",
                  "site": "old", "title": "이전 근거", "date": "2025-01-01", "date_is_access": False,
                  "access_date": RUN_DATE, "author": None, "snippet": "이전 노드가 모은 근거", "agent": "discovery",
                  "query": "x", "body": ""}}


def wid(url: str) -> str:
    return SourceRegistry._id("W", "web:" + url)


# ── 가짜 부품
class FakeLLM:
    """structured(schema) 대역. 스키마 이름별로 준비한 답을 차례로 돌려주고, 받은 프롬프트를 기록한다."""

    def __init__(self, answers: dict[str, list]):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, schema, role: str = "generator"):
        llm = self

        class _Bound:
            def invoke(self, prompt: str):
                llm.prompts.append((schema.__name__, prompt))
                return llm.answers[schema.__name__].pop(0)
        return _Bound()


class FakeWeb:
    """web_search 대역: 쿼리마다 준비한 결과(url → 스니펫·게시일)를 근거로 등록하고 호출을 기록한다."""

    def __init__(self, results: dict[str, list[dict]] | None = None):
        self.results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, query, reg, agent, **kw):
        self.calls.append((query, kw))
        return [reg.add_web(r, agent, query, RUN_DATE) for r in self.results.get(query, [])]


@pytest.fixture
def judge_calls(monkeypatch):
    calls = []

    def fake_judge(dim_id, company, pool_ids, reg, analysis, run_date):
        calls.append({"dim": dim_id, "company": company["official_name"], "pool": list(pool_ids),
                      "analysis": analysis, "run_date": run_date})
        return {"dim": dim_id, "name": dim_id, "weight": 10, "owner": "x", "rows": [], "yes": 1, "no": 0,
                "unknown": 3, "na": 0, "n": 4, "rejected_yes": [], "quote_retried": 0}
    monkeypatch.setattr(core.judge, "judge_dimension", fake_judge)
    return calls


@pytest.fixture
def no_enrich(monkeypatch):
    for m in (fd, tc):
        monkeypatch.setattr(m, "enrich", lambda *a, **k: 0)


def kr_company(**kw) -> dict:
    c = {"name": "테스트팜", "official_name": "테스트팜", "name_en": "Testfarm", "region": "KR", "segment_id": "greenhouse",
         "stage": "Seed", "round_date": "2025-05", "round_amount": "10억 원", "founded_year": 2020, "founded_date": None,
         "ceo": "김대표", "one_line": "AI 온실 환경 제어", "evidence_ids": [],
         "nps": {"status": "matched", "ym": "2026-08", "members": 12, "first_date": "2020-06-01", "withdrawn": False}}
    c.update(kw)
    return c


def state_for(c: dict, **kw) -> dict:
    s = {"current": c, "registry": dict(OLD), "run_date": RUN_DATE, "founder": {}, "tech": {}, "market": {},
         "competition": {}}
    s.update(kw)
    return s


# ── 검색 쿼리: 캐시 보존
def _v1_queries_from_source() -> tuple[list, list]:
    """v1-safe agents/tech.py 의 queries 목록 두 개(국내·해외)를 AST 로 읽어 '{name}'·'{q}' 템플릿으로 되돌린다."""
    out = subprocess.run(["git", "show", "v1-safe:agents/tech.py"], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        pytest.skip("v1-safe 태그 없음")
    lists = []
    for n in ast.walk(ast.parse(out.stdout.decode("utf-8"))):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "queries" for t in n.targets):
            rows = []
            for tup in n.value.elts:
                s = "".join(v.value if isinstance(v, ast.Constant) else "{" + v.value.id + "}"
                            for v in tup.elts[0].values)
                rows.append((s, tup.elts[1].value))
            lists.append(rows)
    return lists[0], lists[1]


def test_query_union_equals_v1_tech_queries():
    kr = {"official_name": "{name}", "region": "KR"}
    gl = {"official_name": "{name}", "name_en": "{q}", "region": "GLOBAL"}
    assert fd.founder_queries(kr) + tc.tech_queries(kr) == V1_KR
    assert fd.founder_queries(gl) + tc.tech_queries(gl) == V1_GLOBAL
    v1_kr, v1_gl = _v1_queries_from_source()  # 상수 픽스처 자체도 v1 원본과 같은지
    assert (v1_kr, v1_gl) == (V1_KR, V1_GLOBAL)


def test_nodes_call_web_search_with_v1_arguments(monkeypatch, judge_calls, no_enrich):
    """쿼리 문자열뿐 아니라 검색 인자(topic·recent·deep·raw)도 v1 과 같아야 캐시 키가 같다."""
    web = FakeWeb()
    for m in (fd, tc):
        monkeypatch.setattr(m, "web_search", web)
    monkeypatch.setattr(fd, "structured", FakeLLM({"FounderAnalysis": [fd.FounderAnalysis(
        people=[], milestones=[], team_assessment="확인 불가", evidence_ids=[])]}))
    _patch_tech(monkeypatch, _tech_answer())
    c = kr_company(name="에이", official_name="에이")
    fd.founder_node(state_for(c))
    tc.tech_node(state_for(c))
    got = [(q, kw["deep"]) for q, kw in web.calls]
    assert got == [(q.format(name="에이"), d) for q, d in V1_KR]
    assert all(kw == {"topic": "news", "recent": False, "deep": kw["deep"], "raw": True} for _, kw in web.calls)


# ── 👤 창업자
def _founder_web():
    interview = "https://news.example.com/interview"
    award = "https://news.example.com/award"
    return interview, award, FakeWeb({
        "테스트팜 대표 창업자 이력 인터뷰": [{
            "url": interview, "title": "테스트팜 김대표 대표 인터뷰", "published_date": "2025-03-10",
            "content": "테스트팜 김대표 대표는 2012년 서울대 원예학과를 졸업하고 2014~2019 농촌진흥청 연구원으로 근무했다. "
                       "테스트팜은 온실 AI 제어기를 만든다."}],
        "테스트팜 수상 선정 혁신상 우수기업 출시": [{
            "url": award, "title": "테스트팜, 스마트농업 혁신상 수상", "published_date": "2025-03-20",
            "content": "테스트팜이 2025년 3월 스마트농업 혁신상을 받았다."}],
    })


def _founder_answer(i_id: str, a_id: str):
    P, M = fd.Person, fd.Milestone
    return fd.FounderAnalysis(
        people=[P(name="김대표", role="대표", background=f"2014~2019 농촌진흥청 연구원 [{i_id}]", before_t0=None,
                  evidence_ids=[i_id]),
                P(name="홍길동", role="CTO", background="확인 불가", before_t0=True, evidence_ids=[i_id])],
        milestones=[M(date="2025-03", what="스마트농업 혁신상 수상", evidence_ids=[a_id]),
                    M(date="", what="날짜 없는 사건", evidence_ids=[a_id]),
                    M(date="2025", what="월 없는 사건", evidence_ids=[a_id]),
                    M(date="2027-01", what="해외 진출 예정", evidence_ids=[a_id]),
                    M(date="2025-04", what="근거 id 가 저장소에 없음", evidence_ids=["Wzzzzz"]),
                    M(date="2019-05", what="연도가 근거에 없음", evidence_ids=[a_id])],
        team_assessment=f"분야 연구 경력이 있다 [{i_id}]", evidence_ids=[i_id])


def test_founder_node_contract_and_grounding(monkeypatch, judge_calls, no_enrich):
    i_url, a_url, web = _founder_web()
    i_id, a_id = wid(i_url), wid(a_url)
    llm = FakeLLM({"FounderAnalysis": [_founder_answer(i_id, a_id)]})
    monkeypatch.setattr(fd, "web_search", web)
    monkeypatch.setattr(fd, "structured", llm)
    out = fd.founder_node(state_for(kr_company()))

    assert set(out) == {"founder", "registry", "log"}
    f = out["founder"]
    c8 = {"t0", "t0_source", "years_since_t0", "people", "milestones", "milestones_24m", "headcount", "hires_per_year",
          "team_assessment", "evidence_ids", "pool_ids", "criterion"}
    assert c8 <= set(f)
    # 근거에 이름이 없는 인물(홍길동)은 버린다
    assert [p["name"] for p in f["people"]] == ["김대표"]
    # t0 = 국민연금 최초 가입일(TIPS 설립일 없음) → 이력의 2014년은 창업 전
    assert f["t0"] == "2020-06-01" and f["t0_source"].startswith("국민연금")
    assert f["people"][0]["before_t0"] is True
    # 날짜·근거 id·근거 속 연도가 모두 있는 마일스톤만 남고, 기준일 뒤(예정)는 버린다
    assert f["milestones"] == [{"date": "2025-03", "what": "스마트농업 혁신상 수상", "evidence_ids": [a_id]}]
    assert f["milestones_24m"] == 1
    assert f["headcount"] == 12 and f["years_since_t0"] == 6.3 and f["hires_per_year"] == round(12 / 6.3, 1)
    assert set(f["evidence_ids"]) <= set(f["pool_ids"]) and i_id in f["evidence_ids"]
    # 판정: founder 차원 1회, 이 노드에서 새로 모은 근거가 풀에 들어간다
    assert [j["dim"] for j in judge_calls] == ["founder"]
    assert {i_id, a_id} <= set(judge_calls[0]["pool"]) and "[창업 시점 t0] 2020-06-01" in judge_calls[0]["analysis"]
    assert f["criterion"]["dim"] == "founder"
    # 근거 저장소는 추가분이 아니라 전체(reg.data)를 돌려준다
    assert "Wold01" in out["registry"] and i_id in out["registry"]
    # 프롬프트에 t0 와 누적 인기 지표 금지 규칙이 들어간다
    prompt = llm.prompts[0][1]
    assert "창업 시점 t0: 2020-06-01" in prompt and "팔로워" in prompt


def test_founder_retry_and_tips_ceo_fallback(monkeypatch, judge_calls, no_enrich):
    """인물이 근거로 확인되지 않으면 한 번 다시 묻고, 그래도 없으면 TIPS 대표자를 근거와 함께 넣는다 (v1 규칙)."""
    i_url, _, web = _founder_web()
    empty = fd.FounderAnalysis(people=[], milestones=[], team_assessment="확인 불가", evidence_ids=[])
    llm = FakeLLM({"FounderAnalysis": [empty, empty]})
    monkeypatch.setattr(fd, "web_search", web)
    monkeypatch.setattr(fd, "structured", llm)
    f = fd.founder_node(state_for(kr_company()))["founder"]
    assert len(llm.prompts) == 2 and "people 이 비어 있다" in llm.prompts[1][1]
    assert f["people"] == [{"name": "김대표", "role": "대표", "background": "확인 불가 (TIPS 공개 목록의 대표자)",
                            "before_t0": None, "evidence_ids": [wid(i_url)]}]


def test_t0_priority():
    reg = SourceRegistry()
    a = reg.add_web({"url": "https://n.example.com/1", "title": "t", "published_date": "2024-01-02",
                     "content": "2019년 설립된 테스트팜은 온실 제어기를 만든다."}, "founder", "q", RUN_DATE)
    b = reg.add_web({"url": "https://n.example.com/2", "title": "t", "published_date": "2024-01-02",
                     "content": "테스트팜은 2021년 창업했다. 2018년 창업도약패키지에 선정된 다른 회사도 있다."}, "founder", "q", RUN_DATE)
    names, ids = ["테스트팜", "Testfarm"], [b, a]
    nps = {"status": "matched", "first_date": "2020-06-01", "evidence_id": "Wnps01"}
    t = fd.pick_t0(kr_company(founded_date="2020-04-17", nps=nps), reg, ids, names, RUN_DATE)
    assert (t["t0"], t["t0_source"]) == ("2020-04-17", "TIPS 공개 목록 설립일(estDt)")
    t = fd.pick_t0(kr_company(nps=nps), reg, ids, names, RUN_DATE)
    assert t["t0"] == "2020-06-01" and t["t0_evidence_ids"] == ["Wnps01"] and "Wnps01" in t["t0_source"]
    t = fd.pick_t0(kr_company(nps={"status": "not_found"}, founded_year=2021), reg, ids, names, RUN_DATE)
    assert t["t0"] == "2021" and t["t0_evidence_ids"] == [b]  # 적격성 설립연도 + 같은 연도의 원문 설립 표현
    t = fd.pick_t0(kr_company(nps={"status": "not_found"}), reg, ids, names, RUN_DATE)
    assert t["t0"] == "2020" and t["t0_evidence_ids"] == [] and "원문 인용 확인 안 됨" in t["t0_source"]
    t = fd.pick_t0(kr_company(nps={"status": "not_found"}, founded_year=None), reg, ids, names, RUN_DATE)
    assert t["t0"] == "2019" and t["t0_evidence_ids"] == [a]  # 가장 이른 설립 표현, '창업도약패키지'는 설립이 아님
    t = fd.pick_t0(kr_company(nps={}, founded_year=2018), SourceRegistry(), [], names, RUN_DATE)
    assert t["t0"] == "2018" and "원문 인용 확인 안 됨" in t["t0_source"]
    t = fd.pick_t0(kr_company(nps={}, founded_year=0), SourceRegistry(), [], names, RUN_DATE)
    assert t == {"t0": None, "t0_basis": "없음", "t0_source": "확인 불가", "t0_evidence_ids": []}


def test_founding_mentions_english_and_future():
    reg = SourceRegistry()
    s = reg.add_web({"url": "https://n.example.com/3", "title": "t",
                     "content": "Testfarm, founded in March 2017, builds robots. Founded in 2030 by nobody."}, "f", "q",
                    RUN_DATE)
    got = fd.founding_mentions(reg, [s], ["Testfarm"], RUN_DATE)
    assert [g["date"] for g in got] == ["2017-03"]  # 영문 월 이름을 읽고, 기준일보다 뒤의 연도는 버린다


# ── 🗜️ 기술 요약
def _tech_answer(claim_id: str = "Wnone0") -> tc.TechAnalysis:
    return tc.TechAnalysis(
        product="온실 AI 제어기", core_technology="생육 예측", maturity="상용", maturity_evidence="2025 설치",
        pros=["에너지 절감"], cons=["시설 온실에만 적용"],
        claims=[f"경쟁사 대비 난방비 30% 절감 [{claim_id}]", "근거 없는 주장"], ip_evidence="확인 불가",
        license_check="해당 없음: 소프트웨어", growth_signals=[f"2025-05 농가 30곳 설치 [{claim_id}]", "2025-06 근거 없음"],
        trend_fit="기준선 수준", evidence_ids=[claim_id])


class FakeTool:
    def __init__(self, results: dict[str, str]):
        self.results, self.calls = results, []

    def invoke(self, args: dict) -> str:
        self.calls.append(args)
        return self.results.get(args["source"], "")


def _patch_tech(monkeypatch, answer, rag_calls: list | None = None, tool: FakeTool | None = None):
    llm = FakeLLM({"TechAnalysis": [answer]})
    monkeypatch.setattr(tc, "structured", llm)
    monkeypatch.setattr(tc, "get_chunks", lambda: [])

    def fake_answer(question, purpose, registry, agent, **kw):
        (rag_calls if rag_calls is not None else []).append((question, purpose, agent, kw))
        d = registry.add_doc({"doc_id": "doc1", "page": 3, "title": "스마트농업 보고서", "publisher": "기관", "year": 2025},
                             "스마트팜 기술 동향: 상용화 초기", agent, question)
        return {"answer": f"분야 기준선 답변 [{d}]", "evidence_ids": [d], "cited_ids": [d], "status": "grounded",
                "route": "docs", "rewrites": 0, "regenerations": 0, "trace": [{"step": "agent"}]}
    monkeypatch.setattr(rag.agentic_rag, "answer_question", fake_answer)
    tool = tool or FakeTool({})
    monkeypatch.setattr(tools.agent_tools, "make_tools", lambda reg, agent: {"summarize_document": tool})
    monkeypatch.setattr(tools.channels, "tips_agtech", lambda min_year=2021: [])
    return llm


def test_tech_node_contract(monkeypatch, judge_calls, no_enrich):
    art = "https://news.example.com/tech"
    web = FakeWeb({"테스트팜 매출 고객 농가 수 설치": [{
        "url": art, "title": "테스트팜 농가 30곳 설치", "published_date": "2025-05-02",
        "content": "테스트팜이 온실 AI 제어기를 농가 30곳에 설치했다. " + "테스트팜 제어기는 난방비를 줄인다. " * 20,
        "raw_content": "테스트팜이 온실 AI 제어기를 농가 30곳에 설치했다. " + "테스트팜 제어기는 난방비를 줄인다. " * 30}]})
    monkeypatch.setattr(tc, "web_search", web)
    art_id = wid(art)
    # 홈페이지(TIPS 목록)는 읽기 실패('') → 회사 기사 본문 요약으로 대신한다
    tool = FakeTool({art_id: f"회사 기사 요약: 온실 제어기 [{art_id}]"})
    rag_calls: list = []
    llm = _patch_tech(monkeypatch, _tech_answer(art_id), rag_calls, tool)
    monkeypatch.setattr(tools.channels, "tips_agtech",
                        lambda min_year=2021: [{"name": "테스트팜", "homepage": "www.testfarm.co.kr"}])
    out = tc.tech_node(state_for(kr_company()))

    assert set(out) == {"tech", "registry", "rag_traces", "log"}
    t = out["tech"]
    assert "founders" not in t and "team_assessment" not in t
    c8 = {"product", "core_technology", "maturity", "maturity_evidence", "pros", "cons", "claims", "ip_evidence",
          "license_check", "baseline_answer", "trend_fit", "homepage_summary", "growth_signals", "evidence_ids",
          "pool_ids", "criterion"}
    assert c8 <= set(t)
    # 근거 id 가 달린 주장·성장 신호만 남는다
    assert t["claims"] == [f"경쟁사 대비 난방비 30% 절감 [{art_id}]"]
    assert t["growth_signals"] == [f"2025-05 농가 30곳 설치 [{art_id}]"]
    # 기준선: v1 과 같은 질문으로 answer_question 1회, 직접 답 경로는 끈다
    assert rag_calls == [("스마트팜·시설원예 AI 환경제어 분야의 기술 동향, 상용화 수준, 기술적 과제",
                          "스타트업 기술 수준을 비교할 기준선", "tech",
                          {"allow_direct": False})]
    assert t["baseline_answer"].startswith("분야 기준선 답변")
    assert out["rag_traces"][0]["route"] == "docs" and out["rag_traces"][0]["trace"] == [{"step": "agent"}]
    # 홈페이지 → 실패 → 기사 요약
    assert [x["source"] for x in tool.calls] == ["https://www.testfarm.co.kr", art_id]
    assert t["homepage_summary"].startswith("회사 기사 요약") and t["homepage_source"] == art_id
    assert "회사 기사 요약" in llm.prompts[0][1] and "분야 기준선 답변" in llm.prompts[0][1]
    assert [j["dim"] for j in judge_calls] == ["product"]
    assert "Wold01" in out["registry"] and art_id in out["registry"]


def test_tech_summary_failure_does_not_stop(monkeypatch, judge_calls, no_enrich):
    """문서 요약 도구가 없거나(P2 병합 전) 예외를 던져도 기술 요약은 끝까지 간다."""
    monkeypatch.setattr(tc, "web_search", FakeWeb())
    _patch_tech(monkeypatch, _tech_answer())

    def broken(reg, agent):
        raise NotImplementedError
    monkeypatch.setattr(tools.agent_tools, "make_tools", broken)
    t = tc.tech_node(state_for(kr_company()))["tech"]
    assert t["homepage_summary"] is None and t["criterion"]["dim"] == "product"


def test_homepage_url_from_web_evidence(monkeypatch):
    monkeypatch.setattr(tools.channels, "tips_agtech", lambda min_year=2021: [])
    reg = SourceRegistry()
    a = reg.add_web({"url": "https://news.example.com/x", "title": "t"}, "tech", "q", RUN_DATE)
    b = reg.add_web({"url": "https://www.testfarm.co.kr/about", "title": "t"}, "tech", "q", RUN_DATE)
    assert tc.homepage_url(kr_company(), reg, [a, b]) == "https://www.testfarm.co.kr/about"
    assert tc.homepage_url(kr_company(name_en=""), reg, [a, b]) is None


# ── 📊 시장성
def _market_patch(monkeypatch, subs: list[tuple[str, str]], size_source: str | None = None):
    answered: list = []

    def fake_answer(question, purpose, registry, agent, **kw):
        answered.append((question, purpose, agent, kw))
        d = registry.add_doc({"doc_id": f"d{len(answered)}", "page": 1, "title": "보고서", "publisher": "기관",
                              "year": 2025}, f"{question} 답 근거: 국내 시장 1조 원(2024)", agent, question)
        return {"answer": f"{purpose} 답 [{d}]", "evidence_ids": [d], "cited_ids": [d], "status": "grounded",
                "route": "docs", "rewrites": 0, "regenerations": 0, "trace": []}
    monkeypatch.setattr(rag.agentic_rag, "answer_question", fake_answer)
    monkeypatch.setattr(rag.agentic_rag, "corpus_catalog", lambda: "- 기관 (2025) 보고서", raising=False)
    first_doc = SourceRegistry._id("D", "doc:d1:1:" + hashlib.md5(
        f"{subs[0][0]} 답 근거: 국내 시장 1조 원(2024)".encode()).hexdigest()[:8])
    size = mk.MarketSize(value=10000, unit="KRW_EOK", year=2024, scope="국내 스마트팜 시장",
                         source_id=size_source or first_doc, raw="1조 원")
    ans = mk.MarketAnalysis(market_size="국내 1조 원(2024)", size=size, growth="성장", growth_rate_pct=12.5,
                            growth_year=2025, growth_source_id="Wnotin", investment_trend="감소",
                            demand_drivers=["인력난"], willingness_to_pay="확인 불가", policy_regulation="지원",
                            adoption_barriers=["비용"], evidence_ids=[first_doc])
    dec = mk.Decomposition(sub_questions=[mk.SubQuestion(question=q, purpose=p) for q, p in subs])
    llm = FakeLLM({"Decomposition": [dec], "MarketAnalysis": [ans]})
    monkeypatch.setattr(mk, "structured", llm)
    templates: list[str] = []

    def spy_render(name, **kw):
        templates.append(name)
        return render(name, **kw)
    monkeypatch.setattr(mk, "render", spy_render)
    return answered, llm, templates, first_doc


def test_market_node_contract_and_cache(monkeypatch, judge_calls):
    subs = [("국내 스마트팜 시장 규모", "시장 규모"), ("글로벌 성장률", "성장률"), ("농가 수요", "수요")]
    answered, llm, templates, first_doc = _market_patch(monkeypatch, subs)
    c = kr_company()
    out = mk.market_node(state_for(c, tech={"product": "온실 AI 제어기", "pros": ["난방비 절감"]}))

    assert set(out) == {"market", "market_cache", "registry", "rag_traces", "log"}
    m = out["market"]
    # 경로 선택은 RAG 서브그래프가 한다: market_route 프롬프트를 쓰지 않는다
    assert "market_route" not in templates and not (ROOT / "prompts" / "market_route.md").exists()
    # 하위 질문 3개 + 투자 동향 1개(분해에 없어서 추가) = answer_question 4회, 직접 답 경로는 끈다
    assert [a[:2] for a in answered] == subs + [("애그테크 벤처 투자 동향 (2025년 투자액 증감, 국가별 증감)", "투자 동향")]
    assert all(a[2] == "market" and a[3] == {"allow_direct": False} for a in answered)
    assert [a["purpose"] for a in m["answers"]] == ["시장 규모", "성장률", "수요", "투자 동향"]
    assert set(m["answers"][0]) == {"question", "purpose", "answer", "route", "status", "evidence_ids"}
    # 시장 규모 구조화: 1조 원 = 10,000억 원, 환율 1400 → 714.3 백만 달러
    assert m["size"]["value_usd_m"] == 714.3 and m["size"]["source_id"] == first_doc
    assert m["growth_source_id"] is None  # 근거 저장소에 없는 id
    c8 = {"segment", "segment_id", "market_size", "size", "growth", "growth_rate_pct", "growth_year", "growth_source_id",
          "investment_trend", "demand_drivers", "willingness_to_pay", "policy_regulation", "adoption_barriers",
          "answers", "evidence_ids", "pool_ids", "criterion"}
    assert c8 <= set(m)
    cached = out["market_cache"]["greenhouse"]
    assert "criterion" not in cached and cached["size"] == m["size"]
    assert [j["dim"] for j in judge_calls] == ["market"] and "[기술 요약] 제품: 온실 AI 제어기" in judge_calls[0]["analysis"]
    assert [t["step"] for t in out["rag_traces"]] == ["decompose", "answer", "answer", "answer", "answer"]
    assert "Wold01" in out["registry"]

    # 같은 분야의 두 번째 후보: 분석은 재사용(RAG·LLM 호출 없음), 시장성 판정은 다시 한다
    c2 = kr_company(name="둘째팜", official_name="둘째팜", name_en="")
    out2 = mk.market_node(state_for(c2, market_cache=out["market_cache"], registry=out["registry"]))
    assert len(answered) == 4 and len(llm.prompts) == 2
    assert [j["dim"] for j in judge_calls] == ["market", "market"] and judge_calls[1]["company"] == "둘째팜"
    assert out2["market"]["size"] == m["size"] and "market_cache" not in out2


def test_market_decompose_prompt_is_v1(monkeypatch, judge_calls):
    """질의 분해 프롬프트 입력(큰 질문·카탈로그)과 스키마가 v1 과 같다 (LLM 캐시 키)."""
    subs = [("a", "시장 규모"), ("b", "성장률"), ("c", "투자 동향")]
    answered, llm, _, _ = _market_patch(monkeypatch, subs)
    mk.market_node(state_for(kr_company()))
    seg = "스마트팜·시설원예 AI 환경제어"
    assert llm.prompts[0] == ("Decomposition", render("market_decompose", segment=seg, question=V1_MARKET_Q.format(seg=seg),
                                                      run_date=RUN_DATE, catalog="- 기관 (2025) 보고서"))
    assert len(answered) == 3  # 투자 동향 하위 질문이 이미 있으면 더하지 않는다
    out = subprocess.run(["git", "show", "v1-safe:agents/market.py"], cwd=ROOT, capture_output=True)
    if out.returncode == 0:
        old = {n.name: ast.dump(n) for n in ast.parse(out.stdout.decode("utf-8")).body if isinstance(n, ast.ClassDef)}
        new = {n.name: ast.dump(n) for n in ast.parse((ROOT / "agents" / "market.py").read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef)}
        assert new["SubQuestion"] == old["SubQuestion"] and new["Decomposition"] == old["Decomposition"]


def test_to_usd_m():
    assert mk.to_usd_m(10000, "KRW_EOK", 1400) == 714.3
    assert mk.to_usd_m(1200, "USD_M", 1400) == 1200
    assert mk.to_usd_m(5, "", 1400) is None and mk.to_usd_m(None, "USD_M", 1400) is None


# ── 🥊 경쟁사
def test_competition_node_claims_and_downgrade(monkeypatch, judge_calls):
    news, own = "https://news.example.com/rival", "https://www.testfarm.co.kr/about"
    web = FakeWeb({"테스트팜 경쟁사": [
        {"url": news, "title": "온실 제어 경쟁", "content": "테스트팜과 경쟁사 그린랩스 비교: 난방비 30% 절감 확인"},
        {"url": own, "title": "테스트팜 소개", "content": "테스트팜 제어기는 난방비를 30% 줄입니다"}]})
    monkeypatch.setattr(comp, "web_search", web)
    n_id, o_id = wid(news), wid(own)
    VC = comp.VerifiedClaim
    ans = comp.CompetitionAnalysis(
        competitors=[comp.Competitor(name="그린랩스", country="한국", offering="온실 플랫폼", scale="확인 불가",
                                     vs_target="경쟁사: 온실 플랫폼 / 대상: 회사 측 주장: 절감", evidence_ids=[n_id])],
        differentiation="회사 측 주장: 절감", entry_barriers="약함", threats=["대기업 진입"],
        verified_claims=[VC(claim="A", status="제3자 확인", evidence_ids=[]),
                         VC(claim="B", status="제3자 확인", evidence_ids=[o_id]),
                         VC(claim="C", status="제3자 확인", evidence_ids=[n_id]),
                         VC(claim="D", status="반대 근거", evidence_ids=["Wnotin"]),
                         VC(claim="E", status="회사 주장", evidence_ids=[])],
        evidence_ids=[n_id])
    plan = comp.SearchPlan(product_type_ko="온실 AI 제어기", product_type_en="greenhouse AI controller", queries=[])
    llm = FakeLLM({"SearchPlan": [plan], "CompetitionAnalysis": [ans]})
    monkeypatch.setattr(comp, "structured", llm)
    tech = {"product": "온실 AI 제어기", "core_technology": "생육 예측", "claims": ["난방비 30% 절감 [Wabc12]"],
            "pool_ids": [], "evidence_ids": []}
    out = comp.competition_node(state_for(kr_company(), tech=tech))

    assert set(out) == {"competition", "registry", "log"}
    k = out["competition"]
    prompt = next(p for s, p in llm.prompts if s == "CompetitionAnalysis")
    assert "- 난방비 30% 절감 [Wabc12]" in prompt  # tech.claims 가 프롬프트에 들어간다
    assert [(v["claim"], v["status"]) for v in k["verified_claims"]] == [
        ("A", "회사 주장"), ("B", "회사 주장"), ("C", "제3자 확인"), ("D", "회사 주장"), ("E", "회사 주장")]
    assert [v["downgraded"] for v in k["verified_claims"]] == [True, True, False, True, False]
    assert {"competitors", "differentiation", "entry_barriers", "verified_claims", "evidence_ids", "pool_ids",
            "search_plan", "mentioned_competitors", "criterion"} <= set(k)
    assert [j["dim"] for j in judge_calls] == ["competition"]
    assert "Wold01" in out["registry"] and n_id in out["registry"]


# ── 순서대로 이어 돌리기: 뒤 에이전트의 판정 근거 풀에 앞 에이전트가 모은 근거가 들어간다 (계약 C3)
def test_pool_accumulates_founder_to_tech(monkeypatch, judge_calls, no_enrich):
    i_url, a_url, web = _founder_web()
    monkeypatch.setattr(fd, "web_search", web)
    monkeypatch.setattr(fd, "structured", FakeLLM({"FounderAnalysis": [_founder_answer(wid(i_url), wid(a_url))]}))
    monkeypatch.setattr(tc, "web_search", FakeWeb())
    _patch_tech(monkeypatch, _tech_answer())
    state = state_for(kr_company())
    out = fd.founder_node(state)
    state = {**state, "founder": out["founder"], "registry": {**state["registry"], **out["registry"]}}
    tc.tech_node(state)
    founder_pool, product_pool = judge_calls[0]["pool"], judge_calls[1]["pool"]
    assert [j["dim"] for j in judge_calls] == ["founder", "product"]
    assert set(founder_pool) <= set(product_pool) and len(product_pool) > len(founder_pool)


# ── 통합 1건: 실제 Agentic RAG(v1 호환 구현, 재현용 캐시)
def test_tech_baseline_with_real_answer_question_v1(monkeypatch, judge_calls, no_enrich):
    if not hasattr(rag.agentic_rag, "answer_question_v1"):
        pytest.skip("answer_question_v1 없음")
    monkeypatch.setattr(tc, "web_search", FakeWeb())
    _patch_tech(monkeypatch, _tech_answer())
    monkeypatch.setattr(rag.agentic_rag, "answer_question", rag.agentic_rag.answer_question_v1)
    try:
        out = tc.tech_node(state_for(kr_company()))
    except RuntimeError as e:  # 재현용 캐시에 없는 LLM 호출 (캐시를 정리한 clone 등)
        pytest.skip(str(e)[:80])
    docs = [i for i in out["tech"]["pool_ids"] if i.startswith("D")]
    assert docs and all(i in out["registry"] for i in docs)
    assert out["rag_traces"][0]["trace"] and out["rag_traces"][0]["status"] == "grounded"
