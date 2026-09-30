"""Agentic RAG 서브그래프 검사 (P2). LLM·검색기·웹 검색은 가짜로 바꿔 끼워 키·네트워크 없이 돈다.

- v2 answer_question: 문서 경로 · 재작성 소진 → 웹 보완 · not_grounded 재생성 · 코드 수치 대조 · direct · both · web
- 근거 저장소 하나(B6): web 경로에서 등록한 근거가 generate 에 보이고, 호출한 에이전트의 registry 에 남는다
- 라우터 순수 함수 표 · 간선 = RAG_EDGES · 최악 경로도 rag_recursion_limit 안에서 끝남
- v1 agentic_rag: v1-safe 코드와 같은 LLM 입력·결과 (발굴 캐시 보존), generate·check 를 부르지 않음
- @pytest.mark.api: bind_tools 응답이 SQLiteCache 로 재생되는지 (키 필요, 기본 실행에서 제외)
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import types

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import rag.agentic_rag as ar
from core.config import ROOT, get_config
from rag.loader import load_manifest
from tools.sources import SourceRegistry

QUESTION = "국내 스마트팜 시장 규모는?"
PURPOSE = "시장 규모"


def _doc(i: int, text: str | None = None) -> Document:
    return Document(page_content=text or f"조각 {i}: 국내 스마트팜 시장 규모는 5,000억 원(2023년)이다.",
                    metadata={"doc_id": f"doc{i % 3}", "page": i + 1, "publisher": "기관", "year": 2024,
                              "title": f"보고서 {i % 3}", "chunk_id": f"c{i}", "url": "https://example.org"})


class FakeRetriever:
    def __init__(self, n: int = 8):
        self.n, self.queries = n, []

    def invoke(self, query: str):
        self.queries.append(query)
        return [_doc(i + 10 * (len(self.queries) - 1)) for i in range(self.n)]


class Env:
    """가짜 LLM 판단부·검색기·웹 검색. calls 에 호출 기록을 남긴다."""

    def __init__(self, monkeypatch, *, decide=None, grade=None, generate=None, check=None, n_web=2):
        self.calls: dict[str, list] = {"agent": [], "grade": [], "rewrite": [], "generate": [], "check": [], "web": []}
        self.retriever = FakeRetriever()
        self.n_web = n_web
        self._decide = decide or {"route": "docs", "query": "스마트팜 시장 규모", "web_query": QUESTION,
                                  "recent": True, "answer": "", "reason": "", "tools": ["search_documents"]}
        self._grade = grade if grade is not None else (lambda n: {0, 1, 2})
        self._generate = generate or (lambda ctx, fb, n: self.cite_all(ctx))
        self._check = check or (lambda n: (True, True, "ok"))
        monkeypatch.setattr(ar, "get_hybrid_retriever", lambda: self.retriever)
        monkeypatch.setattr(ar, "web_search", self.web_search)
        monkeypatch.setattr(ar, "_agent_decide", self.agent)
        monkeypatch.setattr(ar, "_grade_llm", self.grade)
        monkeypatch.setattr(ar, "_rewrite_llm", self.rewrite)
        monkeypatch.setattr(ar, "_generate_llm", self.generate)
        monkeypatch.setattr(ar, "_check_llm", self.check)

    @staticmethod
    def cite_all(ctx: str) -> tuple[str, list[str]]:
        ids = re.findall(r"^\[([DW][0-9a-f]{5})\]", ctx, re.M)
        if not ids:
            return "근거에서 확인되지 않음", []
        return "국내 스마트팜 시장 규모는 5000억 원이다(2023년, 기관) " + " ".join(f"[{i}]" for i in ids), ids

    def agent(self, question, purpose, run_date, allow_web, allow_direct):
        self.calls["agent"].append((allow_web, allow_direct))
        return dict(self._decide)

    def grade(self, question, purpose, chunks):
        self.calls["grade"].append(len(chunks))
        return self._grade(len(self.calls["grade"]))

    def rewrite(self, question, purpose, tried):
        self.calls["rewrite"].append(list(tried))
        return f"재작성 {len(self.calls['rewrite'])}"

    def generate(self, question, purpose, run_date, context, feedback):
        self.calls["generate"].append((context, feedback))
        return self._generate(context, feedback, len(self.calls["generate"]))

    def check(self, question, purpose, answer, context):
        self.calls["check"].append(answer)
        return self._check(len(self.calls["check"]))

    def web_search(self, query, registry, agent, **kw):
        self.calls["web"].append({"query": query, "registry": registry, **kw})
        return [registry.add_web({"url": f"https://news.example.com/{len(self.calls['web'])}/{j}",
                                  "title": f"기사 {j}", "content": f"스마트팜 기사 {j} 본문 5,000억 원",
                                  "published_date": "2026-05-01"}, agent=agent, query=query, access_date="2026-09-30")
                for j in range(self.n_web)]

    @property
    def llm_calls(self) -> int:
        return sum(len(self.calls[k]) for k in ("agent", "grade", "rewrite", "generate", "check"))


def _steps(out: dict) -> list[str]:
    return [t.get("step") for t in out["trace"]]


# ── v2 경로
def test_docs_path_grounded(monkeypatch):
    env = Env(monkeypatch)
    reg = SourceRegistry()
    out = ar.answer_question(QUESTION, PURPOSE, reg, "market")
    assert set(out) == set(ar.RESULT_KEYS)
    assert env.llm_calls == 4  # agent · grade · generate · check
    assert _steps(out) == ["agent", "grade", "generate", "check"]
    assert (out["status"], out["route"], out["rewrites"], out["regenerations"]) == ("grounded", "docs", 0, 0)
    assert env.retriever.queries == ["스마트팜 시장 규모"]  # agent 가 고른 검색어
    assert out["evidence_ids"] and set(out["evidence_ids"]) <= set(reg.data)
    assert out["cited_ids"] == out["evidence_ids"] and all(i.startswith("D") for i in out["cited_ids"])
    assert len(out["evidence_ids"]) == 3 and not env.calls["web"]


def test_insufficient_rewrites_twice_then_web(monkeypatch):
    env = Env(monkeypatch, grade=lambda n: set())
    reg = SourceRegistry()
    out = ar.answer_question(QUESTION, PURPOSE, reg, "market")
    assert out["rewrites"] == 2 and len(env.calls["rewrite"]) == 2  # 3회 이상 재작성하지 않음
    assert env.retriever.queries == ["스마트팜 시장 규모", "재작성 1", "재작성 2"]
    assert env.calls["rewrite"] == [["스마트팜 시장 규모"], ["스마트팜 시장 규모", "재작성 1"]]  # v1 과 같은 tried 형식
    (web,) = env.calls["web"]  # 교정형 웹 보완: v1 과 같은 인자
    assert web["query"] == QUESTION and (web["topic"], web["recent"], web["max_results"]) == ("general", False, 4)
    assert _steps(out) == ["agent", "grade", "rewrite", "grade", "rewrite", "grade", "web", "generate", "check"]
    assert any("web_fallback" in t for t in out["trace"])
    assert out["status"] == "grounded" and all(i.startswith("W") for i in out["evidence_ids"])


def test_not_grounded_regenerates_once_then_partial(monkeypatch):
    env = Env(monkeypatch, check=lambda n: (False, True, "근거에 없는 투자액이 있음"))
    out = ar.answer_question(QUESTION, PURPOSE, SourceRegistry(), "market")
    assert out["regenerations"] == 1 and len(env.calls["generate"]) == 2 and len(env.calls["check"]) == 2
    assert env.calls["generate"][0][1] == "" and env.calls["generate"][1][1] == "근거에 없는 투자액이 있음"
    assert env.calls["generate"][0][0] == env.calls["generate"][1][0]  # 검색은 두고 답변만 다시
    assert out["status"] == "partial" and out["rewrites"] == 0


def test_keeps_best_answer_when_later_answers_get_worse(monkeypatch):
    """1차 답은 근거는 맞지만 질문에 덜 맞음(not_useful) → 재작성 후 새 답은 근거 밖(not_grounded) 두 번 → 1차 답을 낸다."""
    def generate(ctx, fb, n):
        ans, ids = Env.cite_all(ctx)
        return f"답{n} " + ans, ids

    checks = {1: (True, False, "질문과 다름"), 2: (False, True, "근거 밖"), 3: (False, True, "근거 밖")}
    env = Env(monkeypatch, generate=generate, check=lambda n: checks[n])
    out = ar.answer_question(QUESTION, PURPOSE, SourceRegistry(), "market")
    assert len(env.calls["check"]) == 3 and out["rewrites"] == 1 and out["regenerations"] == 1
    assert out["answer"].startswith("답1 ") and (out["check"], out["status"]) == ("not_useful", "partial")
    assert out["evidence_ids"]


def test_numeric_check_skips_llm_and_feeds_back(monkeypatch):
    def generate(ctx, fb, n):
        ans, ids = Env.cite_all(ctx)
        return (ans + " 연평균 37% 성장한다.", ids) if n == 1 else (ans, ids)

    env = Env(monkeypatch, generate=generate)
    out = ar.answer_question(QUESTION, PURPOSE, SourceRegistry(), "market")
    first = next(t for t in out["trace"] if t.get("step") == "check")
    assert first["check"] == "not_grounded" and first["numbers"] == ["37"]
    assert len(env.calls["check"]) == 1  # 37% 답변에는 LLM 점검을 부르지 않았다
    assert "37" in env.calls["generate"][1][1]
    assert (out["status"], out["regenerations"]) == ("grounded", 1)


@pytest.mark.parametrize("answer, evidence, known, bad", [
    ("시장 규모는 37%다 [D1a2b3]", "시장 규모 12%", "", ["37"]),
    ("5,000억 원 [D12a45]", "5000억원", "", []),                  # 쉼표 차이, 인용 id 속 숫자(12·45)는 무시
    ("2023년 기준 5000억 원", "5,000억 원", "", []),              # 4자리 연도 제외
    ("24개월 안에 3배", "", "창업 후 24개월 마일스톤", []),        # 질문에 나온 수치 제외, 한 자리 수 제외
    ("37.0% 증가", "37% 증가", "", []),                           # 37.0 = 37
    ("1.5배, 12.5%", "1.5배", "", ["12.5"]),
    ("1조 원", "", "", []),                                        # 한 자리(표기 차이 많음)는 비교하지 않음
    ("137억", "37억", "", ["137"]),                               # 부분 문자열 일치로 통과시키지 않음
])
def test_unsupported_numbers_rules(answer, evidence, known, bad):
    assert ar.unsupported_numbers(answer, evidence, known) == bad


def test_direct_passthrough_then_retrieve_when_not_useful(monkeypatch):
    decide = {"route": "direct", "query": QUESTION, "web_query": QUESTION, "recent": True,
              "answer": "스마트팜은 정보통신기술을 접목한 농장이다.", "reason": "", "tools": []}
    env = Env(monkeypatch, decide=decide, check=lambda n: (True, n > 1, "질문에 답하지 않음" if n == 1 else "ok"))
    out = ar.answer_question(QUESTION, PURPOSE, SourceRegistry(), "eval", allow_direct=True)
    assert env.calls["agent"] == [(True, True)]
    assert env.calls["check"][0] == decide["answer"]  # 직접 답은 generate LLM 없이 그대로 점검
    assert len(env.calls["generate"]) == 1            # 검색 뒤 생성 1회만
    assert env.retriever.queries == [QUESTION]         # not_useful → 질문 그대로 문서 검색 1회
    assert _steps(out) == ["agent", "generate", "check", "grade", "generate", "check"]
    assert (out["route"], out["status"], out["rewrites"]) == ("direct", "grounded", 0)


def test_direct_with_number_goes_to_retrieve_without_llm_check(monkeypatch):
    decide = {"route": "direct", "query": QUESTION, "web_query": QUESTION, "recent": True,
              "answer": "국내 스마트팜 시장은 약 42조 원이다.", "reason": "", "tools": []}
    env = Env(monkeypatch, decide=decide)
    out = ar.answer_question(QUESTION, PURPOSE, SourceRegistry(), "eval", allow_direct=True)
    assert out["trace"][2]["numbers"] == ["42"] and len(env.calls["check"]) == 1  # 검색 뒤 점검만 LLM
    assert env.retriever.queries == [QUESTION] and out["status"] == "grounded"


def test_both_route_runs_docs_then_web_with_tool_args(monkeypatch):
    decide = {"route": "both", "query": "스마트팜 시장 규모", "web_query": "스마트팜 보급 2026", "recent": True,
              "answer": "", "reason": "", "tools": ["search_documents", "web_search"]}
    env = Env(monkeypatch, decide=decide)
    reg = SourceRegistry()
    out = ar.answer_question(QUESTION, PURPOSE, reg, "market")
    (web,) = env.calls["web"]
    assert web["query"] == "스마트팜 보급 2026" and (web["topic"], web["recent"]) == ("news", True)
    assert _steps(out) == ["agent", "grade", "web", "generate", "check"]
    ctx = env.calls["generate"][0][0]
    assert re.search(r"^\[D", ctx, re.M) and re.search(r"^\[W", ctx, re.M)
    assert {i[0] for i in out["cited_ids"]} == {"D", "W"} and out["route"] == "both"


def test_web_route_uses_one_registry_object(monkeypatch):
    """B6: web 경로에서 등록한 근거가 generate 에 보이고, 인용 id 가 호출한 에이전트의 registry 에 있다."""
    decide = {"route": "web", "query": QUESTION, "web_query": "팜랩 투자 유치", "recent": False, "answer": "",
              "reason": "", "tools": ["web_search"]}
    env = Env(monkeypatch, decide=decide)
    reg = SourceRegistry({"W00000": {"id": "W00000", "kind": "web"}})
    out = ar.answer_question(QUESTION, PURPOSE, reg, "tech")
    (web,) = env.calls["web"]
    assert web["registry"] is reg and (web["query"], web["recent"], web["topic"]) == ("팜랩 투자 유치", False, "news")
    assert not env.calls["grade"] and _steps(out) == ["agent", "web", "generate", "check"]
    assert out["cited_ids"] and set(out["cited_ids"]) <= set(reg.data) and "W00000" in reg.data
    assert reg.brief(out["cited_ids"]).strip()


def test_web_route_not_useful_finishes_partial(monkeypatch):
    decide = {"route": "web", "query": QUESTION, "web_query": "q", "recent": True, "answer": "", "reason": "",
              "tools": ["web_search"]}
    env = Env(monkeypatch, decide=decide, check=lambda n: (True, False, "다른 내용"))
    out = ar.answer_question(QUESTION, PURPOSE, SourceRegistry(), "tech")
    assert out["status"] == "partial" and not env.calls["rewrite"] and len(env.calls["check"]) == 1


def test_not_found_status_and_evidence_fallback(monkeypatch):
    Env(monkeypatch, generate=lambda ctx, fb, n: ("근거에서 확인되지 않음", []), check=lambda n: (True, False, "없음"),
        grade=lambda n: {0, 1, 2})
    reg = SourceRegistry()
    out = ar.answer_question(QUESTION, PURPOSE, reg, "market")
    assert out["status"] == "not_found" and out["cited_ids"] == []
    assert out["evidence_ids"] and set(out["evidence_ids"]) <= set(reg.data)  # 인용이 없으면 generate 가 본 근거


def test_not_useful_rewrites_and_resets_relevant(monkeypatch):
    env = Env(monkeypatch, check=lambda n: (True, n > 1, "질문과 다른 내용" if n == 1 else "ok"))
    out = ar.answer_question(QUESTION, PURPOSE, SourceRegistry(), "market")
    assert _steps(out) == ["agent", "grade", "generate", "check", "rewrite", "grade", "generate", "check"]
    assert out["rewrites"] == 1 and out["regenerations"] == 0 and out["status"] == "grounded"
    # 두 번째 생성은 새 검색의 조각만 본다 (이전 조각을 비움)
    first, second = env.calls["generate"][0][0], env.calls["generate"][1][0]
    assert "조각 0:" in first and "조각 0:" not in second and "조각 10:" in second


def test_allow_web_false_never_searches_web(monkeypatch):
    env = Env(monkeypatch, grade=lambda n: set())
    out = ar.answer_question(QUESTION, PURPOSE, SourceRegistry(), "market", allow_web=False)
    assert not env.calls["web"] and env.calls["agent"] == [(False, False)]
    assert _steps(out)[-3:] == ["grade", "generate", "check"] and out["status"] == "not_found"


def test_worst_case_ends_within_recursion_limit(monkeypatch):
    """모든 점검이 실패해도 상한(재작성 2·재생성 1·direct 검색 1)으로 끝나고, 단계 수가 rag_recursion_limit 보다 작다."""
    decide = {"route": "direct", "query": QUESTION, "web_query": QUESTION, "recent": True, "answer": "모름",
              "reason": "", "tools": []}
    # direct 실패 → 검색 → not_useful 2회(재작성 2회) → 관련 부족 → 웹 보완 → not_grounded 2회(재생성 1회) → finish
    seq = {1: (False, False, "x"), 2: (True, False, "x"), 3: (True, False, "x")}
    env = Env(monkeypatch, decide=decide, check=lambda n: seq.get(n, (False, True, "x")),
              grade=lambda n: {0, 1, 2} if n < 3 else set())
    reg = SourceRegistry()
    graph = ar.build_rag_graph(reg)
    state = ar.initial_state(QUESTION, PURPOSE, "eval", "2026-09-30", allow_web=True, allow_direct=True)
    limit = get_config().rag.rag_recursion_limit
    steps = list(graph.stream(state, config={"recursion_limit": limit}, stream_mode="updates"))
    assert len(steps) == 21 < limit and "finish" in steps[-1]
    assert len(env.calls["rewrite"]) == 2 and len(env.calls["web"]) == 1 and len(env.calls["check"]) == 5
    assert steps[-1]["finish"]["status"] == "partial"


def test_answer_question_signature_and_defaults():
    import inspect

    sig = inspect.signature(ar.answer_question)
    assert list(sig.parameters) == ["question", "purpose", "registry", "agent", "allow_web", "allow_direct"]
    assert sig.parameters["allow_direct"].default is False and sig.parameters["allow_web"].default is True
    assert get_config().rag.allow_direct_answer is False



def test_judge_eval_summary_keys():
    """judge_eval.json 요약 키는 계약대로 (설계서·README 생성기가 읽는다)."""
    from eval.eval_judge import summarize_rows

    rows = [{"relevance": True, "faithfulness": True, "correctness": False, "route": "docs", "rewrites": 1,
             "regenerations": 0, "web_fallback": False, "status": "grounded"},
            {"relevance": False, "faithfulness": True, "correctness": False, "route": "direct", "rewrites": 0,
             "regenerations": 1, "web_fallback": True, "status": "not_found"}]
    s = summarize_rows(rows)
    assert list(s) == ["relevance", "faithfulness", "correctness", "n", "route_dist", "rewrite_rate",
                       "web_fallback_rate", "regen_rate", "not_found_rate"]
    assert (s["relevance"], s["faithfulness"], s["correctness"], s["n"]) == (0.5, 1.0, 0.0, 2)
    assert s["route_dist"] == {"docs": 1, "web": 0, "both": 0, "direct": 1}
    assert (s["rewrite_rate"], s["web_fallback_rate"], s["regen_rate"], s["not_found_rate"]) == (0.5, 0.5, 0.5, 0.5)

# ── 라우터 순수 함수 (12 경우)
def _st(**kw) -> dict:
    base = {"route": "docs", "relevant": [], "rewrites": 0, "regenerations": 0, "web_done": False, "allow_web": True,
            "check": "", "trace": [{"step": "grade"}]}
    return {**base, **kw}


R3 = [{}, {}, {}]


@pytest.mark.parametrize("fn, state, want", [
    ("route_after_agent", _st(route="docs"), "retrieve"),
    ("route_after_agent", _st(route="both"), "retrieve"),
    ("route_after_agent", _st(route="web"), "web"),
    ("route_after_agent", _st(route="direct"), "generate"),
    ("route_after_grade", _st(relevant=R3), "generate"),
    ("route_after_grade", _st(relevant=R3, route="both"), "web"),
    ("route_after_grade", _st(relevant=R3, route="both", web_done=True), "generate"),
    ("route_after_grade", _st(relevant=[{}], rewrites=1), "rewrite"),
    ("route_after_grade", _st(relevant=[{}], rewrites=2), "web"),
    ("route_after_grade", _st(relevant=[{}], rewrites=2, web_done=True), "generate"),
    ("route_after_grade", _st(relevant=[{}], rewrites=2, allow_web=False), "generate"),
    ("route_after_check", _st(check="grounded"), "finish"),
    ("route_after_check", _st(check="not_grounded"), "generate"),
    ("route_after_check", _st(check="not_grounded", regenerations=1), "finish"),
    ("route_after_check", _st(check="not_useful"), "rewrite"),
    ("route_after_check", _st(check="not_useful", rewrites=2), "finish"),
    ("route_after_check", _st(check="not_useful", route="web", trace=[]), "finish"),
    ("route_after_check", _st(check="not_useful", route="direct", trace=[]), "retrieve"),
    ("route_after_check", _st(check="not_grounded", route="direct", trace=[]), "retrieve"),
    ("route_after_check", _st(check="not_useful", route="direct"), "rewrite"),  # 검색한 뒤의 direct 는 문서 경로처럼
])
def test_router_table(fn, state, want):
    assert getattr(ar, fn)(state) == want


def test_graph_edges_match_design_constant():
    g = ar.build_rag_graph().get_graph()
    assert {(e.source, e.target) for e in g.edges} == ar.RAG_EDGES


# ── agent 노드: 도구 바인딩과 tool_calls 해석
@pytest.mark.parametrize("calls, allow_web, allow_direct, route, query, web_query, recent", [
    ([{"name": "search_documents", "args": {"query": "시장 규모"}}], True, False, "docs", "시장 규모", QUESTION, True),
    ([{"name": "web_search", "args": {"query": "팜랩", "recent": False}}], True, False, "web", QUESTION, "팜랩", False),
    ([{"name": "search_documents", "args": {"query": "a"}}, {"name": "web_search", "args": {"query": "b"}},
      {"name": "search_documents", "args": {"query": "c"}}], True, False, "both", "a", "b", True),
    ([], True, True, "direct", QUESTION, QUESTION, True),
    ([], True, False, "docs", QUESTION, QUESTION, True),                    # 직접 답 비허용 → 질문 그대로 문서
    ([{"name": "web_search", "args": {"query": "b"}}], False, False, "docs", QUESTION, QUESTION, True),
])
def test_parse_tool_calls(calls, allow_web, allow_direct, route, query, web_query, recent):
    d = ar.parse_tool_calls(calls, "", QUESTION, allow_web, allow_direct)
    assert (d["route"], d["query"], d["web_query"], d["recent"]) == (route, query, web_query, recent)


def test_agent_decide_binds_tools(monkeypatch):
    seen = {}

    class FakeLLM:
        def bind_tools(self, tools, **kw):
            seen["tools"], seen["kw"] = [t.name for t in tools], kw
            return self

        def invoke(self, prompt):
            seen["prompt"] = prompt
            return AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "q"}, "id": "1"}])

    monkeypatch.setattr(ar, "get_llm", lambda role="generator": FakeLLM())
    monkeypatch.setattr(ar, "corpus_catalog", lambda: "- 기관 (2024) 보고서")
    d = ar._agent_decide(QUESTION, PURPOSE, "2026-09-30", True, False)
    assert seen["tools"] == ["search_documents", "web_search"] and seen["kw"] == {"tool_choice": "required"}
    assert "기관 (2024) 보고서" in seen["prompt"] and "반드시 도구를" in seen["prompt"]
    assert (d["route"], d["query"]) == ("docs", "q")
    ar._agent_decide(QUESTION, PURPOSE, "2026-09-30", False, True)
    assert seen["tools"] == ["search_documents"] and seen["kw"] == {}
    assert "web_search" not in seen["prompt"] and "직접 답" in seen["prompt"]


def test_prompts_render():
    from core.prompts import render

    g = render("rag_generate", question="q", purpose="p", run_date="2026-09-30", context="[D1a2b3] ...", feedback="")
    assert "직전 답변" not in g and "(YYYY년 발표 전망)" in g and "지시문" in g
    assert "직전 답변" in render("rag_generate", question="q", purpose="p", run_date="d", context="c", feedback="37")
    c = render("rag_check", question="q", purpose="p", answer="a", context="c")
    assert "grounded" in c and "answers_question" in c


# ── v1 교정형 경로 (탐색 전용): v1-safe 코드와 같은 LLM 입력·결과
def _git_show(rel: str) -> str:
    out = subprocess.run(["git", "show", f"v1-safe:{rel}"], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        pytest.skip("v1-safe 태그 없음")
    return out.stdout.decode("utf-8")


def _v1_module(monkeypatch) -> types.ModuleType:
    mod = types.ModuleType("agentic_rag_v1")
    monkeypatch.setitem(sys.modules, mod.__name__, mod)  # pydantic 이 스키마의 지연 주석(Judgment)을 풀 때 찾는다
    exec(compile(_git_show("rag/agentic_rag.py"), "agentic_rag_v1", "exec"), mod.__dict__)
    return mod


def _run_corrective(mod, monkeypatch, relevant_per_call):
    """mod.agentic_rag 를 가짜 structured·검색기·웹 검색으로 돌리고 (결과, 렌더된 LLM 입력, 웹 호출)을 돌려준다."""
    prompts, webs = [], []

    def structured(schema, role="generator"):
        class R:
            @staticmethod
            def invoke(prompt):
                prompts.append((schema.__name__, role, prompt))
                if schema.__name__ == "Grades":
                    ok = relevant_per_call[min(sum(p[0] == "Grades" for p in prompts), len(relevant_per_call)) - 1]
                    return schema.model_validate({"judgments": [{"idx": i, "relevant": i in ok} for i in range(8)]})
                return schema.model_validate({"query": f"재작성 {sum(p[0] == 'Rewrite' for p in prompts)}"})
        return R()

    def web_search(query, registry, agent, **kw):
        webs.append((query, agent, kw))
        return [registry.add_web({"url": "https://n.example.com/a", "title": "기사", "content": "본문"},
                                 agent=agent, query=query, access_date="2026-09-30")]

    retriever = FakeRetriever()
    monkeypatch.setattr(mod, "structured", structured)
    monkeypatch.setattr(mod, "get_hybrid_retriever", lambda: retriever)
    monkeypatch.setattr(mod, "web_search", web_search)
    reg = SourceRegistry({"W00000": {"id": "W00000", "key": "x", "kind": "web"}})
    ids, trace = mod.agentic_rag("국내외 애그테크 스타트업 사례", "발굴", reg, "discovery")
    return (ids, trace, reg.data), prompts, webs, retriever.queries


@pytest.mark.parametrize("relevant_per_call", [[{0, 1, 2, 5}], [{1}, set(), {0, 3}], [set()]])
def test_agentic_rag_matches_v1(monkeypatch, relevant_per_call):
    def boom(*a, **k):
        raise AssertionError("v1 경로는 도구 선택·생성·점검을 부르지 않는다")

    for name in ("_agent_decide", "_generate_llm", "_check_llm"):
        monkeypatch.setattr(ar, name, boom)
    monkeypatch.setattr(ar, "_RAG", None)
    v1 = _v1_module(monkeypatch)
    old = _run_corrective(v1, monkeypatch, relevant_per_call)
    new = _run_corrective(ar, monkeypatch, relevant_per_call)
    assert new == old  # 근거 id·trace·registry, rag_grade/rag_rewrite 렌더 입력 문자열, 웹 보완 인자, 검색 질의
    assert all(p[0] in ("Grades", "Rewrite") for p in new[1])


def test_corpus_catalog_same_as_v1_market():
    src = _git_show("agents/market.py")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "corpus_catalog")
    ns: dict = {"get_chunks": ar.get_chunks}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "market_v1", "exec"), ns)
    ar.corpus_catalog.cache_clear()
    n_docs = len(load_manifest())  # 문서마다 한 줄
    assert ar.corpus_catalog() == ns["corpus_catalog"]() and ar.corpus_catalog().count("\n") == n_docs - 1


# ── bind_tools 응답 재생 (키 필요: uv run pytest -m api tests/test_rag_subgraph.py)
@pytest.mark.api
def test_bind_tools_decision_replays_from_llm_cache(tmp_path, monkeypatch, set_cfg):
    """도구 선택 LLM(bind_tools) 응답의 tool_calls 가 SQLiteCache 에 저장되고, REPLAY_OFFLINE(네트워크 차단)에서
    같은 호출이 캐시로 재생되는지 확인한다. 실제 호출은 1회(gpt-4.1-mini, 약 $0.0003). 캐시는 임시 디렉터리."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY 없음")
    import core.llm as llm
    from langchain_core.globals import get_llm_cache, set_llm_cache
    from langchain_openai import ChatOpenAI

    ar.corpus_catalog()  # 조각 캐시는 replay/ 에서 먼저 읽어 둔다 (아래에서 캐시 경로를 바꾸므로)
    monkeypatch.setattr(ChatOpenAI, "_generate", ChatOpenAI._generate)  # 재생 단계의 네트워크 차단을 끝나고 되돌림
    monkeypatch.setattr(llm, "_cache_ready", False)
    saved = get_llm_cache()
    set_cfg("cache.dir", str(tmp_path / "cache"))
    try:
        llm.get_llm.cache_clear()
        live = ar._agent_decide(QUESTION, PURPOSE, "2026-09-30", True, False)
        monkeypatch.setenv("REPLAY_OFFLINE", "1")
        llm.get_llm.cache_clear()
        replayed = ar._agent_decide(QUESTION, PURPOSE, "2026-09-30", True, False)
    finally:
        llm.get_llm.cache_clear()
        set_llm_cache(saved)
    print("live:", live, "\nreplayed:", replayed)
    assert live["route"] in ("docs", "both") and live["tools"]
    assert replayed == live
