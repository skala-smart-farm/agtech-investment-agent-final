"""Agentic RAG. 경로가 두 개 있다.

1) agentic_rag() — v1 교정형 경로 (build_evidence_graph, 탐색 에이전트용, 답변 생성 없음)
   retrieve(하이브리드 검색) → grade(Judge LLM 이 조각별 관련성 O/X) →
     관련 조각이 충분하면(min_relevant 이상) finish
     부족하면 rewrite(질의 재작성) 후 다시 retrieve  (최대 max_rewrites 회)
     그래도 부족하면 web(웹 검색으로 보완, Corrective RAG) → finish
   관련 조각은 검색기 순위(Kiwi BM25 + FAISS 의 RRF 순위) 그대로 두고 앞에서부터 top_k 개만 컨텍스트로 쓴다.
   top_k 때문에 버린 관련 조각 수는 trace 에 kept/dropped 로 남긴다 (순위가 실제로 무엇을 버렸는지 보이게).
   LLM 입력(프롬프트·조각 목록 형식)과 웹 보완 인자가 v1 과 같다. 그래야 v1 캐시가 맞아 발굴 결과가 바뀌지 않는다.

2) answer_question() — v2 서브그래프 (build_rag_graph, 기술·시장성 에이전트와 답변 평가 eval_judge 가 쓴다)
   agent   : LLM 이 도구를 고른다 (search_documents / web_search / 둘 다 / 직접 답은 allow_direct 일 때만)
   retrieve·grade·rewrite·web : 1) 과 같은 코드·프롬프트 (web 은 도구 인자 또는 v1 보완 인자)
   generate: 근거 id 를 문장마다 인용한 답변
   check   : Self-RAG 점검. 코드가 답변 속 수치가 근거에 있는지 먼저 보고, 통과하면 LLM 이 이진 2문항으로 판정
             grounded → finish / not_grounded → 답변만 다시(max_regenerations) / not_useful → 질의를 고쳐 다시 검색
   근거 저장소(SourceRegistry)는 호출한 에이전트의 객체 하나를 노드들이 제자리에서 갱신한다(복사본을 만들지 않음).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.config import get_config, run_date
from core.llm import get_llm, structured
from core.prompts import render
from rag.index import get_chunks, get_hybrid_retriever
from tools.agent_tools import make_tools
from tools.sources import SourceRegistry
from tools.web_search import web_search


class RAGState(TypedDict):
    question: str
    purpose: str
    query: str
    retrieved: list[dict]
    relevant: list[dict]
    rewrites: int
    web_ids: list[str]
    evidence_ids: list[str]
    registry: dict
    trace: list[dict]
    agent: str


class Judgment(BaseModel):
    idx: int = Field(description="조각 번호")
    relevant: bool = Field(description="질문에 답하는 데 직접 쓸 수 있는 정보가 있으면 true")


class Grades(BaseModel):
    judgments: list[Judgment]


class Rewrite(BaseModel):
    query: str = Field(description="검색에 더 잘 걸리도록 고친 질의 (핵심 명사·수치 표현 포함)")


# ── LLM 판단부 (테스트에서 monkeypatch 로 바꿔 끼운다)
def _grade_llm(question: str, purpose: str, chunks: list[dict]) -> set[int]:
    """조각별 관련 O/X. 관련 조각 번호 집합을 돌려준다 (프롬프트·조각 목록 형식은 v1 그대로)."""
    listing = "\n\n".join(
        f"[{i}] ({c['meta'].get('publisher')} {c['meta'].get('year')}, p.{c['meta'].get('page')})\n{c['text'][:900]}"
        for i, c in enumerate(chunks))
    res: Grades = structured(Grades, "judge").invoke(
        render("rag_grade", question=question, purpose=purpose, chunks=listing))
    return {j.idx for j in res.judgments if j.relevant}


def _rewrite_llm(question: str, purpose: str, tried: list[str]) -> str:
    """이미 시도한 질의와 다른 검색 질의 (프롬프트·입력 형식은 v1 그대로)."""
    res: Rewrite = structured(Rewrite).invoke(
        render("rag_rewrite", question=question, purpose=purpose, tried="\n".join(tried)))
    return res.query


# ── 1) v1 교정형 경로 노드 (v2 서브그래프도 retrieve·grade 를 그대로 쓴다)
def _retrieve(state: RAGState) -> dict:
    cfg = get_config()
    docs = get_hybrid_retriever().invoke(state["query"])[: cfg.rag.candidate_k]
    got = [{"text": d.page_content, "meta": d.metadata} for d in docs]
    return {"retrieved": got}


def _grade(state: RAGState) -> dict:
    cfg = get_config()
    chunks = state["retrieved"]
    if not chunks:
        return {"trace": state["trace"] + [{"query": state["query"], "retrieved": 0, "relevant": 0,
                                            "kept": 0, "dropped": 0}]}
    ok = _grade_llm(state["question"], state["purpose"], chunks)
    seen = {c["meta"]["chunk_id"] for c in state["relevant"]}
    # 관련 판정 조각을 검색기 순위 순서 그대로 (재정렬 없음)
    new = [(i, c) for i, c in enumerate(chunks) if i in ok and c["meta"]["chunk_id"] not in seen]
    room = max(0, cfg.rag.top_k - len(state["relevant"]))
    kept = new[:room]  # top_k 를 넘는 관련 조각은 버린다
    trace = state["trace"] + [{"query": state["query"], "retrieved": len(chunks), "relevant": len(new),
                               "kept": len(kept), "dropped": len(new) - len(kept),
                               "kept_ranks": [i + 1 for i, _ in kept], "top_k": cfg.rag.top_k}]
    return {"relevant": state["relevant"] + [c for _, c in kept], "trace": trace}


def _route(state: RAGState) -> Literal["finish", "rewrite", "web"]:
    cfg = get_config()
    if len(state["relevant"]) >= cfg.rag.min_relevant:
        return "finish"
    if state["rewrites"] < cfg.rag.max_rewrites:
        return "rewrite"
    return "web" if cfg.rag.web_fallback else "finish"


def _rewrite(state: RAGState) -> dict:
    tried = [t["query"] for t in state["trace"]]
    return {"query": _rewrite_llm(state["question"], state["purpose"], tried), "rewrites": state["rewrites"] + 1}


def _web(state: RAGState) -> dict:
    reg = SourceRegistry(state["registry"])
    ids = web_search(state["question"], reg, agent=state["agent"], topic="general", recent=False, max_results=4)
    return {"web_ids": ids, "registry": reg.data,
            "trace": state["trace"] + [{"query": state["question"], "web_fallback": len(ids)}]}


def _finish(state: RAGState) -> dict:
    reg = SourceRegistry(state["registry"])
    ids = [reg.add_doc(c["meta"], c["text"], agent=state["agent"], query=state["question"])
           for c in state["relevant"]]
    return {"evidence_ids": ids + state["web_ids"], "registry": reg.data}


def build_evidence_graph():
    """v1 교정형 경로 그래프 (생성 없음). agentic_rag() 가 쓴다."""
    g = StateGraph(RAGState)
    g.add_node("retrieve", _retrieve)
    g.add_node("grade", _grade)
    g.add_node("rewrite", _rewrite)
    g.add_node("web", _web)
    g.add_node("finish", _finish)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route, {"finish": "finish", "rewrite": "rewrite", "web": "web"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web", "finish")
    g.add_edge("finish", END)
    return g.compile()


_RAG = None


def agentic_rag(question: str, purpose: str, registry: SourceRegistry, agent: str) -> tuple[list[str], list[dict]]:
    """질문 하나에 대한 근거 id 목록과 수행 기록(trace)을 돌려준다. registry 는 제자리에서 갱신된다.
    v1 교정형 경로(검색·채점·재작성·웹 보완, 답변 생성 없음) 그대로다. 탐색 에이전트가 기업명 추출용 근거를 모을 때 쓴다
    (기술·시장성 에이전트는 v2 에서 answer_question 을 쓴다)."""
    global _RAG
    if _RAG is None:
        _RAG = build_evidence_graph()
    out = _RAG.invoke({"question": question, "purpose": purpose, "query": question, "retrieved": [],
                       "relevant": [], "rewrites": 0, "web_ids": [], "evidence_ids": [],
                       "registry": registry.data, "trace": [], "agent": agent})
    registry.data.update(out["registry"])
    return out["evidence_ids"], out["trace"]


@lru_cache(maxsize=1)
def corpus_catalog() -> str:
    """도구 선택 LLM 이 코퍼스에 무엇이 있는지 알도록 문서 목록(기관·연도·제목)을 만든다.
    출력 문자열은 v1 agents/market.py 의 corpus_catalog 와 같다 (market_decompose 프롬프트 입력 = 캐시 키)."""
    docs: dict[str, str] = {}
    for ch in get_chunks():
        m = ch.metadata
        docs.setdefault(m.get("doc_id"), f"- {m.get('publisher')} ({m.get('year')}) {m.get('title')}")
    return "\n".join(docs.values())


# ── 2) v2 서브그래프: 도구 선택 → 검색 → 채점 → 재작성/웹 보완 → 인용 답변 생성 → Self-RAG 점검
class AnswerState(TypedDict):
    question: str
    purpose: str
    agent: str
    run_date: str
    allow_web: bool
    allow_direct: bool
    route: str             # agent 가 고른 경로 docs | web | both | direct (이후 바뀌지 않음)
    query: str             # 문서 검색 질의 (search_documents 인자, 재작성하면 바뀜)
    web_query: str         # web_search 도구 인자
    recent: bool           # web_search 도구 인자 (True 면 최근 1년 뉴스 우선)
    retrieved: list[dict]
    relevant: list[dict]
    web_ids: list[str]
    web_done: bool         # 웹 검색을 이미 했는지 (한 번만)
    context_ids: list[str]  # generate 가 본 근거 id
    answer: str
    cited_ids: list[str]
    check: str             # '' | grounded | not_grounded | not_useful
    feedback: str          # not_grounded 일 때 재생성에 넘기는 점검 결과
    best: dict             # 지금까지 점검한 답 중 가장 나은 것 (grounded > not_useful > not_grounded, 같으면 나중 것)
    rewrites: int
    regenerations: int
    evidence_ids: list[str]
    status: str
    trace: list[dict]


class Generated(BaseModel):
    answer: str = Field(description="근거만으로 쓴 답변. 문장마다 끝에 [근거 id]")
    cited_ids: list[str] = Field(description="답변에서 실제로 인용한 근거 id")


class Check(BaseModel):
    reason: str = Field(description="판정 이유 한두 문장 (근거에 없는 내용이 있으면 그 내용을 적는다)")
    grounded: bool = Field(description="답변의 모든 사실·수치가 근거에 있으면 true")
    answers_question: bool = Field(description="답변이 질문이 묻는 것에 직접 답하면 true")


CITE = re.compile(r"\[([DW][0-9a-f]{5})\]")
NOT_FOUND = "확인되지 않음"
CHECK_RANK = {"grounded": 2, "not_useful": 1, "not_grounded": 0}  # not_useful 은 근거는 맞고 질문에 덜 맞는 답


def _agent_decide(question: str, purpose: str, run_date: str, allow_web: bool, allow_direct: bool) -> dict:
    """도구 선택 LLM. 도구 이름·설명(docstring)·인자를 bind_tools 로 알려 주고, 고른 도구 호출(tool_calls)을 읽는다.
    도구 실행은 여기서 하지 않는다(retrieve·web 노드가 같은 근거 저장소로 실행). allow_direct 가 False 면
    tool_choice='required' 로 도구를 반드시 하나 이상 고르게 한다."""
    tools = make_tools(SourceRegistry(), "rag")  # 스키마만 쓴다
    bound = [tools["search_documents"]] + ([tools["web_search"]] if allow_web else [])
    llm = get_llm("generator").bind_tools(bound, **({} if allow_direct else {"tool_choice": "required"}))
    msg = llm.invoke(render("rag_agent", question=question, purpose=purpose, run_date=run_date,
                            catalog=corpus_catalog(), allow_web=allow_web, allow_direct=allow_direct))
    return parse_tool_calls(msg.tool_calls, msg.content if isinstance(msg.content, str) else "", question,
                            allow_web, allow_direct)


def parse_tool_calls(calls: list[dict], content: str, question: str, allow_web: bool, allow_direct: bool) -> dict:
    """tool_calls → 경로. 두 도구를 함께 부르면 both(문서 기준 수치 + 최신 뉴스). 같은 도구를 여러 번 부르면 첫 호출만 쓴다.
    도구를 고르지 않았는데 직접 답이 허용되지 않으면 질문 그대로 문서 검색(docs)."""
    docs = next((c.get("args") or {} for c in calls if c.get("name") == "search_documents"), None)
    web = next((c.get("args") or {} for c in calls if c.get("name") == "web_search"), None) if allow_web else None
    if docs is not None and web is not None:
        route = "both"
    elif docs is not None:
        route = "docs"
    elif web is not None:
        route = "web"
    else:
        route = "direct" if allow_direct else "docs"
    reason = (content or "").strip()
    if route == "docs" and docs is None:
        reason = "도구를 고르지 않음 → 직접 답 비허용이라 질문 그대로 문서 검색"
    return {"route": route, "query": str((docs or {}).get("query") or question).strip(),
            "web_query": str((web or {}).get("query") or question).strip(),
            "recent": bool((web or {}).get("recent", True)),
            "answer": reason if route == "direct" else "", "reason": reason if route != "direct" else "",
            "tools": [c.get("name") for c in calls]}


def _generate_llm(question: str, purpose: str, run_date: str, context: str, feedback: str) -> tuple[str, list[str]]:
    res: Generated = structured(Generated).invoke(
        render("rag_generate", question=question, purpose=purpose, run_date=run_date, context=context,
               feedback=feedback))
    return res.answer, res.cited_ids


def _check_llm(question: str, purpose: str, answer: str, context: str) -> tuple[bool, bool, str]:
    res: Check = structured(Check, "judge").invoke(
        render("rag_check", question=question, purpose=purpose, answer=answer, context=context))
    return res.grounded, res.answers_question, res.reason


# ── 코드 수치 대조
_NUM = re.compile(r"\d+(?:\.\d+)?")
_YEAR = re.compile(r"(?:19|20)\d{2}")


def _canon(n: str) -> str:
    """'37.0' → '37', '09' → '9' (같은 값을 같은 문자열로)."""
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n.lstrip("0") or "0"


def numbers(text: str) -> list[str]:
    """글 속 수치(정규화). 인용 괄호 [D1a2b3] 안의 숫자는 빼고, 천 단위 쉼표는 없앤다('5,000억' → 5000)."""
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", re.sub(r"\[[^\]]*\]", " ", text or ""))
    return [_canon(n) for n in _NUM.findall(t)]


def unsupported_numbers(answer: str, evidence: str, known: str = "") -> list[str]:
    """답변 속 수치 중 근거에 없는 것. 오탐을 줄이려고 두 자리 이상 수치만 보고(한 자리·'1조' 같은 표기 차이 제외),
    4자리 연도와 질문·목적·기준일(known)에 이미 나온 수치는 제외한다."""
    have = set(numbers(evidence)) | set(numbers(known))
    out = []
    for n in numbers(answer):
        if len(n.replace(".", "")) < 2 or _YEAR.fullmatch(n) or n in have or n in out:
            continue
        out.append(n)
    return out


def _searched(state: AnswerState) -> bool:
    """문서 검색(채점)을 한 번이라도 했는지."""
    return any(t.get("step") == "grade" for t in state["trace"])


def _context(reg: SourceRegistry, ids: list[str]) -> str:
    """generate·check 에 넘기는 근거 목록 (reg.brief): 문서 조각은 rag.context_doc_chars(조각이 800자라 사실상 전부),
    웹 근거는 스니펫 rag.context_web_chars 까지."""
    r = get_config().rag
    docs = reg.brief([i for i in ids if i.startswith("D")], r.context_doc_chars)
    webs = reg.brief([i for i in ids if i.startswith("W")], r.context_web_chars)
    return "\n\n".join(x for x in (docs, webs) if x)


def build_rag_graph(reg: SourceRegistry | None = None):
    """v2 Agentic RAG 서브그래프. 노드는 reg(호출한 에이전트의 근거 저장소 객체 하나)를 제자리에서 갱신한다.
    reg 를 주지 않으면 빈 저장소를 쓴다(그래프 모양 확인용)."""
    reg = reg if reg is not None else SourceRegistry()

    def agent(state: AnswerState) -> dict:
        d = _agent_decide(state["question"], state["purpose"], state["run_date"], state["allow_web"],
                          state["allow_direct"])
        entry = {"step": "agent", "route": d["route"], "tools": d.get("tools", []), "reason": d.get("reason", "")}
        if d["route"] in ("docs", "both"):
            entry["query"] = d["query"]
        if d["route"] in ("web", "both"):
            entry.update(web_query=d["web_query"], recent=d["recent"])
        return {"route": d["route"], "query": d["query"], "web_query": d["web_query"], "recent": d["recent"],
                "answer": d.get("answer", ""), "trace": state["trace"] + [entry]}

    def retrieve(state: AnswerState) -> dict:
        return {**_retrieve(state), "check": "", "feedback": ""}  # 새 검색: 이전 점검 결과는 지운다

    def grade(state: AnswerState) -> dict:
        out = _grade(state)
        out["trace"][-1] = {"step": "grade", **out["trace"][-1]}
        return out

    def rewrite(state: AnswerState) -> dict:
        tried = [t["query"] for t in state["trace"] if t.get("step") == "grade"]
        out = {"query": _rewrite_llm(state["question"], state["purpose"], tried), "rewrites": state["rewrites"] + 1}
        if state["check"] == "not_useful":  # 답변이 질문에 맞지 않음 → 지금 조각은 비우고 고친 질의로 새로 모은다
            out["relevant"] = []
        out["trace"] = state["trace"] + [{"step": "rewrite", "query": out["query"], "after": state["check"] or "grade"}]
        return out

    def web(state: AnswerState) -> dict:
        if state["route"] in ("web", "both") and not state["web_done"]:  # agent 가 고른 web_search 도구 인자
            q = state["web_query"]
            ids = web_search(q, reg, agent=state["agent"], topic="news", recent=state["recent"])
            entry = {"step": "web", "query": q, "recent": state["recent"], "web_results": len(ids)}
        else:  # 교정형 보완: v1 과 같은 인자
            q = state["question"]
            ids = web_search(q, reg, agent=state["agent"], topic="general", recent=False, max_results=4)
            entry = {"step": "web", "query": q, "web_fallback": len(ids)}
        return {"web_ids": list(dict.fromkeys(state["web_ids"] + ids)), "web_done": True,
                "trace": state["trace"] + [entry]}

    def generate(state: AnswerState) -> dict:
        if state["route"] == "direct" and not _searched(state):  # 검색 없이 agent 가 쓴 답을 그대로 점검에 넘긴다
            return {"cited_ids": [], "context_ids": [],
                    "trace": state["trace"] + [{"step": "generate", "direct": True}]}
        doc_ids = [reg.add_doc(c["meta"], c["text"], agent=state["agent"], query=state["question"])
                   for c in state["relevant"]]
        ids = list(dict.fromkeys(doc_ids + state["web_ids"]))
        context = _context(reg, ids)
        regen = state["check"] == "not_grounded"  # 점검에서 바로 돌아온 경우만 재생성으로 센다
        answer, cited = _generate_llm(state["question"], state["purpose"], state["run_date"],
                                      context or "(근거 없음)", state["feedback"] if regen else "")
        cited = [i for i in dict.fromkeys(list(cited) + CITE.findall(answer)) if i in ids]
        return {"answer": answer, "cited_ids": cited, "context_ids": ids,
                "regenerations": state["regenerations"] + int(regen),
                "trace": state["trace"] + [{"step": "generate", "context": len(ids), "cited": len(cited),
                                            "regeneration": regen}]}

    def check(state: AnswerState) -> dict:
        ids = state["context_ids"]
        context = _context(reg, ids)
        # 수치 대조 대상: generate 가 받은 근거 목록 + 그 근거들의 본문 (인용 id 가 틀린 경우는 LLM 점검이 본다)
        evidence = context + "\n" + " ".join(reg.text(i) for i in ids)
        known = " ".join((state["question"], state["purpose"], state["run_date"]))
        bad = unsupported_numbers(state["answer"], evidence, known)
        entry = {"step": "check"}
        if bad:  # 코드 수치 대조에서 걸리면 LLM 점검 없이 not_grounded
            verdict = "not_grounded"
            feedback = (f"근거에 없는 수치: {', '.join(bad)}. 근거에 있는 수치만 근거 표기 그대로(단위 환산 없이) 쓰고, "
                        "근거에 없는 내용은 빼라.")
            entry["numbers"] = bad
        else:
            grounded, useful, reason = _check_llm(state["question"], state["purpose"], state["answer"],
                                                  context or "(근거 없음)")
            verdict = "not_grounded" if not grounded else ("not_useful" if not useful else "grounded")
            feedback = reason if verdict == "not_grounded" else ""
            entry["reason"] = reason
        entry["check"] = verdict
        out = {"check": verdict, "feedback": feedback, "trace": state["trace"] + [entry]}
        best = state.get("best") or {}
        if not best or CHECK_RANK[verdict] >= CHECK_RANK[best["check"]]:  # 재작성·재생성한 답이 더 나빠지면 앞의 답을 남긴다
            out["best"] = {"check": verdict, "answer": state["answer"], "cited_ids": state["cited_ids"],
                           "context_ids": state["context_ids"]}
        return out

    def finish(state: AnswerState) -> dict:
        b = state.get("best") or {k: state[k] for k in ("check", "answer", "cited_ids", "context_ids")}
        cited = [i for i in b["cited_ids"] if i in reg.data]
        evidence = cited or [i for i in b["context_ids"] if i in reg.data]
        if NOT_FOUND in b["answer"] and not cited:
            status = "not_found"
        elif b["check"] == "grounded" and evidence and NOT_FOUND not in b["answer"]:
            status = "grounded"
        else:
            status = "partial"
        return {"answer": b["answer"], "check": b["check"], "evidence_ids": evidence, "cited_ids": cited,
                "status": status}

    g = StateGraph(AnswerState)
    for name, fn in (("agent", agent), ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite), ("web", web),
                     ("generate", generate), ("check", check), ("finish", finish)):
        g.add_node(name, fn)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"retrieve": "retrieve", "web": "web", "generate": "generate"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite", "web": "web"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web", "generate")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", route_after_check,
                            {"finish": "finish", "generate": "generate", "rewrite": "rewrite", "retrieve": "retrieve"})
    g.add_edge("finish", END)
    return g.compile()


# 설계서 그림과 비교하는 v2 서브그래프 간선 (tests/test_rag_subgraph.py 가 컴파일된 그래프와 같은지 확인)
RAG_EDGES = {(START, "agent"), ("agent", "retrieve"), ("agent", "web"), ("agent", "generate"), ("retrieve", "grade"),
             ("grade", "generate"), ("grade", "rewrite"), ("grade", "web"), ("rewrite", "retrieve"),
             ("web", "generate"), ("generate", "check"), ("check", "finish"), ("check", "generate"),
             ("check", "rewrite"), ("check", "retrieve"), ("finish", END)}


# ── 라우터 (순수 함수)
def route_after_agent(state: AnswerState) -> Literal["retrieve", "web", "generate"]:
    """docs·both → 문서 검색, web → 웹 검색, direct → 검색 없이 점검으로."""
    return {"docs": "retrieve", "both": "retrieve", "web": "web"}.get(state["route"], "generate")


def route_after_grade(state: AnswerState) -> Literal["generate", "rewrite", "web"]:
    """관련 조각이 충분하면 generate(both 면 웹 검색을 먼저). 부족하면 재작성, 재작성을 다 쓰면 웹 보완(한 번)."""
    cfg = get_config()
    web_left = state["allow_web"] and not state["web_done"]
    if len(state["relevant"]) >= cfg.rag.min_relevant:
        return "web" if state["route"] == "both" and web_left else "generate"
    if state["rewrites"] < cfg.rag.max_rewrites:
        return "rewrite"
    return "web" if web_left and (cfg.rag.web_fallback or state["route"] == "both") else "generate"


def route_after_check(state: AnswerState) -> Literal["finish", "generate", "rewrite", "retrieve"]:
    """Self-RAG 분기.
    grounded → finish
    direct 답이 점검을 못 넘음(아직 검색 전) → retrieve (질문 그대로 문서 검색, 1회)
    not_grounded, 재생성 여유 있음 → generate (검색은 두고 답변만 다시)
    not_useful, 문서 경로이고 재작성 여유 있음 → rewrite (질의를 고쳐 다시 검색)
    그 밖(상한 도달, 웹 경로의 not_useful) → finish (partial)"""
    cfg = get_config()
    c = state["check"]
    if c == "grounded":
        return "finish"
    if state["route"] == "direct" and not _searched(state):
        return "retrieve"
    if c == "not_grounded" and state["regenerations"] < cfg.rag.max_regenerations:
        return "generate"
    if c == "not_useful" and state["route"] != "web" and state["rewrites"] < cfg.rag.max_rewrites:
        return "rewrite"
    return "finish"


RESULT_KEYS = ("answer", "evidence_ids", "cited_ids", "status", "check", "route", "rewrites", "regenerations", "trace")


def answer_question(question: str, purpose: str, registry: SourceRegistry, agent: str, *,
                    allow_web: bool = True, allow_direct: bool = False) -> dict:
    """질문 하나에 근거를 달아 답한다 (계약 C4, v2 서브그래프).
    반환: {'answer': str, 'evidence_ids': [str], 'cited_ids': [str], 'status': 'grounded'|'partial'|'not_found',
           'check': 마지막으로 고른 답의 점검 결과 grounded|not_useful|not_grounded,
           'route': 'docs'|'web'|'both'|'direct', 'rewrites': int, 'regenerations': int, 'trace': [dict]}
    - evidence_ids: 답변이 인용한 근거 id (인용이 없으면 generate 가 본 근거 전부)
    - answer: 점검한 답 중 가장 나은 것(grounded > not_useful > not_grounded, 같으면 나중 것)
    - status: 인용 없이 '확인되지 않음'이면 not_found, 점검을 통과하고 빠진 부분이 없으면 grounded, 그 밖은 partial
    - route: agent 가 고른 경로. both 는 두 도구를 함께 부른 경우(문서 기준 수치 + 최신 뉴스)
    registry 는 제자리에서 갱신된다(노드가 같은 객체에 근거를 등록). allow_direct(검색 없이 답하기)의 기본값은
    False 다(수치 환각 방지, 분석 에이전트 호출). allow_web=False 면 web_search 도구와 웹 보완을 쓰지 않는다."""
    out = build_rag_graph(registry).invoke(
        initial_state(question, purpose, agent, run_date(), allow_web=allow_web, allow_direct=allow_direct),
        config={"recursion_limit": get_config().rag.rag_recursion_limit})
    return {k: out[k] for k in RESULT_KEYS}


def initial_state(question: str, purpose: str, agent: str, date: str, *, allow_web: bool,
                  allow_direct: bool) -> AnswerState:
    """v2 서브그래프 입력 State."""
    return {"question": question, "purpose": purpose, "agent": agent, "run_date": date, "allow_web": allow_web,
            "allow_direct": allow_direct, "route": "", "query": question, "web_query": question, "recent": True,
            "retrieved": [], "relevant": [], "web_ids": [], "web_done": False, "context_ids": [], "answer": "",
            "cited_ids": [], "check": "", "feedback": "", "best": {}, "rewrites": 0, "regenerations": 0, "evidence_ids": [],
            "status": "", "trace": []}


def answer_question_v1(question: str, purpose: str, registry: SourceRegistry, agent: str, *,
                       allow_web: bool = True, allow_direct: bool = False) -> dict:
    """v1 호환 구현(폴백용): 교정형 경로(agentic_rag)로 근거 id 만 모은다. 답변 생성·점검은 하지 않는다.
    - answer 는 빈 문자열, cited_ids 는 빈 목록 (호출한 에이전트가 근거를 직접 읽고 분석한다)
    - status: 근거 id 가 있으면 'grounded', 없으면 'not_found' / route: 항상 'docs'
    - rewrites: 질의 재작성 횟수 (trace 의 검색 기록 수 − 1), regenerations: 0
    - allow_web·allow_direct 는 v2 와 같은 시그니처를 위해 받기만 한다. v1 경로에는 direct 가 없고,
      웹 보완 여부는 config rag.web_fallback 이 정한다.
    v2 서브그래프에 문제가 생기면 answer_question 대신 이 함수를 쓰면 된다. registry 는 제자리에서 갱신된다."""
    ids, trace = agentic_rag(question, purpose, registry, agent)
    rewrites = max(0, sum(1 for t in trace if "retrieved" in t) - 1)
    return {"answer": "", "evidence_ids": ids, "cited_ids": [], "status": "grounded" if ids else "not_found",
            "route": "docs", "rewrites": rewrites, "regenerations": 0, "trace": trace}
