"""목적별 도구(tools/agent_tools.py) 검사 (P2). 검색기·웹 검색·원문 수집·LLM 은 가짜로 바꿔 끼운다.

- make_tools: 도구 3개의 이름·설명(용도·한계·최신성)·인자
- search_documents: content_and_artifact, 아티팩트 = [{'text','meta'}]
- web_search: content_and_artifact, 아티팩트 = [근거 id], 같은 근거 저장소 객체에 등록
- summarize_document: URL → 원문 근거 등록 + 요약 끝에 [W…], 근거 id → 등록 본문 요약, 실패하면 ''
"""
from __future__ import annotations

import re

import yaml
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import tools.agent_tools as at
from core.config import ROOT
from tools.sources import SourceRegistry


def _call(t, args: dict):
    """LLM 이 부른 것처럼 ToolCall 로 실행한다 (content_and_artifact 는 ToolMessage.artifact 로 받는다)."""
    return t.invoke({"type": "tool_call", "id": "call_1", "name": t.name, "args": args})


def test_make_tools_names_and_descriptions():
    tools = at.make_tools(SourceRegistry(), "tech")
    assert tuple(tools) == at.TOOL_NAMES == ("search_documents", "web_search", "summarize_document")
    for name, t in tools.items():
        assert t.name == name and t.description
    sd, ws, sm = tools["search_documents"], tools["web_search"], tools["summarize_document"]
    assert "13종 196쪽" in sd.description and "2023~2026" in sd.description and "web_search" in sd.description
    assert "개별 스타트업" in sd.description                       # 한계
    assert "최근" in ws.description                                  # 최신성
    assert sd.response_format == ws.response_format == "content_and_artifact"
    assert set(sd.args) == {"query"} and set(ws.args) == {"query", "recent"} and set(sm.args) == {"source", "focus"}
    assert ws.args["recent"]["default"] is True and "최근 1년" in ws.args["recent"]["description"]
    assert "빈 문자열" in sm.description


def test_search_documents_description_matches_manifest():
    """설명의 '13종 196쪽, 2023~2026' 이 실제 코퍼스 목록과 같은지 (코퍼스가 바뀌면 설명도 고쳐야 함)."""
    docs = yaml.safe_load((ROOT / "data/manifest.yaml").read_text(encoding="utf-8"))["documents"]
    years = [d["year"] for d in docs]
    desc = at.make_tools(SourceRegistry(), "x")["search_documents"].description
    want = f"{len(docs)}종 {sum(d['pages_total'] for d in docs)}쪽, {min(years)}~{max(years)}"
    assert want in desc


def test_search_documents_artifact_is_chunk_dicts(monkeypatch):
    class Retriever:
        def invoke(self, q):
            return [Document(page_content=f"조각 {i} 시장 규모", metadata={"publisher": "기관", "year": 2024, "page": i})
                    for i in range(10)]

    monkeypatch.setattr(at, "get_hybrid_retriever", lambda: Retriever())
    msg = _call(at.make_tools(SourceRegistry(), "tech")["search_documents"], {"query": "시장 규모"})
    assert isinstance(msg.artifact, list) and len(msg.artifact) == 8  # rag.candidate_k
    assert all(set(c) == {"text", "meta"} for c in msg.artifact)
    assert "[0] (기관 2024, p.0)" in msg.content


def test_web_search_artifact_is_ids_in_same_registry(monkeypatch):
    seen = {}

    def fake(query, registry, agent, **kw):
        seen.update(query=query, registry=registry, agent=agent, **kw)
        return [registry.add_web({"url": "https://news.example.com/a", "title": "팜랩 투자 유치", "content": "20억 원"},
                                 agent=agent, query=query, access_date="2026-09-30")]

    monkeypatch.setattr(at, "search_web", fake)
    reg = SourceRegistry()
    msg = _call(at.make_tools(reg, "tech")["web_search"], {"query": "팜랩 투자"})
    assert isinstance(msg.artifact, list) and all(isinstance(i, str) and i.startswith("W") for i in msg.artifact)
    assert seen["registry"] is reg and set(msg.artifact) <= set(reg.data)  # 같은 객체에 등록
    assert (seen["topic"], seen["recent"], seen["agent"]) == ("news", True, "tech")
    assert msg.artifact[0] in msg.content
    _call(at.make_tools(reg, "tech")["web_search"], {"query": "팜랩", "recent": False})
    assert seen["recent"] is False


class FakeLLM:
    def __init__(self, fail: bool = False):
        self.prompts, self.fail = [], fail

    def invoke(self, prompt):
        if self.fail:
            raise RuntimeError("--offline: 재현용 캐시에 없는 LLM 호출입니다")
        self.prompts.append(prompt)
        return AIMessage(content=f"회사 주장: 요약 {len(self.prompts)}.")


def test_summarize_document_url_registers_and_cites(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(at, "get_llm", lambda role="generator": llm)
    monkeypatch.setattr(at, "fetch", lambda url: {"text": "가" * 7000, "date": "2026-03-02", "title": "팜랩 소개"})
    reg = SourceRegistry()
    out = at.make_tools(reg, "tech")["summarize_document"].invoke({"source": "https://farmlab.example.com",
                                                                  "focus": "핵심 기술과 장단점"})
    m = re.search(r"\[(W[0-9a-f]{5})\]$", out)
    assert m and m.group(1) in reg.data and reg.data[m.group(1)]["date"] == "2026-03-02"
    assert len(llm.prompts) == 2  # 6000자로 자른 뒤 3000자씩 2조각
    assert all("핵심 기술과 장단점" in p and "가" * 3000 in p and "가" * 3001 not in p for p in llm.prompts)
    assert out.startswith("회사 주장: 요약 1. 회사 주장: 요약 2.")


def test_summarize_document_evidence_id(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(at, "get_llm", lambda role="generator": llm)
    reg = SourceRegistry()
    sid = reg.add_web({"url": "https://news.example.com/b", "title": "팜랩 기사", "content": "스니펫",
                       "raw_content": "본문 내용"}, agent="tech", query="q", access_date="2026-09-30")
    out = at.make_tools(reg, "tech")["summarize_document"].invoke({"source": sid, "focus": "팀"})
    assert out.endswith(f"[{sid}]") and len(llm.prompts) == 1 and "본문 내용" in llm.prompts[0]


def test_summarize_document_failures_return_empty(monkeypatch):
    reg = SourceRegistry()
    monkeypatch.setattr(at, "get_llm", lambda role="generator": FakeLLM())
    monkeypatch.setattr(at, "fetch", lambda url: None)
    tool = at.make_tools(reg, "tech")["summarize_document"]
    assert tool.invoke({"source": "https://blocked.example.com", "focus": "기술"}) == ""  # 원문 수집 실패
    assert tool.invoke({"source": "W99999", "focus": "기술"}) == ""                        # 없는 근거 id
    assert not reg.data
    monkeypatch.setattr(at, "fetch", lambda url: {"text": "본문", "date": "", "title": ""})
    monkeypatch.setattr(at, "get_llm", lambda role="generator": FakeLLM(fail=True))
    assert tool.invoke({"source": "https://ok.example.com", "focus": "기술"}) == ""       # LLM 실패도 예외 없이 ''
