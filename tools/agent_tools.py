"""목적별 도구 정의 (LangChain @tool). 실습 목표 '외부 정보 검색, 문서 요약 등의 목적에 맞는 도구 정의'에 해당한다.

계약 C4. make_tools(reg, agent) 가 돌려주는 도구 3개:
- search_documents(query: str)
    문서 코퍼스(PDF) 하이브리드 검색. response_format='content_and_artifact' — content 는 LLM 이 읽을 조각 목록,
    artifact 는 [{'text', 'meta'}] (근거 등록은 RAG finish 단계가 한다).
- web_search(query: str, recent: bool = True)
    외부 정보 검색. tools.web_search.web_search 로 결과를 reg 에 근거로 등록하고,
    content_and_artifact 로 요약 목록과 [근거 id] 를 돌려준다. recent=True 면 최근 1년 뉴스 우선.
- summarize_document(source: str, focus: str) -> str
    문서 요약. source 는 근거 id 또는 URL. focus 관점으로 요약하고 끝에 [근거 id] 를 붙인다. 실패하면 빈 문자열.
    LLM 에 바인딩하지 않고 기술 요약 에이전트가 코드에서 직접 부른다(홈페이지 요약, 실패하면 기사 본문 근거 요약).
reg 는 하나의 SourceRegistry 객체를 클로저로 잡아 제자리에서 갱신한다(RAG 노드와 같은 객체를 써야 근거가 사라지지 않음).

현재 상태(P0 계약 커밋): 시그니처만 있다. 구현은 P2 가 한다.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from tools.sources import SourceRegistry

TOOL_NAMES = ("search_documents", "web_search", "summarize_document")


def make_tools(reg: SourceRegistry, agent: str) -> dict[str, BaseTool]:
    """에이전트(agent 이름은 근거 등록 기록용)가 쓸 도구 3개를 {이름: 도구} 로 돌려준다. 키는 TOOL_NAMES 와 같다."""
    raise NotImplementedError("P2: search_documents·web_search·summarize_document @tool 구현")
