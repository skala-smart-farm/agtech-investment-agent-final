"""메인 그래프: 노션 Graph(안)을 모양 그대로 따르고 👤 창업자 평가 한 노드만 더했다.

    Graph(안)               이 코드의 노드
    A 스타트업 탐색     →   discover     (🔍 서브그래프 collect → screen → pick / exhausted, graph/discovery_graph.py)
    (추가) 창업자 평가  →   founder
    B 기술 요약         →   tech
    C 시장성 평가       →   market
    D 경쟁사 비교       →   competition
    E 투자 판단         →   decide
    F 보고서 생성       →   report

    START → discover ─(평가 대상 있음)→ founder → tech → market → competition → decide
               │                                                                 ├─(투자·보류, 평가 상한 전)→ discover
               └─(후보 소진)→ report → END                                        └─(평가 상한 도달)→ report → END

- 간선 집합은 설계서의 DESIGN_EDGES(계약 C2)와 같다. 병렬 간선은 없다(경쟁사가 tech.claims 를, 투자 판단이 market 을 받아 쓰는 순서 의존).
- 첫 투자 추천에서 멈추지 않는다(workflow.stop_on_invest=false). 먼저 평가한 후보가 결론이 되지 않도록 상한까지 평가하고,
  보고서가 투자 기준 통과 후보 중 배수 1위를 대상으로 삼는다. true 로 두면 Graph(안)의 '투자 추천 → 보고서 생성' 그대로다.
- 분기 규칙은 graph/routes.py 의 순수 함수다. 보정 실행도 report 노드에서 끝난다(report 가 보정 모드에서는 렌더링하지 않음).
- 에이전트 모듈은 함수 안에서 import 한다. 그래서 nodes 로 가짜 노드를 넣은 테스트는 에이전트 모듈(검색·LLM 의존성)을 읽지 않는다.
"""
from __future__ import annotations

import importlib

from langgraph.graph import END, START, StateGraph

from graph.routes import route_after_decide, route_after_discover
from graph.state import InvestmentState

NODE_NAMES = ("discover", "founder", "tech", "market", "competition", "decide", "report")

# 노드 이름 → (모듈, 함수). discover 는 서브그래프라 build_discovery_graph() 로 만든다
_AGENT_NODES = {
    "founder": ("agents.founder", "founder_node"),              # 👤 창업자 평가 (창업자 30%)
    "tech": ("agents.tech", "tech_node"),                       # 🗜️ 기술 요약 (제품/기술력 15%)
    "market": ("agents.market", "market_node"),                 # 📊 시장성 평가 (시장성 25%, Agentic RAG)
    "competition": ("agents.competition", "competition_node"),  # 🥊 경쟁사 비교 (경쟁 우위 10%)
    "decide": ("agents.decision", "decision_node"),             # 🧮 투자 판단 (실적·투자조건 20% + 동종 대비 배수)
    "report": ("agents.report", "report_node"),                 # 📝 보고서 생성
}


def _default_node(name: str):
    if name == "discover":  # 🔍 스타트업 탐색 (서브그래프)
        from graph.discovery_graph import build_discovery_graph

        return build_discovery_graph()
    module, fn = _AGENT_NODES[name]
    return getattr(importlib.import_module(module), fn)


def build_graph(nodes: dict | None = None):
    """메인 그래프를 컴파일한다. nodes={'decide': 가짜 함수, …} 로 일부 노드를 바꿔 끼울 수 있다(테스트용)."""
    nodes = dict(nodes or {})
    unknown = set(nodes) - set(NODE_NAMES)
    if unknown:
        raise ValueError(f"알 수 없는 노드 이름: {sorted(unknown)} (가능: {NODE_NAMES})")

    g = StateGraph(InvestmentState)
    for name in NODE_NAMES:
        g.add_node(name, nodes[name] if name in nodes else _default_node(name))

    g.add_edge(START, "discover")
    g.add_conditional_edges("discover", route_after_discover, {"founder": "founder", "report": "report"})
    g.add_edge("founder", "tech")
    g.add_edge("tech", "market")
    g.add_edge("market", "competition")
    g.add_edge("competition", "decide")
    g.add_conditional_edges("decide", route_after_decide, {"report": "report", "discover": "discover"})
    g.add_edge("report", END)
    return g.compile()
