"""🔍 탐색 서브그래프(graph/discovery_graph.py) 검사. 키·네트워크 없이 collect·screen 은 가짜 함수로 바꿔 돌린다.

- 기본 노드가 v1 발굴·관문 함수 그대로인지 (캐시 키 보존)
- 진입 분기(pick / collect / exhausted)와 pick 의 후보별 키 초기화, exhausted 의 종료 사유
- 입력 규격에 누적 키가 없어 서브그래프가 새 기록만 돌려주는지, 부모 Reducer 가 중복 없이 붙이는지
"""
from __future__ import annotations

import typing

from langgraph.graph import END, START

from graph.discovery_graph import build_discovery_graph, exhausted_node, pick_node
from graph.state import DiscoveryOut

SUB_EDGES = {(START, "pick"), (START, "collect"), (START, "exhausted"), ("collect", "screen"), ("screen", "pick"),
             ("screen", "collect"), ("screen", "exhausted"), ("pick", END), ("exhausted", END)}
RESET_KEYS = ("founder", "tech", "market", "competition", "scorecard")


def _cand(name: str) -> dict:
    return {"official_name": name, "region": "KR", "stage": "Seed", "segment_id": "robotics"}


def fake_discovery(rounds: list[list[str]], eligible: set[str] | None = None):
    """rounds[i] = (i+1)라운드에 발굴되는 후보 이름. eligible 이 None 이면 모두 관문 통과.
    calls 에 실행된 노드와 그때 받은 State 를 남긴다."""
    calls: list[tuple[str, dict]] = []

    def collect(state: dict) -> dict:
        calls.append(("collect", dict(state)))
        r = state.get("discovery_rounds", 0) + 1
        names = [n for n in (rounds[r - 1] if r <= len(rounds) else []) if n not in state.get("seen", [])]
        return {"raw_candidates": [{"name": n} for n in names], "discovery_rounds": r, "seen": names,
                "registry": {f"W{r}": {"title": f"발굴 {r}"}}, "rag_traces": [{"agent": "discovery", "round": r}],
                "log": [f"[발굴 {r}라운드] 후보 {len(names)}곳"]}

    def screen(state: dict) -> dict:
        calls.append(("screen", dict(state)))
        recs = [{"name": c["name"], "eligible": eligible is None or c["name"] in eligible}
                for c in state.get("raw_candidates", [])]
        passed = [_cand(r["name"]) for r in recs if r["eligible"]]
        return {"screened": recs, "queue": list(state.get("queue", [])) + passed,
                "log": [f"[적격성 검증 {state.get('discovery_rounds')}라운드] 통과 {len(passed)}곳"]}

    return collect, screen, calls


def test_default_nodes_are_v1_discovery_and_eligibility():
    """collect·screen 은 v1 함수를 감싸지 않고 그대로 쓴다 → 발굴·관문의 입력과 LLM·검색 캐시 키가 v1 과 같다."""
    from agents.discovery import discovery_node
    from agents.eligibility import eligibility_node

    sub = build_discovery_graph()
    assert sub.builder.nodes["collect"].runnable.func is discovery_node
    # screen 은 관문 함수를 입력 그대로 부르고 결과의 G2 최근 투자(기준일 이후·예정 라운드)만 다시 거르는 감싸기다
    assert sub.builder.nodes["screen"].runnable.func.__wrapped__ is eligibility_node
    assert {(e.source, e.target) for e in sub.get_graph().edges} == SUB_EDGES


def test_queue_present_picks_without_collect():
    collect, screen, calls = fake_discovery([["A"]])
    sub = build_discovery_graph(collect, screen)
    out = sub.invoke({"queue": [_cand("B"), _cand("C")], "iterations": 2, "discovery_rounds": 1,
                      "log": ["이전 기록"], "screened": [{"name": "이전"}], "rag_traces": [{"old": 1}],
                      "founder": {"t0": "2020"}, "tech": {"x": 1}})
    assert calls == []
    assert out["current"]["official_name"] == "B"
    assert [c["official_name"] for c in out["queue"]] == ["C"]
    assert out["iterations"] == 3
    assert all(out[k] == {} for k in RESET_KEYS)  # 이전 후보의 분석 결과가 섞이지 않게 비운다
    # 입력의 누적 키(log·screened·rag_traces)는 서브그래프에 들어가지 않는다 → 출력에는 새 기록만
    # (Reducer 가 있는 키는 새 기록이 없으면 빈 목록으로 나온다. 부모의 operator.add 에 빈 목록을 붙여도 그대로)
    assert out["log"] == ["[평가 3] B (KR, Seed, robotics)"]
    assert out.get("screened", []) == [] and out.get("rag_traces", []) == []
    assert set(out) <= set(typing.get_type_hints(DiscoveryOut))


def test_empty_queue_collects_screens_then_picks():
    collect, screen, calls = fake_discovery([["A", "B"]])
    sub = build_discovery_graph(collect, screen)
    out = sub.invoke({"domain": "AgTech", "run_date": "2026-09-30", "registry": {"D1": {"title": "문서"}},
                      "discovery_rounds": 0, "iterations": 0, "queue": [], "seen": ["old"]})
    assert [c[0] for c in calls] == ["collect", "screen"]
    seen_by_collect = calls[0][1]
    assert seen_by_collect["registry"] == {"D1": {"title": "문서"}} and seen_by_collect["seen"] == ["old"]
    assert calls[1][1]["raw_candidates"] == [{"name": "A"}, {"name": "B"}]  # screen 은 같은 호출의 발굴 결과를 받는다
    assert out["current"]["official_name"] == "A" and out["iterations"] == 1
    assert [r["name"] for r in out["screened"]] == ["A", "B"]
    assert out["registry"] == {"D1": {"title": "문서"}, "W1": {"title": "발굴 1"}}
    assert out["seen"] == ["old", "A", "B"]
    assert out["log"] == ["[발굴 1라운드] 후보 2곳", "[적격성 검증 1라운드] 통과 2곳", "[평가 1] A (KR, Seed, robotics)"]


def test_second_round_when_first_round_has_no_eligible():
    collect, screen, calls = fake_discovery([["A"], ["B"]], eligible={"B"})
    out = build_discovery_graph(collect, screen).invoke({"discovery_rounds": 0, "iterations": 0, "queue": []})
    assert [c[0] for c in calls] == ["collect", "screen", "collect", "screen"]
    assert out["discovery_rounds"] == 2 and out["current"]["official_name"] == "B"
    assert [r["name"] for r in out["screened"]] == ["A", "B"]


def test_exhausted_end_reason():
    collect, screen, calls = fake_discovery([["A"], ["B"]], eligible=set())
    out = build_discovery_graph(collect, screen).invoke({"discovery_rounds": 0, "iterations": 0, "queue": []})
    assert [c[0] for c in calls] == ["collect", "screen", "collect", "screen"]  # 발굴 라운드 상한(2)에서 멈춘다
    assert out["current"] is None and out["end_reason"] == "no_eligible"

    out = build_discovery_graph(collect, screen).invoke({"discovery_rounds": 2, "iterations": 3, "queue": []})
    assert out["current"] is None and out["end_reason"] == "exhausted"


def test_pick_and_exhausted_nodes_return_contract_keys():
    out = pick_node({"queue": [_cand("A")], "iterations": 0})
    assert set(out) == {"current", "queue", "iterations", *RESET_KEYS, "log"}
    assert set(exhausted_node({"iterations": 1, "discovery_rounds": 2})) == {"current", "end_reason", "log"}


def test_parent_reducers_do_not_duplicate_subgraph_records():
    """부모 그래프 안에서 탐색을 여러 번 돌아도 screened·log·rag_traces·seen·registry 에 같은 항목이 두 번 붙지 않는다."""
    from graph.builder import build_graph

    collect, screen, _ = fake_discovery([["A", "B"], ["C"]])
    analysis = {k: (lambda k: lambda s: {"log": [f"[{k}] {s['current']['official_name']}"]})(k)
                for k in ("founder", "tech", "market", "competition")}
    app = build_graph({
        "discover": build_discovery_graph(collect, screen), **analysis,
        "decide": lambda s: {"decision": "보류", "evaluations": [{"name": s["current"]["official_name"]}],
                             "log": [f"[decide] {s['current']['official_name']}"]},
        "report": lambda s: {"report": {"mode": "hold"}, "log": ["[report]"]},
    })
    out = app.invoke({"registry": {}, "discovery_rounds": 0, "iterations": 0, "queue": [], "seen": [],
                      "screened": [], "log": [], "rag_traces": [], "evaluations": []})
    assert [r["name"] for r in out["screened"]] == ["A", "B", "C"]
    assert out["seen"] == ["A", "B", "C"]
    assert out["rag_traces"] == [{"agent": "discovery", "round": 1}, {"agent": "discovery", "round": 2}]
    assert set(out["registry"]) == {"W1", "W2"}
    assert len(out["log"]) == len(set(out["log"]))  # 같은 기록이 두 번 붙지 않았다
    # 노드 실행 수와 기록 수가 같다: 발굴 2 + 관문 2 + pick 3 + 분석·판단 5×3 + exhausted 1 + report 1
    assert len(out["log"]) == 2 + 2 + 3 + 15 + 1 + 1
    assert [e["name"] for e in out["evaluations"]] == ["A", "B", "C"]
    assert out["end_reason"] == "exhausted" and out["iterations"] == 3
