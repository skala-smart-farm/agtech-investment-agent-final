"""분기 함수. 순수 함수이며 State 와 config.yaml(get_config)만 읽고 다음 노드 이름을 돌려준다.

- route_discovery_entry : 🔍 탐색 서브그래프 안. 대기열에 후보가 있으면 pick, 비었고 발굴 라운드가 남았으면 collect, 라운드를 다 썼으면 exhausted
- route_after_discover  : 메인 그래프. 평가 대상이 있으면 founder, 후보 소진이면 report
- route_after_decide    : 메인 그래프. 투자 추천이면 report, 보류이면서 평가 상한에 닿았으면 report, 그 밖의 보류는 discover

보정 실행(app.py --calibrate)도 마지막은 report 노드다. 보고서 노드가 보정 모드에서는 렌더링하지 않고 끝나므로,
END 로 바로 가는 간선을 따로 두지 않는다. 그래서 코드의 간선이 설계서 그림(Graph(안) + 창업자)과 같다.
"""
from __future__ import annotations

from typing import Literal

from core.config import get_config


def route_discovery_entry(state: dict) -> Literal["pick", "collect", "exhausted"]:
    """탐색 서브그래프 진입과 관문(screen) 뒤에 같은 규칙을 쓴다."""
    if state.get("queue"):
        return "pick"
    if state.get("discovery_rounds", 0) < get_config().workflow.max_discovery_rounds:
        return "collect"
    return "exhausted"


def route_after_discover(state: dict) -> Literal["founder", "report"]:
    """current 가 없으면 후보 소진이다(종료 사유는 exhausted 노드가 기록)."""
    return "founder" if state.get("current") else "report"


def route_after_decide(state: dict) -> Literal["report", "discover"]:
    wf = get_config().workflow
    # 보정 실행은 투자여도 멈추지 않는다 (app.py --calibrate 가 stop_on_invest 도 끄지만, calibrate 만 켜도 같게 동작)
    stop_on_invest = wf.get("stop_on_invest", True) and not wf.get("calibrate", False)
    if state.get("decision") == "투자" and stop_on_invest:
        return "report"
    if state.get("iterations", 0) >= wf.max_evaluations:  # 반복 상한 (무한 루프 방지)
        return "report"
    return "discover"  # 가이드: 보류면 다른 스타트업 탐색으로 돌아간다
