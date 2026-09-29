"""👤 창업자 평가 에이전트 (가이드 에이전트 정의(안)에 추가한 에이전트).

Scorecard 에서 비중이 가장 큰 '창업자 (Owner) 30%' 를 혼자 맡는다 (1 에이전트 = 1 기준, 문항 F1~F4).
창업 시점(t0)을 먼저 고정하고, 창업 **전** 전문성(F1·F2)과 창업 **후** 실행력(F3)을 나눠 판정한다.
- RAG 미사용: 사람에 관한 사실은 PDF 코퍼스에 없어 웹 검색(인터뷰·기사 본문)과 국민연금 사업장 정보(첫 고용·인원)를 쓴다.
- 홈페이지 요약은 하지 않는다(기술 요약 에이전트만 한다).

현재 상태(P0 계약 커밋): 시그니처만 있다. 구현은 P4 가 한다.
"""
from __future__ import annotations

AGENT = "founder"


def founder_node(state: dict) -> dict:
    """state['current'] 후보의 창업자·팀을 조사하고 창업자 기준을 판정한다.

    반환 키: founder, registry(근거 저장소 전체 reg.data), log
    founder = {
        't0': str|None, 't0_source': str, 'years_since_t0': float|None,       # 창업 시점과 그 근거
        'people': [{'name','role','background','before_t0': bool|None,'evidence_ids'}],
        'milestones': [{'date': 'YYYY-MM','what','evidence_ids'}], 'milestones_24m': int,
        'headcount': int|None, 'hires_per_year': float|None, 'team_assessment': str,
        'evidence_ids': [...], 'pool_ids': [...],                               # pool_ids: 판정에 쓰는 근거 id
        'criterion': DimResult,                                                 # core.judge.judge_dimension('founder', …)
    }
    """
    raise NotImplementedError("P4: 창업자 평가 에이전트 구현")
