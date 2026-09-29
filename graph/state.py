"""그래프 전체가 공유하는 State (에이전트 사이의 인터페이스 규격).

- 노드는 바뀐 키만 반환한다. 다음 노드가 이어 써야 하는 키에만 Reducer 를 붙이고, 나머지는 덮어쓴다.
- 후보별 분석 결과(founder/tech/market/competition/scorecard)는 새 후보를 꺼낼 때(탐색의 pick 단계) {} 로 초기화한다.
- 키마다 끝 주석은 '설명  [쓰는 노드 → 읽는 노드]' 형식이다. 설계서 State 표가 이 주석을 읽으므로 키는 한 줄에 하나씩 둔다.
- DiscoveryIn/DiscoveryOut 은 🔍 탐색 서브그래프의 입력·출력 규격이다. 입력에 누적 키(screened·log·rag_traces)를 넣지 않아
  서브그래프는 새 기록만 돌려주고, 부모 Reducer 가 중복 없이 붙인다.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def merge_dict(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


def add_unique(a: list | None, b: list | None) -> list:
    out = list(a or [])
    for x in b or []:
        if x not in out:
            out.append(x)
    return out


class InvestmentState(TypedDict, total=False):
    # ── 실행 설정
    domain: str                                        # 도메인 이름 "AgTech"  [app → 실행 기록]
    run_date: str                                      # 평가 기준일 YYYY-MM-DD (24개월 판정·근거 조회일)  [app → 전 노드]

    registry: Annotated[dict, merge_dict]              # 근거 저장소 {근거 id: 문서·웹 메타데이터}, id → REFERENCE  [전 에이전트 → 전 에이전트·report]

    # ── 🔍 스타트업 탐색 (서브그래프: collect 발굴 → screen 적격성 관문 → pick 선택 / exhausted 소진)
    discovery_rounds: int                              # 발굴 라운드 수 (max_discovery_rounds 로 제한)  [discover.collect → discover 라우터]
    raw_candidates: list[dict]                         # 이번 라운드 발굴 후보 (이름, 분야, 발굴 채널, 근거 id)  [discover.collect → discover.screen]
    seen: Annotated[list[str], add_unique]             # 이미 다룬 후보의 정규화 이름 (중복 발굴 방지)  [discover.collect → discover.collect]
    screened: Annotated[list[dict], operator.add]      # 적격성 관문 판정 기록 전체 (통과·탈락 사유)  [discover.screen → report]
    queue: list[dict]                                  # 적격 후보 평가 대기열 (우선순위 순, 국내·해외 몫)  [discover.screen·pick → discover]
    current: dict | None                               # 평가 대상 프로필 (단계·라운드·금액·대표·설립일·국민연금), None = 후보 소진  [discover.pick·exhausted → 분석 에이전트·decide]
    iterations: int                                    # 심층 평가한 후보 수 (max_evaluations 로 제한)  [discover.pick → 라우터]

    # ── 분석 에이전트 (1 에이전트 = 1 Scorecard 기준, 결과에 criterion 판정 포함)
    founder: dict                                      # 창업 시점 t0·인물(창업 전 경력)·마일스톤·고용 추이·창업자 기준 판정  [founder → decide·report]
    tech: dict                                         # 제품·핵심 기술·장단점·차별점 주장(claims)·기술 기준선·제품/기술력 판정  [tech → market·competition·decide·report]
    market: dict                                       # 시장 규모(수치·연도·출처)·성장·수요·정책·시장성 판정  [market → decide·report]
    competition: dict                                  # 경쟁사 비교표·차별점 검증·경쟁 우위 판정  [competition → decide·report]
    market_cache: Annotated[dict, merge_dict]          # 세부 분야별 시장 분석 캐시 (같은 분야 재분석 방지)  [market → market]

    # ── 🧮 투자 판단과 반복 제어
    scorecard: dict                                    # 6개 기준 비교율·배수 M·결정·보류 유형·뒤집힘·실사·ROI·민감도·순위  [decide → report(evaluations 사본)]
    decision: str                                      # 이번 후보의 결정: '투자' 또는 '보류'  [decide → 라우터]
    evaluations: Annotated[list[dict], operator.add]   # 후보별 평가 누적 (보고서 비교표·보류 사유의 재료)  [decide → report]
    end_reason: str                                    # 종료 사유 invest_found·max_evaluations·exhausted·no_eligible  [discover.exhausted·decide → report]

    # ── 산출물
    report: dict                                       # 보고서 모드·파일 경로·쪽수·형식 검사 결과  [report → app]
    rag_traces: Annotated[list[dict], operator.add]    # Agentic RAG 수행 기록 (재작성·웹 보완 여부)  [discover·tech·market → app(run_log)]
    log: Annotated[list[str], operator.add]            # 분기 이유가 포함된 진행 기록  [전 노드 → app(run_log)]


class DiscoveryIn(TypedDict, total=False):
    """🔍 탐색 서브그래프 입력 규격. 누적 키(screened·log·rag_traces·evaluations)는 넣지 않는다."""
    domain: str
    run_date: str
    registry: Annotated[dict, merge_dict]
    discovery_rounds: int
    seen: Annotated[list[str], add_unique]
    queue: list[dict]
    iterations: int


class DiscoveryOut(TypedDict, total=False):
    """🔍 탐색 서브그래프 출력 규격. 누적 키는 서브그래프 안에서 새로 생긴 항목만 담긴다."""
    registry: Annotated[dict, merge_dict]
    discovery_rounds: int
    raw_candidates: list[dict]
    seen: Annotated[list[str], add_unique]
    screened: Annotated[list[dict], operator.add]
    queue: list[dict]
    current: dict | None
    iterations: int
    founder: dict
    tech: dict
    market: dict
    competition: dict
    scorecard: dict
    end_reason: str
    rag_traces: Annotated[list[dict], operator.add]
    log: Annotated[list[str], operator.add]
