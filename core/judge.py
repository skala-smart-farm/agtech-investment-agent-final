"""평가표 문항 판정 (모든 분석 에이전트와 투자 판단이 함께 쓴다).

계약 C3. 에이전트마다 자기 Scorecard 기준(차원) 하나를 judge_dimension 으로 판정해 state[키]['criterion'] 에 둔다.
- founder → 창업자, market → 시장성, tech → 제품/기술력, competition → 경쟁 우위, decide → 실적·투자조건
  (rubric.yaml 차원의 owner 필드)
- 판정 규칙(인용 원문 대조, 제3자·최근 24개월·상용 운영 검사, D1 코드 판정)은 v1 agents/decision.py 와 같다.

현재 상태(P0 계약 커밋): load_rubric·company_keys·evidence_pool 은 동작하고, judge_dimension 은 시그니처만 있다.
v1 판정 코드(agents/decision.py)를 동작 변경 없이 옮기는 작업은 P3 가 한다.
"""
from __future__ import annotations

from typing import Literal, TypedDict

import yaml

from core.config import ROOT
from tools.grounding import norm
from tools.sources import SourceRegistry

DIMENSIONS = ("founder", "market", "product", "competition", "traction", "deal")


class Row(TypedDict):
    """문항 1개 판정 결과."""
    dim: str
    qid: str
    short: str
    point: str
    question: str
    bessemer_q: list[int]
    answer: Literal["YES", "NO", "UNKNOWN", "N/A"]
    x: int | None                   # 신호: YES 1, NO -1, UNKNOWN 0, N/A None
    evidence_ids: list[str]
    quote: str
    rationale: str


class DimResult(TypedDict):
    """차원(Scorecard 기준) 1개 판정 결과. 분석 에이전트는 이 값을 state[키]['criterion'] 에 둔다."""
    dim: str
    name: str
    weight: int
    owner: str
    rows: list[Row]
    yes: int
    no: int
    unknown: int
    na: int
    n: int                          # 분모 = 문항 수 − N/A
    rejected_yes: list[dict]        # 코드가 UNKNOWN 으로 강등한 YES [{qid, reason, quote}]
    quote_retried: int              # 인용이 원문에 없어 다시 물은 문항 수


def load_rubric() -> dict:
    """rubric.yaml 전체 (dimensions·bessemer·gates·deal_killers·decision_rule)."""
    with open(ROOT / "rubric.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def company_keys(c: dict) -> list[str]:
    """회사 이름 표기들(영문명·공식명·통칭)을 정규화한 목록. 인용 주변 회사명 확인과 회사 도메인 판별에 쓴다.
    v1 decision_node 의 company_keys 계산과 같다(정규화 후 2글자 이상만)."""
    return [norm(x) for x in (c.get("name_en"), c.get("official_name"), c.get("name")) if x and len(norm(x)) >= 2]


def evidence_pool(state: dict, extra: list[str] = ()) -> list[str]:
    """판정에 쓸 근거 id 목록: current.evidence_ids + founder/tech/market/competition 의 pool_ids + extra.
    순서를 유지하고 중복을 없애며, 근거 저장소(registry)에 있는 id 만 남긴다.
    각 에이전트는 자기 차례까지 모인 근거로 판정한다(뒤 에이전트의 pool_ids 는 아직 비어 있음)."""
    reg = SourceRegistry(state.get("registry"))
    ids = list((state.get("current") or {}).get("evidence_ids", []))
    for key in ("founder", "tech", "market", "competition"):
        ids += (state.get(key) or {}).get("pool_ids", [])
    ids += list(extra)
    return [i for i in dict.fromkeys(ids) if reg.get(i)]


def judge_dimension(dim_id: str, company: dict, pool_ids: list[str], reg: SourceRegistry, analysis: str,
                    run_date: str) -> DimResult:
    """차원 하나(dim_id: founder | market | product | competition | traction | deal)의 문항을 판정한다.

    - LLM(judge)은 문항별 판정(YES/NO/UNKNOWN/N/A)·근거 id·원문 인용·한 문장 이유만 답한다 (prompts/decision.md, v1 그대로).
    - 판정을 빠뜨린 문항은 한 번 더 묻고, 인용이 원문에 없는 YES·NO 도 한 번만 다시 묻는다.
    - 코드 강등: 인용이 본문에 없거나 회사명이 인용 ±300자 밖이면 UNKNOWN, 제3자·최근 24개월·문서 근거·상용 운영 요건 검사.
    - D1(최근 라운드)은 LLM 답을 쓰지 않고 적격성 관문에서 인용 검증을 마친 값으로 코드가 판정한다.
    반환: DimResult (rows 는 Row 목록, x 는 YES 1 / NO -1 / UNKNOWN 0 / N/A None).
    """
    raise NotImplementedError("P3: v1 agents/decision.py 의 차원 판정 루프를 옮겨 구현")
