"""📝 보고서 생성 에이전트 (노드 report).

앞 단계 결과(탐색·창업자·기술·시장·경쟁·투자 판단)를 장마다 연결해 5쪽 이내 PDF 로 만들고, 과제 형식을 코드로 검사한다.
- 모드: invest(투자 추천 후보 있음 → A안) / hold(평가한 후보 모두 보류 → B안) / none(적격 후보 0 → C안)
  / calibrate(보정 실행 → 렌더링 없이 모드만 반환)
- A안 본문 5개 장 = 과제 '보고서 주요 내용' 5항목: 사업 아이디어(·기술·경쟁 차별성) / 시장 규모 / 팀의 구성 /
  투자 판단과 사업 리스크 / 한계점. B안은 후보마다 '왜 안 되는지'와 '무엇이 확인되면 되는지'로 구성한다
- SUMMARY 5줄: 상황 → 결론 → 근거(→n장) → 리스크·원인 → 요청('~할까요?'). 상황·결론·요청 줄과 '(→n장)'은 코드가,
  근거·리스크 구절은 LLM 이 쓴다. 개요 문장 금지, A4 절반 이내(렌더링 후 높이 측정)
- 표·점수·순위·ROI·뒤집힘 조건·실사 항목은 코드가 State(evaluations 의 scorecard)에서 만든다. LLM 은 문장 칸만 채운다
- 본문이 평가표와 어긋나면(예: 특허 문항이 미확인인데 "특허 보유", C2 가 YES 가 아닌데 "앞선다") 이유를 붙여 다시 쓰게 한다
- REFERENCE: 본문에 실제로 인용된 근거만, 기관 보고서 / 학술 논문 / 웹페이지 형식. 같은 자료(포털 전재본 포함)는 한 번만
- 5쪽을 넘으면 조판 밀도를 높이고, 그래도 넘으면 줄여 쓴다
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field

from core.config import ROOT, get_config, get_segment, path
from core.judge import company_keys
from core.llm import structured
from core.prompts import render
from report.render import html_to_pdf, render_html
from tools import web_search as search_tool
from tools.grounding import norm
from tools.sources import (ACCESS_DATE_NOTE, GROUPS, REFERENCE_FORMATS, SourceRegistry, citable, format_reference,
                           merge_duplicates, reference_group, reference_key, title_key, uses_access_date)

AGENT = "report"
CITE = re.compile(r"\[\s*([WD][0-9a-f]{5}(?:\s*[,，]\s*[WD][0-9a-f]{5})*)\s*\]")
PAREN_CITE = re.compile(r"\(\s*([WD][0-9a-f]{5}(?:\s*[,，]\s*[WD][0-9a-f]{5})*)\s*\)")  # LLM 이 (W1a2b3) 로 쓴 인용
NUM_CITE = re.compile(r"\[(\d+(?:, \d+)*)\]")  # 번호로 바뀐 본문 인용
CHAPTER_REF = re.compile(r"\(→\s*([\d·,\s]+)장\)")

# 장 제목 (고정 문자열, 형식 검사 대상). 과제 '보고서 주요 내용' 5항목이 장 제목(A안)이나 소제목(B안)에 들어간다
A_CHAPTERS = ("SUMMARY", "1. 사업 아이디어(핵심 컨셉)·기술·경쟁 차별성", "2. 시장 규모와 성장성",
              "3. 팀의 구성(핵심 창업자·기술 역량)", "4. 투자 판단과 사업 리스크", "5. 한계점", "REFERENCE")
B_CHAPTERS = ("SUMMARY", "1. 후보 풀과 선정 과정", "2. 후보별 보류 사유", "3. 최고점 후보 상세", "4. 투자 판단 요약",
              "5. 한계점", "REFERENCE")
B_DETAIL = ("3.1 사업 아이디어(핵심 컨셉)와 기술", "3.2 시장 규모와 성장성", "3.3 팀의 구성(핵심 창업자·기술 역량)",
            "3.4 경쟁 구도와 차별성", "3.5 사업 리스크(시장·기술·규제·경쟁)")
C_CHAPTERS = ("SUMMARY", "1. 탐색 경과와 탈락 사유", "2. 한계점", "REFERENCE")
REQUIRED_ITEMS = ("사업 아이디어", "사업 리스크", "시장 규모", "팀의 구성", "한계점")
DESIGN_THRESHOLD = 1.10  # 설계서 C.4 에 고정한 투자 기준 배수 (시나리오 실행 --threshold 와 구분해 문구를 고른다)
# 이 말이 있는 줄(ROI 수치)에는 '가정' 표기가 있어야 한다 ('유료 전환율'의 '환율'은 제외)
ASSUMPTION_TERMS = re.compile(r"(?<!전)환율|지분율|post-money|필요 Exit|회수 배수")
STAGE_ORDER = ("Seed", "Pre-A", "Series A", "Pre-B", "Series B", "Pre-C", "Series C")


def _fix_cites(x):
    """초안 전체에서 소괄호 인용 (W1a2b3) 을 대괄호 [W1a2b3] 로 바꾼다 (검사·번호 매기기가 대괄호만 알아보므로)."""
    if isinstance(x, str):
        return PAREN_CITE.sub(r"[\1]", x)
    if isinstance(x, list):
        return [_fix_cites(i) for i in x]
    if isinstance(x, dict):
        return {k: _fix_cites(v) for k, v in x.items()}
    return x


BANNED = ["본 보고서", "이 보고서", "보고서는", "보고서에서", "보고서의 목적", "판단하는 보고서", "평가하는 보고서",
          "목적으로 작성", "살펴보", "다음과 같", "개요", "소개하", "UNKNOWN", "이 장에서", "본 장"]  # UNKNOWN 은 내부 용어 → "미확인"
EVALUATIVE = ["우수", "탁월", "뛰어난", "선도적", "독보적", "혁신적인", "압도적"]
# 평가표에서 확인되지 않은 사실을 본문이 단정하는지 찾는 규칙: (문항, 패턴, 설명)
CLAIMS = [
    ("C1", r"특허[^.。\n]{0,15}(보유|확인|등록|출원|확보)", "특허"),
    ("R1", r"매출[^.。\n]{0,12}(\d[\d,.]*\s*(억|만|천|원|달러|%)|기록|달성)", "매출"),  # "제3자" 같은 숫자는 제외
    ("R2", r"(유료 고객|고객 수|설치 농가|도입 농가)[^.。\n]{0,12}\d[\d,.]*\s*(곳|명|개|농가|호|대|ha|㎡|%|만|천|억)",
     "고객·설치 규모"),
    ("P4", r"(인허가|검정|인증)[^.。\n]{0,10}(획득|취득|받았|완료)", "인허가·검정"),
    ("C2", r"차별화(된다|되어 있|를 갖|했다)|경쟁 우위(를|가)? ?(확보|갖)", "경쟁사 대비 차별성"),
    ("P1", r"상용 (운영|판매)[^.。\n]{0,6}(중이|하고 있)", "상용 운영"),
]
NEGATION = re.compile(r"않|불가|미확인|없|못|필요|여부|되면|이면|하면|경우|예정|계획")  # 부정·조건(재검토 조건)은 단정이 아니다
SUPERIORITY = re.compile(r"앞선다|앞서 있|우위|차별화된다|차별화되어")  # C2 가 YES 가 아니면 우열 단정 금지
SUP_NEGATION = re.compile(r"않|없|미확인|불가|못|확인되지")  # 우열 단정을 부정하는 말만 (조건·필요는 면제하지 않음)
# 판정 단계의 내부 기록(강등 사유)을 임원이 읽을 짧은 이유로 바꾼다
DEMOTED = {"인용문이 근거 본문에서 확인되지 않음": "근거 원문에서 해당 내용이 확인되지 않음",
           "인용 주변": "근거가 이 회사가 아닌 업계 일반 내용임",
           "공공·연구기관 문서 근거 없음": "공공·연구기관 문서 근거 없음",
           "회사 자체 발표만 있음": "회사 자체 발표만 있고 제3자 근거 없음",
           "최근 24개월 이내 근거 아님": "최근 24개월 이내 근거 없음",
           "언급된 사건 날짜가 모두": "언급된 사건이 모두 24개월보다 오래됨"}


# ── LLM 문안 스키마 (문장 칸만. 표·수치·결론은 코드가 쓴다)

class Risk(BaseModel):
    type: Literal["시장", "기술", "규제", "경쟁"]
    content: str = Field(description="리스크 내용 한 문장 (근거 id 인용)")
    evidence_ids: list[str] = Field(description="리스크 근거 id (1개 이상)")
    due_diligence: str = Field(description="실사에서 회사에 확인할 질문 한 구절")


class CompetitorNote(BaseModel):
    name: str = Field(description="경쟁사 표의 회사명 그대로")
    vs_target: str = Field(description="대상과 비교한 사실 한 문장 (근거 id). C2 가 YES 가 아니면 우열 표현 없이 차이만")


class CandidateNote(BaseModel):
    name: str = Field(description="평가한 후보 회사명 그대로")
    business: str = Field(description="사업 한 줄 요약 (한국어, 50자 이내, 근거가 영어여도 한국어로)")
    why_not: str = Field(description="왜 안 되는지 1~2문장: 결정적 반대 근거와 확인되지 않은 핵심 항목 (근거 id)")


class _Body(BaseModel):
    problem: str = Field(description="사업 아이디어: 해결하는 문제 한두 문장 (근거 id)")
    product: str = Field(description="사업 아이디어: 제품·핵심 컨셉 한두 문장 (근거 id)")
    revenue_model: str = Field(description="사업 아이디어: 수익 방식 (근거 id), 확인 안 되면 '확인 불가'")
    industry_baseline: str = Field(description="업계 기술 수준 대비 위치 한 문장 ([D..] 문서 인용 필수)")
    market: str = Field(description="시장 규모·성장·수요·투자 조정 국면 2~4문장 (국내·글로벌 수치는 [D..] 문서 인용)")
    team: str = Field(description="팀의 구성: 대표·핵심 인력의 창업 전 경력과 기술 역량 2~3문장 (근거 id)")
    competition: str = Field(description="경쟁 구도·진입장벽 요약 1~2문장 (경쟁사 표는 코드가 붙임)")
    competitor_notes: list[CompetitorNote] = Field(description="경쟁사 표의 회사마다 대상과 비교한 한 문장")
    risks: list[Risk] = Field(description="시장·기술·규제·경쟁 리스크 각 1개 이상")
    data_limits: list[str] = Field(description="이 회사·시장 데이터의 한계 1개")


class Draft(_Body):
    """A안(투자 추천) 문안."""
    ev_tech: str = Field(description="SUMMARY 근거①: 기술·차별성에서 확인된 강점 한 구절 (근거 id, 70자 이내)")
    ev_market: str = Field(description="SUMMARY 근거②: 시장 규모·성장 한 구절 ([D..] 문서 인용, 70자 이내)")
    ev_team: str = Field(description="SUMMARY 근거③: 팀(창업 전 경력·기술 역량) 한 구절 (근거 id, 70자 이내)")
    risk_line: str = Field(description="SUMMARY 리스크: 가장 큰 리스크 1~2개를 한 구절로 (근거 id, 90자 이내)")
    lead_idea: str = Field(description="1장 첫 줄 결론 한 문장 (60자 이내)")
    lead_market: str = Field(description="2장 첫 줄 결론 한 문장 (60자 이내)")
    lead_team: str = Field(description="3장 첫 줄 결론 한 문장 (60자 이내)")


class HoldDraft(_Body):
    """B안(모두 보류) 문안: 최고점 후보 상세 + 후보별 '왜 안 되는지' + 공통 원인."""
    common_cause: str = Field(description="SUMMARY 공통 원인: 후보들이 기준에 못 미친 공통 원인 한 구절 (근거 id, 90자 이내)")
    common_detail: str = Field(description="4장: 공통 원인과 시장 맥락(투자 조정 국면 포함) 2~3문장 (근거 id)")
    candidates: list[CandidateNote] = Field(description="평가한 후보마다 사업 한 줄과 왜 안 되는지")


# ── 공용 도우미

def _same(a: str, b: str) -> bool:
    x, y = norm((a or "").split("(")[0]), norm((b or "").split("(")[0])
    return bool(x) and bool(y) and (x == y or (min(len(x), len(y)) >= 2 and (x in y or y in x)))


def _names(e: dict) -> list[str]:
    p = e.get("profile") or {}
    return [x for x in (e.get("name"), p.get("official_name"), p.get("name"), p.get("name_en")) if x]


def _s100(m: float | None) -> str:
    """배수 → 보고서 표기 (동종 평균 = 100)."""
    return "-" if m is None else f"{m * 100:.0f}"


def _num(x) -> str:
    """숫자는 불필요한 소수점 없이, 숫자가 아니면 그대로 (분석 에이전트 출력 형식이 달라도 보고서가 멈추지 않게)."""
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def _dim_short(name: str) -> str:
    return name.split(" (")[0]


def _bare(x: str) -> str:
    """LLM 이 칸 이름을 앞에 또 붙이거나 끝에 마침표를 찍은 구절을 SUMMARY 줄에 맞게 다듬는다."""
    x = re.sub(r"^\s*(상황|결론|근거|리스크|원인|공통 원인|요청)\s*[:：]\s*", "", x or "").strip()
    return re.sub(r"[.。]\s*$", "", x)


def _pool(state: dict, evals: list[dict], screened: list[dict], cfg, mode: str) -> dict:
    """발굴 → 검증 → 적격 → 평가 단계별 후보 수와, 적격인데 평가하지 않은 후보."""
    def split(rows):
        return sum(r.get("region") == "KR" for r in rows), sum(r.get("region") != "KR" for r in rows)

    rounds = state.get("discovery_rounds") or 1
    counts = [int(m.group(1)) for x in state.get("log", []) if (m := re.match(r"\[발굴 \d+라운드\].*?후보 (\d+)곳", x))]
    raw = state.get("raw_candidates") or []
    discovered = sum(counts) if counts else (len(raw) if rounds == 1 and raw else None)
    disc_split = split(raw) if rounds == 1 and raw and discovered == len(raw) else (None, None)
    eligible = [r for r in screened if r.get("eligible")]
    done = [n for e in evals for n in _names(e)]
    left = [r for r in eligible if not any(_same(r.get("official_name") or r["name"], n) or _same(r["name"], n)
                                           for n in done)]
    capped = state.get("end_reason") == "max_evaluations" or len(evals) >= cfg.workflow.max_evaluations
    if mode == "invest":
        why = "투자 추천 후보가 나와 평가를 멈춤"
    elif capped:
        why = f"평가 상한({cfg.workflow.max_evaluations}곳, 비용 관리) 도달"
    else:
        why = "평가 전 종료"
    return {
        "rounds": rounds, "discovered": discovered, "disc_split": disc_split,
        "screened": len(screened), "scr_split": split(screened),
        "rejected": [(r.get("official_name") or r["name"], re.sub(r"(^|;\s*)G\d(?:/G\d)?\s*", r"\1", r.get("reason") or ""))
                     for r in screened if not r.get("eligible")],
        "eligible": len(eligible), "el_split": split(eligible),
        "evaluated": len(evals), "ev_split": split([{"region": e.get("region") or (e.get("profile") or {}).get("region")}
                                                   for e in evals]),
        "unevaluated": [{"name": r.get("official_name") or r["name"], "stage": r.get("stage") or "-",
                         "search_failed": bool(r.get("gate_search_failed"))} for r in left],
        "why": why, "capped": capped, "per_round": cfg.workflow.max_candidates_per_round,
    }


def _pool_rows(p: dict) -> list[list]:
    def n(x):
        return "-" if x is None else x

    rej = "; ".join(f"{name} — {reason}" for name, reason in p["rejected"])
    return [
        ["발굴", n(p["discovered"]), n(p["disc_split"][0]), n(p["disc_split"][1]),
         f"발굴 {p['rounds']}라운드" + (" (라운드 합계, 중복 포함)" if p["rounds"] > 1 else "")
         + f", 라운드당 최대 {p['per_round']}곳을 적격성 관문으로 넘김"],
        ["적격성 관문 (G1~G6)", p["screened"], p["scr_split"][0], p["scr_split"][1],
         f"탈락 {len(p['rejected'])}곳" + (f": {rej}" if rej else "")],
        ["적격 (비상장·Seed~C·Exit 전)", p["eligible"], p["el_split"][0], p["el_split"][1],
         f"미평가 {len(p['unevaluated'])}곳 ({p['why']})" if p["unevaluated"] else "모두 평가"],
        ["평가 (창업자·기술·시장·경쟁·투자 판단)", p["evaluated"], p["ev_split"][0], p["ev_split"][1],
         "6개 기준 24문항 판정 → 동종 평균 대비 배수"],
    ]


def _funnel(pool: dict) -> str:
    disc = f"{pool['discovered']}곳 발굴 → " if pool.get("discovered") else ""
    return f"국내외 AgTech AI 스타트업 {disc}적격성 관문 통과 {pool['eligible']}곳(모두 Seed~Series C 투자 유치 확인)"


def _hold_counts(evals: list[dict]) -> str:
    order = ("동종 대비 열위", "정보 부족", "창업자 근거 없음", "Deal-killer")
    types = [e.get("hold_type") or (e.get("scorecard") or {}).get("hold_type") for e in evals]
    return " · ".join(f"{t} {types.count(t)}" for t in order if types.count(t))


# ── 검사: SUMMARY 규칙, 평가표와의 일치

def _summary_problems(lines: list[str], limit: int) -> list[str]:
    text = " ".join(lines)
    probs = [f"SUMMARY 에 금지 표현 '{b}'" for b in BANNED if b in text]
    plain = CITE.sub("", text)
    if len(plain) > limit:
        probs.append(f"SUMMARY {len(plain)}자 > {limit}자 (각 칸 길이 상한을 지켜라)")
    return probs


def _superiority(text: str) -> re.Match | None:
    return next((m for m in SUPERIORITY.finditer(text) if not SUP_NEGATION.search(text[m.end(): m.end() + 12])), None)


def _claim_problems(text: str, verdict: dict, who: str = "") -> list[str]:
    """평가표에서 YES 가 아닌 사실을 단정하는 문장."""
    probs = []
    for qid, pat, label in CLAIMS:
        if verdict.get(qid) == "YES":
            continue
        for m in re.finditer(pat, text):
            if not NEGATION.search(text[m.start(): m.end() + 12]):
                probs.append(f"{who}'{m.group(0)}' — {qid}({label})는 평가표에서 {_ko(verdict.get(qid))} 이므로 확인됐다고 쓰지 마라")
                break
    if verdict.get("C2") != "YES" and (m := _superiority(text)):
        probs.append(f"{who}'{m.group(0)}' — 경쟁사 대비 차별성(C2)이 평가표에서 {_ko(verdict.get('C2'))} 이므로 "
                     f"'앞선다·우위·차별화된다' 같은 우열 표현 없이 차이만 사실로 써라 (경쟁사 표·리스크 표 포함)")
    return probs


def _ko(answer: str | None) -> str:
    return {"UNKNOWN": "미확인", "NO": "반대 근거", None: "미판정"}.get(answer, answer)


def _maturity_problems(text: str, p1: str | None) -> list[str]:
    """SUMMARY·장 결론의 성숙도 표현이 평가표 P1(상용 운영)과 맞는지."""
    if p1 == "YES":
        if m := re.search(r"PoC|실증 단계|시제품", text):
            return [f"'{m.group(0)}' — 평가표 P1(상용 운영)이 YES 이므로 성숙도를 평가표와 같게 써라"]
        return []
    for m in re.finditer(r"상용", text):
        if not re.search(r"예정|계획|목표|앞두|준비|않|미확인|불가|없|이전|화", text[m.end(): m.end() + 12]):
            return [f"'상용' — 평가표 P1(상용 운영)이 {_ko(p1)} 이므로 상용 운영 중이라고 쓰지 마라"]
    return []


def _stale_forecasts(text: str, run_date: str) -> list[str]:
    """이미 지난 기간의 전망('2025년 … 전망')을 미래처럼 쓴 문장. '(2022년 발표 전망)'처럼 발표 시점을 밝히면 허용."""
    year, out = int(run_date[:4]), []
    for sent in re.split(r"(?<=[.。])\s+|(?<=다)\s+|\n", CITE.sub("", text)):
        if "발표" in sent or re.search(r"확인 불가|미확인", sent):
            continue
        for m in re.finditer(r"전망", sent):
            years = [int(y) for y in re.findall(r"(20\d{2})년?(?!\s*(?:기준|에서|대비))", sent[max(0, m.start() - 60): m.start()])]
            if years and max(years) < year:
                out.append(f"'{sent.strip()[:40]}…' — {max(years)}년은 이미 지났다. 지난 기간의 전망이면 "
                           f"'(20XX년 발표 전망)'처럼 발표 시점을 밝히거나 최신 수치로 바꿔라")
                break
    return out[:3]


def _rejected_yes(e: dict) -> list[dict]:
    """판정 단계에서 원문 확인에 실패해 기각된 주장 (투자 판단 scorecard 와 분석 에이전트 criterion 모두)."""
    sc = e.get("scorecard") or {}
    out = list(sc.get("rejected_yes") or [])
    for k in ("founder", "tech", "market", "competition"):
        out += ((e.get(k) or {}).get("criterion") or {}).get("rejected_yes") or []
    return out


def _consistency_problems(draft: _Body, target: dict, evals: list[dict], run_date: str) -> list[str]:
    """본문·표가 평가표와 어긋나는 단정, 평가성 형용사, 지난 전망, 근거 없는 리스크·시장 수치를 찾는다."""
    verdict = {r["qid"]: r["answer"] for r in target["scorecard"]["rows"]}
    body = [draft.problem, draft.product, draft.revenue_model, draft.market, draft.team, draft.industry_baseline,
            draft.competition]
    tables = [c.vs_target for c in draft.competitor_notes] + [r.content for r in draft.risks]  # 실사 질문 칸은 확인할 일이라 제외
    notes = ""
    if isinstance(draft, Draft):
        head = [draft.ev_tech, draft.ev_market, draft.ev_team, draft.risk_line, draft.lead_idea, draft.lead_market,
                draft.lead_team]
        probs = _claim_problems("\n".join(head + body + tables), verdict)
        probs += _maturity_problems(f"{draft.ev_tech} {draft.lead_idea}", verdict.get("P1"))
        probs += [f"장 결론에 금지 표현 '{b}'" for b in BANNED if b in " ".join(head[4:])]
    else:
        head = [draft.common_cause, draft.common_detail]
        probs = _claim_problems("\n".join(body + tables), verdict)  # 최고점 후보 상세(3장)는 그 후보의 평가표로
        verdicts = {e["name"]: {r["qid"]: r["answer"] for r in e["scorecard"]["rows"]} for e in evals}
        # 후보 이름이 없는 공통 원인 문장은 한 후보라도 YES 면 허용
        any_yes = {q: "YES" for v in verdicts.values() for q, a in v.items() if a == "YES"}
        for sent in re.split(r"(?<=[.。])\s+|(?<=다)\s+", " ".join(head)):
            named = [e for e in evals if any(n and n in sent for n in _names(e))]
            for e in named:
                probs += _claim_problems(sent, verdicts[e["name"]], f"[{e['name']}] ")
            if not named:
                probs += _claim_problems(sent, any_yes)
        for note in draft.candidates:  # 후보별 보류 사유는 그 후보의 평가표와 맞춘다
            e = next((e for e in evals if any(_same(note.name, n) for n in _names(e))), None)
            if e is not None:
                probs += _claim_problems(note.why_not, verdicts[e["name"]], f"[{e['name']}] ")
        missing = [e["name"] for e in evals if not any(_same(n.name, x) for n in draft.candidates for x in _names(e))]
        if missing:
            probs.append(f"candidates 에 {', '.join(missing)} 가 없다 — 평가한 후보마다 하나씩 써라")
        notes = "\n".join(n.why_not for n in draft.candidates)
    text = "\n".join(head + body + tables)
    probs += _stale_forecasts(f"{text}\n{notes}", run_date)
    for rj in _rejected_yes(target):  # 판정 단계에서 원문 확인에 실패해 기각된 주장이 다시 나오면 안 된다
        q = norm(rj.get("quote", ""))
        if len(q) >= 15 and q[:15] in norm(f"{text}\n{notes}"):
            probs.append(f"판정 단계에서 기각된 주장('{rj['quote'][:30]}…')을 쓰지 마라")
    plain = re.sub(r"우수기업|우수 기업|우수벤처|우수 벤처", "", f"{text}\n{notes}")  # 공식 프로그램 이름은 평가성 표현이 아니다
    probs += [f"평가성 표현 '{w}' 대신 사실과 판정 근거로 써라" for w in EVALUATIVE if w in plain]
    if not re.search(r"\[[^\]]*D[0-9a-f]{5}", draft.market):
        probs.append("시장 수치에 공공·연구기관 문서([D..]) 인용이 없다")
    if not re.search(r"\[[^\]]*D[0-9a-f]{5}", draft.industry_baseline):
        probs.append("업계 기술 수준 문장에 문서([D..]) 인용이 없다")
    if any(not r.evidence_ids for r in draft.risks):
        probs.append("근거 id 가 없는 리스크가 있다")
    if {r.type for r in draft.risks} < {"시장", "기술", "규제", "경쟁"}:
        probs.append("리스크 유형(시장·기술·규제·경쟁)을 하나씩은 다 써라")
    return probs


# ── 표 재료 (코드가 State 로 만든다)

def _human_reason(text: str) -> str:
    """판정 이유에서 내부 기록('→ UNKNOWN 강등 (…)', '(코드 판정)')을 빼고 짧은 이유만 남긴다."""
    if "→ UNKNOWN 강등" in text:
        head = text.split(" → ")[0]
        return next((v for k, v in DEMOTED.items() if head.startswith(k)), re.sub(r"\s*\(.*\)\s*$", "", head))
    if "→ NO 대신 UNKNOWN" in text:
        return "반대 근거가 원문에서 확인되지 않음"
    text = text.removeprefix("N/A 불가 문항 — ")
    return re.sub(r"\s*\((?:코드 판정|코드 날짜 검사)\)", "", text).strip()


def _usable(reg: SourceRegistry, ids) -> list[str]:
    return [i for i in dict.fromkeys(ids or []) if isinstance(i, str) and reg.get(i) and citable(reg.get(i))]


def _cite(reg: SourceRegistry, ids, limit: int = 3) -> str:
    ids = _usable(reg, ids)[:limit]
    return f" [{', '.join(ids)}]" if ids else ""


def _own_source(s: dict, keys: list[str]) -> bool:
    """회사 자체 홈페이지 근거인지 (호스트에 회사 영문 키가 들어 있음)."""
    host = re.sub(r"[^a-z0-9]", "", urlparse(s.get("url") or "").netloc.lower().removeprefix("www."))
    return s["kind"] == "web" and any(k.isascii() and len(k) >= 4 and k in host for k in keys)


def _pros_cons(target: dict, reg: SourceRegistry, c2_yes: bool) -> list[list[str]]:
    """장점/단점 표. 회사 홈페이지만 근거이거나 근거가 없는 장점은 '회사 주장'으로 표시한다."""
    t = target.get("tech") or {}
    keys = company_keys(target.get("profile") or {})

    def mark(x: str) -> str:
        ids = [i for m in CITE.finditer(x) for i in re.split(r"\s*[,，]\s*", m.group(1))]
        srcs = [reg.get(i) for i in ids if reg.get(i)]
        if not srcs or all(_own_source(s, keys) for s in srcs):
            return f"(회사 주장) {x}"
        if not c2_yes and _superiority(x):
            return f"(회사 측 주장, 제3자 비교 근거 없음) {x}"
        return x

    pros, cons = [mark(x) for x in (t.get("pros") or [])[:3]], list((t.get("cons") or [])[:3])
    return [[pros[i] if i < len(pros) else "", cons[i] if i < len(cons) else ""] for i in range(max(len(pros), len(cons)))]


def _competitor_rows(target: dict, reg: SourceRegistry, notes: list[CompetitorNote], c2_yes: bool) -> list[list[str]]:
    """근거 본문에 이름이 실제로 나오는 경쟁사만 표에 남기고(지어낸 경쟁사 차단), 그 근거를 인용으로 붙인다.
    비교 문장은 보고서 문안(평가표 일치 검사를 거친 것)을 쓰고, C2 가 YES 가 아닌데 남은 우열 표현은 회사 측 주장으로 표시한다."""
    comp = target.get("competition") or {}
    ids = _usable(reg, (comp.get("evidence_ids") or []) + (comp.get("pool_ids") or []))
    texts = {i: norm(reg.text(i)) for i in ids}
    tkeys = [norm(x) for x in _names(target) if len(norm(x)) >= 2]
    rows, found = [], []
    for c in (comp.get("competitors") or [])[:5]:
        key = norm(c["name"].split("(")[0])
        hit = [i for i in ids if len(key) >= 2 and key in texts[i]][:2]
        vs = next((n.vs_target for n in notes if _same(n.name, c["name"])), c.get("vs_target") or "")
        if not c2_yes and _superiority(vs):
            vs = f"(회사 측 주장, 제3자 비교 근거 없음) {vs}"
        offering = c.get("offering") or "-"
        if hit and all(any(k in texts[i] for k in tkeys) for i in hit):
            # 경쟁사 이름이 대상 회사 기사에만 지나가듯 나오면, 그 경쟁사의 제품 설명은 이 표의 근거로 확인할 수 없다
            offering += " (대상 회사 기사에서 이름만 언급)"
        row = [c["name"], c.get("country") or "-", offering, vs, f"[{', '.join(hit)}]" if hit else ""]
        rows.append(row)
        if hit:
            found.append(row)
    return (found or rows)[:4]


def _risk_rows(risks: list[Risk], reg: SourceRegistry) -> list[list[str]]:
    """리스크 근거 id 를 내용 끝에 붙여 REFERENCE 로 이어지게 한다."""
    out = []
    for r in risks:
        cite = _cite(reg, r.evidence_ids) if not CITE.search(r.content) else ""
        out.append([r.type, r.content + cite, r.due_diligence])
    return out


def _first_source(reg: SourceRegistry, ids: list[str], needle: str | None) -> str | None:
    return next((i for i in ids if needle and reg.get(i) and citable(reg.get(i)) and needle in reg.text(i)), None)


def _t0(e: dict) -> str | None:
    f, p = e.get("founder") or {}, e.get("profile") or {}
    return f.get("t0") or p.get("founded_date") or (str(p["founded_year"]) if p.get("founded_year") else None)


def _team_line(e: dict, reg: SourceRegistry) -> str:
    """'대표 OOO (창업 시점 YYYY-MM-DD) [근거]'. 대표는 적격성 기록·창업자 에이전트, t0 는 창업자 에이전트에서 온다."""
    p, f = e.get("profile") or {}, e.get("founder") or {}
    people = f.get("people") or []
    ceo = p.get("ceo") or next((x["name"] for x in people if re.search(r"대표|CEO|창업자", x.get("role") or "", re.I)), None)
    t0 = _t0(e)
    tips = next((x for n in (p.get("name"), p.get("official_name"), e.get("name")) if n and (x := reg._find(f"tips:{n}"))), None)
    ids = ([tips] if tips else []) + list(dict.fromkeys(
        (p.get("evidence_ids") or []) + [i for x in people for i in x.get("evidence_ids") or []] + (f.get("evidence_ids") or [])))
    cites = [x for x in (_first_source(reg, ids, ceo), _first_source(reg, ids, t0)) if x]
    line = f"대표 {ceo}" if ceo else "대표 확인 불가"
    line += f" (창업 시점 {t0})" if t0 else " (창업 시점 확인 불가)"
    return line + (f" [{', '.join(dict.fromkeys(cites))}]" if cites else "")


def _nps_line(p: dict) -> str:
    n = p.get("nps") or {}
    if n.get("status") != "matched" or not n.get("evidence_id"):
        return ""
    return f"국민연금 가입자 {n['members']}명({n['ym']} 기준) [{n['evidence_id']}]"


def _stage_line(p: dict, reg: SourceRegistry) -> str:
    return (f"{p.get('stage') or '단계 미상'} ({p.get('round_date') or '시점 미상'}, {p.get('round_amount') or '금액 미공개'})"
            + _cite(reg, p.get("stage_evidence_ids"), 2))


def _failed_by_company(failed: list[dict], companies: list[list[str]]) -> dict[str, int]:
    """실패한 검색 질의(중복 제외)를 회사별로 센다 (companies: [대표 이름, 다른 표기...]). 회사명이 없는 질의는 '분야·경쟁사 검색'."""
    alias = sorted(((a, c[0]) for c in companies for a in c if len(norm(a)) >= 2), key=lambda x: -len(norm(x[0])))
    out: dict[str, int] = {}
    for q in dict.fromkeys(f.get("query", "") for f in failed):
        who = next((c for a, c in alias if norm(a) in norm(q)), "분야·경쟁사 검색")
        out[who] = out.get(who, 0) + 1
    return dict(sorted(out.items()))  # 병렬 실행 순서와 관계없이 같은 보고서가 나오게


def _short(r: dict) -> str:
    return r.get("short") or r["qid"]


def _verdict_facts(rows: list[dict], dim: str) -> str:
    """기준(차원)별 핵심 사실: 확인 / 반대 / 미확인 문항의 짧은 이름. F4 는 대리지표임을 밝힌다."""
    def nm(r):
        return _short(r) + ("(대리지표)" if r["qid"] == "F4" else "")

    rs = [r for r in rows if r["dim"] == dim]
    parts = [(lab, [nm(r) for r in rs if r["answer"] == a]) for lab, a in (("확인", "YES"), ("반대", "NO"), ("미확인", "UNKNOWN"))]
    return " / ".join(f"{lab} {'·'.join(v)}" for lab, v in parts if v) or "-"


def _criteria_rows(sc: dict) -> list[list[str]]:
    rows = sc["rows"]
    out = [[f"{_dim_short(c['name'])} ({c['weight']}%)", f"{c['yes']} / {c['no']} / {c['unknown']}", _s100(c["pct"]),
            f"{c['contribution'] * 100:.1f}", _verdict_facts(rows, c["dim"])] for c in sc["criteria"]]
    out.append(["합계 (가중합)", "", "", f"{sc['multiplier'] * 100:.1f}", f"동종 평균 = 100, 기준 {_s100(sc['threshold'])}"])
    return out


def _rule_line(sc: dict, cfg) -> str:
    t = sc["threshold"]
    why = ("두 참고자료 예시 배수 1.1205(Eqvista)·1.155(ACA 2019)보다 낮은 1.10, 설계 가정"
           if abs(t - DESIGN_THRESHOLD) < 1e-9 else f"이번 실행에 지정한 기준(설계 기준 {DESIGN_THRESHOLD:.2f}과 다름)")
    return (f"투자 추천 ⇔ 배수 ≥ {t:.2f} ∧ 창업자 문항 YES ≥ {cfg.decision.min_founder_yes} ∧ Deal-killer 없음. "
            f"기준 {t:.2f}: {why}. 기준별 배수 = 1 + {cfg.decision.step} × 평균(문항 신호 − 동종 평균 신호)"
            f"({cfg.decision.clip[0]}~{cfg.decision.clip[1]}로 자름), 신호 YES +1 · NO −1 · 미확인 0")


def _reference_members(ref: dict, cfg) -> list[dict]:
    """기준 집단 파일(보정 실행이 쓴 decision.reference_file)에서 이번 기준 집단 구성원의 단계·지역을 읽는다."""
    try:
        data = json.loads((ROOT / cfg.decision.reference_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    names = set(ref.get("members") or [])
    return [m for m in data.get("members") or [] if m.get("name") in names]


def _reference_line(sc: dict, cfg) -> str:
    """기준 집단 설명. 같은 단계·지역 기업이 아니라 단계·지역이 섞인 근사 기준임을 밝힌다."""
    ref = sc.get("reference") or {}
    rows = _reference_members(ref, cfg)
    stages = [x for x in STAGE_ORDER if any(r.get("stage") == x for r in rows)]
    kr, gl = sum(r.get("region") == "KR" for r in rows), sum(r.get("region") not in (None, "KR") for r in rows)
    mix = (f" — {'·'.join(stages)} 혼합, 국내 {kr}·해외 {gl}의 근사 기준" if rows
           else " — 단계(Seed~Series C)·지역(국내·해외)이 섞인 근사 기준")
    if ref.get("source") != "calibration":
        return (f"기준 집단 {ref.get('n', 0)}곳 < {cfg.decision.reference_min_n}곳 → 동종 평균 신호를 0(미확인 수준)으로 둔 "
                f"설계 가정으로 계산")
    return (f"기준 집단: 적격성 관문을 통과해 같은 파이프라인으로 평가한 {ref.get('n')}곳"
            f"({'대상 제외 평균' if ref.get('loo') else '대상 미포함'}, 보정 실행 {ref.get('run_date') or '-'}){mix}")


def _sensitivity_line(sc: dict) -> str:
    return "민감도(기준 배수별 결정): " + " · ".join(f"{k} {v}" for k, v in (sc.get("sensitivity") or {}).items())


def _bessemer_line(sc: dict) -> str:
    mark = {"YES": "✔", "NO": "✖"}
    items = " ".join(f"Q{b['q']}{mark.get(b['answer'], '?')}" for b in sc.get("bessemer") or [])
    proxy = [f"Q{b['q']}" for b in sc.get("bessemer") or [] if b.get("proxy")]
    return f"Bessemer 10문: {items} (✔ YES · ✖ NO · ? 미확인{', 대리지표: ' + '·'.join(proxy) if proxy else ''})"


def _ranking_line(sc: dict) -> str:
    rk = sc.get("ranking") or []
    top = " · ".join(f"{i}위 {r['name']} {_s100(r['multiplier'])}" for i, r in enumerate(rk[:3], 1))
    tail = f" — 대상 {sc['target_rank']}위/{sc.get('peer_n') or len(rk)}곳" if sc.get("target_rank") else ""
    return f"동종 순위(상위 3): {top}{tail}" if rk else ""


def _agfunder_id(reg: SourceRegistry, cfg) -> str | None:
    """ROI 단계별 중앙값의 출처(AgFunder 2026 보고서)를 근거 id 로. 이미 검색된 조각이 있으면 그것을, 없으면 코퍼스 목록의
    문서 정보로 등록한다(REFERENCE 는 문서 단위로 한 줄)."""
    hit = next((sid for sid, s in reg.data.items() if s.get("kind") == "doc" and s.get("doc_id") == "agfunder2026"), None)
    if hit:
        return hit
    with open(ROOT / cfg.rag.manifest, encoding="utf-8") as f:
        meta = next((d for d in yaml.safe_load(f)["documents"] if d["doc_id"] == "agfunder2026"), None)
    if not meta:
        return None
    return reg.add_doc({**meta, "page": 13}, cfg.roi.stage_median_source, AGENT, "roi stage median")


def _roi_items(sc: dict, reg: SourceRegistry, cfg) -> list[list[str]]:
    """ROI 참고치: 라운드 금액 대 단계별 중앙값, VC Method 필요 Exit. 모든 가정 수치에 '가정'을 붙인다."""
    r = sc.get("roi") or {}
    if not r.get("computable"):
        return [["라운드 금액", f"{r.get('round_amount_raw') or '비공개'} → ROI 산정 불가, 실사 항목으로 넘김"]]
    med, vs = r.get("stage_median_usd_m"), r.get("vs_stage_median")
    src = _agfunder_id(reg, cfg) if med else None
    fx = "원화 금액은 환율 가정으로 환산" if not re.search(r"\$|달러|million", r.get("round_amount_raw") or "", re.I) else ""
    amount = (f"{r['round_amount_raw']} ≈ ${r['round_amount_usd_m']:.1f}M" + (f" ({fx})" if fx else "") + _cite(reg, r.get("source_ids"), 2)
              + (f" — {r.get('stage')} 단계 중앙값 ${med:g}M의 {vs:.1f}배" + (f" [{src}]" if src else "") if med and vs else ""))
    lo, hi = r.get("stake_assumption") or [None, None]
    pm, ex = r.get("post_money_usd_m"), r.get("required_exit_usd_m")
    items = [["라운드 금액", amount]]
    if pm and ex and lo and hi:
        items.append(["VC Method (가정)", f"라운드 투자자 지분율 {lo:.0%}~{hi:.0%} 가정 → post-money ${pm[0]:.1f}M~${pm[1]:.1f}M, "
                                        f"목표 회수 배수 {r.get('target_multiple')}배 가정 → 필요 Exit ${ex[0]:,.0f}M~${ex[1]:,.0f}M"])
    items.append(["가정", "; ".join(r.get("assumptions") or []) + " (후속 라운드 희석 미반영, 점수·결정에 넣지 않음)"])
    return items


def _dd_items(sc: dict) -> list[str]:
    return [f"{d['item']} — {d['why']}" for d in sc.get("dd_items") or []]


def _flip_text(e: dict) -> str:
    """뒤집힘 조건: 어떤 미확인 문항이 확인되면 배수가 얼마가 되는지 (투자 판단 에이전트가 계산). Deal-killer 면 해소가 먼저다."""
    sc = e.get("scorecard") or {}
    f = e.get("flip") or sc.get("flip")
    if not f:
        ks = sc.get("deal_killers") or []
        return f"Deal-killer {', '.join(ks)} 해소 필요 (배수와 관계없이 보류)" if ks else "-"
    return f.get("note") or ("·".join(f.get("items") or []) + (f" 확인 시 동종 평균 대비 {_s100(f['new_multiplier'])}"
                                                            if f.get("new_multiplier") else ""))


def _limitations(cfg, pool: dict, failed_by: dict[str, int], data_limits: list[str], mode: str,
                 access_dates: list[str]) -> list[str]:
    """한계점: 검색 실패, 미평가 적격 후보, 상대 평가, 가정, 데이터 한계, 판정 모델 순으로 최대 6개 + 조회일 표기 1줄
    (REFERENCE 는 목록만 싣고, 게시일 대신 조회일을 쓴 사실은 여기서 밝힌다)."""
    out = []
    if failed_by:
        n = sum(failed_by.values())
        parts = ", ".join(f"{k} {v}건" for k, v in sorted(failed_by.items(), key=lambda kv: (kv[0] == "분야·경쟁사 검색", -kv[1])))
        out.append(f"웹 검색 질의 {n}건이 실패(검색 한도 초과 등)해 빈 결과로 진행했다 — {parts}. "
                   f"해당 후보의 미확인 판정 일부는 근거 부재가 아니라 검색 실패 때문일 수 있다")
    if pool["unevaluated"]:
        out.append(f"적격 후보 {pool['eligible']}곳 중 {len(pool['unevaluated'])}곳은 평가하지 않았다({pool['why']})"
                   + (f" — '투자 추천 없음'은 평가한 {pool['evaluated']}곳에 대한 결론이다" if mode == "hold" else ""))
    if mode != "none":
        out.append("동종 기준 집단 평균 대비 상대 평가라 기준 집단 전체의 질이 낮으면 상대적으로 나은 기업이 추천될 수 있다. "
                   "관문(실제 투자 유치 확인)·창업자 근거 요건·Deal-killer·실사 조건으로 보완한다")
        out.append("ROI 는 환율·지분율·회수 배수 가정에 따른 참고치이며 점수와 결정에 넣지 않았다")
    out += data_limits[:1]
    out.append(f"문항 판정은 {cfg.models.judge} 가 하고, 코드는 인용이 원문에 있는지·제3자·최근 24개월·상용 운영 요건만 검사한다"
               " (근거 해석 오류는 남을 수 있음)")
    access = [f"{ACCESS_DATE_NOTE}({len(access_dates)}건, 조회일 {', '.join(sorted(set(access_dates)))})"] if access_dates else []
    return out[:6] + access


# ── REFERENCE: 인용 번호 매기기

def _same_title(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 15 and (a.startswith(b) or b.startswith(a)))


def _renumber(view: dict, reg: SourceRegistry) -> tuple[dict, list[dict], list[str]]:
    """[W..]/[D..] 인용을 REFERENCE 번호로 바꾸고, 실제 인용된 근거만 REFERENCE 로 만든다.
    - 같은 자료(같은 URL, 포털 전재본처럼 제목이 같은 기사)는 한 번호로 묶는다
    - 번호는 기관 보고서 → 학술 논문 → 웹페이지 순으로 이어지고, 같은 유형 안에서는 처음 인용된 순서
    - 뉴스레터·모음 메일은 인용하지 않는다 (인용 표시에서도 뺀다)
    반환: (번호로 바꾼 view, REFERENCE 항목, 인용된 근거 id)"""
    cited: list[str] = []

    def collect(x):
        if isinstance(x, str):
            cited.extend(i for m in CITE.finditer(x) for i in re.split(r"\s*[,，]\s*", m.group(1)))
        elif isinstance(x, (list, tuple)):
            for v in x:
                collect(v)
        elif isinstance(x, dict):
            for v in x.values():
                collect(v)

    collect(view)
    groups: list[dict] = []
    for sid in dict.fromkeys(cited):
        s = reg.get(sid)
        if not s or not citable(s):
            continue
        k, tk = reference_key(s), title_key(s)
        g = next((g for g in groups if k in g["keys"] or (tk and any(_same_title(tk, x) for x in g["titles"]))), None)
        if g is None:
            g = {"ids": [], "srcs": [], "keys": set(), "titles": []}
            groups.append(g)
        g["ids"].append(sid)
        g["srcs"].append(s)
        g["keys"].add(k)
        if tk:
            g["titles"].append(tk)
    groups.sort(key=lambda g: GROUPS.index(reference_group(g["srcs"][0])))  # 안정 정렬: 유형 안에서는 인용 순서 유지
    num, refs = {}, []
    for n, g in enumerate(groups, 1):
        num.update({sid: n for sid in g["ids"]})
        best = merge_duplicates(g["srcs"])
        refs.append({"n": n, "group": reference_group(g["srcs"][0]), "text": format_reference(best),
                     "access_date": best.get("access_date") if uses_access_date(best) else None})

    def repl(m: re.Match) -> str:
        nums = sorted({num[i] for i in re.split(r"\s*[,，]\s*", m.group(1)) if i in num})
        return "[" + ", ".join(map(str, nums)) + "]" if nums else ""

    def walk(x):
        if isinstance(x, str):
            x = CITE.sub(repl, x)
            x = re.sub(r"\[(?![\d, ]+\])[^\[\]]{1,20}\]", "", x)  # 근거 id 가 아닌 임의 대괄호 표기 제거
            x = re.sub(r"\[\d+(?:, \d+)*\](?:\s*\[\d+(?:, \d+)*\])+",  # "[1][2, 3]" → "[1, 2, 3]"
                       lambda m: "[" + ", ".join(map(str, sorted({int(n) for n in re.findall(r"\d+", m.group(0))}))) + "]",
                       x)
            return re.sub(r"\s+\.", ".", x).strip()
        if isinstance(x, (list, tuple)):
            return [walk(i) for i in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    return walk(view), refs, [i for g in groups for i in g["ids"]]


def _ref_groups(refs: list[dict]) -> list[dict]:
    """과제 형식의 세 소제목을 항상 둔다 (인용한 자료가 없는 유형은 '없음'으로 표시)."""
    return [{"name": g, "items": [r for r in refs if r["group"] == g]} for g in GROUPS]


# ── 보고서 본문 (장 = {title, lead, blocks}. 블록은 md·HTML 이 같은 내용으로 그린다)

def _p(text: str) -> dict:
    return {"t": "p", "text": text}


def _kv(items: list[list[str]]) -> dict:
    return {"t": "kv", "items": [[k, v] for k, v in items if v]}


def _ul(items: list[str], title: str = "") -> dict:
    return {"t": "ul", "title": title, "items": [x for x in items if x]}


def _table(head: list[str], rows: list[list], widths: list[str] | None = None, small: bool = False) -> dict:
    return {"t": "table", "head": head, "rows": [[str(c) for c in r] for r in rows], "widths": widths or [], "small": small}


def _note(text: str) -> dict:
    return {"t": "note", "text": text}


def _idea_blocks(target: dict, d: dict, reg: SourceRegistry, c2_yes: bool, notes: list[CompetitorNote],
                 with_competition: bool) -> list[dict]:
    """사업 아이디어·기술 (A안 1장이면 경쟁 차별성까지)."""
    prof = target.get("profile") or {}
    blocks = [_kv([["해결하는 문제", d["problem"]], ["제품·핵심 컨셉", d["product"]], ["수익 방식", d["revenue_model"]],
                   ["투자 단계", _stage_line(prof, reg)]])]
    pc = _pros_cons(target, reg, c2_yes)
    if pc:
        blocks.append(_table(["핵심 기술의 장점", "단점·한계"], pc, ["50%", "50%"], small=True))
    blocks.append(_p(f"업계 기술 수준 대비: {d['industry_baseline']}"))
    if with_competition:
        blocks += _competition_blocks(target, d, reg, c2_yes, notes)
    return blocks


def _competition_blocks(target: dict, d: dict, reg: SourceRegistry, c2_yes: bool, notes: list[CompetitorNote]) -> list[dict]:
    comp = target.get("competition") or {}
    blocks = [_p(f"경쟁 구도: {d['competition']}")]
    rows = _competitor_rows(target, reg, notes, c2_yes)
    if rows:
        blocks.append(_table(["경쟁사", "국가", "제품·접근", "대상 대비" if c2_yes else "대상 대비 (회사 측 주장 포함)", "근거"],
                             rows, ["14%", "7%", "27%", "45%", "7%"], small=True))
    claims = [f"{c.get('claim')} — {c.get('status')}{_cite(reg, c.get('evidence_ids'), 2)}"
              for c in (comp.get("verified_claims") or [])[:3] if c.get("claim")]
    if claims:
        blocks.append(_ul(claims, "차별점 주장 검증 (경쟁사 에이전트)"))
    return blocks


def _market_blocks(target: dict, d: dict, reg: SourceRegistry) -> list[dict]:
    m = target.get("market") or {}
    size, facts = m.get("size") or {}, []
    if size.get("raw") and reg.get(size.get("source_id") or ""):
        facts.append(["세부 시장 규모", f"{size['raw']}" + (f" ({size['scope']})" if size.get("scope") else "")
                      + f" [{size['source_id']}]"])
    if m.get("growth_rate_pct") and reg.get(m.get("growth_source_id") or ""):
        facts.append(["성장률", f"연평균 {_num(m['growth_rate_pct'])}%" + (f" ({m['growth_year']})" if m.get("growth_year") else "")
                      + f" [{m['growth_source_id']}]"])
    return [_p(d["market"])] + ([_kv(facts)] if facts else [])


def _team_blocks(target: dict, d: dict, reg: SourceRegistry, detail: bool) -> list[dict]:
    """팀의 구성: 창업 시점 t0 표, 인물(창업 전 경력), 창업 후 마일스톤, 팀 평가 문장."""
    f, prof = target.get("founder") or {}, target.get("profile") or {}
    tips = next((x for n in (prof.get("name"), prof.get("official_name"), target.get("name")) if n and (x := reg._find(f"tips:{n}"))), None)
    t0 = _t0(target)
    t0_ids = [tips] if tips and "TIPS" in (f.get("t0_source") or "TIPS") else []
    nps = _nps_line(prof)
    years = f.get("years_since_t0")
    hires = f.get("hires_per_year")
    blocks = [_table(["창업 시점 t0", "t0 근거", "경과", "고용(국민연금)", "최근 24개월 마일스톤"],
                     [[t0 or "확인 불가", (f.get("t0_source") or "-") + _cite(reg, t0_ids, 1),
                       f"{_num(years)}년" if years else "-",
                       (nps + (f", 창업 후 연평균 {_num(hires)}명(가입자 ÷ 경과 연수)" if hires else "")) if nps else "-",
                       f"{f.get('milestones_24m', 0)}건"]], ["16%", "22%", "9%", "33%", "20%"], small=True)]
    people = [[x.get("name") or "-", x.get("role") or "-",
               {True: "창업 전 · ", False: "창업 후 · "}.get(x.get("before_t0"), "") + (x.get("background") or "경력 확인 불가")
               + (_cite(reg, x.get("evidence_ids"), 2) if not CITE.search(x.get("background") or "") else "")]
              for x in (f.get("people") or [])[:4]]
    if people and detail:
        blocks.append(_table(["인물", "역할", "경력 (t0 기준 창업 전/후)"], people, ["12%", "12%", "76%"], small=True))
    ms = [f"{m.get('date')} {m.get('what')}{_cite(reg, m.get('evidence_ids'), 2)}" for m in (f.get("milestones") or [])[:4]
          if m.get("date") and m.get("what")]
    if ms and detail:
        blocks.append(_ul(ms, "창업 후 마일스톤"))
    blocks.append(_p(d["team"]))
    return blocks


def _chapters_invest(target: dict, d: dict, t: dict, reg: SourceRegistry, cfg, pool: dict) -> list[dict]:
    sc = target["scorecard"]
    c2_yes = t["verdict"].get("C2") == "YES"
    ch4_lead = (f"동종 평균 대비 {_s100(sc['multiplier'])}(평균 100, 기준 {_s100(sc['threshold'])}), 창업자 근거 "
                f"{sc.get('founder_yes', 0)}개 확인, Deal-killer 없음 — 투자 추천(실사 조건부)")
    judge = [_table(["기준 (비중)", "YES / NO / 미확인", "동종 평균 대비", "기여", "핵심 사실"], _criteria_rows(sc),
                    ["18%", "13%", "11%", "8%", "50%"], small=True),
             _note(_rule_line(sc, cfg)), _note(_reference_line(sc, cfg)), _note(_sensitivity_line(sc)),
             _note(_bessemer_line(sc)), _note(_ranking_line(sc)),
             {"t": "box", "title": "ROI 참고치 (점수·결정에 넣지 않음)", "items": _roi_items(sc, reg, cfg)},
             {"t": "h3", "text": "사업 리스크(시장·기술·규제·경쟁)와 실사 조건"},
             _table(["유형", "내용", "실사 질문"], t["risks"], ["8%", "57%", "35%"], small=True),
             _ul(_dd_items(sc), f"실사 확인 항목 {len(sc.get('dd_items') or [])}개 (미확인·반대 문항 중 배수 영향이 큰 순)")]
    return [
        {"title": A_CHAPTERS[1], "lead": d["lead_idea"],
         "blocks": _idea_blocks(target, d, reg, c2_yes, t["notes"], with_competition=True)},
        {"title": A_CHAPTERS[2], "lead": d["lead_market"], "blocks": _market_blocks(target, d, reg)},
        {"title": A_CHAPTERS[3], "lead": d["lead_team"], "blocks": _team_blocks(target, d, reg, detail=True)},
        {"title": A_CHAPTERS[4], "lead": ch4_lead, "blocks": judge},
        {"title": A_CHAPTERS[5], "lead": "결론은 공개 정보와 동종 기준 집단 비교에 기반하므로, 아래 한계는 실사로 보완한다.",
         "blocks": [_ul(t["limitations"])]},
    ]


def _candidate_blocks(evals: list[dict], notes: list[CandidateNote], reg: SourceRegistry, failed_by: dict[str, int]) -> list[dict]:
    """후보별 보류 사유 블록(4줄): 사업 / 팀(대표·t0) / 왜 안 되는지 / 뒤집힘 조건."""
    out = []
    for idx, e in enumerate(evals, 1):
        p, sc = e.get("profile") or {}, e["scorecard"]
        rows = sc["rows"]
        note = next((n for n in notes if any(_same(n.name, x) for x in _names(e))), None)
        one_line = p.get("one_line") or "-"
        no_ids = _usable(reg, [x for r in rows if r["answer"] == "NO" for x in r.get("evidence_ids") or []])
        if note:
            why = note.why_not
            if no_ids and not CITE.search(why):  # 반대 근거가 있는데 번호를 빠뜨리면 코드가 그 근거를 붙인다
                why = f"{why.rstrip('.')}. 반대 근거 [{', '.join(no_ids[:3])}]"
            if note.business and re.search(r"[가-힣]", note.business):  # 영어 원문 요약 대신 한국어 한 줄
                one_line = note.business
        else:
            no = [_short(r) for r in rows if r["answer"] == "NO"]
            unk = [_short(r) for r in rows if r["answer"] == "UNKNOWN"]
            why = " / ".join(([f"반대 근거: {', '.join(no)}" + _cite(reg, no_ids)] if no else [])
                             + ([f"미확인: {', '.join(unk[:6])}"] if unk else []))
        fail = next((v for k, v in failed_by.items() if any(_same(k, x) for x in _names(e))), 0)
        team = " · ".join(x for x in (_team_line(e, reg), _nps_line(p), _stage_line(p, reg),
                                      f"대상 웹 검색 {fail}건 실패" if fail else "") if x)
        seg = get_segment(e.get("segment_id") or p.get("segment_id") or "")["name"]
        out.append({"t": "cand", "title": f"2.{idx} {e['name']} — 동종 평균 대비 {_s100(e.get('multiplier', sc.get('multiplier')))} · "
                                          f"{e.get('hold_type') or sc.get('hold_type') or '보류'}", "sub": seg,
                    "items": [["사업", one_line], ["팀", team],
                              ["왜 안 되는가", f"{why} ({'; '.join(e.get('reasons') or sc.get('reasons') or []).replace('UNKNOWN', '미확인')})"],
                              ["뒤집힘 조건", _flip_text(e)]]})
    return out


def _chapters_hold(evals: list[dict], target: dict, d: dict, t: dict, reg: SourceRegistry, cfg, pool: dict,
                   failed_by: dict[str, int]) -> list[dict]:
    sc = target["scorecard"]
    c2_yes = t["verdict"].get("C2") == "YES"
    thr = sc["threshold"]
    un = pool["unevaluated"]
    pool_blocks = [_table(["단계", "후보 수", "국내", "해외", "비고"], _pool_rows(pool), ["22%", "8%", "7%", "7%", "56%"])]
    if un:
        pool_blocks.append(_note(f"미평가 적격 후보 {len(un)}곳 ({pool['why']}): " + ", ".join(
            f"{u['name']}({u['stage']}{', 관문 검색 실패' if u['search_failed'] else ''})" for u in un)))
    ht = target.get("hold_type") or sc.get("hold_type")
    top_why = (f"기준 {_s100(thr)}에 못 미쳤다" if sc["multiplier"] < thr else f"배수 기준은 넘었지만 보류 유형 '{ht}'에 해당한다")
    detail = [{"t": "h3", "text": B_DETAIL[0]}] + _idea_blocks(target, d, reg, c2_yes, t["notes"], with_competition=False)
    detail += [{"t": "h3", "text": B_DETAIL[1]}] + _market_blocks(target, d, reg)
    detail += [{"t": "h3", "text": B_DETAIL[2]}] + _team_blocks(target, d, reg, detail=False)
    detail += [{"t": "h3", "text": B_DETAIL[3]}] + _competition_blocks(target, d, reg, c2_yes, t["notes"])
    detail += [{"t": "h3", "text": B_DETAIL[4]}, _table(["유형", "내용", "실사 질문"], t["risks"], ["8%", "57%", "35%"], small=True)]
    dims = [_dim_short(c["name"]) for c in sc["criteria"]]
    score_rows = [[e["name"]] + [_s100(c["pct"]) for c in e["scorecard"]["criteria"]]
                  + [_s100(e["scorecard"]["multiplier"]), e.get("hold_type") or e["scorecard"].get("hold_type") or "-"]
                  for e in evals]
    sens_keys = list((sc.get("sensitivity") or {}).keys())
    sens = " · ".join(f"{k} → {sum((e['scorecard'].get('sensitivity') or {}).get(k) == '투자' for e in evals)}곳" for k in sens_keys)
    summary = [_table(["후보"] + dims + ["배수", "보류 유형"], score_rows, small=True),
               _note(_rule_line(sc, cfg)), _note(_reference_line(sc, cfg)),
               _note(f"민감도(평가 {len(evals)}곳 중 투자 추천 수): {sens}") if sens else None,
               _p(f"공통 원인과 시장 맥락: {d['common_detail']}")]
    return [
        {"title": B_CHAPTERS[1], "lead": f"{_funnel(pool)} 중 {pool['evaluated']}곳을 평가했다.", "blocks": pool_blocks},
        {"title": B_CHAPTERS[2], "lead": f"평가한 {len(evals)}곳 모두 보류 — {_hold_counts(evals)}.",
         "blocks": _candidate_blocks(evals, t["cands"], reg, failed_by)},
        {"title": f"{B_CHAPTERS[3]} — {target['name']}",
         "lead": f"평가한 {len(evals)}곳 중 최고점(동종 평균 대비 {_s100(sc['multiplier'])})이지만 {top_why}.",
         "blocks": detail},
        {"title": B_CHAPTERS[4], "lead": f"최고점 {_s100(sc['multiplier'])}(기준 {_s100(thr)}) — 투자 추천 대상 없음.",
         "blocks": [b for b in summary if b]},
        {"title": B_CHAPTERS[5], "lead": "보류는 '공개 정보로 확인되지 않음'을 포함하므로, 아래 한계를 함께 읽어야 한다.",
         "blocks": [_ul(t["limitations"])]},
    ]


# ── 마크다운 (HTML 과 같은 view 로 만든다)

def _cell(x: str) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _block_md(b: dict) -> list[str]:
    if b["t"] == "p":
        return [b["text"], ""]
    if b["t"] == "note":
        return [f"> {b['text']}", ""]
    if b["t"] == "kv":
        return [f"- {k}: {v}" for k, v in b["items"]] + [""]
    if b["t"] == "ul":
        return ([f"{b['title']}", ""] if b.get("title") else []) + [f"- {x}" for x in b["items"]] + [""]
    if b["t"] == "table":
        return ["| " + " | ".join(_cell(h) for h in b["head"]) + " |", "|" + "---|" * len(b["head"]),
                *["| " + " | ".join(_cell(c) for c in r) + " |" for r in b["rows"]], ""]
    if b["t"] == "h3":
        return [f"### {b['text']}", ""]
    if b["t"] == "cand":
        return [f"### {b['title']} ({b['sub']})", *[f"- {k}: {v}" for k, v in b["items"]], ""]
    if b["t"] == "box":
        return [f"**{b['title']}**", "", *[f"- {k}: {v}" for k, v in b["items"]], ""]
    return []


def _to_markdown(v: dict) -> str:
    L = [f"# {v['title']} ({v['run_date']})", "", v["meta"], "", "## SUMMARY", *[f"- {k}: {x}" for k, x in v["summary"]], ""]
    for ch in v["chapters"]:
        L += [f"## {ch['title']}", "", f"**{ch['lead']}**", ""]
        for b in ch["blocks"]:
            L += _block_md(b)
    L += ["## REFERENCE"]
    for g in v["ref_groups"]:
        L += ["", f"### {g['name']}", *([f"{r['n']}. {r['text']}" for r in g["items"]] or ["- 본문에 인용한 자료 없음"])]
    return "\n".join(L) + "\n"


# ── 형식 검사 (렌더링한 md·REFERENCE·평가 결과로)

def _checks(md: str, view: dict, refs: list[dict], mode: str, expect: dict, draft_probs: list[str], limit: int) -> dict:
    heads = re.findall(r"^## (.+)$", md, re.M)
    subs = re.findall(r"^### (.+)$", md.split("\n## REFERENCE")[0], re.M)
    numbers = {int(m.group(1)) for h in heads if (m := re.match(r"(\d+)\. ", h))}
    refd = [int(n) for m in CHAPTER_REF.finditer(md) for n in re.split(r"[·,\s]+", m.group(1)) if n]
    body = md.split("\n## REFERENCE")[0]
    cited = {int(n) for m in NUM_CITE.finditer(body) for n in m.group(1).split(", ")}
    listed = {r["n"] for r in refs}
    summary_text = " ".join(x for _, x in view["summary"])
    concl = next((x for k, x in view["summary"] if k == "결론"), "")
    items_in = heads + (subs if mode == "hold" else [])
    fmt_bad = [r["text"] for r in refs
               if not REFERENCE_FORMATS[r["group"]].match(r["text"]) or re.search(r"\s\([^()]*\)$", r["text"])]
    assumption_bad = [ln for ln in md.splitlines() if ASSUMPTION_TERMS.search(ln) and "가정" not in ln]
    return {
        "first_section": heads[0] if heads else None, "last_section": heads[-1] if heads else None,
        "banned_hits": [b for b in BANNED if b in summary_text],
        "summary_rule_violations": _summary_problems([x for _, x in view["summary"]], limit),
        "chapter_refs_ok": bool(refd) and all(n in numbers for n in refd) if mode != "none" else all(n in numbers for n in refd),
        "required_items_ok": mode == "none" or all(any(it in h for h in items_in) for it in REQUIRED_ITEMS),
        "refs_match_citations": cited == listed, "reference_format_ok": not fmt_bad, "reference_format_bad": fmt_bad[:3],
        "conclusion_matches_scorecard": bool(expect["conclusion_ok"](concl)),
        "assumptions_labeled": not assumption_bad,
        "consistency_violations": draft_probs,
    }


# ── 보고서 노드

def _brief_json(x: dict, reg: SourceRegistry) -> str:
    """분석 결과를 LLM 에 넘길 JSON (판정 사본·풀 목록 제외, 뉴스레터 근거 id 제거)."""
    def scrub(v):
        if isinstance(v, str):
            return CITE.sub(lambda m: f"[{', '.join(k)}]" if (k := _usable(reg, re.split(r"\s*[,，]\s*", m.group(1)))) else "", v)
        if isinstance(v, list):
            return (_usable(reg, v) if v and all(isinstance(i, str) and re.fullmatch(r"[WD][0-9a-f]{5}", i) for i in v)
                    else [scrub(i) for i in v])
        if isinstance(v, dict):
            return {k: scrub(i) for k, i in v.items() if k not in ("criterion", "pool_ids", "search_plan", "answers")}
        return v
    return json.dumps(scrub(x or {}), ensure_ascii=False)


def _candidates_context(evals: list[dict], reg: SourceRegistry, failed_by: dict[str, int]) -> str:
    """모두 보류 보고서용: 후보마다 평가표 요약 (보류 유형·반대 근거·확인된 사실·미확인 항목)."""
    blocks = []
    for e in evals:
        p, sc = e.get("profile") or {}, e["scorecard"]
        rows = sc["rows"]

        def fmt(r):
            return f"{r['qid']} {_short(r)}: \"{(r.get('quote') or '')[:80]}\"" + _cite(reg, r.get("evidence_ids"))

        fail = next((v for k, v in failed_by.items() if any(_same(k, x) for x in _names(e))), 0)
        blocks.append("\n".join([
            f"### {e['name']} — 보류 유형: {e.get('hold_type') or sc.get('hold_type')} (동종 평균 대비 {_s100(sc['multiplier'])})",
            f"- 사업: {p.get('one_line')} / 단계: {p.get('stage')} ({p.get('round_date') or '시점 미상'}, "
            f"{p.get('round_amount') or '금액 미공개'}) / 대표: {p.get('ceo') or '확인 불가'} / 창업 시점: {_t0(e) or '확인 불가'}",
            "- 반대 근거(NO): " + ("; ".join(f"{fmt(r)} — {_human_reason(r['rationale'])}" for r in rows if r["answer"] == "NO") or "없음"),
            "- 확인된 사실(YES): " + ("; ".join(fmt(r) for r in rows if r["answer"] == "YES") or "없음"),
            "- 공개 근거 없음(미확인): " + (", ".join(f"{r['qid']} {_short(r)}" for r in rows if r["answer"] == "UNKNOWN") or "없음"),
        ] + ([f"- 이 회사 대상 웹 검색 {fail}건이 실패했다 (근거 부족의 일부는 검색 실패 탓일 수 있음)"] if fail else [])))
    return "\n\n".join(blocks)


def _context(mode: str, target: dict, evals: list[dict], reg: SourceRegistry, pool: dict, conclusion: str,
             comp_names: str, run_date: str, failed_by: dict[str, int]) -> dict:
    prof, sc = target.get("profile") or {}, target["scorecard"]
    rows = sc["rows"]
    pool_ids = _usable(reg, (prof.get("evidence_ids") or []) + (prof.get("stage_evidence_ids") or [])
                       + [i for k in ("founder", "tech", "market", "competition") for i in (target.get(k) or {}).get("evidence_ids") or []]
                       + [i for r in rows for i in (r.get("evidence_ids") or [])])
    other_ids = [i for i in _usable(reg, [i for e in evals if e is not target for i in
                                          ((e.get("profile") or {}).get("stage_evidence_ids") or [])
                                          + [x for r in e["scorecard"]["rows"] for x in (r.get("evidence_ids") or [])]])
                 if i not in pool_ids]
    return {
        "mode": mode, "run_date": run_date, "conclusion": conclusion, "target": target["name"],
        "pool": (f"발굴 {pool['discovered'] if pool['discovered'] is not None else '-'}곳 → 관문 {pool['screened']}곳 → "
                 f"적격 {pool['eligible']}곳 → 평가 {pool['evaluated']}곳 (미평가 적격 {len(pool['unevaluated'])}곳: {pool['why']})"),
        "profile": json.dumps({k: prof.get(k) for k in ("official_name", "region", "one_line", "founded_date", "ceo", "stage",
                                                        "round_date", "round_amount")}, ensure_ascii=False),
        "founder": _brief_json(target.get("founder"), reg), "tech": _brief_json(target.get("tech"), reg),
        "market": _brief_json(target.get("market"), reg), "competition": _brief_json(target.get("competition"), reg),
        "competitor_names": comp_names,
        "scorecard": json.dumps({"동종 평균 대비": _s100(sc["multiplier"]), "기준": _s100(sc["threshold"]),
                                 "결정": sc["decision"], "보류 유형": sc.get("hold_type"), "사유": sc.get("reasons"),
                                 "기준별": {_dim_short(c["name"]): _s100(c["pct"]) for c in sc["criteria"]},
                                 "실사 항목": [x["item"] for x in sc.get("dd_items") or []]}, ensure_ascii=False),
        "verified": "\n".join(f"- {r['qid']} {_short(r)}: \"{r['quote']}\"{_cite(reg, r.get('evidence_ids'))}"
                              for r in rows if r["answer"] == "YES" and r.get("quote")) or "(없음)",
        "unverified": ", ".join(f"{r['qid']}({_short(r)}: {_ko(r['answer'])})" for r in rows if r["answer"] not in ("YES", "N/A")),
        "rejected": "\n".join(f"- {r['qid']}: \"{r['quote']}\"" for r in _rejected_yes(target) if r.get("quote")) or "(없음)",
        "candidates": _candidates_context(evals, reg, failed_by) if mode == "hold" else "",
        "evidence": reg.brief(pool_ids, 420) + ("\n\n" + reg.brief(other_ids, 300) if mode == "hold" and other_ids else ""),
    }


def _out_paths(cfg):
    out_dir = path(f"{cfg.report.output_dir}/.keep").parent
    team = cfg.submission
    submit = out_dir / f"RAG-Output_{team.campus}-{team['class']}_{'+'.join(sorted(team.members))}.pdf"
    return out_dir, out_dir / "investment_report.pdf", submit


def _finish(view: dict, md: str, result: dict, cfg, checks: dict, mode: str, cited: list[str], title: str,
            target: str | None, decision: str) -> dict:
    out_dir, pdf_path, submit = _out_paths(cfg)
    md_path = out_dir / "investment_report.md"
    md_path.write_text(md, encoding="utf-8")
    submit_pdf = None
    if not cfg.report.get("scenario"):  # 시나리오 실행(--out)은 제출용 파일을 만들지 않는다
        shutil.copyfile(pdf_path, submit)
        submit_pdf = str(submit)
    checks = {"pages": result["pages"], "max_pages": cfg.report.max_pages, "pages_ok": result["pages"] <= cfg.report.max_pages,
              "summary_ratio": result["summary_ratio"], "summary_ok": result["summary_ratio"] <= 0.5, **checks}
    # v1 이름 (v1 README 생성기 docs/README.md.j2 가 읽는다. README 를 v2 키로 바꾸면 지워도 된다)
    checks |= {"summary_ratio_of_a4": checks["summary_ratio"],
               "scorecard_consistency_violations": checks.get("consistency_violations", [])}
    bad = [k for k, v in checks.items() if (k.endswith("_ok") or k in ("refs_match_citations", "conclusion_matches_scorecard",
                                                                        "assumptions_labeled")) and v is False]
    msg = (f"[보고서] {mode} {pdf_path.name} {result['pages']}쪽 (≤{cfg.report.max_pages}), SUMMARY A4 대비 "
           f"{result['summary_ratio']:.0%}, REFERENCE {sum(len(g['items']) for g in view['ref_groups'])}건, "
           f"형식 검사 {'통과' if not bad else '실패: ' + ', '.join(bad)}"
           + (f", 평가표 불일치 {len(checks['consistency_violations'])}건" if checks.get("consistency_violations") else "")
           + (f" → {submit.name}" if submit_pdf else " (시나리오 실행: 제출 파일 없음)"))
    print(msg)
    return {"report": {"mode": mode, "md": str(md_path), "html": result["html"], "pdf": str(pdf_path), "submit_pdf": submit_pdf,
                       "pages": result["pages"], "summary_ratio": result["summary_ratio"], "cited_ids": cited,
                       "checks": checks, "title": title, "target": target, "decision": decision,
                       "generated_at": datetime.now().isoformat(timespec="seconds")},
            "log": [msg]}


def report_node(state: dict) -> dict:
    cfg = get_config()
    if cfg.workflow.get("calibrate"):  # 보정 실행: 기준 집단만 만들고 보고서는 쓰지 않는다 (그래프 간선은 본 실행과 같다)
        msg = "[보고서] 보정 실행 — 렌더링하지 않음 (기준 집단은 app.py 가 evaluations 로 기록)"
        print(msg)
        return {"report": {"mode": "calibrate", "md": None, "html": None, "pdf": None, "submit_pdf": None, "checks": {}},
                "log": [msg]}
    reg = SourceRegistry(state.get("registry"))
    evals = state.get("evaluations") or []
    screened = state.get("screened") or []
    run_date = state.get("run_date") or datetime.now().strftime("%Y-%m-%d")
    if not evals:
        return _no_candidate_report(state, cfg, screened, reg, run_date)
    invested = [e for e in evals if e["decision"] == "투자"]
    mode = "invest" if invested else "hold"
    ranked = sorted(evals, key=lambda e: -e["scorecard"]["multiplier"])
    target = invested[0] if invested else ranked[0]
    prof, sc = target.get("profile") or {}, target["scorecard"]
    pool = _pool(state, evals, screened, cfg, mode)
    companies = [_names(e) for e in evals] + [
        [n for n in (r.get("official_name") or r.get("name"), r.get("name"), r.get("name_en")) if n] for r in screened]
    failed_by = _failed_by_company(list(getattr(search_tool, "FAILED_QUERIES", []) or []), companies)
    verdict = {r["qid"]: r["answer"] for r in sc["rows"]}
    k_dd = len(sc.get("dd_items") or [])
    seg = get_segment(target.get("segment_id") or prof.get("segment_id") or "")["name"]
    un = pool["unevaluated"]

    # 결론·상황·요청 줄은 코드가 평가 결과로 쓴다 (LLM 이 바꿀 수 없음)
    if mode == "invest":
        rank = sc.get("target_rank")
        peer = f"동종 {sc.get('peer_n')}곳 중 {rank}위" if rank else f"기준 집단 {(sc.get('reference') or {}).get('n', 0)}곳"
        conclusion = (f"{target['name']} 투자 추천(실사 조건부) — 동종 평균 대비 {_s100(sc['multiplier'])}"
                      f"(평균 100, 기준 {_s100(sc['threshold'])}), {peer}")
        situation = f"{_funnel(pool)} → {evals.index(target) + 1}번째 평가 대상 {target['name']}({prof.get('stage') or '-'}, {seg})"
        request = (f"실사 확인 항목 {k_dd}개를 조건으로 투자심의 상정을 진행할까요?" if k_dd else "투자심의 상정을 진행할까요?")
        title = f"{target['name']} 투자 검토 — 투자 추천(실사 조건부)"
        badge = "투자 추천"
        expect = lambda c: (c.startswith(f"{target['name']} 투자 추천") and target["decision"] == "투자"  # noqa: E731
                            and f"동종 평균 대비 {_s100(sc['multiplier'])}" in c)
    else:
        scope = (f"적격 {pool['eligible']}곳 중 비용 상한 {len(evals)}곳 평가" if un and pool["capped"]
                 else f"적격 {pool['eligible']}곳 중 {len(evals)}곳 평가" if un else f"적격 {pool['eligible']}곳 모두 평가")
        conclusion = (f"투자 추천 없음 — 평가한 {len(evals)}곳 모두 보류({scope}), 최고점 {target['name']} 동종 평균 대비 "
                      f"{_s100(sc['multiplier'])}(기준 {_s100(sc['threshold'])})")
        situation = f"{_funnel(pool)} → {len(evals)}곳 평가(국내 {pool['ev_split'][0]}·해외 {pool['ev_split'][1]})"
        request = (f"{target['name']} 실사 착수 또는 미평가 적격 {len(un)}곳 추가 평가를 승인할까요?" if un
                   else f"{target['name']} 실사 착수를 승인할까요?")
        title = f"AgTech AI 스타트업 투자 검토 — 투자 추천 없음 (평가 {len(evals)}곳 모두 보류)"
        badge = "투자 추천 없음"
        expect = lambda c: (c.startswith("투자 추천 없음") and all(e["decision"] == "보류" for e in evals)  # noqa: E731
                            and f"평가한 {len(evals)}곳 모두 보류" in c)
    c2_yes = verdict.get("C2") == "YES"
    comp_names = ", ".join(r[0] for r in _competitor_rows(target, reg, [], c2_yes))
    ctx = _context(mode, target, ranked, reg, pool, conclusion, comp_names, run_date, failed_by)
    schema = Draft if mode == "invest" else HoldDraft

    def summary(d: _Body) -> list[tuple[str, str]]:
        if isinstance(d, Draft):
            return [("상황", situation), ("결론", conclusion),
                    ("근거", f"{_bare(d.ev_tech)}(→1장) · {_bare(d.ev_market)}(→2장) · {_bare(d.ev_team)}(→3장)"),
                    ("리스크", f"{_bare(d.risk_line)} — 실사 확인 항목 {k_dd}개(→4장)"), ("요청", request)]
        return [("상황", situation), ("결론", conclusion),
                ("원인", f"보류 유형 {_hold_counts(evals)} — {_bare(d.common_cause)}(→4장)"),
                ("재검토", f"가장 가까운 후보 {target['name']}: {_flip_text(target)}(→2장)"), ("요청", request)]

    # 1) 초안 → SUMMARY 규칙 + 평가표 일치 검사. 어긋나면 이유를 붙여 다시 쓴다
    #    (재작성 상한은 분량 초과 때의 축약 재작성과 같은 report.max_rewrites 를 쓴다)
    max_rw = int(cfg.report.get("max_rewrites", 2))
    draft, feedback, probs, rewrites = None, "", [], 0
    for attempt in range(max_rw + 1):
        draft = structured(schema).invoke(render("report", **ctx, feedback=feedback, shorten=False))
        draft = schema.model_validate(_fix_cites(draft.model_dump()))
        probs = (_summary_problems([x for _, x in summary(draft)], cfg.report.summary_max_chars)
                 + _consistency_problems(draft, target, ranked, run_date))
        if not probs:
            break
        if attempt < max_rw:
            rewrites += 1
            feedback = "직전 초안의 문제를 모두 고쳐라:\n- " + "\n- ".join(probs)

    out_dir, pdf_path, _ = _out_paths(cfg)
    meta = (f"{cfg.domain.name} · 평가 {len(evals)}곳 · 기준 배수 {sc['threshold']:.2f}(동종 평균 = 1.00) · 작성일 {run_date}"
            if mode == "hold" else
            f"{cfg.domain.name} · {prof.get('official_name') or target['name']} ({prof.get('stage') or '-'}"
            f"{', ' + prof['round_date'] if prof.get('round_date') else ''}) · {seg} · 작성일 {run_date}")

    # 2) 렌더링 → 분량 검증. 5쪽을 넘으면 조판 밀도를 올리고, 그래도 넘으면 줄여 쓴다
    result, density, shorten_round = {}, 0, 0
    while True:
        d = draft.model_dump()
        t = {"verdict": verdict, "notes": draft.competitor_notes, "risks": _risk_rows(draft.risks, reg),
             "cands": getattr(draft, "candidates", []), "limitations": []}
        view = {"title": title, "badge": badge, "mode": mode, "meta": meta, "run_date": run_date,
                "summary": summary(draft), "chapters": []}
        # 한계점의 조회일 건수는 REFERENCE 를 만든 뒤에 알 수 있어 두 번 번호를 매긴다 (인용은 한계점에 없다)
        view["chapters"] = (_chapters_invest(target, d, t, reg, cfg, pool) if mode == "invest"
                            else _chapters_hold(ranked, target, d, t, reg, cfg, pool, failed_by))
        _, refs0, _ = _renumber(view, reg)
        t["limitations"] = _limitations(cfg, pool, failed_by, d["data_limits"], mode,
                                        [r["access_date"] for r in refs0 if r["access_date"]])
        view["chapters"] = (_chapters_invest(target, d, t, reg, cfg, pool) if mode == "invest"
                            else _chapters_hold(ranked, target, d, t, reg, cfg, pool, failed_by))
        numbered, refs, cited = _renumber(view, reg)
        numbered["ref_groups"] = _ref_groups(refs)
        html = render_html(numbered, density=density)
        result = html_to_pdf(html, pdf_path)
        if result["pages"] <= cfg.report.max_pages and result["summary_ratio"] <= 0.5:
            break
        if density < 3:
            density += 1
            continue
        if shorten_round >= max_rw:
            break
        shorten_round += 1
        rewrites += 1
        density = 1
        draft = structured(schema).invoke(render("report", **ctx, feedback="분량 초과", shorten=True))
        draft = schema.model_validate(_fix_cites(draft.model_dump()))
        probs = (_summary_problems([x for _, x in summary(draft)], cfg.report.summary_max_chars)
                 + _consistency_problems(draft, target, ranked, run_date))

    md = _to_markdown(numbered)
    checks = _checks(md, numbered, refs, mode, {"conclusion_ok": expect}, probs, cfg.report.summary_max_chars)
    checks |= {"density_level": density, "rewrites": rewrites, "references": len(refs),
               "reference_groups": {g["name"]: len(g["items"]) for g in numbered["ref_groups"]}}
    return _finish(numbered, md, result, cfg, checks, mode, cited, title, target["name"],
                   "투자 추천" if mode == "invest" else "투자 추천 대상 없음")


def _no_candidate_report(state: dict, cfg, screened: list[dict], reg: SourceRegistry, run_date: str) -> dict:
    """C안 — 적격 후보가 하나도 없을 때: 탐색 경과와 탈락 사유만 담은 보고서 (LLM 없이 코드로 작성)."""
    pool = _pool(state, [], screened, cfg, "none")
    reasons: dict[str, int] = {}
    for r in screened:
        key = re.sub(r"^G\d(?:/G\d)?\s*", "", (r.get("reason") or "-")).split("(")[0].split(":")[0].split(" — ")[0][:30].strip()
        reasons[key] = reasons.get(key, 0) + 1
    disc = f"{pool['discovered']}곳을 발굴해 " if pool.get("discovered") else ""
    view = {
        "title": "AgTech AI 스타트업 투자 검토 — 적격 후보 없음", "badge": "투자 추천 없음", "mode": "none",
        "meta": f"{cfg.domain.name} · 적격 후보 0곳 · 작성일 {run_date}", "run_date": run_date,
        "summary": [("상황", f"국내외 AgTech AI 스타트업 {disc}{len(screened)}곳을 적격성 관문(비상장·Seed~Series C·Exit 전)으로 "
                            "검증했으나 통과한 곳이 없다"),
                    ("결론", "투자 추천 대상 없음 — 적격 후보 0곳"),
                    ("근거", "탈락 사유 " + (", ".join(f"{k} {v}곳" for k, v in reasons.items()) or "-") + "(→1장)"),
                    ("요청", "발굴 채널·세부 분야를 넓혀 다시 탐색할까요?")],
        "chapters": [
            {"title": C_CHAPTERS[1], "lead": f"검증한 {len(screened)}곳 모두 과제 기준(비상장·Seed~Series C·Exit 전)을 충족하지 못했다.",
             "blocks": [_table(["후보", "단계", "관문 판정"], [[r.get("official_name") or r["name"], r.get("stage") or "-",
                                                               r.get("reason") or "-"] for r in screened], ["22%", "12%", "66%"])]},
            {"title": C_CHAPTERS[2], "lead": "공개 정보로 투자 단계를 확인하지 못한 후보는 보수적으로 제외했다.",
             "blocks": [_ul(_limitations(cfg, pool, {}, [], "none", []))]},
        ],
    }
    numbered, refs, cited = _renumber(view, reg)
    numbered["ref_groups"] = _ref_groups(refs)
    _, pdf_path, _ = _out_paths(cfg)
    result = html_to_pdf(render_html(numbered, density=0), pdf_path)
    md = _to_markdown(numbered)
    checks = _checks(md, numbered, refs, "none", {"conclusion_ok": lambda c: c.startswith("투자 추천 대상 없음")}, [],
                     cfg.report.summary_max_chars)
    checks |= {"density_level": 0, "rewrites": 0, "references": len(refs),
               "reference_groups": {g["name"]: len(g["items"]) for g in numbered["ref_groups"]}}
    return _finish(numbered, md, result, cfg, checks, "none", cited, view["title"], None, "적격 후보 없음")
