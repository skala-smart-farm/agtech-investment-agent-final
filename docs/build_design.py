"""설계 산출물 v2 생성: docs/design.md (GitHub 에서 읽는 용) + RAG-Design_...pdf (제출용).

설계 내용(State 키·그래프 간선·에이전트·도구·평가 포인트 매핑)은 이 파일의 상수에서 만든다.
코드(graph/, agents/, tools/)가 이 상수와 같은지는 코드 반영 뒤 비교한다.
문서 목록·검색기 실측·적격성 정확도는 data/manifest.yaml 과 outputs/eval/ 에서 읽는다.

    uv run python -m docs.build_design
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import markdown
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader

from core.config import ROOT, get_config, path
from rag.loader import load_manifest, total_pages

START, END = "__start__", "__end__"

# ── 메인 그래프 (가이드 Graph(안) + 👤 창업자 평가) ─────────────────────────────
NODE_LABELS = {
    START: "START",
    "discover": "🔍 스타트업 탐색<br/>발굴 → 적격성 관문 → 다음 후보 1곳",
    "founder": "👤 창업자 평가<br/>창업자 30%",
    "tech": "🗜️ 기술 요약<br/>제품/기술력 15% · Agentic RAG",
    "market": "📊 시장성 평가<br/>시장성 25% · Agentic RAG",
    "competition": "🥊 경쟁사 비교<br/>경쟁 우위 10%",
    "decide": "🧮 투자 판단<br/>실적·투자조건 20% · 동종 대비 배수",
    "report": "📝 보고서 생성<br/>A안 투자 · B안 모두 보류 · C안 후보 없음",
    END: "END",
}
# (출발, 도착, 간선 이름) — 설계서 그림과 코드 간선이 1:1 이어야 한다
MAIN_EDGES = [
    (START, "discover", None),
    ("discover", "founder", "평가 대상 있음"),
    ("discover", "report", "후보 소진 · 적격 후보 없음"),
    ("founder", "tech", None),
    ("tech", "market", None),
    ("market", "competition", None),
    ("competition", "decide", None),
    ("decide", "report", "투자 추천 또는 평가 상한 도달"),
    ("decide", "discover", "보류 → 다른 스타트업"),
    ("report", END, None),
]
DESIGN_EDGES = {(a, b) for a, b, _ in MAIN_EDGES}

# ── State (계약 C1) ─────────────────────────────────────────────────────────
# key, 타입, reducer(None = 덮어쓰기), 쓰는 노드, 읽는 노드, 설명
DESIGN_STATE = [
    ("domain", "str", None, "app(초기값)", "discover", "평가 도메인(AgTech)"),
    ("run_date", "str", None, "app(초기값)", "전 노드", "평가 기준일 — 24개월 판정·조회일의 기준"),
    ("registry", "dict", "merge_dict", "모든 에이전트", "모든 에이전트 · report",
     "근거 저장소 {근거 id: 출처 정보}. id = URL·문서 해시(W/D). REFERENCE 의 재료"),
    ("discovery_rounds", "int", None, "discover(발굴)", "discover(진입 판단)", "발굴 라운드 수(최대 2)"),
    ("raw_candidates", "list[dict]", None, "discover(발굴)", "discover(관문)", "이번 라운드에 발굴한 후보"),
    ("seen", "list[str]", "add_unique", "discover(발굴)", "discover(발굴)", "이미 다룬 후보 이름 — '다른 스타트업'을 보장"),
    ("screened", "list[dict]", "operator.add", "discover(관문)", "report", "관문 판정 기록(통과·탈락 사유) — 보고서의 탐색 경과 표"),
    ("queue", "list[dict]", None, "discover(관문·선택)", "discover", "적격 후보 대기열(국내·해외 몫을 섞음)"),
    ("current", "dict / None", None, "discover(선택·소진)", "founder · tech · market · competition · decide · 라우터",
     "평가 대상 프로필(단계·라운드·금액·대표·설립일·국민연금). None = 후보 소진"),
    ("iterations", "int", None, "discover(선택)", "라우터 · report", "심층 평가한 후보 수(최대 10)"),
    ("founder", "dict", None, "founder", "tech · market · competition(근거 풀) · decide",
     "창업 시점 t0, 인물(창업 전 경력), 날짜 있는 마일스톤, 고용 추이, 창업자 기준 판정"),
    ("tech", "dict", None, "tech", "market(근거 풀) · competition(차별점 주장) · decide",
     "제품·핵심 기술·장점/단점·차별점 주장·기술 기준선, 제품/기술력 판정"),
    ("market", "dict", None, "market", "competition(근거 풀) · decide", "시장 규모(수치·연도·출처)·성장·수요·정책, 시장성 판정"),
    ("competition", "dict", None, "competition", "decide", "경쟁사 비교표·차별점 검증 결과, 경쟁 우위 판정"),
    ("market_cache", "dict", "merge_dict", "market", "market", "세부 분야별 시장 분석 재사용(같은 분야 후보의 비용 절감)"),
    ("scorecard", "dict", None, "decide", "없음(report 는 evaluations 안의 사본을 읽음)",
     "6개 기준 비교율·배수 M·결정·보류 유형·뒤집힘 조건·실사 항목·ROI·민감도·순위"),
    ("decision", "'투자' / '보류'", None, "decide", "라우터", "이번 후보의 결정"),
    ("evaluations", "list[dict]", "operator.add", "decide", "report", "후보별 평가 누적 — 보류 사유·순위 보고서의 재료"),
    ("end_reason", "str", None, "discover(소진) · decide", "report", "종료 사유 invest_found / max_evaluations / exhausted / no_eligible"),
    ("report", "dict", None, "report", "app(run_log)", "보고서 모드·파일·쪽수·형식 검사 결과"),
    ("rag_traces", "list[dict]", "operator.add", "discover · tech · market", "app(run_log)",
     "RAG 경로 기록(도구 선택·재작성·웹 보완·답변 점검 결과)"),
    ("log", "list[str]", "operator.add", "전 노드", "app(run_log)", "분기 이유가 들어간 진행 기록"),
]
RESET_BY_PICK = {"founder", "tech", "market", "competition", "scorecard"}

# ── 에이전트 (가이드 6개 + 👤 창업자) ───────────────────────────────────────
AGENTS = [
    {"name": "🔍 스타트업 탐색", "node": "discover", "guide": "🔍 스타트업 탐색",
     "role": "평가할 **다음 적격 스타트업 1곳**을 정한다. 내부 3단계: 발굴 → 적격성 관문 G1~G6 → 다음 후보 선택",
     "criterion": "— (평가 전 관문)", "rag": "O — 문서 속 기관 선정 기업(교정형 검색)",
     "tools": "웹 검색, TIPS 목록·투자 기사 피드, 상장 목록 대조, 국민연금 사업장",
     "state_keys": "current, queue, screened, seen, raw_candidates, discovery_rounds, iterations, end_reason"},
    {"name": "👤 창업자 평가", "node": "founder", "guide": "**추가**",
     "role": "창업 시점(t0)을 먼저 고정하고, 창업 **전** 전문성과 창업 **후** 실행력을 나눠 판정",
     "criterion": "창업자 30% (F1~F4)", "rag": "X — 사람에 관한 사실은 문서 코퍼스에 없음",
     "tools": "웹 검색(인터뷰·기사), 기사 원문 수집, 국민연금 조회 결과(첫 고용·인원)", "state_keys": "founder"},
    {"name": "🗜️ 기술 요약", "node": "tech", "guide": "🗜️ 기술 요약",
     "role": "핵심 기술·장단점을 요약하고, 경쟁사 비교가 검증할 **차별점 주장**을 정리",
     "criterion": "제품/기술력 15% (P1~P4)", "rag": "O — 분야 기술 기준선(논문·보고서)",
     "tools": "문서 요약(홈페이지·기사), Agentic RAG, 웹 검색", "state_keys": "tech"},
    {"name": "📊 시장성 평가", "node": "market", "guide": "📊 시장성 평가",
     "role": "세부 분야의 시장 크기·성장·수요(실제 문제·지불 의향)",
     "criterion": "시장성 25% (M1~M4)", "rag": "**O (주 사용처)** — 하위 질문 3~6개마다 Agentic RAG",
     "tools": "Agentic RAG, 웹 검색", "state_keys": "market, market_cache"},
    {"name": "🥊 경쟁사 비교", "node": "competition", "guide": "🥊 경쟁사 비교",
     "role": "같은 제품 유형의 국내외 경쟁사와 비교하고, 기술 요약의 차별점 주장을 검증",
     "criterion": "경쟁 우위 10% (C1~C4)", "rag": "X (가이드 표 그대로)",
     "tools": "웹 검색, 기사 속 경쟁사 추출", "state_keys": "competition"},
    {"name": "🧮 투자 판단", "node": "decide", "guide": "🧮 투자 판단",
     "role": "실적·투자조건을 판정하고 6개 기준을 **종합**(동종 대비 배수). 리스크(Deal-killer)·ROI·후보 리스트를 곁들여 **투자/보류** 결정",
     "criterion": "실적 10%·투자조건 10% (R1~R4, D1~D4)", "rag": "X — 문항별 원문 인용 대조(RAG 아님)",
     "tools": "코드 규칙, 인용 검증", "state_keys": "scorecard, decision, evaluations, end_reason"},
    {"name": "📝 보고서 생성", "node": "report", "guide": "📝 보고서 생성",
     "role": "**단계별 결과를 장마다 연결**해 5쪽 이내 PDF를 만들고 형식을 검사. 새로 조사하지 않음",
     "criterion": "—", "rag": "X", "tools": "템플릿·PDF 렌더러·REFERENCE 포맷터", "state_keys": "report"},
]

# ── 도구 ────────────────────────────────────────────────────────────────────
DESIGN_TOOLS = [
    {"name": "`search_documents(query)`", "purpose": "PDF 자료 기반 정보 추출",
     "kind": "@tool — Agentic RAG 안에서 LLM 이 고름", "user": "기술 요약·시장성 (Agentic RAG)",
     "key": "없음 (로컬 KURE-v1·FAISS·Kiwi BM25)"},
    {"name": "`web_search(query, recent)`", "purpose": "외부 정보 검색",
     "kind": "@tool — Agentic RAG 안에서 LLM 이 고름. 에이전트 코드도 직접 호출", "user": "탐색·창업자·기술·시장성·경쟁사",
     "key": "Serper·Tavily (재현 때는 캐시)"},
    {"name": "`summarize_document(source, focus)`", "purpose": "문서 요약",
     "kind": "@tool 로 정의, 코드가 호출", "user": "기술 요약: 회사 홈페이지·기사 → 핵심 기술·장단점 (홈페이지를 못 읽으면 수집한 기사 본문)",
     "key": "OpenAI (gpt-4.1-mini)"},
    {"name": "상장 목록 대조 (KRX KIND)", "purpose": "스타트업 기준 '비상장'", "kind": "결정적 함수", "user": "탐색(관문 G1)", "key": "없음"},
    {"name": "국민연금 사업장 조회", "purpose": "창업 시점(첫 고용)·인원·탈퇴", "kind": "결정적 함수",
     "user": "탐색(관문 G4)에서 조회, 창업자 평가가 t0·고용 추이에 사용", "key": "없음"},
    {"name": "TIPS 목록 · 투자 기사 피드", "purpose": "발굴 채널", "kind": "결정적 함수", "user": "탐색(발굴)", "key": "없음"},
    {"name": "기사 원문 수집", "purpose": "근거 본문·게시일 확보", "kind": "결정적 함수",
     "user": "탐색(관문 근거 보강), 창업자·기술(기사 본문 발췌)", "key": "없음"},
]

# ── 평가표 매핑 (계약 C5): 문항 → 가이드 평가 포인트 · Bessemer 번호 · 실사 문구 ────
OWNER = {"founder": "founder", "market": "market", "product": "tech", "competition": "competition",
         "traction": "decide", "deal": "decide"}
OWNER_LABEL = {"founder": "👤 창업자 평가", "market": "📊 시장성 평가", "tech": "🗜️ 기술 요약",
               "competition": "🥊 경쟁사 비교", "decide": "🧮 투자 판단", "decision": "🧮 투자 판단"}
RUBRIC_MAP = {
    "F1": ("전문성", [5, 10], "창업자·경영진 이력과 창업 전 경력 증빙"),
    "F2": ("전문성(기술 역량)", [5], "기술 책임자 이력·특허 발명자·논문 목록"),
    "F3": ("실행력", [10], "최근 24개월 마일스톤 증빙"),
    "F4": ("커뮤니케이션(대외 발표 수치의 제3자 확인)", [5], "회사 공개 수치의 원자료"),
    "M1": ("시장 크기", [1], "세부 시장 규모 추정의 기준 연도·출처"),
    "M2": ("성장 가능성", [8], "세부 분야 최근 2년 투자·도입 추이"),
    "M3": ("수요(실제 문제 해결)", [2], "농가 투자 회수기간(payback)·효과 실측"),
    "M4": ("수요(지불 이유·수익 모델)", [3, 7], "가격표·과금 구조·지불 주체"),
    "P1": ("구현 가능성(상용 운영)", [2], "PoC·시범이 아닌 유료 설치·계약 목록"),
    "P2": ("구현 가능성(제3자 실증)", [2], "기관·대학 실증 결과서"),
    "P3": ("독창성", [4], "자체 개발 증빙"),
    "P4": ("구현 가능성(인허가)", [9], "필수 검정·등록 현황"),
    "C1": ("특허", [4], "특허 등록·출원 번호"),
    "C2": ("진입장벽(경쟁사 대비 차별점)", [4], "경쟁 제품 대비 성능·가격 비교"),
    "C3": ("네트워크 효과", [], "데이터 축적·연동 구조"),
    "C4": ("진입장벽(전략 파트너)", [], "전략 파트너 계약서(MOU 제외)"),
    "R1": ("매출", [7], "연도별 매출·매출총이익률"),
    "R2": ("유저수", [3, 6], "유료 고객 수·설치 규모"),
    "R3": ("계약(재계약·확대)", [6], "재계약률·이탈률(churn)"),
    "R4": ("매출(민간)", [7], "보조금 외 민간 매출 비중"),
    "D1": ("Valuation(최근 라운드)", [], "최근 라운드 금액·일자"),
    "D2": ("투자조건(기관투자자)", [], "투자자 명단·조건"),
    "D3": ("Valuation·지분율", [], "밸류에이션·지분율·term sheet"),
    "D4": ("투자조건(자본집약도)", [9], "상업화까지 필요 자금·설비 계획"),
}
# Bessemer 10문 (가이드 원문) → 연결 문항
BESSEMER = [
    (1, "이 시장은 얼마나 큰가?", ["M1"], False),
    (2, "제품이 시장의 실제 문제를 해결하는가?", ["M3", "P1"], False),
    (3, "고객이 실제로 이 제품에 비용을 지불할 이유가 있는가?", ["M4", "R2"], False),
    (4, "경쟁사 보다 뚜렷한 차별성이 있는가?", ["C2", "P3", "C1"], False),
    (5, "창업자와 팀은 이 분야에서 믿을만한가?", ["F1", "F2"], False),
    (6, "초기 고객의 반응은 어떠한가?", ["R2", "R3"], False),
    (7, "수익 모델은 명확한가?", ["M4", "R4"], False),
    (8, "이 스타트업이 성공한다면, 정말 큰 기회가 될까?", ["M2"], False),
    (9, "기술, 운영, 법률적 리스크는 무엇인가?", ["P4", "D4"], False),
    (10, "이 창업자가 다음 10년을 이 분야에 쏟아부을 각오가 있는가?", ["F1", "F3"], True),
]
BESSEMER_NOTE = {8: "M2 (ROI 는 참고치로 함께 표시, C.7)", 9: "P4, D4 + Deal-killer",
                 10: "F1 + F3 — 대리지표: 창업 전 같은 분야 경력 + 창업 후 이어진 마일스톤"}

# 결정 규칙 기본값 (config.yaml 에 decision.threshold 등이 들어오면 그 값을 쓴다)
DECISION_DEFAULT = {"threshold": 1.10, "step": 0.5, "clip": [0.5, 1.5], "min_founder_yes": 1, "info_gap_ratio": 0.6,
                    "reference_min_n": 5, "sensitivity": [1.00, 1.05, 1.10, 1.15, 1.20], "flip_max_items": 3, "dd_max_items": 5}
RAG_DEFAULT = {"max_regenerations": 1, "rag_recursion_limit": 25}
WORKFLOW_DEFAULT = {"recursion_limit": 100}
ROI_DEFAULT = {"fx_krw_per_usd": 1400, "stake_assumption": [0.10, 0.20], "target_multiple": 10}


def _merged(cfg_section, default: dict) -> dict:
    out = dict(default)
    for k, v in (cfg_section or {}).items():
        if k in default:
            out[k] = v
    return out


def _q(label: str) -> str:
    return '"' + label.replace('"', "'") + '"'


def _mid(n: str) -> str:
    return {START: "S", END: "T"}.get(n, n)


def _main_mermaid(cfg) -> str:
    """메인 그래프 그림. app.py 가 README Architecture 이미지를 만들 때도 쓴다 (이름 유지)."""
    lines = ["graph TD"]
    for n, lab in NODE_LABELS.items():
        shape = f"(({_q(lab)}))" if n in (START, END) else f"[{_q(lab)}]"
        lines.append(f"  {_mid(n)}{shape}")
    for a, b, lab in MAIN_EDGES:
        arrow = f"-->|{_q(lab)}|" if lab else "-->"
        lines.append(f"  {_mid(a)} {arrow} {_mid(b)}")
    return "\n".join(lines)


def _discovery_mermaid(cfg) -> str:
    rounds = cfg.workflow.max_discovery_rounds
    return f"""graph LR
  I(["탐색 진입"]) --> Q{{"대기열에 적격 후보가 있나"}}
  Q -->|있음| P["다음 후보 선택<br/>1곳 꺼냄 · 후보별 결과 키 비움"]
  Q -->|"없음 · 발굴 {rounds}라운드 미만"| C["발굴<br/>채널 8개 · 교차 신호"]
  Q -->|"없음 · 라운드 소진"| X["후보 소진<br/>current 없음 · 종료 사유 기록"]
  C --> V["적격성 관문<br/>G1~G6 · 확인 못 하면 탈락"]
  V --> Q
  P --> O(["평가 대상 1곳 → 창업자 평가"])
  X --> Z(["보고서 생성으로"])"""


def _rag_mermaid(cfg) -> str:
    r = cfg.rag
    rg = _merged(r, RAG_DEFAULT)
    w = r.ensemble_weights
    return f"""graph TD
  Q(["질문 · 목적"]) --> AG["agent: LLM 이 문서 검색·웹 검색 중 고름<br/>수치·정책·기술 기준선은 문서 우선"]
  AG -->|문서 검색| RT["retrieve<br/>BM25 {w[0]} + KURE-v1 {w[1]} · 후보 {r.candidate_k}"]
  AG -->|웹 검색| WB["web<br/>웹 검색 · 최근 1년 우선"]
  RT --> GR["grade<br/>조각별 관련 O/X"]
  GR -->|"관련 {r.min_relevant}개 이상 · 앞 {r.top_k}개"| GN["generate<br/>인용 달린 답변"]
  GR -->|"부족 · 재작성 {r.max_rewrites}회 미만"| RW["rewrite<br/>질의 재작성"]
  GR -->|"부족 · 재작성 소진"| WB
  RW --> RT
  WB --> GN
  GN --> CK["check<br/>수치 대조 + LLM 이진 점검"]
  CK -->|grounded| FN(["finish<br/>답변 · 근거 id · 상태"])
  CK -->|"not_grounded · 재생성 {rg['max_regenerations']}회 미만"| GN
  CK -->|"not_useful · 재작성 {r.max_rewrites}회 미만"| RW
  CK -->|"상한 도달"| FN"""


def _state_rows() -> list[dict]:
    out = []
    for key, typ, red, writer, reader, desc in DESIGN_STATE:
        t = f"{typ} ({red})" if red else f"{typ} (덮어쓰기)"
        if key in RESET_BY_PICK:
            desc += " — 새 후보를 꺼낼 때 비움"
        out.append({"name": key, "type": t.replace("|", "/"), "writer": writer, "reader": reader, "desc": desc})
    return out


def _rubric_view(rubric: dict) -> list[dict]:
    """rubric.yaml 의 문항 text·need 에 평가 포인트·담당·Bessemer 번호를 붙인다 (rubric 에 있으면 그 값 우선)."""
    dims = []
    for d in rubric["dimensions"]:
        owner = d.get("owner") or OWNER[d["id"]]
        qs = []
        for q in d["questions"]:
            point, bq, dd = RUBRIC_MAP.get(q["id"], ("", [], ""))
            qs.append({"id": q["id"], "text": q["text"], "need": q["need"], "point": q.get("point") or point,
                       "bessemer_q": ", ".join(f"Q{n}" for n in (q.get("bessemer_q") or bq)) or "—",
                       "dd_item": q.get("dd_item") or dd})
        dims.append({"id": d["id"], "name": d["name"], "weight": d["weight"], "owner": OWNER_LABEL.get(owner, owner),
                     "questions": qs})
    return dims


def _retrieval_table() -> str:
    f = path("outputs/eval/retrieval_eval.csv")
    if not f.exists():
        return "(eval/eval_retrieval.py 실행 결과 없음)"
    df = pd.read_csv(f).dropna(subset=["MRR@5"])
    piv = (df.groupby(["retriever", "embedding"])[["Hit@1", "Hit@3", "Hit@5", "MRR@5"]].mean().round(3)
           .sort_values("MRR@5", ascending=False).reset_index())
    piv["embedding"] = piv["embedding"].str.replace("dragonkue/", "").str.replace("intfloat/", "")
    best = piv.drop_duplicates("embedding", keep="first")  # 임베딩마다 MRR@5 가 가장 높은 검색 방식 1줄
    return (best.to_markdown(index=False) + f"\n\n임베딩마다 가장 좋은 검색 방식 한 줄씩 보였다(전체 {len(piv)}개 조합은 "
            "`outputs/eval/retrieval_eval.md`).")


def _runtime_table() -> str:
    """실제 파이프라인 설정(후보 8개, 앞 4개 사용)으로 잰 검색기 비교 (eval/eval_final_retriever.py 결과)."""
    d = _json("outputs/eval/runtime_retriever.json")
    rows = {r["name"]: r for r in d.get("rows") or []}
    if not rows:
        return "(eval/eval_final_retriever.py 실행 결과 없음)"
    pick = d.get("recommendation", {}).get("name")
    # '(선택)' 은 지금 config 의 임베딩·가중치와 같은 행 (다시 잰 결과에서 규칙 1위가 달라져도 적용 설정을 빠뜨리지 않게)
    cfg = get_config()
    w = [float(x) for x in cfg.rag.ensemble_weights]
    sel = next((r["name"] for r in rows.values() if r.get("embedding") == cfg.embedding.model
                and [float(x) for x in r.get("weights[bm25,dense]", [])] == w), pick)
    show = ["final (hybrid)", "dense only", "hybrid 0.5:0.5", "KURE-v1 dense", "KURE-v1 hybrid", sel, pick,
            "KURE-v1 hybrid 0.5:0.5", "Kiwi BM25 only"]
    label = {"final (hybrid)": "snowflake + 하이브리드 0.3:0.7 (처음 선택)", "dense only": "snowflake Dense",
             "hybrid 0.5:0.5": "snowflake + 하이브리드 0.5:0.5", "KURE-v1 dense": "KURE-v1 Dense",
             "KURE-v1 hybrid": "KURE-v1 + 하이브리드 0.3:0.7", "Kiwi BM25 only": "Kiwi BM25 단독"}
    out = ["| 설정 (후보 8개) | Hit@1 | Hit@3 | **Hit@4** | **MRR@4** | Hit@8 | 한국어 문서 Hit@4 | 영어 문서 Hit@4 |",
           "|---|---|---|---|---|---|---|---|"]
    for name in dict.fromkeys(n for n in show if n in rows):
        a, ko, en = rows[name]["all"], rows[name]["ko"], rows[name]["en"]
        lab = label.get(name, name.replace("KURE-v1 hybrid", "KURE-v1 + 하이브리드"))
        lab = f"**{lab} (선택)**" if name == sel else f"{lab} (규칙 1위)" if name == pick else lab
        out.append(f"| {lab} | {a['Hit@1']:.3f} | {a['Hit@3']:.3f} | {a['Hit@4']:.3f} | {a['MRR@4']:.3f} | {a['Hit@8']:.3f} | "
                   f"{ko['Hit@4']:.3f} | {en['Hit@4']:.3f} |")
    n = ((d.get("config") or {}).get("questions") or {}).get("all", 70)  # 표제 지표 = 기존 문항 (추가 문항은 따로)
    return "\n".join(out) + f"\n\n질문 {n}개 기준이며 1문항 = {1 / n:.3f}. 한국어/영어는 정답 문서의 언어다(질문은 모두 한국어)."


def _json(rel: str):
    f = path(rel)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _read(rel: str, strip_title: bool = True) -> str:
    f = path(rel)
    if not f.exists():
        return ""
    t = f.read_text(encoding="utf-8")
    return re.sub(r"^# .*\n", "", t) if strip_title else t


def _judge_line() -> str:
    j = (_json("outputs/eval/judge_eval.json") or {}).get("summary")
    if not j:
        return "(미실행)"
    return (f"Relevance {j['relevance']:.2f} · Faithfulness {j['faithfulness']:.2f} · Correctness {j['correctness']:.2f} "
            f"(질문 {j.get('n', 20)}개)")


def _elig(rel: str) -> str:
    j = (_json(rel) or {}).get("summary")
    return f"정확도 {j['accuracy']:.2f}" if j else "(미실행)"


_LIST = re.compile(r"^\s*(?:[-*] |\d+\. )")


def _blank_before_lists(md: str) -> str:
    """문단 바로 다음 줄에 목록이 오면 빈 줄을 넣는다 (Python-Markdown 은 빈 줄 없는 목록을 문단으로 합친다)."""
    out: list[str] = []
    for line in md.split("\n"):
        prev = out[-1] if out else ""
        if _LIST.match(line) and prev.strip() and not _LIST.match(prev) and not prev.startswith((" ", "|", ">", "#")):
            out.append("")
        out.append(line)
    return "\n".join(out)


def context() -> dict:
    """설계서 md(GitHub 용)와 HTML(제출 PDF 용)이 같이 쓰는 값."""
    cfg = get_config()
    team = cfg.submission
    members = sorted(team.members)
    with open(ROOT / "rubric.yaml", encoding="utf-8") as f:
        rubric = yaml.safe_load(f)
    dec = _merged(cfg.get("decision"), DECISION_DEFAULT)
    return dict(
        team=team, members=members, members_line=" · ".join(members), today=datetime.now().strftime("%Y-%m-%d"), cfg=cfg,
        segments=cfg.domain.segments, corpus=load_manifest(), total_pages=total_pages(), rubric=rubric,
        dims=_rubric_view(rubric), bessemer=BESSEMER, bessemer_note=BESSEMER_NOTE,
        agents=AGENTS, tools=DESIGN_TOOLS, state_rows=_state_rows(), n_state=len(DESIGN_STATE),
        dec=dec, rag2=_merged(cfg.rag, RAG_DEFAULT), wf2=_merged(cfg.workflow, WORKFLOW_DEFAULT),
        roi=_merged(cfg.get("roi"), ROI_DEFAULT),
        retrieval_table=_retrieval_table(), runtime_table=_runtime_table(),
        retrieval_decision=_read("outputs/eval/retrieval_decision.md").replace("## ", "#### "),
        eligibility_history=_read("outputs/eval/eligibility_eval_history.md"),
        judge_line=_judge_line(), elig_gold=_elig("outputs/eval/eligibility_eval_gold.json"),
        elig_holdout=_elig("outputs/eval/eligibility_eval_holdout.json"),
        main_mermaid=_main_mermaid(cfg), discovery_mermaid=_discovery_mermaid(cfg), rag_mermaid=_rag_mermaid(cfg),
        main_edges=MAIN_EDGES, node_labels=NODE_LABELS, rubric_map=RUBRIC_MAP, owner_label=OWNER_LABEL,
        calib=_calibration(), run=_run_result(), rt=_runtime_numbers(cfg))


def _calibration() -> dict:
    """v2 보정 실행 결과(data/reference_class.json): 구성원마다 자기 제외 동종 평균으로 다시 계산한 배수·결정.
    파일이 없으면 빈 값 (설계서는 규칙만 싣는다)."""
    from agents.decision import _founder_yes, _killers, _reference, _rows_from_signals, decide_rule, payne_multiplier
    from core.judge import load_rubric

    ref, sens = _json("data/reference_class.json"), _json("outputs/eval/threshold_sensitivity.json")
    if not ref:
        return {"rows": [], "sens": {}, "n": 0, "run_date": None}
    cfg, rubric = get_config(), load_rubric()
    d = cfg.decision
    rows = []
    for m in ref["members"]:
        r = _rows_from_signals(m["signals"], rubric)
        mean = _reference(ref, m["name"], rubric, d.reference_min_n)["mean"]
        M, _ = payne_multiplier(r, mean, rubric, d.step, d.clip)
        fy, k = _founder_yes(r), _killers(r, rubric)
        judged = [x for x in r if x["answer"] != "N/A"]
        unk = sum(x["answer"] == "UNKNOWN" for x in judged) / len(judged)
        dec, hold, _ = decide_rule(M, fy, k, unk, cfg)
        rows.append({"name": m["name"], "region": m.get("region"), "stage": m.get("stage"), "M": M, "founder_yes": fy,
                     "yes": sum(x["answer"] == "YES" for x in r), "no": sum(x["answer"] == "NO" for x in r),
                     "unknown_ratio": unk, "decision": dec, "hold_type": hold, "killers": k})
    rows.sort(key=lambda x: -x["M"])
    return {"rows": rows, "sens": (sens or {}).get("invest_count_at", {}), "n": ref.get("n"), "run_date": ref.get("run_date")}


def _run_result() -> dict:
    """제출 실행(outputs/run_log.json) 요약: 결론·대상·배수·보고서 쪽수 (없으면 빈 값)."""
    run = _json("outputs/run_log.json") or {}
    ev = [e for e in run.get("evaluations", []) if e.get("decision") == "투자"]
    rep = run.get("report") or {}
    return {"mode": rep.get("mode"), "target": ev[0]["name"] if ev else None,
            "multiplier": ev[0]["multiplier"] if ev else None, "pages": (rep.get("checks") or {}).get("pages"),
            "summary_ratio": (rep.get("checks") or {}).get("summary_ratio_of_a4"), "end_reason": run.get("end_reason"),
            "evaluated": len(run.get("evaluations", [])),
            "screened": len(run.get("screened", [])), "eligible": sum(1 for x in run.get("screened", []) if x.get("eligible"))}


def _runtime_numbers(cfg) -> tuple[str, str, str]:
    from docs.build_readme import _runtime

    return _runtime(cfg)


EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]\\s?")


def _paper_body(md: str) -> str:
    """design.md → 학회지 양식 본문: 제목·메타 줄은 제목 블록으로 옮기고, 표·그림에 번호 캡션을 붙이고, 이모지를 뺀다."""
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    html = re.sub(r"^\s*<h1.*?</h1>\s*<blockquote>.*?</blockquote>", "", html, count=1, flags=re.S)
    html = re.sub(r"<h2[^>]*>목차</h2>\s*(<ul>.*?</ul>)", r'<nav class="toc">\1</nav>', html, count=1, flags=re.S)
    n = iter(range(1, 100))
    html = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                  lambda m: f'<pre class="mermaid m{next(n)}">{m.group(1)}</pre>', html, flags=re.S)
    html = EMOJI.sub("", html)
    out, last, t_no, f_no, pos = [], "", 0, 0, 0
    for m in re.finditer(r"<h[234][^>]*>(.*?)</h[234]>|<p><strong>([^<]{1,40})</strong></p>\s*(?=<table)|<table>|</pre>", html, flags=re.S):
        tok = m.group(0)
        if tok.startswith("<h"):
            last = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            continue
        if tok.startswith("<p><strong>"):  # 표 바로 위의 굵은 한 줄 = 그 표의 제목
            out.append(html[pos:m.start()])
            pos = m.end()
            last = m.group(2).strip()
            continue
        if tok == "<table>":
            t_no += 1
            out.append(html[pos:m.start()] + f'<div class="cap"><b>표 {t_no}.</b>{last}</div><table>')
        else:  # 그림(mermaid) 뒤 캡션
            f_no += 1
            out.append(html[pos:m.end()] + f'<div class="figcap"><b>그림 {f_no}.</b>{last}</div>')
        pos = m.end()
    out.append(html[pos:])
    return "".join(out)


def _html_env() -> Environment:
    from markupsafe import Markup

    env = Environment(loader=FileSystemLoader(ROOT / "docs"))
    # 표·목록이 든 markdown 조각(검색기 실측표 등)을 HTML 로
    env.filters["md"] = lambda s: Markup(markdown.markdown(s or "", extensions=["tables", "sane_lists"])
                                         .replace("<table>", '<table class="t compact">'))
    # **굵게**·`코드` 가 든 한 줄 설명(에이전트 역할 등)을 HTML 로
    env.filters["inline"] = lambda s: Markup(re.sub(r"^<p>|</p>$", "", markdown.markdown(s or "").strip()))
    return env


def build() -> tuple[str, str]:
    ctx = context()
    team, members = ctx["team"], ctx["members"]
    md = Environment(loader=FileSystemLoader(ROOT / "docs")).get_template("design.md.j2").render(**ctx)
    md = _blank_before_lists(md)
    md_path = path("docs/design.md")
    md_path.write_text(md, encoding="utf-8")

    from markupsafe import Markup

    html = _html_env().get_template("design_html/paper.html.j2").render(**ctx, body=Markup(_paper_body(md)))
    pdf = path(f"docs/RAG-Design_{team.campus}-{team['class']}_{'+'.join(members)}.pdf")
    _to_pdf(html, pdf)
    return str(md_path), str(pdf)


def _to_pdf(html: str, pdf) -> None:
    from playwright.sync_api import sync_playwright

    tmp = pdf.with_suffix(".html")
    tmp.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.resolve().as_uri(), wait_until="networkidle")
        pg.wait_for_function("() => document.querySelectorAll('pre.mermaid svg').length === "
                             "document.querySelectorAll('pre.mermaid').length", timeout=30000)
        bad = pg.evaluate("() => [...document.querySelectorAll('pre.mermaid')]"
                          ".filter(e => /Syntax error|Parse error/i.test(e.textContent)).length")
        if bad:
            b.close()
            raise RuntimeError(f"mermaid 그림 {bad}개가 렌더링되지 않았습니다 (문법 오류)")
        pg.pdf(path=str(pdf), format="A4", print_background=True, display_header_footer=True,
               header_template='<div style="font-family:Pretendard,sans-serif;font-size:7px;width:100%;padding:0 20mm;'
                               'display:flex;justify-content:space-between;color:#777">'
                               '<span>AgTech AI 스타트업 투자 평가 에이전트 — 설계 산출물</span>'
                               '<span>SKALA 울산캠퍼스 2반 1조</span></div>',
               footer_template='<div style="font-family:Pretendard,sans-serif;font-size:7.5px;width:100%;text-align:center;color:#555">'
                               '— <span class="pageNumber"></span> —</div>',
               margin={"top": "18mm", "bottom": "17mm", "left": "20mm", "right": "20mm"})
        b.close()


HTML = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pretendard@1.3.9/dist/web/static/pretendard.css">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
 body{font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-size:9pt;line-height:1.45;color:#1b2330}
 h1{font-size:18pt;color:#12213f;border-bottom:3px solid #12213f;padding-bottom:6px;margin-top:0}
 h2{font-size:13.5pt;color:#12213f;border-bottom:1.5px solid #d8dee9;padding-bottom:3px;margin-top:16px;break-after:avoid}
 h3{font-size:11pt;color:#2f5bd3;margin:12px 0 4px;break-after:avoid}
 h4{font-size:10pt;margin:10px 0 4px}
 p,ul,ol{margin:3px 0} ul,ol{padding-left:18px} li{margin:1px 0}
 table{width:100%;border-collapse:collapse;font-size:7.9pt;margin:4px 0 8px}
 th,td{border:0.6pt solid #cfd6e2;padding:2px 4px;vertical-align:top;text-align:left}
 th{background:#eef2f8} tr{break-inside:avoid}
 code{font-family:Menlo,monospace;font-size:8.1pt;background:#f3f5f9;padding:0 3px;border-radius:3px}
 blockquote{border-left:4px solid #2f5bd3;background:#f5f7fb;margin:8px 0;padding:6px 12px;color:#344054}
 pre.mermaid{background:#fff;text-align:center;break-inside:avoid;page-break-inside:avoid;margin:6px 0}
 pre.mermaid svg{max-height:560px;max-width:100%}
 pre.m1 svg{max-height:540px} pre.m2 svg{max-height:230px} pre.m3 svg{max-height:500px}
 a{color:#2f5bd3;text-decoration:none}
</style></head><body>{{BODY}}
<script>mermaid.initialize({startOnLoad:true,theme:"default",themeVariables:{fontSize:"15px"},flowchart:{htmlLabels:true,curve:"basis",nodeSpacing:28,rankSpacing:30,padding:6,wrappingWidth:260}});</script>
</body></html>"""


if __name__ == "__main__":
    print(build())
