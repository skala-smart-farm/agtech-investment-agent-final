"""👤 창업자 평가 에이전트 (가이드 에이전트 정의(안)에 추가한 에이전트).

Scorecard 에서 비중이 가장 큰 '창업자 (Owner) 30%' 를 혼자 맡는다 (1 에이전트 = 1 기준, 문항 F1~F4).
창업 시점(t0)을 먼저 코드로 고정하고, 창업 **전** 전문성(F1·F2)과 창업 **후** 실행력(F3)을 나눠 본다.
- t0 우선순위: TIPS 공개 목록 설립일(estDt) → 국민연금 사업장 최초 가입일(첫 고용) → 적격성 검증 단계의 설립연도
  (같은 연도의 설립 표현이 근거에 있으면 그 근거 id 를 붙인다. 보고서 안에서 설립연도가 둘로 갈리지 않게 한다)
  → 근거 속 가장 이른 설립 표현(근거 id) → 확인 불가. 어느 값을 썼는지 t0_source 에 남긴다.
- 창업자는 인물 검색이 아니라 "회사가 알려진 기사·인터뷰 속 창업자 이력"으로 확인한다. 검색 쿼리 문자열은 v1 그대로다.
- 검색 스니펫에는 창업자 이름·이력이 잘 안 나온다(예: CTO 이름이 기사 본문 1,300자 뒤에만 있음).
  그래서 기사 본문에서 회사명·창업자 표현(대표, CTO, 창업 …) 주변 문단을 골라 스니펫과 함께 LLM 에 넘긴다.
- 근거에 이름이 없는 인물, 날짜·근거 id 가 없는 마일스톤은 코드가 버린다. 인원(headcount)은 국민연금 가입자 수다.
- RAG 미사용: 사람에 관한 사실은 PDF 코퍼스에 없다. 홈페이지 요약도 하지 않는다(기술 요약 에이전트만 한다).
"""
from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field

from agents.tech import CITE_ID, candidate_line, evidence_blocks, judge_own
from core.config import get_segment, run_date
from core.llm_bounded import bounded
from core.prompts import render
from tools.fetch import enrich
from tools.grounding import norm
from tools.sources import SourceRegistry
from tools.web_search import web_search

AGENT = "founder"

# 창업자·핵심 인력이 나오는 문단을 찾는 표현 ("대표적"은 제외)
FOUNDER_TERMS = re.compile(r"대표(?!적)|CEO|CTO|공동\s?창업|창업자|창업|[Cc]o-?[Ff]ounder|[Ff]ounder|기술이사|연구소장")
_TITLE = r"(?:대표이사|대표(?!적)|CEO|CTO|COO|공동창업자|창업자|기술이사|연구소장)"
# 사람 이름(한글 3자) + 직함: "이규화 대표", "이규화(28) 메타파머스 대표", "(대표 이규화)", "대표자는 이원준입니다",
# "윤원재 CTO", "Jane Doe, CEO". 회사명이 사이에 끼는 "이원준 조벡스 대표"는 _mentions 가 회사명으로 따로 찾는다
PERSON = re.compile(
    rf"(?<![가-힣])(?P<ko>[가-힣]{{3}})(?:\(\d{{2}}\) (?:[가-힣A-Za-z]{{2,12}} )?| ){_TITLE}"
    rf"|(?:대표이사|대표|CEO|CTO) (?P<ko2>[가-힣]{{3}})(?=[),])"
    r"|대표자는 (?P<ko3>[가-힣]{3})(?=입니다|[\s.,)])"
    r"|(?P<en>[A-Z][a-z]+ [A-Z][a-z]+),? (?:the )?(?:CEO|CTO|[Cc]o-?[Ff]ounder|[Ff]ounder)"
    r"|(?:CEO|CTO|[Cc]o-?[Ff]ounder|[Ff]ounder)(?: and CEO)?,? (?P<en2>[A-Z][a-z]+ [A-Z][a-z]+)")
# 직함 앞에 오지만 사람 이름이 아닌 말
NOT_NAME = {"비롯한", "투자사", "관계자", "운영사", "스타트", "창업주", "공동의", "신임의"}
# 창업자 이력 문단에 자주 나오는 말 (학력·경력) — 이름만 나열된 문단보다 이력 문단을 먼저 고르게 한다
BACKGROUND = re.compile(r"대학|학과|학부|박사|석사|전공|출신|경력|경험|근무|졸업|University|PhD|former", re.I)

# 근거 속 설립 표현: "2022년 9월 설립", "설립일: 2020-04-17", "founded in 2019", "Founded: 2019"
# ("2023년 창업도약패키지 선정", "창업자", "창업기업" 처럼 설립 사실이 아닌 말은 뺀다)
_MONTHS = {m: i for i, m in enumerate(("january february march april may june july august september october "
                                       "november december").split(), 1)}
_NOT_FOUNDING = r"(?!\s?예정|[가-힣]*(?:지원|사업|패키지|대회|경진|보육|교육|프로그램|센터|펀드|기업|도약|성장|진흥|자))"
FOUNDED = [
    re.compile(r"(?P<y>(?:19|20)\d{2})\s?년(?:\s?(?P<m>\d{1,2})\s?월)?(?:\s?\d{1,2}\s?일)?(?:\s?에)?\s?(?:설립|창업|창립)"
               + _NOT_FOUNDING),
    re.compile(r"(?:설립|창업|창립)(?:일자?|연도|년도|연월일)?\s*[:：]?\s*(?P<y>(?:19|20)\d{2})(?:\s?[.\-/년]\s?(?P<m>\d{1,2}))?"),
    re.compile(r"(?P<y>(?:19|20)\d{2})\.(?P<m>\d{2})\s?설립"),                      # 기업 DB 표기 "2009.08설립"
    re.compile(r"(?:[Ff]ounded|[Ee]stablished)(?:\s?[:：])?\s(?:in\s)?(?:(?P<mn>[A-Z][a-z]+)\s)?(?P<y>(?:19|20)\d{2})"),
    # "founded Upside Robotics in 2024" (회사명이 사이에 낀 표현, 이름 3단어까지)
    re.compile(r"[Ff]ounded\s(?:[A-Z][\w&.\-]*\s){1,3}in\s(?:(?P<mn>[A-Z][a-z]+)\s)?(?P<y>(?:19|20)\d{2})"),
]
YEAR = re.compile(r"(?<!\d)((?:19[5-9]|20[0-4])\d)(?!\d)(?!\s?년생|생)")  # 이력 속 연도 (출생 연도 제외)
YM = re.compile(r"^((?:19|20)\d{2})-(\d{2})")


def founder_queries(c: dict) -> list[tuple[str, bool]]:
    """창업자·마일스톤 검색 쿼리 (쿼리, deep). v1 tech_node 쿼리 중 창업자·수상용이며 문자열은 v1 그대로다."""
    name = c["official_name"]
    if c["region"] == "KR":
        return [(f"{name} 대표 창업자 이력 인터뷰", True), (f"{name} CTO 연구소장 기술 개발", False),
                (f"{name} 수상 선정 혁신상 우수기업 출시", False)]
    q = c.get("name_en") or name
    return [(f"{q} founder CEO background interview", True), (f"{q} CTO technology team", False)]


def _mentions(text: str, companies: list[str] | None = None) -> list[re.Match]:
    """이름+직함 표현. companies 가 있으면 "이원준 조벡스 대표", "조벡스 대표 이원준" 형태도 찾고, 회사명 자체(3자)는 뺀다."""
    companies = [c for c in companies or [] if c]
    pats = [PERSON] + [re.compile(rf"(?<![가-힣])(?P<ko>[가-힣]{{3}}) {re.escape(c)} {_TITLE}"
                                  rf"|{re.escape(c)} {_TITLE} (?P<ko2>[가-힣]{{3}})(?=(?:는|은|이|가|의)?(?![가-힣]))")
                       for c in companies]
    own = {norm(c) for c in companies}
    out = []
    for pat in pats:
        for m in pat.finditer(text or ""):
            who = next(v for v in m.groupdict().values() if v)
            if who not in NOT_NAME and norm(who) not in own:
                out.append(m)
    return sorted(out, key=lambda m: m.start())


def person_mentions(text: str, companies: list[str] | None = None) -> list[str]:
    """사람 이름과 직함이 붙은 표현 (창업자 누락 재시도 판단용)."""
    return list(dict.fromkeys(m.group(0) for m in _mentions(text, companies)))


def _leader_hints(blocks: dict[str, str], keys: list[str], ceo: str | None, limit: int = 8) -> list[str]:
    """회사명이 나오는 근거 안의 '이름 + 직함' 표현. 프로필 대표자 → 회사명 바로 옆(±80자) → 나머지 순으로 고른다
    (같은 기사에 나온 투자사 대표 같은 다른 회사 사람이 앞에 오지 않게)."""
    nk = [norm(k) for k in keys if k and len(norm(k)) >= 2]
    ck = norm(ceo or "")
    ranked = []
    for sid, text in blocks.items():
        if not any(k in norm(text) for k in nk):
            continue
        found = _mentions(text, keys)
        for m in found:
            near = any(k in norm(text[max(0, m.start() - 80): m.end() + 80]) for k in nk)
            rank = 0 if len(ck) >= 2 and ck in norm(m.group(0)) else (1 if near else 2)
            ranked.append((rank, f"'{m.group(0)}' [{sid}]"))
        if len(ck) >= 2 and ck in norm(text) and not any(ck in norm(m.group(0)) for m in found):
            ranked.append((0, f"'{ceo}'(프로필 대표자) [{sid}]"))
    return list(dict.fromkeys(h for _, h in sorted(ranked, key=lambda x: x[0])))[:limit]


def _ground_founders(people: list, blocks: dict[str, str]) -> list[dict]:
    """근거 텍스트에 이름이 실제로 있는 인물만 남기고, evidence_ids 는 그 이름이 나오는 근거로 맞춘다 (지어낸 인물 차단)."""
    out = []
    for f in people:
        key = norm(re.split(r"[(/]", f.name)[0])
        found = [sid for sid, t in blocks.items() if len(key) >= 2 and key in norm(t)]
        if not found:
            continue
        ev = [i for i in f.evidence_ids if i in found] or found[:3]
        out.append({**f.model_dump(), "evidence_ids": ev})
    return out


def _to_date(s: str | None) -> date | None:
    """'YYYY-MM-DD' / 'YYYY-MM' / 'YYYY' → date. 월·일이 없으면 1일·1월로 본다 (경과 연수를 가장 길게 잡는 쪽)."""
    m = re.match(r"^((?:19|20)\d{2})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", s or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))
    except ValueError:
        return None


def founding_mentions(reg: SourceRegistry, ids: list[str], keys: list[str], until: str) -> list[dict]:
    """근거 본문 속 설립 표현 중 회사명이 바로 옆(앞 80자·뒤 40자)에 있는 것 → [{date, evidence_id, quote}], 이른 날짜 순.
    until(평가 기준일)보다 뒤의 연도는 버린다."""
    nk = [norm(k) for k in keys if k and len(norm(k)) >= 2]
    out = []
    for sid in dict.fromkeys(ids):
        text = re.sub(r"\s+", " ", reg.text(sid))
        for pat in FOUNDED:
            for m in pat.finditer(text):
                if not any(k in norm(text[max(0, m.start() - 80): m.end() + 40]) for k in nk):
                    continue
                y = m.group("y")
                mon = int(m.groupdict().get("m") or 0) or _MONTHS.get((m.groupdict().get("mn") or "").lower(), 0)
                when = f"{y}-{mon:02d}" if 1 <= mon <= 12 else y
                if when[:7] <= until[:7]:
                    out.append({"date": when, "evidence_id": sid, "quote": text[max(0, m.start() - 30): m.end() + 10]})
    return sorted(out, key=lambda x: x["date"])


def pick_t0(c: dict, reg: SourceRegistry, ids: list[str], keys: list[str], until: str) -> dict:
    """창업 시점 t0 를 코드 규칙으로 정한다 → {t0, t0_basis(짧은 이름표), t0_source(설명), t0_evidence_ids}."""
    nps = c.get("nps") or {}
    if c.get("founded_date"):
        return {"t0": c["founded_date"], "t0_basis": "TIPS 설립일", "t0_source": "TIPS 공개 목록 설립일(estDt)",
                "t0_evidence_ids": []}
    if nps.get("status") == "matched" and nps.get("first_date"):
        ev = [nps["evidence_id"]] if nps.get("evidence_id") else []
        return {"t0": nps["first_date"], "t0_basis": "국민연금 최초 가입일", "t0_evidence_ids": ev,
                "t0_source": "국민연금 사업장 최초 가입일(첫 고용 시점, 실제 설립일보다 늦을 수 있음)"
                             + (f" [{ev[0]}]" if ev else "")}
    found = founding_mentions(reg, ids, keys, until)
    if c.get("founded_year"):
        y = str(c["founded_year"])
        same = next((f for f in found if f["date"][:4] == y), None)   # 설립 표현은 근거를 붙이는 데만 쓴다
        if same:
            return {"t0": y, "t0_basis": "적격성 검증 설립연도", "t0_evidence_ids": [same["evidence_id"]],
                    "t0_source": f"적격성 검증 단계의 설립연도, 원문 \"{same['quote'].strip()}\" [{same['evidence_id']}]"}
        return {"t0": y, "t0_basis": "적격성 검증 설립연도", "t0_evidence_ids": [],
                "t0_source": "적격성 검증 단계에서 LLM 이 근거에서 읽은 설립연도 (원문 인용 확인 안 됨)"}
    if found:
        f = found[0]
        return {"t0": f["date"], "t0_basis": "근거 속 설립 표현", "t0_evidence_ids": [f["evidence_id"]],
                "t0_source": f"근거 속 가장 이른 설립 표현 \"{f['quote'].strip()}\" [{f['evidence_id']}]"}
    return {"t0": None, "t0_basis": "없음", "t0_source": "확인 불가", "t0_evidence_ids": []}


def _before_t0(background: str, llm_value: bool | None, t0: date | None) -> bool | None:
    """창업 전 경력 여부. t0 를 모르면 None. 이력에 t0 연도보다 이른 연도가 있으면 True, 없으면 LLM 판단을 따른다
    (연도 없이 '창업 전 ○○ 근무'처럼 적힌 이력은 LLM 만 읽을 수 있다)."""
    if t0 is None:
        return None
    years = [int(y) for y in YEAR.findall(CITE_ID.sub("", background or ""))]
    if any(y < t0.year for y in years):
        return True
    return llm_value


def _dated_ids(m: dict, valid: set[str], reg: SourceRegistry) -> list[str]:
    """마일스톤 근거 id 중 저장소에 있고, 본문이나 게시일에 그 연도가 나오는 것."""
    y = m["date"][:4]
    return [i for i in m["evidence_ids"] if i in valid
            and (y in reg.text(i) or ((reg.get(i) or {}).get("date") or "").startswith(y))]


def clean_milestones(items: list, valid: set[str], reg: SourceRegistry, until: str) -> list[dict]:
    """날짜('YYYY-MM')와 그 연도가 확인되는 근거 id 가 있는 마일스톤만 남긴다. 기준일보다 뒤(예정)는 버린다.
    같은 날짜·같은 사건(공백·기호 무시)은 하나로 합친다. 날짜 순."""
    out: dict[tuple[str, str], dict] = {}
    for x in items:
        m = YM.match(x.date.strip())
        if not m or not 1 <= int(m.group(2)) <= 12:
            continue
        row = {"date": f"{m.group(1)}-{m.group(2)}", "what": x.what, "evidence_ids": list(x.evidence_ids)}
        if row["date"] > until[:7]:
            continue
        if ev := _dated_ids(row, valid, reg):
            key = (row["date"], norm(row["what"]))
            if key in out:
                out[key]["evidence_ids"] = list(dict.fromkeys(out[key]["evidence_ids"] + ev))
            else:
                out[key] = {**row, "evidence_ids": ev}
    return sorted(out.values(), key=lambda r: r["date"])


def months_between(ym: str, until: str) -> int:
    """'YYYY-MM' 에서 기준일(YYYY-MM-DD)까지의 개월 수."""
    return (int(until[:4]) - int(ym[:4])) * 12 + int(until[5:7]) - int(ym[5:7])


def _nps_line(n: dict) -> str:
    if n.get("status") != "matched":
        return "국민연금 가입 사업장 목록에서 확인 안 됨 (3인 미만 법인이거나 사명 다름, 해외 법인은 대상 아님)"
    return (f"국민연금 가입자 {n['members']}명({n['ym']}), 최초 가입 {n['first_date']}, "
            f"{'탈퇴' if n['withdrawn'] else '가입 중'}")


class Person(BaseModel):
    name: str = Field(description="근거에 나온 이름 그대로")
    role: str = Field(description="대표, 공동창업자, CTO, 기술이사, 연구소장 등")
    background: str = Field(description="학력·경력·이전 창업을 연도와 함께 (예: '2015~2020 농촌진흥청 연구원 [W1a2b3]'), "
                                        "근거에 없으면 '확인 불가'")
    before_t0: bool | None = Field(description="창업 시점(t0) 이전의 학력·경력이 근거에 있으면 true, "
                                               "창업 후 활동만 있으면 false, 판단할 수 없으면 null")
    evidence_ids: list[str] = Field(description="이 인물 이름이 나오는 근거 id")


class Milestone(BaseModel):
    date: str = Field(description="사건 연-월 'YYYY-MM'. 본문에 사건 날짜가 없으면 그 근거의 게시일 연-월")
    what: str = Field(description="이미 일어난 사건 한 줄 (투자 유치·수상·정부 과제·TIPS 선정·출시·설치·인증·수출)")
    evidence_ids: list[str] = Field(description="이 사건이 나오는 근거 id")


class FounderAnalysis(BaseModel):
    people: list[Person] = Field(description="근거에 이름이 나온 이 회사의 대표·공동창업자·기술 책임자")
    milestones: list[Milestone] = Field(description="회사의 날짜 있는 마일스톤 (예정·계획 제외)")
    team_assessment: str = Field(description="창업 전 전문성과 창업 후 실행력 평가 2~3문장 (근거 id 인용)")
    evidence_ids: list[str] = Field(description="실제로 인용한 근거 id 전체")


def _analysis_text(c: dict, f: dict) -> str:
    """창업자 기준 판정에 넘기는 요약문."""
    tag = {True: "창업 전 이력 있음", False: "창업 후 활동만 확인", None: "창업 전후 판단 불가"}
    people = "; ".join(f"{p['name']}({p['role']}, {tag[p['before_t0']]}): {p['background']}" for p in f["people"])
    stones = "; ".join(f"{m['date']} {m['what']} [{', '.join(m['evidence_ids'])}]" for m in f["milestones"])
    years = f", 기준일까지 {f['years_since_t0']}년" if f["years_since_t0"] is not None else ""
    return "\n".join([
        candidate_line(c),
        f"[창업 시점 t0] {f['t0'] or '확인 불가'} — {f['t0_source']}{years}",
        f"[팀] {f['team_assessment']}",
        f"[창업자·핵심 인력] {people or '확인 불가'}",
        f"[마일스톤] {stones or '없음'} (최근 24개월 {f['milestones_24m']}건)",
        f"[고용] {_nps_line(c.get('nps') or {})}",
    ])


def founder_node(state: dict) -> dict:
    """state['current'] 후보의 창업자·팀을 조사하고 창업자 기준(F1~F4)을 판정한다.

    반환 키: founder, registry(근거 저장소 전체 reg.data), log
    founder = {
        't0': str|None, 't0_basis': str, 't0_source': str, 't0_evidence_ids': [...],   # 창업 시점과 그 근거
        'years_since_t0': float|None,                                           # t0 → 평가 기준일 경과 연수
        'people': [{'name','role','background','before_t0': bool|None,'evidence_ids'}],
        'milestones': [{'date': 'YYYY-MM','what','evidence_ids'}], 'milestones_24m': int,
        'headcount': int|None, 'hires_per_year': float|None, 'team_assessment': str,
        'evidence_ids': [...], 'pool_ids': [...],                               # pool_ids: 판정에 쓰는 근거 id
        'criterion': DimResult,                                                 # core.judge.judge_dimension('founder', …)
    }
    """
    c = state["current"]
    reg = SourceRegistry(state.get("registry"))
    name = c["official_name"]
    until = state.get("run_date") or run_date()
    ids = list(c.get("evidence_ids", []))
    for query, deep in founder_queries(c):
        ids += web_search(query, reg, AGENT, topic="news", recent=False, deep=deep, raw=True)
    # 인터뷰·기사 원문을 받아 창업자 이력처럼 스니펫에 없는 사실을 보강 (키 불필요)
    enrich(reg, ids, [norm(name), norm(c.get("name_en") or "")], limit=8)
    ids = list(dict.fromkeys(ids))
    names = [name, c.get("name_en") or ""]

    t0 = pick_t0(c, reg, ids, names, until)
    t0_date = _to_date(t0["t0"])
    years = round((_to_date(until) - t0_date).days / 365.25, 1) if t0_date and _to_date(until) else None

    # 스니펫 + 본문 속 회사명·창업자 문단 (창업자 이름·이력은 대개 본문에만 있다)
    blocks = evidence_blocks(reg, ids, names, terms=FOUNDER_TERMS, boost=BACKGROUND, person=PERSON)
    ceo = c.get("ceo")
    nps = c.get("nps") or {}
    ctx = dict(name=name, one_line=c.get("one_line", ""), segment=get_segment(c["segment_id"])["name"],
               ceo=ceo or "미확인", t0=t0["t0"] or "확인 불가", t0_source=t0["t0_source"], run_date=until,
               nps=_nps_line(nps), evidence="\n\n".join(blocks.values()))
    llm = bounded(FounderAnalysis)
    res: FounderAnalysis = llm.invoke(render("founder", **ctx, feedback=""))
    hints = _leader_hints(blocks, names, ceo)
    retried = False
    # 근거에 '이름 + 대표' 같은 표현이 있는데 (근거로 확인되는) 인물이 비었으면 한 번만 다시 묻는다
    if not _ground_founders(res.people, blocks) and hints:
        retried = True
        feedback = ("직전 답의 people 이 비어 있다. 근거에 다음 인물 표현이 있다: " + "; ".join(hints)
                    + "\n근거 본문에서 이 회사의 대표·공동창업자·기술 책임자인지 확인해 people 을 채워라. "
                      "사람 이름이 아니거나 다른 회사 사람이면 넣지 마라.")
        res = llm.invoke(render("founder", **ctx, feedback=feedback))

    valid = set(ids)
    people = _ground_founders(res.people, blocks)
    for p in people:
        p["before_t0"] = _before_t0(p["background"], p["before_t0"], t0_date)
    ck = norm(ceo or "")
    if not people and len(ck) >= 2:  # 그래도 비었으면 공공 데이터(TIPS)의 대표자를 근거와 함께 넣는다
        found = [sid for sid, t in blocks.items() if ck in norm(t)]
        if found:
            people = [{"name": ceo, "role": "대표", "background": "확인 불가 (TIPS 공개 목록의 대표자)",
                       "before_t0": None, "evidence_ids": found[:3]}]
    milestones = clean_milestones(res.milestones, valid, reg, until)
    headcount = nps.get("members") if nps.get("status") == "matched" else None
    out = {
        **t0, "years_since_t0": years, "people": people, "milestones": milestones,
        "milestones_24m": sum(0 <= months_between(m["date"], until) <= 24 for m in milestones),
        "headcount": headcount,
        "hires_per_year": round(headcount / max(years, 0.5), 1) if headcount is not None and years is not None else None,
        "team_assessment": res.team_assessment,
    }
    refs = (list(res.evidence_ids) + t0["t0_evidence_ids"] + [i for p in people for i in p["evidence_ids"]]
            + [i for m in milestones for i in m["evidence_ids"]] + CITE_ID.findall(res.team_assessment))
    out["evidence_ids"] = [i for i in dict.fromkeys(refs) if i in valid]
    out["pool_ids"] = ids
    out["criterion"] = judge_own("founder", state, reg, ids, _analysis_text(c, out))
    crit = out["criterion"]
    who = ", ".join(f"{p['name']}({p['role']})" for p in people) or "없음"
    msg = (f"[창업자] {name}: t0 {out['t0'] or '확인 불가'}({t0['t0_basis']}), 인물 {len(people)}명({who})"
           f"{' — 재시도' if retried else ''}, 마일스톤 {len(milestones)}건(최근 24개월 {out['milestones_24m']}건), "
           f"근거 {len(out['evidence_ids'])}건 → 창업자 YES {crit['yes']}/{crit['n']}")
    print(msg)
    return {"registry": reg.data, "founder": out, "log": [msg]}
