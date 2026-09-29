"""🥊 경쟁사 비교 에이전트 (Scorecard '경쟁 우위 10%', 문항 C1~C4).

기술 요약 결과(제품·핵심 기술·회사 측 차별점 주장 claims)를 받아, 국내·해외 경쟁사와 실제로 비교해 차별성과 진입장벽을 검증한다.
- 경쟁사는 분야 이름("애그테크")이 아니라 대상의 **제품 유형**(예: 온실 과채류 수확 로봇)으로 찾는다.
  LLM 이 제품 설명에서 제품 유형과 검색 질의를 정하고(검색 계획), 에이전트가 그 질의로 검색한다.
- 대상 회사 기사 본문에 경쟁사로 직접 언급된 회사(예: "영국 더그투스 등 … 외국 경쟁사")는 코드로 뽑아 반드시 검토한다.
- 경쟁사 행마다 근거 id 를 달고, 이름이 근거에 없는 경쟁사는 버린다. 대상의 우위는 단정하지 않고 "회사 측 주장"으로 쓴다.
- 기술 요약의 차별점 주장(claims)마다 '제3자 확인'·'회사 주장'·'반대 근거'를 붙인다(verified_claims).
  근거 id 가 없거나 회사 자체 사이트 근거뿐인 '제3자 확인', 근거 id 가 없는 '반대 근거'는 코드가 '회사 주장'으로 낮춘다.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from agents.tech import candidate_line, evidence_blocks, judge_own
from core.config import get_segment
from core.llm import structured
from core.llm_bounded import bounded
from core.prompts import render
from tools.grounding import norm
from tools.sources import SourceRegistry
from tools.web_search import web_search

AGENT = "competition"
MAX_QUERIES = 4  # 검색 계획 질의 상한 (+ "{회사} 경쟁사" 1건)

RIVAL_TERMS = re.compile(r"경쟁|competitor|rival|대비|비교|점유율|market share", re.I)  # 본문 문단 고르기용
RIVAL = re.compile(r"경쟁\s?(?:사|업체|기업)|competitors?|rivals?", re.I)                  # 경쟁사를 말하는 문장
_COUNTRY = "영국|미국|네덜란드|이스라엘|일본|중국|독일|프랑스|호주|스페인|벨기에|캐나다|이탈리아|덴마크|노르웨이|스웨덴|대만|싱가포르|인도"
_NAME = r"[가-힣A-Za-z][가-힣A-Za-z0-9&\-]{1,20}"
# 경쟁 문장 안의 회사 이름: "영국 더그투스", "더그투스 등 … 경쟁사", "경쟁사인 ○○", "competitors such as Tevel"
MENTION = [re.compile(rf"(?:{_COUNTRY})의?\s(?P<name>{_NAME})"),
           re.compile(rf"(?<![가-힣A-Za-z])(?P<name>{_NAME})\s?등\s?[^.]{{0,30}}?경쟁\s?(?:사|업체|기업)"),
           re.compile(rf"경쟁\s?(?:사|업체|기업)(?:인|로는|으로는)\s(?P<name>{_NAME}(?:\s?,\s?{_NAME}){{0,4}})"),
           re.compile(r"(?:competitors?|rivals?)(?: such as| like| including)? (?P<name>[A-Z][\w&\-]+(?: [A-Z][\w&\-]+)?"
                      r"(?:(?:, | and )[A-Z][\w&\-]+(?: [A-Z][\w&\-]+)?){0,4})")]
LIST_SEP = re.compile(r"\s?,\s?|\s(?:and|or)\s")
NOT_COMPANY = {"시장", "기업", "업체", "회사", "스타트업", "경쟁사", "경쟁", "제품", "기술", "정부", "농가", "농업", "외국", "해외",
               "국내", "업계", "로봇", "등", "대비", "비슷한", "유사한", "다른", "기존", "현지", "The", "Other", "Many",
               "진출", "수출", "작물", "딸기", "토마토", "오이", "파프리카", "과일", "채소", "서비스", "솔루션", "플랫폼", "대표",
               "한국", "정부", "전시회", "박람회", "CES", "투자", "고객", "농장", "온실", "스마트팜"}
# 이름 끝의 조사 (한글·영문 이름 모두: '테벨과', 'Tevel보다', '시장에')
PARTICLE = re.compile(r"(?<=[가-힣A-Za-z]{2})(?:보다|에서|에게|으로|와|과|의|는|은|를|을|도|로|가|이|에)$")
# 대상이 경쟁사보다 낫다는 단정 (검증 전에는 쓰지 않는다). '회사 측 주장:' 뒤의 말은 출처를 밝힌 것이라 허용.
# '선도 기업'·'앞서 언급한' 같은 경쟁사 사실 서술은 단정이 아니므로 비교 표현만 잡는다
SUPERIOR = re.compile(r"앞선다|앞서 있|앞서는|우위(?:에|를|가)|우월|뛰어나|능가|압도|더 낫|차별화된다|outperform|superior|ahead of",
                      re.I)
CLAIMED = re.compile(r"회사\s?측\s?주장\s?[:：][^/\n.]*")  # 주장은 문장 끝('.')이나 '/' 까지
CITE = re.compile(r"\[[WD][0-9a-f]{5}")


def _plain(s: str | None) -> str:
    """근거 id 표기([W..])를 뺀 문장 (검색 계획 입력용)."""
    return re.sub(r"\[[^\]]*\]", "", s or "").strip()


def _asserts(text: str) -> bool:
    """출처를 밝히지 않은 우열 단정이 있으면 True. '회사 측 주장: …' 부분(그 문장 끝까지)은 빼고 나머지만 본다."""
    return bool(SUPERIOR.search(CLAIMED.sub("", text or "")))


def mentioned_competitors(reg: SourceRegistry, ids: list[str], keys: list[str]) -> list[dict]:
    """대상 회사가 나오는 근거 본문에서 '경쟁' 문장에 이름이 나온 회사 → [{name, evidence_ids, quote}]."""
    nk = [norm(k) for k in keys if k and len(norm(k)) >= 2]
    found: dict[str, dict] = {}
    for sid in dict.fromkeys(ids):
        text = reg.text(sid)
        if not text or not any(k in norm(text) for k in nk):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
            if not RIVAL.search(sent):
                continue
            for pat in MENTION:
                for part in (p for m in pat.finditer(sent) for p in LIST_SEP.split(m.group("name"))):
                    nm = PARTICLE.sub("", part.strip())
                    n_ = norm(nm)
                    if nm in NOT_COMPANY or len(n_) < 2 or any(k in n_ or n_ in k for k in nk):
                        continue
                    row = found.setdefault(n_, {"name": nm, "evidence_ids": [], "quote": sent.strip()[:200]})
                    if sid not in row["evidence_ids"]:
                        row["evidence_ids"].append(sid)
    return list(found.values())


class SearchPlan(BaseModel):
    product_type_ko: str = Field(description="대상 제품 유형 (예: '온실 과채류 수확·수분 로봇')")
    product_type_en: str = Field(description="영문 제품 유형 (예: 'greenhouse fruit harvesting robot')")
    queries: list[str] = Field(description="경쟁사 검색 질의 2~4개 (국내 1개 이상·해외 1개 이상, 언급된 경쟁사 이름 포함)")


class Competitor(BaseModel):
    name: str
    country: str
    offering: str = Field(description="제품·접근 방식 (근거 id)")
    scale: str = Field(description="투자 단계·매출·고객 규모 등 알려진 규모, 모르면 '확인 불가'")
    vs_target: str = Field(description="중립 비교 한 문장: 경쟁사 사실 [근거 id] / 대상은 '회사 측 주장: ~' [근거 id]")
    evidence_ids: list[str] = Field(description="이 경쟁사 이름이 실제로 나오는 근거 id (1개 이상)")


class VerifiedClaim(BaseModel):
    claim: str = Field(description="기술 요약이 정리한 차별점 주장 (문구 그대로, 근거 id 표기는 빼도 됨)")
    status: Literal["제3자 확인", "회사 주장", "반대 근거"] = Field(
        description="제3자 확인: 회사가 아닌 출처(기관·언론 취재·실증 결과·경쟁사 비교)가 확인 / "
                    "회사 주장: 회사 발표·대표 발언뿐 / 반대 근거: 경쟁사가 같은 기능을 이미 제공하는 등 주장과 어긋나는 근거가 있음")
    evidence_ids: list[str] = Field(description="판단에 쓴 근거 id (제3자 확인·반대 근거는 필수)")


class CompetitionAnalysis(BaseModel):
    competitors: list[Competitor] = Field(description="3~5곳")
    differentiation: str = Field(description="대상의 차별성 판단: 제3자 근거로 확인된 것과 회사 측 주장을 구분 (근거 id)")
    entry_barriers: str = Field(description="특허·데이터·네트워크 효과·인증 등 진입장벽 (근거 id), 약하면 약하다고 쓴다")
    threats: list[str] = Field(description="경쟁 위협 (근거 id)")
    verified_claims: list[VerifiedClaim] = Field(description="기술 요약의 차별점 주장마다 1개 (주장이 없으면 빈 목록)")
    evidence_ids: list[str]


def _ground(res: CompetitionAnalysis, ids: list[str], reg: SourceRegistry, named: list[dict]) -> list[dict]:
    """이름이 근거에 실제로 나오는 경쟁사만 남기고, 행마다 근거 id 를 맞춘다 (지어낸 경쟁사 차단).
    LLM 이 근거 id 를 비우면 대상 기사 속 언급(named) → 나머지 근거 순으로 채운다."""
    named_ids = {norm(m["name"]): m["evidence_ids"] for m in named}
    rows = []
    for c in res.competitors:
        keys = [norm(x) for x in re.split(r"[()/,·]", c.name) if len(norm(x)) >= 2]
        found = [i for i in ids if any(k in norm(reg.text(i)) for k in keys)]
        first = [i for k in keys for i in named_ids.get(k, []) if i in found]
        ev = [i for i in c.evidence_ids if i in found] or list(dict.fromkeys(first + found))[:3]
        if not ev:
            continue
        row = {**c.model_dump(), "evidence_ids": ev}
        if _asserts(row["vs_target"]):
            row["vs_target"] = "회사 측 주장(제3자 비교 근거 없음): " + row["vs_target"]
        if not CITE.search(row["vs_target"]):  # 표 칸에도 출처가 이어지게
            row["vs_target"] += f" [{', '.join(ev[:2])}]"
        rows.append(row)
    return rows


def _own_site(s: dict | None, keys: list[str]) -> bool:
    """회사 자체 사이트 근거인지 (호스트에 회사 영문명·이름이 들어감)."""
    url = (s or {}).get("url") or ""
    host = norm(url.split("/")[2]) if url.count("/") >= 2 else ""
    return bool(host) and any(k in host for k in keys)


def verify_claims(items: list[VerifiedClaim], valid: set[str], reg: SourceRegistry, company: list[str]) -> list[dict]:
    """주장 검증 결과를 코드로 점검한다. 근거 id 는 저장소에 있는 것만 남기고,
    - '제3자 확인'인데 근거 id 가 없거나 회사 자체 사이트 근거뿐이면 → '회사 주장'
    - '반대 근거'인데 근거 id 가 없으면 → '회사 주장'
    낮춘 항목은 downgraded=True 로 표시한다."""
    keys = [norm(k) for k in company if k and len(norm(k)) >= 3]
    out = []
    for v in items:
        ev = [i for i in dict.fromkeys(v.evidence_ids) if i in valid]
        third = [i for i in ev if not _own_site(reg.get(i), keys)]
        low = (v.status == "제3자 확인" and not third) or (v.status == "반대 근거" and not ev)
        out.append({"claim": v.claim, "status": "회사 주장" if low else v.status, "evidence_ids": ev, "downgraded": low})
    return out


def _analysis_text(c: dict, tech: dict, k: dict) -> str:
    """경쟁 우위 판정에 넘기는 요약문."""
    rows = "; ".join(f"{r['name']}({r['country']}): {r['offering']}" for r in k["competitors"])
    claims = "; ".join(f"{v['claim']} → {v['status']}" for v in k["verified_claims"])
    return "\n".join([
        candidate_line(c),
        f"[기술 요약] 제품: {tech.get('product') or '확인 불가'} / 핵심 기술: {tech.get('core_technology') or '확인 불가'} "
        f"/ 특허·인증: {tech.get('ip_evidence') or '확인 불가'}",
        f"[경쟁] 제품 유형: {k['search_plan']['product_type_ko']} / 경쟁사: {rows or '없음'}",
        f"[차별성] {k['differentiation']} / 진입장벽: {k['entry_barriers']}",
        f"[차별점 주장 검증] {claims or '검증할 주장 없음'}",
    ])


def competition_node(state: dict) -> dict:
    c, tech = state["current"], state.get("tech", {})
    reg = SourceRegistry(state.get("registry"))
    seg = get_segment(c["segment_id"])
    name = c["official_name"]
    names = [name, c.get("name_en") or ""]
    # 1) 대상 회사 근거 본문에 경쟁사로 직접 언급된 회사 (적격성 근거 + 창업자·기술 요약 근거 전체, v1 기술·팀 근거와 같은 범위)
    own = list(dict.fromkeys(c.get("evidence_ids", []) + state.get("founder", {}).get("pool_ids", [])
                             + tech.get("pool_ids", [])))
    named = mentioned_competitors(reg, own, names)
    # 2) 검색 계획: 제품 유형을 정하고 그 유형의 경쟁사를 찾는 질의를 LLM 이 만든다
    plan: SearchPlan = structured(SearchPlan).invoke(render(
        "competition_plan", name=name, segment=seg["name"], product=_plain(tech.get("product")),
        core_technology=_plain(tech.get("core_technology")),
        named="\n".join(f"- {m['name']}: \"{m['quote']}\"" for m in named) or "(없음)"))
    queries = list(dict.fromkeys(q.strip() for q in plan.queries if q.strip()))[:MAX_QUERIES]
    ids = []
    for q in queries:
        ids += web_search(q, reg, AGENT, topic="general", recent=False)
    ids += web_search(f"{name} 경쟁사", reg, AGENT, topic="news", recent=False, raw=True)
    ids = list(dict.fromkeys(ids + tech.get("evidence_ids", []) + [i for m in named for i in m["evidence_ids"]]))
    # 3) 근거: 스니펫 + 대상·경쟁사 이름이 나오는 기사 본문의 '경쟁·비교' 문단
    keys = names + [m["name"] for m in named]
    evidence = "\n\n".join(evidence_blocks(reg, ids, keys, terms=RIVAL_TERMS, max_chars=600, boost=None).values())
    ctx = dict(name=name, segment=seg["name"], product=tech.get("product", ""), product_type=plan.product_type_ko,
               claims="\n".join(f"- {x}" for x in tech.get("claims", [])) or "(없음)",
               named="\n".join(f"- {m['name']} [{', '.join(m['evidence_ids'])}]: \"{m['quote']}\"" for m in named)
               or "(없음)", evidence=evidence)
    llm = bounded(CompetitionAnalysis)
    res: CompetitionAnalysis = llm.invoke(render("competition", **ctx, feedback=""))
    bad = [x.name for x in res.competitors if _asserts(x.vs_target)] + (
        ["differentiation"] if _asserts(res.differentiation) else [])
    if bad:  # 우열 단정이 있으면 한 번만 고쳐 쓰게 한다 (남으면 _ground 가 '회사 측 주장'으로 감싼다)
        res = llm.invoke(render("competition", **ctx, feedback=(
            f"다음 칸에 '앞선다·우위·뛰어나다' 같은 우열 단정이 있다: {', '.join(bad)}. "
            "경쟁사는 근거에 적힌 사실로, 대상의 강점은 '회사 측 주장: ~' 으로 고쳐 써라.")))
    out = res.model_dump()
    out["competitors"] = _ground(res, ids, reg, named)
    if _asserts(out["differentiation"]):
        out["differentiation"] = "회사 측 주장(제3자 비교 근거 없음): " + out["differentiation"]
    out["evidence_ids"] = [i for i in dict.fromkeys(res.evidence_ids + [i for r in out["competitors"]
                                                                        for i in r["evidence_ids"]]) if i in set(ids)]
    out["pool_ids"] = ids
    out["search_plan"] = {"product_type_ko": plan.product_type_ko, "product_type_en": plan.product_type_en,
                          "queries": queries}
    out["mentioned_competitors"] = named
    out["verified_claims"] = verify_claims(res.verified_claims, set(ids), reg, names)
    out["evidence_ids"] = [i for i in dict.fromkeys(out["evidence_ids"] + [i for v in out["verified_claims"]
                                                                          for i in v["evidence_ids"]])]
    out["criterion"] = judge_own("competition", state, reg, ids, _analysis_text(c, tech, out))
    crit = out["criterion"]
    status = [v["status"] for v in out["verified_claims"]]
    msg = (f"[경쟁사] {name}: 제품 유형 '{plan.product_type_ko}', 검색 {len(queries) + 1}건, "
           f"근거 속 경쟁사 {', '.join(m['name'] for m in named) or '없음'} → 경쟁사 {len(out['competitors'])}곳 비교"
           f"{' (우열 단정 고쳐 씀)' if bad else ''}, 주장 {len(status)}건(제3자 확인 {status.count('제3자 확인')}·"
           f"반대 근거 {status.count('반대 근거')}) → 경쟁 우위 YES {crit['yes']}/{crit['n']}")
    print(msg)
    return {"registry": reg.data, "competition": out, "log": [msg]}
