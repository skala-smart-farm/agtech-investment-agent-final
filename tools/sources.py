"""근거(Evidence) 레지스트리와 REFERENCE 표기.

모든 에이전트가 모은 근거는 id(D=문서, W=웹)로 등록되고, 분석 결과는 이 id 로만 근거를 인용한다.
보고서 REFERENCE 에는 최종 본문에 실제로 인용된 id 만 들어간다(과제 조건: 실제 활용 자료만).
"""
from __future__ import annotations

import hashlib
import re
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, urlparse

# 언론사·DB 도메인 → 사이트명 (없으면 제목 꼬리나 도메인으로 추정)
SITE_NAMES = {
    "wowtale.net": "와우테일", "startuprecipe.co.kr": "스타트업레시피", "platum.kr": "플래텀",
    "venturesquare.net": "벤처스퀘어", "thevc.kr": "THE VC", "innoforest.co.kr": "혁신의숲",
    "edaily.co.kr": "이데일리", "mk.co.kr": "매일경제", "hankyung.com": "한국경제", "sedaily.com": "서울경제",
    "chosun.com": "조선일보", "joongang.co.kr": "중앙일보", "donga.com": "동아일보", "yna.co.kr": "연합뉴스",
    "zdnet.co.kr": "지디넷코리아", "etnews.com": "전자신문", "nongmin.com": "농민신문", "aflnews.co.kr": "농수축산신문",
    "agrinet.co.kr": "한국농어민신문", "bloter.net": "블로터", "unicornfactory.co.kr": "머니투데이 유니콘팩토리",
    "news.mt.co.kr": "머니투데이", "mt.co.kr": "머니투데이", "hankookilbo.com": "한국일보", "khan.co.kr": "경향신문",
    "techcrunch.com": "TechCrunch", "agfundernews.com": "AgFunderNews", "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg", "businesswire.com": "Business Wire", "prnewswire.com": "PR Newswire",
    "forbes.com": "Forbes", "crunchbase.com": "Crunchbase", "futurefarming.com": "Future Farming",
    "jointips.or.kr": "TIPS 창업기업 목록", "mafra.go.kr": "농림축산식품부", "agnavigator.com": "AgNavigator",
    "data.go.kr": "공공데이터포털", "daum.net": "다음뉴스", "news.naver.com": "네이버 뉴스", "aving.net": "에이빙(AVING)", "newsis.com": "뉴시스",
    "news1.kr": "뉴스1", "asiae.co.kr": "아시아경제", "fnnews.com": "파이낸셜뉴스", "heraldcorp.com": "헤럴드경제",
    "dt.co.kr": "디지털타임스", "etoday.co.kr": "이투데이", "sisajournal-e.com": "시사저널e", "the-pr.co.kr": "더피알",
    "cbinsights.com": "CB Insights", "financialcontent.com": "FinancialContent", "zdnet.co.kr": "지디넷코리아",
    "supplychangecapital.substack.com": "Supply Change Capital", "agfunder.com": "AgFunder", "weforum.org": "World Economic Forum",
}


def _site(url: str, title: str = "") -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.").removeprefix("m.")
    for dom, name in SITE_NAMES.items():
        if host == dom or host.endswith("." + dom):
            return name
    for sep in (" - ", " | ", " : ", " – "):
        if sep in title:
            tail = title.rsplit(sep, 1)[-1].strip()
            if 1 < len(tail) <= 20:
                return tail
    return host


SITE_TAIL = re.compile(r"(news|korea|\.com|\.kr|일보|신문|뉴스|경제|innoforest|혁신의숲|the vc|times|herald|tribune|"
                       r"기업정보|매출·투자·고용)", re.I)


def _clean_title(title: str, site: str) -> str:
    title = title.strip()
    for sep in (" - ", " | ", " : ", " – "):  # 먼저 "제목 - 사이트명" 꼬리를 떼고
        if sep in title:
            tail = title.rsplit(sep, 1)[-1].strip()
            if tail == site or (len(tail) <= 28 and SITE_TAIL.search(tail)):
                title = title.rsplit(sep, 1)[0].strip()
    # 다음으로 "< 산업 < 기사본문" 같은 게시판 경로 꼬리를 뗀다
    return re.sub(r"(\s*<\s*[^<>]{1,20})+\s*<\s*기사본문\s*$", "", title).strip()


def _date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        pass
    m = re.search(r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})", raw)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def _date_from(url: str, text: str) -> str | None:
    """게시일이 없을 때 URL(/2022/12/10/, /20221210) 이나 본문(2022.12.10)에서 날짜를 찾는다.
    설립일("설립일 2022-09-02", "2022년 9월 2일 설립")은 게시일이 아니므로 건너뛴다."""
    for pat in (r"/(20\d{2})/(\d{2})/(\d{2})(?:/|\b)", r"/(20\d{2})(\d{2})(\d{2})\d*", r"[?&]date=(20\d{2})-?(\d{2})-?(\d{2})"):
        m = re.search(pat, url or "")
        if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    head = (text or "")[:1500]
    for pat in (r"(?:입력|등록|게재|승인|발행|기사입력|Published|Posted)\s*[:：]?\s*(20\d{2})[.\-/년 ]+\s?(\d{1,2})[.\-/월 ]+\s?(\d{1,2})",
                r"(20\d{2})\s?년\s?(\d{1,2})\s?월\s?(\d{1,2})\s?일",
                r"(20\d{2})[.-]\s?(\d{1,2})[.-]\s?(\d{1,2})"):
        for m in re.finditer(pat, head):  # "설립일 2022-09-02", "설립연월일: …", "2022년 9월 2일 설립" 은 건너뛴다
            if (re.search(r"(?:설립|창립|창업|Founded)(?:일자?|연월일|년월일)?\s*[:：]?\s*$", head[max(0, m.start() - 12): m.start()])
                    or re.match(r"\s*(?:[-–]\s*)?(?:에\s*)?(?:설립|창립)(?!일|연월일)", head[m.end(): m.end() + 10])):
                continue
            if 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
                return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


NOT_REPORTER = {"사진", "취재", "전문", "객원", "구독", "담당", "정책", "축산", "농업", "산업"}


def _author(text: str) -> str | None:
    """바이라인의 기자 이름. '스마트농업 기자재'·'사진기자' 같은 말은 이름으로 보지 않는다."""
    for m in re.finditer(r"(?<![가-힣])([가-힣]{2,4})\s?기자(?![가-힣])", text or ""):
        if m.group(1) not in NOT_REPORTER:
            return f"{m.group(1)} 기자"
    return None


class SourceRegistry:
    """dict 기반이라 LangGraph State 에 그대로 담을 수 있다."""

    def __init__(self, data: dict | None = None):
        self.data: dict[str, dict] = dict(data or {})

    @staticmethod
    def _id(prefix: str, key: str) -> str:
        # 병렬 노드가 동시에 등록해도 충돌하지 않도록 순번 대신 키 해시로 id 를 만든다
        return prefix + hashlib.sha1(key.encode()).hexdigest()[:5]

    def _find(self, key: str) -> str | None:
        return next((sid for sid, s in self.data.items() if s.get("key") == key), None)

    def add_web(self, result: dict, agent: str, query: str, access_date: str, key: str | None = None) -> str:
        url = result.get("url", "")
        key = key or "web:" + url
        if sid := self._find(key):
            if result.get("raw_content") and not self.data[sid].get("body"):  # 같은 URL 을 본문과 함께 다시 받은 경우
                self.data[sid]["body"] = re.sub(r"\s+", " ", result["raw_content"])[:6000]
                if not self.data[sid]["date"] and (d := _date_from("", result["raw_content"])):
                    self.data[sid].update(date=d, date_is_access=False)
            return sid
        title = result.get("title", "") or url
        site = _site(url, title)
        pub = (_date(result.get("published_date")) or _date_from(url, result.get("content", ""))
               or _date_from("", result.get("raw_content") or ""))
        sid = self._id("W", key)
        self.data[sid] = {
            "id": sid, "key": key, "kind": "web", "url": url, "site": site,
            "title": _clean_title(title, site), "date": pub, "date_is_access": pub is None,
            "access_date": access_date, "author": result.get("author") or _author(result.get("content", "")),
            "snippet": (result.get("content") or "")[:1200], "agent": agent, "query": query,
            "body": re.sub(r"\s+", " ", result.get("raw_content") or "")[:6000],
        }
        return sid

    def add_doc(self, meta: dict, text: str, agent: str, query: str) -> str:
        key = f"doc:{meta.get('doc_id')}:{meta.get('page')}:{hashlib.md5(text.encode()).hexdigest()[:8]}"
        if sid := self._find(key):
            return sid
        sid = self._id("D", key)
        self.data[sid] = {
            "id": sid, "key": key, "kind": "doc", "doc_id": meta.get("doc_id"), "type": meta.get("type", "report"),
            "title": meta.get("title"), "publisher": meta.get("publisher"), "year": meta.get("year"),
            "url": meta.get("url"), "page": meta.get("page"), "authors": meta.get("authors"),
            "journal": meta.get("journal"), "volume": meta.get("volume"), "issue": meta.get("issue"),
            "pages": meta.get("pages"), "snippet": text[:1200], "agent": agent, "query": query,
        }
        return sid

    def get(self, sid: str) -> dict | None:
        return self.data.get(sid)

    def text(self, sid: str) -> str:
        """근거 전체 텍스트 (제목 + 스니펫 + 본문). 인용문 검증과 문항별 재검색에 쓴다."""
        s = self.data.get(sid) or {}
        return " ".join(x for x in (s.get("title"), s.get("snippet"), s.get("body")) if x)

    def brief(self, ids: list[str], max_chars: int = 600) -> str:
        """LLM 에 넘길 근거 목록 텍스트."""
        lines = []
        for sid in ids:
            s = self.data.get(sid)
            if not s:
                continue
            if s["kind"] == "web":
                when = s["date"] or "게시일 미상"  # 조회일을 보여 주면 LLM 이 사건 날짜로 오인한다
                head = f"[{sid}] (웹, {s['site']}, {when}) {s['title']}"
            else:
                head = f"[{sid}] (문서, {s['publisher']} {s['year']}, p.{s['page']}) {s['title']}"
            lines.append(f"{head}\n{s['snippet'][:max_chars]}")
        return "\n\n".join(lines)


# ── REFERENCE 표기 보정
# 포털 전재본: 원 매체·기자를 본문("Copyright ⓒ 조선비즈", "최효정 기자")에서 찾아 표기한다 (URL 은 그대로)
PORTALS = ("v.daum.net", "news.daum.net", "n.news.naver.com", "news.naver.com", "news.nate.com")
# 뉴스레터·모음 메일: 여러 기사를 한데 모은 2차 자료라 인용하지 않는다
NEWSLETTERS = ("stibee.com", "maily.so")
# 목록·DB 페이지: 게시일이 없어 조회일로 표기하고, 기관명은 운영 주체로 쓴다
LIST_PAGES = {"jointips.or.kr": "중소벤처기업부 TIPS", "data.go.kr": "국민연금공단"}
GROUPS = ("기관 보고서", "학술 논문", "웹페이지")  # REFERENCE 소제목과 번호 순서
TRUNCATED = re.compile(r"\s*(?:\.{3,}|…)\s*$")
_PORTAL_OWN = re.compile(r"nate|daum|kakao|naver", re.I)


def _host(url: str) -> str:
    return urlparse(url or "").netloc.lower()


def _origin(s: dict) -> tuple[str | None, str | None]:
    """포털 전재본의 (원 매체, 기자). 못 찾으면 None."""
    text = " ".join(x for x in (s.get("body"), s.get("snippet")) if x)
    for pat in (r"Copyright\s*[ⓒ©]\s*(?:(?:19|20)\d{2}\s*)?([^&.,<>\[\]]{2,20}?)\s*(?:&|\.|,|All rights|무단|$)",
                r"([가-힣A-Za-z0-9]{2,12})\s*원문\s*기사전송",
                r"[ⓒ©]\s*(?:(?:19|20)\d{2}\s*)?(?:['‘\"“][^'’\"”]{0,30}['’\"”]\s*)?([가-힣A-Za-z0-9]{2,12})"):
        for m in re.finditer(pat, text):
            if not _PORTAL_OWN.search(m.group(1)) and not m.group(1).strip().isdigit():
                return m.group(1).strip(), _author(text)
    return None, _author(text)


def citable(s: dict) -> bool:
    """REFERENCE 에 올릴 수 있는 근거인지 (뉴스레터·모음 메일 제외)."""
    return s["kind"] == "doc" or not _host(s.get("url", "")).endswith(NEWSLETTERS)


def reference_group(s: dict) -> str:
    """기관 보고서 / 학술 논문 / 웹페이지. 저자가 있는 문서(학술지·정기간행물 기고)는 학술 논문 형식."""
    if s["kind"] == "web":
        return "웹페이지"
    return "학술 논문" if s.get("type") == "paper" or s.get("authors") else "기관 보고서"


def full_title(s: dict) -> str:
    """검색 결과에서 잘린 제목("... ....")을 복원한다: 원문 캐시의 제목 → 본문 첫머리 → 말줄임표만 제거."""
    title = s.get("title") or ""
    if s["kind"] == "doc" or not TRUNCATED.search(title):
        return title
    from tools.fetch import cached  # fetch → sources 순환 import 방지

    if (d := cached(s.get("url", ""))) and d.get("title"):
        return _clean_title(d["title"], s.get("site", ""))
    base = TRUNCATED.sub("", title).strip()
    body = s.get("body") or ""
    i = body.find(base)
    if len(base) >= 10 and 0 <= i < 40:  # 포털 본문은 제목으로 시작한다: 본문 표지 앞까지(30자 이내) 이어 붙인다
        m = re.match(r"(.{1,30}?)\s(?:전체 맥락을|자동요약|[가-힣]{2,4}\s?기자\s|[가-힣A-Za-z0-9]{2,12}\s원문\s기사전송|"
                     r"20\d{2}\.\s?\d)", body[i + len(base):])
        if m:
            return base + m.group(1)
    return base


def _ref_title(s: dict) -> str:
    """REFERENCE 제목: 잘린 제목 복원 + 기업 DB·사이트 이름 꼬리("- 기업정보 | 투자, 매출, 기업가치", "- 유니콘팩토리") 제거."""
    title, prev = full_title(s), None
    while title != prev:
        prev = title
        for sep in (" | ", " - "):
            if sep in title:
                head, tail = title.rsplit(sep, 1)
                site = (s.get("site") or "").strip()
                if len(tail.strip()) >= 2 and (tail.strip() == site or (site and site.endswith(tail.strip()))
                                               or re.search(r"기업가치|기업정보|스타트업 조회|데이터랩|THE VC|혁신의숲", tail)):
                    title = head.strip()
    return title


def title_key(s: dict) -> str | None:
    """같은 기사(포털 전재본 포함)를 묶는 정규화 제목. 짧거나 목록형 제목이면 None."""
    t = re.sub(r"[\W_]+", "", full_title(s)).lower()
    return t if s["kind"] == "web" and len(t) >= 12 else None


# 게시일이 없는 웹페이지는 REFERENCE 에 조회일을 적는다. 이 사실은 목록 줄이 아니라 보고서 한계점에 한 줄로 밝힌다
# (과제: REFERENCE 는 '실제로 활용한 자료 목록만' 기재)
ACCESS_DATE_NOTE = "게시일이 없는 웹페이지는 REFERENCE 에 조회일을 적었다"
# 과제 지정 형식 3유형 (보고서 형식 검사용). 줄 끝에 괄호 주석을 붙이지 않는다
REFERENCE_FORMATS = {
    "기관 보고서": re.compile(r"^\S.*\((?:19|20)\d{2}\)\. .+\. https?://.+$"),
    "학술 논문": re.compile(r"^\S.*\((?:19|20)\d{2}\)\. .+\. .+\.$"),
    "웹페이지": re.compile(r"^\S.*\((?:19|20)\d{2}-\d{2}-\d{2}\)\. .+\. .+, https?://.+$"),
}


def _list_page_org(host: str) -> str | None:
    return next((v for k, v in LIST_PAGES.items() if host == k or host.endswith("." + k)), None)


def uses_access_date(s: dict) -> bool:
    """REFERENCE 날짜로 게시일 대신 조회일을 쓰는 웹페이지인지 (게시일 미상, 또는 목록·DB 페이지)."""
    if s["kind"] != "web":
        return False
    return bool(_list_page_org(_host(unquote(s["url"]).split("#")[0]))) or not s.get("date")


def format_reference(s: dict) -> str:
    """과제에서 지정한 REFERENCE 표기 형식 그대로 쓴다(줄 끝 주석 없음).
    게시일을 찾지 못한 웹페이지와 목록·DB 페이지는 조회일을 쓴다(uses_access_date, 한계점에서 밝힘)."""
    if s["kind"] == "doc":
        if reference_group(s) == "학술 논문":
            vol = f"{s.get('volume') or ''}({s['issue']})" if s.get("issue") else str(s.get("volume") or "")
            tail = ", ".join(str(x) for x in (s.get("journal") or s.get("publisher"), vol, s.get("pages")) if x)
            return f"{s['authors']}({s['year']}). {s['title']}. {tail}."
        return f"{s['publisher']}({s['year']}). {s['title']}. {s['url']}"
    url = unquote(s["url"]).split("#")[0]  # "#메타파머스" 같은 내부 표지는 URL 이 아니다
    host = _host(url)
    site, who = s["site"], s.get("author")
    if host.endswith(PORTALS):
        outlet, reporter = _origin(s)
        site, who = outlet or site, reporter or who or outlet
    org = _list_page_org(host)
    who = who or org or site
    # 목록·DB 페이지의 날짜(설립일·자료 기준월)는 게시일이 아니다
    when = s["access_date"] if uses_access_date(s) else s["date"]
    return f"{who}({when}). {_ref_title(s)}. {site}, {url}"


def reference_key(s: dict) -> str:
    """같은 문서의 여러 페이지는 한 줄로, 같은 웹페이지(인코딩·쿼리 차이 포함)도 한 줄로 합친다.
    같은 목록 URL 을 공유하는 회사별 항목(TIPS 목록 등)은 등록 키로 구분한다."""
    if s["kind"] == "doc":
        return f"doc:{s['doc_id']}"
    if s.get("key") and not s["key"].startswith("web:"):
        return s["key"]
    u = unquote(s["url"]).split("#")[0].split("?")[0].rstrip("/").lower()
    return "web:" + u.replace("://m.", "://").replace("://www.", "://")


def is_portal(s: dict) -> bool:
    return s["kind"] == "web" and _host(s.get("url", "")).endswith(PORTALS)


def merge_duplicates(srcs: list[dict]) -> dict:
    """같은 자료로 묶인 근거들의 대표 항목: 원 매체 페이지 우선, 제목은 가장 온전한 것, 비어 있는 기자·게시일은 사본에서 보충."""
    best = dict(next((s for s in srcs if not is_portal(s)), srcs[0]))
    if len(srcs) == 1 or best["kind"] == "doc":
        return best
    best["title"] = max((full_title(s) for s in srcs), key=len)
    if not best.get("author") and not is_portal(best):
        best["author"] = next((r for s in srcs if is_portal(s) and (r := _origin(s)[1])), None)
    if not best.get("date"):
        best["date"] = next((s["date"] for s in srcs if s.get("date")), None)
    return best


def today() -> str:
    """조회일 = 평가 기준일 (재현 모드에서는 고정)."""
    from core.config import run_date

    return run_date()
