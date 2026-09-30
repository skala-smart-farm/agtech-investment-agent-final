"""검색 공급자 여러 개 (한 곳이 막혀도 계속 동작하도록).

- Serper (SERPER_API_KEY): 구글 검색·뉴스. 한국어 질의는 gl=kr·hl=ko 로 국내 결과. 가입 시 무료 2,500건
- Tavily (TAVILY_API_KEY): 처음부터 쓰던 공급자 (무료 한도를 다 써서 예비로 둔다)
키가 있는 공급자만 Serper → Tavily 순서로 시도한다.
두 공급자 모두 요청 제한 시간(timeout)을 둔다. 오류는 ProviderError 로 올려 상태 코드로 인증·한도 오류(fatal)와
일시 오류를 구분한다 (tools/web_search.py 가 재시도·공급자 끄기를 정한다).
네이버 검색 API 는 2026-07-31 부터 개발자센터 신규 발급이 끝나(NAVER API HUB 로 이관) 넣지 않았다.
모든 공급자의 결과는 같은 형식 {url, title, content, published_date} 으로 맞춘다.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta

import requests

HANGUL = re.compile(r"[가-힣]")
TAVILY_URL = "https://api.tavily.com/search"
# 다시 불러도 소용없는 오류: 인증 실패(401·403), 요금제 한도(432)·종량제 한도(433) 초과 → 이번 실행에서 그 공급자를 끈다
FATAL_STATUS = {401, 403, 432, 433}


class ProviderError(RuntimeError):
    """검색 공급자 오류. status 는 HTTP 상태 코드(연결·시간 초과는 None)."""

    def __init__(self, provider: str, message: str, status: int | None = None, timed_out: bool = False,
                 retry_after: float = 0.0):
        super().__init__(message)
        self.provider, self.status, self.timed_out, self.retry_after = provider, status, timed_out, retry_after

    @property
    def fatal(self) -> bool:
        return self.status in FATAL_STATUS

    @property
    def retryable(self) -> bool:
        """바로 다시 시도할 만한 오류: 연결 실패, 429(요청 속도 제한), 5xx. 시간 초과는 이미 timeout 만큼 기다렸으므로 제외."""
        return not self.timed_out and (self.status is None or self.status == 429 or self.status >= 500)


def order_for(query: str) -> list[str]:
    return [p for p, key in (("serper", "SERPER_API_KEY"), ("tavily", "TAVILY_API_KEY")) if os.getenv(key)]


def _serper_date(raw: str | None, today: str) -> str | None:
    """Serper 의 날짜 표기("3 days ago", "3일 전", "Jan 5, 2025", "2025. 1. 5.")를 YYYY-MM-DD 로 바꾼다.
    "N일 전" 은 검색한 날(today)에서 뺀다. 변환한 날짜째로 캐시에 저장되므로 재현 때도 같다."""
    if not raw:
        return None
    base = date.fromisoformat(today)
    m = re.search(r"(\d+)\s*(minute|hour|day|week|month|year|분|시간|일|주|개월|달|년)", raw)
    if m and ("ago" in raw or "전" in raw):
        n, unit = int(m.group(1)), m.group(2)
        days = {"minute": 0, "분": 0, "hour": 0, "시간": 0, "day": 1, "일": 1, "week": 7, "주": 7,
                "month": 30, "개월": 30, "달": 30, "year": 365, "년": 365}[unit]
        return (base - timedelta(days=n * days)).isoformat()
    m = re.search(r"(20\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})", raw)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    for fmt in ("%b %d, %Y", "%d %b %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def tavily(query: str, topic: str, time_range: str | None, max_results: int, include_domains: list[str] | None,
           exclude_domains: list[str], depth: str, raw: bool, timeout: float) -> list[dict]:
    """Tavily 검색 REST 호출. 예전에 쓰던 langchain_tavily.TavilySearch 와 같은 요청 본문을 보낸다(같은 검색 결과).
    그 래퍼는 requests.post 에 제한 시간이 없어, 응답 없는 연결에서 실행이 끝없이 멈췄다 → 여기서 timeout 을 건다.
    결과가 없으면 [] (예전 "No search results" 처리와 같음)."""
    body = {"query": query, "max_results": max_results, "search_depth": depth, "include_domains": include_domains or [],
            "exclude_domains": [] if include_domains else list(exclude_domains), "include_images": False, "topic": topic}
    if raw:
        body["include_raw_content"] = "text"  # 본문 전체: 창업자 이력·실적처럼 스니펫에 잘 안 나오는 사실 확보
    if time_range:
        body["time_range"] = time_range
    try:
        r = requests.post(TAVILY_URL, json=body, timeout=(min(10, timeout), timeout),
                          headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"})
    except requests.RequestException as e:  # 연결 실패·시간 초과
        raise ProviderError("tavily", type(e).__name__, None, isinstance(e, requests.Timeout)) from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", {})
            msg = detail.get("error") if isinstance(detail, dict) else str(detail)
        except ValueError:
            msg = r.text[:80]
        wait = r.headers.get("Retry-After", "") if r.status_code == 429 else ""
        raise ProviderError("tavily", f"Error {r.status_code}: {msg}", r.status_code,
                            retry_after=float(wait) if wait.replace(".", "", 1).isdigit() else 0.0)
    return r.json().get("results", []) or []


def serper(query: str, topic: str, include_domains: list[str] | None,
           time_range: str | None, today: str) -> list[dict]:
    q = query + (" (" + " OR ".join(f"site:{d}" for d in include_domains) + ")" if include_domains else "")
    body = {"q": q, "num": 10}  # 제외 도메인을 결과에서 거르므로 넉넉히 받는다
    if HANGUL.search(query):
        body.update(gl="kr", hl="ko")
    if time_range == "year":
        body["tbs"] = "qdr:y"
    endpoint = "news" if topic == "news" else "search"
    try:
        r = requests.post(f"https://google.serper.dev/{endpoint}", json=body,
                          headers={"X-API-KEY": os.environ["SERPER_API_KEY"]}, timeout=20)
    except requests.RequestException as e:
        raise ProviderError("serper", type(e).__name__, None, isinstance(e, requests.Timeout)) from e
    if not r.ok:  # 예전 raise_for_status() 와 같은 기준(4xx·5xx)
        raise ProviderError("serper", f"Error {r.status_code}: {r.text[:80]}", r.status_code)
    items = r.json().get("news" if endpoint == "news" else "organic", [])
    return [{"url": it["link"], "title": it.get("title", ""), "content": it.get("snippet", ""),
             "published_date": _serper_date(it.get("date"), today)} for it in items if it.get("link")]
