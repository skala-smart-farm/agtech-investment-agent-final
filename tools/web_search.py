"""외부 정보 검색 도구 (Serper(구글) → Tavily, 키가 있는 공급자만 순서대로).

- 뉴스는 기본적으로 최근 1년(time_range)만 검색해 오래된 정보가 섞이지 않게 한다.
- 결과가 부족할 때만 기간 제한을 풀어 다시 검색한다.
- 같은 쿼리는 디스크 캐시를 재사용한다(비용·재현성). 캐시 키에 공급자를 넣지 않아, 어느 공급자로 받았든
  재현 때는 저장된 결과를 그대로 쓴다 (결과마다 provider 필드로 출처 공급자를 남긴다).
- 한 공급자가 한도 초과·오류면 다음 공급자로 넘어간다. 모두 실패하면 실패 표시(.failed)를 남기고,
  --retry-failed 로 실행하면 그 검색만 다시 시도한다.
- 실시간 검색만 해당(재현 실행은 캐시만 읽음): 요청마다 제한 시간(search.timeout_sec)을 두고, 일시 오류만 다시 시도한다.
  인증·한도 오류(401·403·432·433)는 1번, 연결 실패·시간 초과·5xx 는 연속 search.breaker_after 번이면 그 공급자를 이번 실행에서 끈다
  (429 속도 제한은 기다렸다 다시 시도할 뿐 끄지 않는다)
  (한도가 끝난 키로 남은 검색 수백 번을 헛되이 부르지 않게). 꺼진 뒤의 검색도 지금과 같이 실패로 기록된다(fail-closed).
- 실패한 검색은 "결과 없음"과 구분해 FAILED_QUERIES 에 남긴다 (적격성 관문의 fail-closed, 보고서 한계점에 사용).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import date

from core.config import get_config, path, require_keys
from tools import search_providers as providers
from tools.sources import SourceRegistry, today

# 보도자료 배포 사이트로 퍼진 유료 시장조사 홍보 기사 (예: "... Market Growing at 24% CAGR, Says Mordor Intelligence")
PR_MARKET = re.compile(r"market (size|share|growing|to reach|worth|report|intelligence|analysis|forecast)|cagr|"
                       r"says .*(research|intelligence|insights)|(mordor|marketsandmarkets|grand view|precedence|imarc|"
                       r"fortune business|researchandmarkets|research and markets)", re.I)
# 검색 결과 목록 페이지는 근거가 아니다 (예: search.zdnet.co.kr?kwd=...)
SEARCH_PAGE = re.compile(r"//search\.|/search[/?]|[?&](kwd|q|query|keyword)=", re.I)
# 실패해서 데이터 없이 끝난 검색 {query, agent}. 실시간 실패와 재현용 캐시의 실패 표시(.failed)를 모두 기록한다.
# 호출할 때마다 붙인다(같은 쿼리가 여러 번 들어갈 수 있음). 다른 모듈이 같은 객체를 참조하므로 다시 대입하지 않는다
FAILED_QUERIES: list[dict] = []
# 이번 실행에서 끈 공급자 {공급자: 사유}와 공급자별 연속 오류 수, 실시간 검색 횟수·누적 시간 (적격성 검증이 병렬이라 잠금)
DISABLED: dict[str, str] = {}
_STREAK: dict[str, int] = {}
_LIVE = {"n": 0, "sec": 0.0}
_LOCK = threading.Lock()


def _cache_file(key: dict):
    h = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return path(f"{get_config().cache.dir}/search/{h}.json")


def _legacy_files(key: dict) -> list:
    """예전 캐시 키(제외 도메인 목록을 키에 넣던 시절)의 파일 후보. 찾으면 새 키로 옮겨 재현용 캐시를 살린다."""
    ex = list(get_config().search.exclude_domains)
    return [_cache_file({**key, "ex": ex[:n]}) for n in (15, 33, len(ex))]


def _raw_search(query: str, topic: str, time_range: str | None, max_results: int,
                include_domains: list[str] | None, depth: str = "basic", raw: bool = False) -> list[dict] | None:
    """검색 결과 목록. 검색 자체가 실패하면(모든 공급자 오류, 또는 캐시의 실패 표시) None — 정상적인 "결과 없음"은 []."""
    cfg = get_config()
    # 제외 도메인은 결과를 받은 뒤 다시 거르므로 캐시 키에 넣지 않는다 (목록을 고쳐도 재현용 캐시가 유지되게)
    key = {"q": query, "topic": topic, "tr": time_range, "n": max_results, "dom": include_domains or [],
           "depth": depth, "raw": raw}
    f = _cache_file(key)
    if cfg.cache.search and not f.exists():
        old = next((o for o in _legacy_files(key) if o.exists()), None)
        if old:
            f.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    if cfg.cache.search and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    failed = f.with_suffix(".failed")
    # 제출 실행 때 실패한 검색은 재현 때도 실패로 (누구 키로 돌려도 같은 결과). --retry-failed 면 다시 시도
    if cfg.cache.search and failed.exists() and not os.getenv("SEARCH_RETRY_FAILED"):
        return None
    if os.getenv("REPLAY_OFFLINE"):
        raise RuntimeError(f"--offline: 재현용 캐시에 없는 검색입니다 → {query!r}")
    require_keys()
    results, ok, errors, t0 = [], False, [], time.time()
    active = active_providers()
    if not active:
        errors.append(f"공급자 모두 꺼짐({', '.join(DISABLED)})")
    for provider in active:
        try:  # 다음 공급자가 있으면 예전처럼 바로 넘어가고, 마지막 공급자일 때만 일시 오류를 다시 시도한다
            got = _call_with_retry(provider, provider == active[-1], query, topic, time_range, max_results,
                                   include_domains, depth, raw)
        except Exception as e:  # 한도 초과·인증 오류 → 다음 공급자
            errors.append(f"{provider}: {str(e)[:60]}")
            _note_error(provider, e)
            continue
        ok = True
        results = [{**r, "provider": provider} for r in got]
        if results:
            break
    if not ok:
        print(f"   (검색 실패, 빈 결과로 진행: {query[:40]} — {'; '.join(errors)[:120]})")
    _progress(time.time() - t0)
    if cfg.cache.search and ok:
        f.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        failed.unlink(missing_ok=True)
    elif cfg.cache.search:  # 오류(한도 초과 등)는 결과 대신 실패 표시만 남긴다. --fresh(새 캐시)에서는 다시 시도한다
        failed.write_text(json.dumps({"query": query, "error": "search failed"}, ensure_ascii=False), encoding="utf-8")
    return results if ok else None


def active_providers() -> list[str]:
    """키가 있고 이번 실행에서 꺼지지 않은 공급자 (Serper → Tavily 순서)."""
    return [p for p in providers.order_for("") if p not in DISABLED]


def _note_error(provider: str, e: Exception) -> None:
    """공급자 오류를 세고, 인증·한도 오류이거나 연속 오류가 search.breaker_after 번이면 이번 실행에서 끈다(한 번만 알림)."""
    fatal = getattr(e, "fatal", False)
    if getattr(e, "status", None) == 429:  # 속도 제한은 기다리면 풀리므로 공급자를 끄는 근거로 세지 않는다
        return
    with _LOCK:
        _STREAK[provider] = _STREAK.get(provider, 0) + 1
        if provider in DISABLED or not (fatal or _STREAK[provider] >= int(get_config().search.get("breaker_after", 3))):
            return
        DISABLED[provider] = str(e)[:80]
        streak = _STREAK[provider]
    why = "인증·사용 한도 오류" if fatal else f"연속 {streak}번 오류"
    print(f"   [검색] {provider} 를 이번 실행에서 끕니다 ({why}: {str(e)[:60]}). 남은 공급자가 없으면 이후 검색은 실패로 기록되고 "
          f"해당 판정은 fail-closed 로 처리됩니다 → 키·한도를 확인한 뒤 --retry-failed 로 실패한 검색만 다시 할 수 있습니다")


def _call_with_retry(provider: str, last: bool, *args) -> list[dict]:
    """마지막 공급자일 때 일시 오류(연결 실패·429·5xx)만 search.retries 번 다시 시도한다.
    인증·한도 오류와 시간 초과는 바로 올린다."""
    retries = int(get_config().search.get("retries", 1)) if last else 0
    for attempt in range(retries + 1):
        try:
            got = _call(provider, *args)
        except providers.ProviderError as e:
            if not e.retryable or attempt == retries:
                raise
            time.sleep(min(10.0, e.retry_after) or 2 * (attempt + 1))  # 429 의 Retry-After 가 있으면 따른다(최대 10초)
            continue
        with _LOCK:
            _STREAK[provider] = 0
        return got
    return []


def _progress(sec: float) -> None:
    """실시간 검색 search.progress_every 회마다 한 줄 (노드가 끝날 때까지 출력이 없어 멈춘 것처럼 보이지 않게)."""
    every = int(get_config().search.get("progress_every", 10) or 0)
    with _LOCK:
        _LIVE["n"] += 1
        _LIVE["sec"] += sec
        n, avg = _LIVE["n"], _LIVE["sec"] / _LIVE["n"]
    if every and (n == 1 or n % every == 0):
        print(f"   (실시간 웹 검색 {n}회 · 공급자 {' → '.join(active_providers()) or '없음'} · 평균 {avg:.1f}초/회)")


def _call(provider: str, query: str, topic: str, time_range: str | None, max_results: int,
          include_domains: list[str] | None, depth: str, raw: bool) -> list[dict]:
    cfg = get_config()
    if provider == "tavily":
        return providers.tavily(query, topic, time_range, max_results, include_domains, cfg.search.exclude_domains,
                                depth, raw, float(cfg.search.get("timeout_sec", 30)))
    blocked = tuple(cfg.search.exclude_domains)
    got = [r for r in providers.serper(query, topic, include_domains, time_range, date.today().isoformat())
           if not _host(r["url"]).endswith(blocked) and not PR_MARKET.search(r["title"])
           and not SEARCH_PAGE.search(r["url"])][:max_results]
    if raw:  # 본문은 원문 페이지에서 직접 받는다 (Tavily raw_content 대신)
        from tools.fetch import fetch

        for r in got:
            if d := fetch(r["url"]):
                r["raw_content"] = d["text"]
                r["published_date"] = r["published_date"] or d["date"][:10] or None
    return got


def _host(url: str) -> str:
    return url.split("/")[2].lower() if url.count("/") >= 2 else ""


def web_search(query: str, registry: SourceRegistry, agent: str, *, topic: str = "news",
               recent: bool = True, max_results: int | None = None,
               include_domains: list[str] | None = None, min_results: int = 2, deep: bool = False,
               raw: bool = False) -> list[str]:
    """검색 결과를 근거로 등록하고 근거 id 목록을 돌려준다.
    deep=True: 특정 회사에 대한 검색처럼 본문 근거가 더 필요할 때 Tavily advanced 검색(관련 문단을 더 길게 가져옴).
    raw=True : 기사 본문 전체도 받아 근거 본문(body)으로 저장한다 (투자 판단 에이전트가 문항별로 다시 검색)."""
    cfg = get_config()
    n = max_results or cfg.search.max_results
    tr = cfg.search.news_time_range if (recent and topic == "news") else None
    depth = "advanced" if deep else "basic"
    got = [_raw_search(query, topic, tr, n, include_domains, depth, raw)]
    if len(got[0] or []) < min_results and tr:
        got.append(_raw_search(query, topic, cfg.search.fallback_time_range, n, include_domains, depth, raw))
    results = [r for g in got for r in g or []]
    if not results and got[-1] is None:  # 가장 넓은 검색까지 실패해 데이터가 없음 → "반증 없음"과 구분해 기록
        FAILED_QUERIES.append({"query": query, "agent": agent})
    ids: list[str] = []
    blocked = tuple(cfg.search.exclude_domains)
    for r in results:
        host = _host(r.get("url", ""))
        if (not r.get("url") or host.endswith(blocked) or PR_MARKET.search(r.get("title") or "")
                or SEARCH_PAGE.search(r["url"])):
            continue
        sid = registry.add_web(r, agent=agent, query=query, access_date=today())
        if sid not in ids:
            ids.append(sid)
    return ids
