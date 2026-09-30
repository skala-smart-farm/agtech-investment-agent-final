"""웹 검색 공급자 오류 처리 (실제 API 호출 없음: 공급자 함수와 requests.post 를 가짜로 바꾼다).

- 재현용 캐시에 있는 검색은 공급자를 부르지 않는다
- Tavily 요청 본문은 예전 langchain_tavily.TavilySearch 가 보내던 것과 같고, 제한 시간(timeout)이 붙는다
- 인증·한도 오류(401·432)는 다시 시도하지 않고 공급자를 끈다. 이후 검색도 지금처럼 실패로 기록된다(fail-closed)
- 일시 오류는 마지막 공급자일 때만 정해진 횟수까지 다시 시도하고, 연속 N번이면 공급자를 끈다
- 두 키가 다 있으면 Serper → Tavily 순서와 "오류면 바로 다음 공급자" 동작은 그대로다
"""
from __future__ import annotations

import json

import pytest
import requests

import tools.search_providers as providers
import tools.web_search as ws
from tools.sources import SourceRegistry


@pytest.fixture
def live(monkeypatch, set_cfg, tmp_path):
    """실시간 검색 상태: 임시 캐시 폴더, 재현 모드 해제, 검색 상태 초기화, .env 의 실제 키는 읽지 않음."""
    monkeypatch.setattr("core.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("REPLAY_OFFLINE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    set_cfg("cache.dir", str(tmp_path))
    monkeypatch.setattr(ws.time, "sleep", lambda s: None)
    monkeypatch.setattr(providers.requests, "post", _no_network)  # 가짜로 바꾸지 않은 호출이 실제 API 로 나가지 않게
    monkeypatch.setattr(ws, "DISABLED", {})
    monkeypatch.setattr(ws, "_STREAK", {})
    monkeypatch.setattr(ws, "_LIVE", {"n": 0, "sec": 0.0})
    n0 = len(ws.FAILED_QUERIES)
    yield tmp_path
    del ws.FAILED_QUERIES[n0:]


def _no_network(*a, **k):
    raise AssertionError("테스트에서 실제 검색 API 를 부르려 함")


def _fake_tavily(calls: list, error: Exception | None = None, results: list | None = None):
    def fake(query, *args):
        calls.append(query)
        if error:
            raise error
        return results if results is not None else [{"url": f"https://news.example.com/{len(calls)}", "title": query,
                                                     "content": "본문"}]
    return fake


def test_cached_search_calls_no_provider(live, monkeypatch):
    key = {"q": "팜랩 투자 유치", "topic": "news", "tr": None, "n": 5, "dom": [], "depth": "basic", "raw": False}
    ws._cache_file(key).write_text(json.dumps([{"url": "https://a.com/1", "title": "t", "content": "c"}]), "utf-8")
    calls: list = []
    monkeypatch.setattr(providers, "tavily", _fake_tavily(calls))
    assert ws._raw_search("팜랩 투자 유치", "news", None, 5, None) == [{"url": "https://a.com/1", "title": "t",
                                                                   "content": "c"}]
    assert calls == []


def test_tavily_request_body_and_timeout(live, monkeypatch):
    sent = {}

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"results": [{"url": "https://a.com/1"}]}

    def fake_post(url, json=None, timeout=None, headers=None):
        sent.update(url=url, body=json, timeout=timeout)
        return Resp()

    monkeypatch.setattr(providers.requests, "post", fake_post)
    got = ws._call("tavily", "q1", "news", "year", 5, None, "advanced", True)
    assert got == [{"url": "https://a.com/1"}]
    body = sent["body"]
    # langchain_tavily 0.2.18 TavilySearch 가 보내던 본문과 같은 키·값 (실측)
    assert {k: v for k, v in body.items() if k != "exclude_domains"} == {
        "query": "q1", "max_results": 5, "search_depth": "advanced", "include_domains": [],
        "include_raw_content": "text", "include_images": False, "topic": "news", "time_range": "year"}
    assert body["exclude_domains"] == list(ws.get_config().search.exclude_domains)
    assert sent["timeout"][1] == ws.get_config().search.timeout_sec
    ws._call("tavily", "q2", "general", None, 5, ["thevc.kr"], "basic", False)
    assert sent["body"]["include_domains"] == ["thevc.kr"] and sent["body"]["exclude_domains"] == []
    assert "include_raw_content" not in sent["body"] and "time_range" not in sent["body"]


@pytest.mark.parametrize("status,fatal,retryable", [(401, True, False), (432, True, False), (433, True, False),
                                                    (429, False, True), (500, False, True), (400, False, False)])
def test_tavily_http_errors_are_classified(live, monkeypatch, status, fatal, retryable):
    class Resp:
        status_code = status
        text = "error"
        headers: dict = {}

        @staticmethod
        def json():
            return {"detail": {"error": "limit"}}

    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: Resp())
    with pytest.raises(providers.ProviderError) as e:
        ws._call("tavily", "q", "news", None, 5, None, "basic", False)
    assert (e.value.fatal, e.value.retryable, e.value.status) == (fatal, retryable, status)


def test_tavily_timeout_is_not_retried(live, monkeypatch):
    calls = []

    def fake_post(*a, **k):
        calls.append(k["timeout"])
        raise requests.ReadTimeout("read timed out")

    monkeypatch.setattr(providers.requests, "post", fake_post)
    assert ws._raw_search("응답 없는 검색", "general", None, 5, None) is None
    assert len(calls) == 1  # 시간 초과는 이미 timeout 만큼 기다렸으므로 다시 부르지 않는다


def test_quota_error_disables_provider_and_keeps_fail_closed(live, monkeypatch):
    calls: list = []
    monkeypatch.setattr(providers, "tavily",
                        _fake_tavily(calls, providers.ProviderError("tavily", "Error 432: limit", 432)))
    reg = SourceRegistry()
    assert ws.web_search("A사 투자 유치", reg, "eligibility", topic="news", recent=True) == []
    assert calls == ["A사 투자 유치"]  # 다시 시도하지 않고, 기간을 넓힌 두 번째 검색은 공급자를 부르지 않는다
    assert "tavily" in ws.DISABLED
    assert ws.web_search("B사 투자 유치", reg, "eligibility", topic="general", recent=False) == []
    assert calls == ["A사 투자 유치"]
    failed = [f["query"] for f in ws.FAILED_QUERIES if f["agent"] == "eligibility"]
    assert failed[-2:] == ["A사 투자 유치", "B사 투자 유치"]  # 결과 없음이 아니라 실패로 기록 → 관문 fail-closed
    assert len(list(live.glob("search/*.failed"))) == 3  # 지금처럼 실패 표시를 남겨 --retry-failed 로 다시 할 수 있다


def test_transient_errors_retry_then_trip_breaker(live, monkeypatch, set_cfg):
    set_cfg("search.retries", 1)
    set_cfg("search.breaker_after", 3)
    calls: list = []
    monkeypatch.setattr(providers, "tavily", _fake_tavily(calls, providers.ProviderError("tavily", "ConnectionError")))
    for i in range(3):
        assert ws._raw_search(f"q{i}", "general", None, 5, None) is None
    assert len(calls) == 6 and "tavily" in ws.DISABLED  # 검색마다 1번 재시도, 연속 3번 실패 뒤 끔
    assert ws._raw_search("q3", "general", None, 5, None) is None
    assert len(calls) == 6


def test_rate_limit_waits_but_never_disables(live, monkeypatch, set_cfg):
    set_cfg("search.retries", 1)
    waits, calls = [], []
    monkeypatch.setattr(ws.time, "sleep", waits.append)
    monkeypatch.setattr(providers, "tavily", _fake_tavily(
        calls, providers.ProviderError("tavily", "Error 429: slow down", 429, retry_after=3.0)))
    for i in range(5):
        assert ws._raw_search(f"q{i}", "general", None, 5, None) is None
    assert len(calls) == 10 and waits == [3.0] * 5 and "tavily" not in ws.DISABLED


def test_success_resets_error_streak(live, monkeypatch, set_cfg):
    set_cfg("search.retries", 0)
    state = {"fail": True}
    calls: list = []

    def flaky(query, *args):
        calls.append(query)
        if state["fail"]:
            raise providers.ProviderError("tavily", "ConnectionError")
        return [{"url": "https://a.com/x", "title": "t", "content": "c"}]

    monkeypatch.setattr(providers, "tavily", flaky)
    ws._raw_search("q0", "general", None, 5, None)
    ws._raw_search("q1", "general", None, 5, None)
    state["fail"] = False
    assert ws._raw_search("q2", "general", None, 5, None)
    state["fail"] = True
    ws._raw_search("q3", "general", None, 5, None)
    ws._raw_search("q4", "general", None, 5, None)
    assert "tavily" not in ws.DISABLED  # 중간 성공으로 연속 오류 수가 0 으로 돌아갔다


def test_both_keys_keep_order_and_fall_through_without_retry(live, monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "serper-test")
    assert ws.active_providers() == ["serper", "tavily"]
    order: list = []

    def fake_serper(*a, **k):
        order.append("serper")
        raise providers.ProviderError("serper", "Error 500: x", 500)

    monkeypatch.setattr(providers, "serper", fake_serper)
    monkeypatch.setattr(providers, "tavily", lambda *a: order.append("tavily") or [{"url": "https://a.com/t"}])
    got = ws._raw_search("q", "general", None, 5, None)
    assert order == ["serper", "tavily"]  # 다음 공급자가 있으면 재시도 없이 바로 넘어간다 (예전과 같음)
    assert got == [{"url": "https://a.com/t", "provider": "tavily"}]


def test_startup_notice_warns_when_serper_missing(live):
    """--fresh 에서 SERPER_API_KEY 가 없으면 시작 줄 아래에 'Tavily 만 씀' 경고와 요청 제한 시간을 알린다."""
    import argparse

    import app

    cfg = ws.get_config()
    fresh = app.search_notice(argparse.Namespace(fresh=True, retry_failed=False), cfg)
    assert "Tavily" in fresh and "SERPER_API_KEY" in fresh and f"{cfg.search.timeout_sec}초" in fresh
    replay = app.search_notice(argparse.Namespace(fresh=False, retry_failed=False), cfg)
    assert replay.startswith("   검색: 재현용 캐시") and "SERPER_API_KEY" not in replay  # 재현 실행은 키와 무관


def test_startup_notice_without_warning_when_both_keys(live, monkeypatch):
    import argparse

    import app

    monkeypatch.setenv("SERPER_API_KEY", "serper-test")
    msg = app.search_notice(argparse.Namespace(fresh=True, retry_failed=False), ws.get_config())
    assert "Serper → Tavily" in msg and "SERPER_API_KEY 가 없어" not in msg
