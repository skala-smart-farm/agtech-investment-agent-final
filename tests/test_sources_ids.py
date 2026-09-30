"""근거 id 충돌 처리: 다른 출처가 같은 짧은 해시를 받아도 먼저 등록된 근거를 덮어쓰지 않는다."""
from __future__ import annotations

import re

import agents.report as R
import agents.tech as T
import rag.agentic_rag as ar
from tools.sources import SourceRegistry


def _web(url: str, title: str) -> dict:
    return {"url": url, "title": title, "content": title + " 본문", "published_date": "2026-05-01"}


def test_no_collision_ids_unchanged():
    reg = SourceRegistry()
    sid = reg.add_web(_web("https://a.example.com/1", "기사 A"), "market", "q", "2026-09-30")
    assert sid == SourceRegistry._id("W", "web:https://a.example.com/1") and len(sid) == 6
    assert reg.add_web(_web("https://a.example.com/1", "기사 A"), "market", "q", "2026-09-30") == sid  # 같은 키는 같은 id


def test_collision_gets_longer_id_and_keeps_first(monkeypatch):
    # 두 키의 짧은 해시를 강제로 같게 만든다: 첫 번째 키의 id 자리를 다른 키가 이미 차지한 상황
    reg = SourceRegistry()
    first = reg.add_web(_web("https://a.example.com/1", "기사 A"), "market", "q", "2026-09-30")
    reg.data[first]["key"] = "web:https://other.example.com/x"      # 다른 출처가 그 id 를 먼저 쓰고 있었다고 가정
    reg.data[first]["title"] = "먼저 온 출처"
    second = reg.add_web(_web("https://a.example.com/1", "기사 A"), "market", "q", "2026-09-30")
    assert second != first and second.startswith(first) and len(second) == 7
    assert reg.data[first]["title"] == "먼저 온 출처"                  # 덮어쓰지 않음
    assert reg.data[second]["key"] == "web:https://a.example.com/1"
    assert reg.add_web(_web("https://a.example.com/1", "기사 A"), "market", "q", "2026-09-30") == second


def test_doc_collision_same_rule():
    reg = SourceRegistry()
    meta = {"doc_id": "d1", "page": 3, "title": "문서"}
    first = reg.add_doc(meta, "조각 본문", "market", "q")
    reg.data[first]["key"] = "doc:other"
    second = reg.add_doc(meta, "조각 본문", "market", "q")
    assert second != first and second.startswith(first)


def test_citation_regexes_accept_longer_ids():
    for sid in ("W1a2b3", "W1a2b3c", "D0f1e2d3c"):
        assert R.CITE.search(f"문장 [{sid}]") and T.CITE_ID.search(f"문장 {sid} 끝")
        assert ar.CITE.findall(f"문장 [{sid}]") == [sid]
    assert R.CITE.search("[W1a2b3, W1a2b3c]").group(1) == "W1a2b3, W1a2b3c"
    assert re.fullmatch(r"[WD][0-9a-f]{5,8}", "W1a2b3c")
