"""최근성 규칙(core/recency.py) 검사: 평가 기준일보다 뒤 날짜(예정·계획)는 '최근'이 아니다.
판정(D1·F3·R1·R2)과 탐색 관문 보정(G2)이 같은 규칙을 쓰는지 본다. LLM·네트워크 없이 돈다."""
from __future__ import annotations

import pytest

import core.judge as judge
from core.judge import Answer, Answers, judge_dimension
from core.recency import (future_round, in_window, is_future, only_future_events, planned_only, round_planned,
                          text_dates)
from graph.discovery_graph import build_discovery_graph, recency_guard
from tools.sources import SourceRegistry

RUN = "2026-09-30"


@pytest.mark.parametrize("when, future", [
    ("2026-10", True), ("2027", True), ("2027-03", True), ("2026년 12월", True),
    ("2026-09", False),   # 기준일과 같은 달은 이전으로 본다 (일자를 모름)
    ("2026", False),      # 연도만: 기준 연도와 같으면 이전
    ("2024-09", False), ("", False), ("미상", False),
])
def test_is_future_uses_written_precision(when, future):
    assert is_future(when, RUN) is future


def test_window_is_past_24_months_only():
    assert in_window(2024, 9, RUN) and in_window(2026, 9, RUN) and in_window(2024, None, RUN)
    assert not in_window(2024, 8, RUN) and not in_window(2026, 10, RUN) and not in_window(2027, None, RUN)


def test_text_rules():
    assert only_future_events("팜랩은 2027년 3월까지 500개 농가에 설치할 예정이다", RUN)
    assert only_future_events("2023년 5월 시드 유치, 2027년 3월 출시 예정", RUN)        # 오래된 날짜 + 미래 날짜뿐
    assert not only_future_events("2026년 4월 시드 유치, 2027년 3월 시리즈A 예정", RUN)  # 창 안 날짜가 있으면 판단하지 않음
    assert not only_future_events("글로벌 시장은 2023년 174억 달러에서 2034년 1,172억 달러로", RUN)  # 금액을 달로 읽지 않음
    assert not only_future_events("팜랩은 2027년형 제어기를 출시했다", RUN)                        # 연식 표기는 사건 날짜 아님
    assert planned_only("팜랩은 올해 매출 50억 원을 목표로 한다", RUN)
    assert not planned_only("전국 12개 농가에서 운영 중이며 내년까지 100곳으로 늘릴 계획이다", RUN)  # 완료 사실이 함께 있음
    assert not planned_only("Upside Robotics plant the seeds for success", RUN)                 # 'plant' 오탐 없음


def test_text_rules_boundaries():
    # 날짜 없는 완료 사실 + 미래 계획: 완료 사실의 최근성은 근거 게시일로 본다. 오래된 날짜가 함께 있으면 최근 사건 아님
    assert not only_future_events("팜랩은 12개 농가에 설치했고 2027년까지 500곳으로 늘릴 계획이다", RUN)
    assert only_future_events("팜랩은 2023년 5월 12개 농가에 설치했고 2027년까지 500곳으로 늘릴 계획이다", RUN)
    assert not only_future_events("팜랩은 2026년 9월 12개 농가에 설치했다", RUN)       # 기준일과 같은 달 = 창 안
    assert only_future_events("팜랩은 2026년 10월 12개 농가에 설치할 예정이다", RUN)   # 기준일 다음 달
    # 영어 기사(해외 후보)
    assert only_future_events("Farm Lab plans to deploy 1,000 robots by 2027", RUN)
    assert only_future_events("Farm Lab will open its first US farm in March 2027", RUN)
    assert not only_future_events("Farm Lab serves 200 farms and expects the market to reach $1B in 2032", RUN)
    assert text_dates("the smart farming market 2027 outlook") == []                     # market 을 March 로 읽지 않음
    assert text_dates("in March 2027") == [(2027, 3)]                                    # 같은 날짜를 두 번 세지 않음
    # 기능 이름 속 '예정·계획·목표'는 계획 문장이 아니다. '하기로 했다'·'추진하고 있다'는 완료가 아니다
    assert not planned_only("목표 온도를 자동으로 맞추는 제어기를 200개 농가에 공급", RUN)
    assert not planned_only("출하 예정일 예측 서비스를 300개 농가가 유료로 이용", RUN)
    assert not planned_only("작물별 재배 계획을 짜 주는 AI 를 120개 농가에 제공", RUN)
    assert planned_only("팜랩은 내년까지 500개 농가에 제어기를 설치하기로 했다", RUN)
    assert planned_only("팜랩은 해외 500개 농가 설치를 추진하고 있다. 연내 출시 예정", RUN)
    assert planned_only("Farm Lab aims to reach 1,000 farms", RUN)
    assert not planned_only("팜랩은 2025년 12월 CES 혁신상을 수상했으며 내년 미국 진출 계획이다", RUN)


def test_round_planned_is_about_the_round_itself():
    assert round_planned("팜랩은 내년 시리즈A 투자 유치를 추진하고 있다")
    assert round_planned("Farm Lab plans to raise a Series A next year")
    assert not round_planned("퓨처커넥트, 86억 원 규모 시리즈A 투자 유치… 북미 시장 공략 본격화")
    assert not round_planned("Farm Lab raised $5M seed and will use the funds to expand")     # 투자금 사용 계획은 아님
    # 끝난 라운드와 다음 라운드 계획이 함께 적힌 인용은 끝난 라운드를 말한다
    assert not round_planned("팜랩은 20억 원 규모 시드 투자를 유치했으며, 내년 시리즈A 유치를 준비하고 있다")
    assert not round_planned("The company is raising awareness of soil health")          # 'is raising' 오탐 없음
    assert round_planned("Farm Lab is raising a Series A")
    assert future_round({"round_date": "2027-01", "stage_quote": "x"}, RUN)
    assert future_round({"round_date": "2026.10", "stage_quote": "x"}, RUN)                # 점 표기도 같은 규칙
    assert future_round({"round_date": "2026-09", "stage_quote": "팜랩, 시드 투자 유치"}, RUN) is None  # 기준일과 같은 달
    assert future_round({"round_date": "2025-08", "stage_quote": "메타파머스, 30억 원 규모 프리A 투자 유치"}, RUN) is None


# ── 판정 (core/judge.py)
CO = {"name": "팜랩", "official_name": "(주)팜랩", "name_en": "Farm Lab", "stage": "Series A", "round_date": "2026-03",
      "round_amount": "30억 원", "stage_evidence_ids": ["W1"], "stage_quote": "팜랩은 30억 원 규모의 시리즈A 투자를 유치했다"}


@pytest.mark.parametrize("rd, quote, answer", [
    ("2026-03", CO["stage_quote"], "YES"),
    ("2026-12", CO["stage_quote"], "UNKNOWN"),      # 기준일 이후 라운드 (음수 개월이 '24개월 안'으로 통과하던 것)
    ("2027", CO["stage_quote"], "UNKNOWN"),
    ("2026-05", "팜랩은 30억 원 규모 시리즈A 유치를 추진 중이다", "UNKNOWN"),  # 예정 라운드
    ("2023-01", CO["stage_quote"], "NO"),
    ("2026-09", CO["stage_quote"], "YES"),         # 기준일과 같은 달
    ("2024-09", CO["stage_quote"], "YES"),         # 정확히 24개월 전
    ("2024-08", CO["stage_quote"], "NO"),          # 25개월 전
    ("2026.10", CO["stage_quote"], "UNKNOWN"),     # 점 표기의 기준일 다음 달 (D1 정규식은 월을 못 읽어 6월로 보지만 미래 검사가 먼저)
    ("2026-05", "팜랩은 시드 투자를 유치했으며 시리즈A 유치를 준비 중이다", "YES"),  # 끝난 라운드 + 다음 계획
])
def test_d1_round_rule_future_is_unknown(rd, quote, answer):
    verdict, ids, why = judge._round_rule({**CO, "round_date": rd, "stage_quote": quote}, RUN)
    assert verdict == answer and (ids == [] if answer == "UNKNOWN" else ids == ["W1"])
    assert "코드 판정" in why


def test_doc_published_after_run_year_is_not_recent():
    assert judge._is_recent({"kind": "doc", "id": "D1", "doc_id": "없는문서", "year": 2026}, RUN)
    assert not judge._is_recent({"kind": "doc", "id": "D1", "doc_id": "없는문서", "year": 2027}, RUN)
    assert judge._is_recent({"kind": "doc", "id": "D1", "doc_id": "없는문서", "year": 2024}, RUN)
    assert not judge._is_recent({"kind": "doc", "id": "D1", "doc_id": "없는문서", "year": 2023}, RUN)


@pytest.mark.parametrize("posted, recent", [
    ("2026-09-30", True), ("2026-10-01", False),    # 기준일 당일은 최근, 다음 날 게시일은 최근 아님
    ("2024-09-30", True), ("2024-09-29", False),    # 730일 경계
])
def test_web_posted_after_run_date_is_not_recent(posted, recent):
    s = {"kind": "web", "id": "W1", "site": "예시신문", "url": "https://news.example.com/1", "date": posted}
    assert judge._is_recent(s, RUN) is recent


class _Fake:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, prompt):
        return Answers(answers=[Answer(**a) for a in self.replies.pop(0)])


def _a(qid, verdict, quote="", ids=("W1",)):
    return {"qid": qid, "verdict": verdict, "quote": quote, "evidence_ids": list(ids), "rationale": "근거"}


PLAN = "팜랩은 2027년 3월까지 전국 500개 농가에 제어기를 설치할 예정이다"
DONE = "팜랩은 2026년 5월 전국 12개 농가에 제어기를 설치했다"
REG = {"W1": {"id": "W1", "kind": "web", "url": "https://news.example.com/1", "site": "예시신문", "date": "2026-08-01",
              "title": "팜랩", "snippet": "", "body": f"{PLAN} {DONE}"}}


@pytest.mark.parametrize("dim, qid, quote, answer", [
    ("traction", "R2", PLAN, "UNKNOWN"), ("traction", "R2", DONE, "YES"),
    ("founder", "F3", PLAN, "UNKNOWN"), ("founder", "F3", DONE, "YES"),
])
def test_recent_questions_reject_future_events(monkeypatch, dim, qid, quote, answer):
    qids = {"traction": ("R1", "R2", "R3", "R4"), "founder": ("F1", "F2", "F3", "F4")}[dim]
    replies = [[_a(q, "YES", quote) if q == qid else _a(q, "UNKNOWN", ids=()) for q in qids]]
    monkeypatch.setattr(judge, "bounded", lambda schema, role="generator", kind="default": _Fake(replies))
    res = judge_dimension(dim, CO, ["W1"], SourceRegistry(REG), "[후보] 팜랩", RUN)
    row = next(r for r in res["rows"] if r["qid"] == qid)
    assert row["answer"] == answer
    if answer == "UNKNOWN":
        assert "평가 기준일 이후" in row["rationale"] and res["rejected_yes"][0]["qid"] == qid


def _judge_one(monkeypatch, dim, qid, quote, reg):
    qids = {"traction": ("R1", "R2", "R3", "R4"), "market": ("M1", "M2", "M3", "M4")}[dim]
    replies = [[_a(q, "YES", quote) if q == qid else _a(q, "UNKNOWN", ids=()) for q in qids]]
    monkeypatch.setattr(judge, "bounded", lambda schema, role="generator", kind="default": _Fake(replies))
    res = judge_dimension(dim, CO, ["W1"], SourceRegistry(reg), "[후보] 팜랩", RUN)
    return next(r for r in res["rows"] if r["qid"] == qid)


def test_recent_question_rejects_evidence_posted_after_run_date(monkeypatch):
    late = {"W1": {**REG["W1"], "date": "2026-10-02"}}
    row = _judge_one(monkeypatch, "traction", "R2", DONE, late)
    assert row["answer"] == "UNKNOWN" and "최근 24개월 이내 근거 아님" in row["rationale"]


def test_undated_planned_sentence_is_not_a_recent_event(monkeypatch):
    plan = "팜랩은 연내 전국 500개 농가에 제어기를 설치할 계획이다"
    row = _judge_one(monkeypatch, "traction", "R2", plan, {"W1": {**REG["W1"], "body": plan}})
    assert row["answer"] == "UNKNOWN" and "예정·계획·목표" in row["rationale"]


def test_market_forecast_keeps_market_level_exemption(monkeypatch):
    fc = "글로벌 스마트 농업 시장은 2023년 174억 달러에서 2034년 1,172억 달러로 연평균 19.1% 성장할 전망이다"
    row = _judge_one(monkeypatch, "market", "M2", fc, {"W1": {**REG["W1"], "body": fc}})
    assert row["answer"] == "YES"                    # 발행 시점 기준의 시장 전망은 미래 연도가 있어도 인정


# ── 탐색 관문 보정 (graph/discovery_graph.py recency_guard)
def test_screen_drops_future_round_candidates():
    recs = [{"name": "A", "eligible": True, "round_date": "2026-05", "stage_quote": "A, 시드 투자 유치", "reason": "통과"},
            {"name": "B", "eligible": True, "round_date": "2027-02", "stage_quote": "B, 시리즈A 유치", "reason": "통과"},
            {"name": "C", "eligible": False, "round_date": "2027-02", "stage_quote": "", "reason": "G5 AgTech 아님"}]

    def collect(state):
        return {"raw_candidates": [{"name": r["name"]} for r in recs], "discovery_rounds": 1, "seen": [],
                "log": ["[발굴]"]}

    def screen(state):
        passed = [{"official_name": r["name"], "name": r["name"], "region": "KR", "stage": "Seed", "segment_id": "x"}
                  for r in recs if r["eligible"]]
        return {"screened": [dict(r) for r in recs], "queue": list(state.get("queue", [])) + passed, "log": ["[관문]"]}

    out = build_discovery_graph(collect, screen).invoke({"run_date": RUN, "discovery_rounds": 0, "iterations": 0,
                                                         "queue": []})
    by = {r["name"]: r for r in out["screened"]}
    assert by["A"]["eligible"] and not by["B"]["eligible"] and by["B"]["reason"].startswith("G2 최근 라운드 시점(2027-02)")
    assert by["C"]["reason"] == "G5 AgTech 아님"          # 이미 탈락한 후보는 그대로
    assert out["current"]["official_name"] == "A" and out["queue"] == []
    assert any(line.startswith("[적격성 보정]") for line in out["log"])


def test_screen_guard_boundaries():
    recs = [{"name": "A", "eligible": True, "round_date": "2026-09", "stage_quote": "A, 시드 투자 유치", "reason": "통과"},
            {"name": "B", "eligible": True, "round_date": "2026-06", "stage_quote": "B는 시리즈A 투자 유치를 추진 중",
             "reason": "통과"}]
    old = [{"name": "Z"}]

    def screen(state):
        return {"screened": recs, "queue": [*state["queue"], *({"name": r["name"]} for r in recs)], "log": ["[관문]"]}

    out = recency_guard(screen)({"run_date": RUN, "queue": old})
    by = {r["name"]: r for r in out["screened"]}
    assert by["A"]["eligible"]                                                  # 기준일과 같은 달 라운드는 통과
    assert not by["B"]["eligible"] and by["B"]["reason"].startswith("G2 단계 인용이 예정·추진 중인 라운드")
    assert out["queue"] == [{"name": "Z"}, {"name": "A"}]                     # 기존 대기열은 그대로, 새 B 만 뺀다
    assert recs[1]["eligible"]                                                  # 관문 함수의 기록 자체는 바꾸지 않음

    ok = {"screened": [recs[0]], "queue": [{"name": "A"}], "log": ["[관문]"]}
    assert recency_guard(lambda state: ok)({"run_date": RUN, "queue": []}) is ok  # 걸리는 후보가 없으면 출력이 그대로
