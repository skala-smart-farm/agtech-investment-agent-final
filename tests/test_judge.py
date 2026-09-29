"""core.judge.judge_dimension 검사 (계약 C3). LLM 은 가짜로 바꾸고, 코드 강등 규칙이 v1 과 같게 동작하는지 본다.
네트워크·API 키 없이 돈다 (Kiwi 형태소 분석기·BM25 는 로컬)."""
from __future__ import annotations

import pytest

import core.judge as judge
from core.judge import Answer, Answers, judge_dimension
from tools.sources import SourceRegistry

RUN_DATE = "2026-09-30"
ROW_KEYS = {"dim", "qid", "short", "point", "question", "bessemer_q", "answer", "x", "evidence_ids", "quote", "rationale"}
DIM_KEYS = {"dim", "name", "weight", "owner", "rows", "yes", "no", "unknown", "na", "n", "rejected_yes", "quote_retried"}

COMPANY = {"name": "팜랩", "official_name": "(주)팜랩", "name_en": "Farm Lab", "stage": "Seed", "round_date": "2026-03",
           "round_amount": "30억 원", "stage_evidence_ids": ["W1"], "stage_quote": "팜랩은 30억 원 규모의 시드 투자를 유치했다"}
FAR = "가나다라마바사아자차카타파하" * 30   # 회사명과 인용 사이를 300자 넘게 벌리는 채움 글자

SOURCES = {
    # 제3자 기사 (최근)
    "W1": {"kind": "web", "url": "https://news.example.com/a/1", "site": "예시신문", "date": "2026-05-01",
           "title": "팜랩, 시드 투자 유치",
           "body": "팜랩은 30억 원 규모의 시드 투자를 유치했다. 팜랩은 전국 12개 농가에 스마트팜 제어기를 설치해 운영 중이다. "
                   "팜랩은 자체 개발한 생육 예측 모델로 특허 3건을 등록했다. "
                   "팜랩은 농협과 업무협약(MOU)만 맺었을 뿐 공급 계약은 없다. "
                   "팜랩 제어기를 쓴 농가의 난방비가 20% 줄었다고 김철수 대표는 말했다."},
    # 인용 앞뒤 300자 밖에만 회사명이 있는 기사 (업계 일반론)
    "W2": {"kind": "web", "url": "https://news.example.com/a/2", "site": "예시신문", "date": "2026-06-01",
           "title": "스마트팜 업계 동향",
           "body": f"팜랩 등 여러 기업이 참가했다. {FAR} 업계 전반적으로 농업용 로봇은 아직 시범 운영 단계에 머물러 있다. "
                   "데이터가 쌓일수록 제어 정확도가 높아지는 구조가 일반적이다."},
    # 예정·실증 표현
    "W3": {"kind": "web", "url": "https://news.example.com/a/3", "site": "예시신문", "date": "2026-07-01",
           "title": "팜랩, 로봇 실증",
           "body": "팜랩은 내년 상용화를 목표로 10개 농가에서 수확 로봇 시범 운영을 하고 있다."},
    # 회사 자체 홈페이지
    "W4": {"kind": "web", "url": "https://www.farmlab.co.kr/about", "site": "팜랩", "date": "2026-04-01",
           "title": "회사 소개", "body": "팜랩의 제어기는 전국 농가에서 수확량을 15% 늘렸다."},
    # 오래된 기사
    "W5": {"kind": "web", "url": "https://news.example.com/a/5", "site": "예시신문", "date": "2020-01-10",
           "title": "팜랩 매출", "body": "팜랩은 2019년 매출 5억 원을 기록했다."},
}


def _reg() -> SourceRegistry:
    return SourceRegistry({k: {"id": k, "snippet": "", **v} for k, v in SOURCES.items()})


class FakeJudge:
    """structured(Answers, 'judge') 대신 쓰는 가짜 LLM. 받은 프롬프트를 기록하고 준비한 답을 차례로 돌려준다."""

    def __init__(self, replies: list[list[dict]]):
        self.replies, self.prompts = list(replies), []

    def invoke(self, prompt: str) -> Answers:
        self.prompts.append(prompt)
        return Answers(answers=[Answer(**a) for a in self.replies.pop(0)])


def _a(qid: str, verdict: str, quote: str = "", ids: tuple = ("W1",), why: str = "근거 문장이 있다") -> dict:
    return {"qid": qid, "verdict": verdict, "quote": quote, "evidence_ids": list(ids), "rationale": why}


@pytest.fixture
def fake(monkeypatch):
    def install(replies):
        f = FakeJudge(replies)
        monkeypatch.setattr(judge, "bounded", lambda schema, role="generator", kind="default": f)
        return f
    return install


def _run(dim: str, company: dict = COMPANY, pool: tuple = ("W1", "W2", "W3", "W4", "W5")):
    res = judge_dimension(dim, company, list(pool), _reg(), "[후보] (주)팜랩 분석 요약", RUN_DATE)
    return res, {r["qid"]: r for r in res["rows"]}


def test_contract_keys_and_quote_rules(fake):
    """반환 키 = 계약 C3. 본문에 없는 인용의 YES → 재질문 1회 뒤 UNKNOWN, 상용 문항의 '예정' 인용 → UNKNOWN."""
    bad = "팜랩 로봇은 수확량을 40% 늘렸다고 농촌진흥청이 발표했다"
    f = fake([
        [_a("P1", "YES", "팜랩은 내년 상용화를 목표로 10개 농가에서 수확 로봇 시범 운영을 하고 있다", ("W3",)),
         _a("P2", "YES", bad), _a("P3", "YES", "팜랩은 자체 개발한 생육 예측 모델로 특허 3건을 등록했다"),
         _a("P4", "N/A", why="인허가 대상 제품이 아님")],
        [_a("P2", "YES", bad)],                                   # 인용 재질문에도 같은 인용
    ])
    res, rows = _run("product")
    assert set(res) == DIM_KEYS and all(set(r) == ROW_KEYS for r in res["rows"])
    assert (res["dim"], res["owner"], res["weight"]) == ("product", "tech", 15)
    assert [r["qid"] for r in res["rows"]] == ["P1", "P2", "P3", "P4"]
    assert rows["P1"]["answer"] == "UNKNOWN" and "실증·예정" in rows["P1"]["rationale"]
    assert rows["P2"]["answer"] == "UNKNOWN" and "확인되지 않음" in rows["P2"]["rationale"]
    assert rows["P3"]["answer"] == "YES" and rows["P3"]["x"] == 1 and rows["P3"]["evidence_ids"] == ["W1"]
    assert rows["P4"]["answer"] == "N/A" and rows["P4"]["x"] is None
    assert (res["yes"], res["no"], res["unknown"], res["na"], res["n"]) == (1, 0, 2, 1, 3)
    assert res["quote_retried"] == 1 and len(f.prompts) == 2 and "직전 인용" in f.prompts[1] and "P2:" in f.prompts[1]
    assert {x["qid"] for x in res["rejected_yes"]} == {"P1", "P2"}
    assert rows["P3"]["short"] == "자체 기술" and rows["P3"]["bessemer_q"] == [4] and rows["P3"]["point"] == "독창성"


def test_prompt_is_v1_template(fake):
    """프롬프트는 v1 prompts/decision.md 그대로: 문항 줄 형식 '- id: text (YES 요건: need)'와 [분석 요약]·[근거]."""
    f = fake([[_a(q, "UNKNOWN", ids=()) for q in ("C1", "C2", "C3", "C4")]])
    _run("competition")
    p = f.prompts[0]
    assert '"경쟁 우위" 항목의 질문에만 판정하라' in p and "평가 기준일: 2026-09-30" in p
    assert "- C1: 핵심 기술 관련 특허(등록 또는 출원)가 1건 이상 확인되는가?" in p and "(YES 요건: 특허 근거)" in p
    assert "[분석 요약]\n[후보] (주)팜랩 분석 요약" in p and "[W1] (예시신문, 2026-05-01)" in p


def test_company_window_and_no_guards(fake):
    """회사명이 인용 ±300자 밖이면 YES·NO 모두 UNKNOWN. NO 는 근거 부족 이유(부정 표현 없는 인용)면 UNKNOWN, 부정 인용이면 유지."""
    fake([[
        _a("C1", "NO", "팜랩은 자체 개발한 생육 예측 모델로 특허 3건을 등록했다", why="특허 등록 여부가 확인되지 않아 추정"),
        _a("C2", "NO", "업계 전반적으로 농업용 로봇은 아직 시범 운영 단계에 머물러 있다", ("W2",), "업계가 시범 단계"),
        _a("C3", "YES", "데이터가 쌓일수록 제어 정확도가 높아지는 구조가 일반적이다", ("W2",)),
        _a("C4", "NO", "팜랩은 농협과 업무협약(MOU)만 맺었을 뿐 공급 계약은 없다", why="MOU 만 있고 계약이 없어 NO"),
    ]])
    _, rows = _run("competition")
    assert rows["C1"]["answer"] == "UNKNOWN" and "근거 부족" in rows["C1"]["rationale"]
    assert rows["C2"]["answer"] == "UNKNOWN" and "회사명 없음" in rows["C2"]["rationale"]
    assert rows["C3"]["answer"] == "UNKNOWN" and "±300자" in rows["C3"]["rationale"]
    assert rows["C4"]["answer"] == "NO" and rows["C4"]["x"] == -1 and rows["C4"]["quote"]


def test_market_level_skips_company_window(fake):
    """시장 문항(M2)은 회사 이야기가 아니어도 된다. M1 은 문서 근거([D…])가 없으면 UNKNOWN."""
    fake([[
        _a("M1", "YES", "업계 전반적으로 농업용 로봇은 아직 시범 운영 단계에 머물러 있다", ("W2",)),
        _a("M2", "YES", "데이터가 쌓일수록 제어 정확도가 높아지는 구조가 일반적이다", ("W2",)),
        _a("M3", "UNKNOWN", ids=()), _a("M4", "UNKNOWN", ids=()),
    ]])
    _, rows = _run("market")
    assert rows["M1"]["answer"] == "UNKNOWN" and "문서 근거 없음" in rows["M1"]["rationale"]
    assert rows["M2"]["answer"] == "YES"


def test_third_party_speaker_and_own_domain(fake):
    """제3자 문항: 대표 발언 인용 → UNKNOWN, 회사 도메인 근거만 → UNKNOWN."""
    fake([[
        _a("F1", "UNKNOWN", ids=()), _a("F2", "UNKNOWN", ids=()), _a("F3", "UNKNOWN", ids=()),
        _a("F4", "YES", "팜랩 제어기를 쓴 농가의 난방비가 20% 줄었다"),
    ]])
    _, rows = _run("founder")
    assert rows["F4"]["answer"] == "UNKNOWN" and "회사 측 발언" in rows["F4"]["rationale"]

    fake([[
        _a("M1", "UNKNOWN", ids=()), _a("M2", "UNKNOWN", ids=()),
        _a("M3", "YES", "팜랩의 제어기는 전국 농가에서 수확량을 15% 늘렸다", ("W4",)), _a("M4", "UNKNOWN", ids=()),
    ]])
    _, rows = _run("market")
    assert rows["M3"]["answer"] == "UNKNOWN" and "회사 자체 발표만" in rows["M3"]["rationale"]


def test_recency_and_missing_reask(fake):
    """최근 문항(R1): 24개월보다 오래된 근거 → UNKNOWN. 판정을 빠뜨린 문항(R4)만 한 번 더 묻는다."""
    f = fake([
        [_a("R1", "YES", "팜랩은 2019년 매출 5억 원을 기록했다", ("W5",)),
         _a("R2", "YES", "팜랩은 전국 12개 농가에 스마트팜 제어기를 설치해 운영 중이다"), _a("R3", "UNKNOWN", ids=())],
        [_a("R4", "UNKNOWN", ids=())],
    ])
    res, rows = _run("traction")
    assert rows["R1"]["answer"] == "UNKNOWN" and "24개월" in rows["R1"]["rationale"]
    assert rows["R2"]["answer"] == "YES"
    assert len(f.prompts) == 2 and "- R4:" in f.prompts[1] and "- R1:" not in f.prompts[1]
    assert res["owner"] == "decision"


def test_doc_recency_uses_publication_year():
    """문서 근거는 발행 연도로 최근 여부를 본다 (기준 연도 − 2 이상)."""
    assert judge._is_recent({"kind": "doc", "id": "D1", "doc_id": "없는문서", "year": 2024}, RUN_DATE)
    assert not judge._is_recent({"kind": "doc", "id": "D1", "doc_id": "없는문서", "year": 2019}, RUN_DATE)


def test_d1_is_code_rule(fake):
    """D1 은 LLM 답(NO·본문에 없는 인용)을 무시하고 적격성 관문 값으로 코드가 판정한다 (재질문도 하지 않음)."""
    f = fake([[_a("D1", "NO", "본문에 없는 문장입니다 아무것도"), _a("D2", "UNKNOWN", ids=()), _a("D3", "UNKNOWN", ids=()),
               _a("D4", "UNKNOWN", ids=())]])
    res, rows = _run("deal")
    assert rows["D1"]["answer"] == "YES" and "코드 판정" in rows["D1"]["rationale"]
    assert rows["D1"]["quote"] == COMPANY["stage_quote"] and rows["D1"]["evidence_ids"] == ["W1"]
    assert len(f.prompts) == 1 and res["quote_retried"] == 0

    fake([[_a(q, "UNKNOWN", ids=()) for q in ("D1", "D2", "D3", "D4")]])
    _, rows = _run("deal", {**COMPANY, "round_date": "2023-01"})
    assert rows["D1"]["answer"] == "NO"
    fake([[_a(q, "UNKNOWN", ids=()) for q in ("D1", "D2", "D3", "D4")]])
    _, rows = _run("deal", {**COMPANY, "round_amount": "비공개"})
    assert rows["D1"]["answer"] == "UNKNOWN"


def test_empty_pool_skips_llm(fake):
    """근거 풀이 비면 LLM 을 부르지 않고 UNKNOWN (D1 은 코드 판정)."""
    f = fake([])
    res, rows = _run("deal", pool=("W없음",))
    assert f.prompts == [] and rows["D1"]["answer"] == "YES"
    assert [rows[q]["answer"] for q in ("D2", "D3", "D4")] == ["UNKNOWN"] * 3


def test_unknown_dimension():
    with pytest.raises(ValueError):
        judge_dimension("team", COMPANY, [], _reg(), "", RUN_DATE)
