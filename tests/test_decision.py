"""투자 판단(agents/decision.py) 검사: Payne 상대 배수·결정 규칙·뒤집힘·실사·Bessemer·ROI·민감도·순위·노드 반환 (계약 C7·C10).

기대값은 v1 제출 실행(outputs/run_log.json)의 판정 24문항×10곳으로 다시 계산한 값이다.
픽스처는 eval/build_reference_class.py 로 만들었다 (tests/fixtures/reference_v1.json, rows_v1.json).
LLM·네트워크 없이 돈다 (judge_dimension 은 가짜로 바꾼다).
"""
from __future__ import annotations

import json
import re

import pytest

import agents.decision as dec
from core.config import ROOT, get_config
from core.judge import load_rubric

REF = "tests/fixtures/reference_v1.json"
ROWS = {c["name"]: c for c in json.loads((ROOT / "tests/fixtures/rows_v1.json").read_text(encoding="utf-8"))["companies"]}
QIDS = [q["id"] for d in load_rubric()["dimensions"] for q in d["questions"]]
DIM_OF = {q["id"]: d["id"] for d in load_rubric()["dimensions"] for q in d["questions"]}
C7_EVAL = {"name", "region", "segment_id", "stage", "round_date", "round_amount", "decision", "hold_type", "multiplier",
           "score100", "reasons", "flip", "criteria", "scorecard", "founder", "tech", "market", "competition", "profile"}
C7_SCORECARD = {"criteria", "rows", "multiplier", "score100", "threshold", "reference", "founder_yes", "founder_c",
                "unknown_ratio", "deal_killers", "decision", "hold_type", "reasons", "flip", "dd_items", "bessemer", "roi",
                "sensitivity", "ranking", "target_rank", "peer_n"}


@pytest.fixture(autouse=True)
def _ref(set_cfg):
    set_cfg("decision.reference_file", REF)


def _rows(answers: dict[str, str] | str) -> list[dict]:
    """{qid: 판정} 또는 모든 문항에 같은 판정 → 판정 행."""
    x = {"YES": 1, "NO": -1, "UNKNOWN": 0, "N/A": None}
    get = (lambda q: answers) if isinstance(answers, str) else (lambda q: answers.get(q, "UNKNOWN"))
    return [{"dim": DIM_OF[q], "qid": q, "answer": get(q), "x": x[get(q)]} for q in QIDS]


def _score(rows: list[dict], exclude: str | None = None) -> tuple[float, float, dict]:
    """(배수 M, 창업자 기준 비율 c_founder, 기준 집단). c_founder 는 창업자 기준의 pct / 100 과 같아야 한다."""
    ref = dec.load_reference(exclude=exclude)
    d = get_config().decision
    M, crit = dec.payne_multiplier(rows, ref["mean"], load_rubric(), d.step, d.clip)
    fc = dec._founder_c(crit)
    assert fc == pytest.approx(next(c["pct"] for c in crit if c["dim"] == "founder") / 100)
    return M, fc, ref


def _unknown_ratio(rows: list[dict]) -> float:
    judged = [r for r in rows if r["answer"] != "N/A"]
    return sum(r["answer"] == "UNKNOWN" for r in judged) / len(judged)


def _decide(rows: list[dict], M: float, fc: float):
    return dec.decide_rule(M, fc, dec._killers(rows, load_rubric()), _unknown_ratio(rows), get_config())


# 보류 유형마다 맨 앞 사유 문장에 들어가는 말 (투자면 첫 사유)
FIRST_REASON = {None: "— 동종 평균보다 높음", "창업자 점수 평균 미만": "— 팀이 동종 평균 미만",
                "동종 평균 이하": "— 동종 평균 이하", "Deal-killer": "Deal-killer K"}


# ── 배수 재계산 (v1 판정 + 자기 제외 기준 집단). 투자 ⇔ M > 1.00 ∧ c_founder ≥ 1.00 ∧ Deal-killer 없음
@pytest.mark.parametrize("name, M, founder_c, decision, hold", [
    ("메타파머스", 1.130, 1.292, "투자", None),
    ("퓨처커넥트", 1.095, 1.153, "투자", None),                     # 옛 기준(M ≥ 1.10)에서는 보류였다
    ("에임비랩", 1.015, 1.014, "투자", None),                       # P4 N/A 는 분모에서 빠짐
    ("Nature Robots", 0.935, 0.875, "보류", "창업자 점수 평균 미만"),  # 창업자 4문항 모두 미확인 → 팀이 동종 평균 미만
    ("바르카", 0.956, 1.014, "보류", "동종 평균 이하"),               # 팀은 평균 이상(F1 YES)이지만 M ≤ 1.00
])
def test_v1_recompute(name, M, founder_c, decision, hold):
    rows = ROWS[name]["rows"]
    got, fc, ref = _score(rows, exclude=name)
    assert (ref["n"], ref["loo"], ref["source"]) == (9, True, "calibration") and name not in ref["members"]
    assert got == pytest.approx(M, abs=0.002) and fc == pytest.approx(founder_c, abs=0.002)
    d, h, reasons = _decide(rows, got, fc)
    assert (d, h) == (decision, hold) and FIRST_REASON[h] in reasons[0], reasons
    if d == "보류":
        assert f"미확인 문항 {_unknown_ratio(rows):.0%}" in reasons[-1] and "(동종 평균 100)" in reasons[-1]


def test_peer_inferior_label_on_real_data(set_cfg):
    """팀은 동종 평균 이상인데 M 이 기준 이하면 '동종 평균 이하'. 기준은 초과(>)라 M = 기준이면 보류다.
    시나리오 기준(--threshold 1.30)에서도 사유 문장의 동종 평균은 100 이고 기준 130 은 따로 적는다."""
    rows = ROWS["메타파머스"]["rows"]
    M, fc, _ = _score(rows, "메타파머스")
    assert (M, fc) == (pytest.approx(1.130, abs=0.002), pytest.approx(1.292, abs=0.002))
    set_cfg("decision.threshold", M)                       # 경계: 배수 = 기준 → 보류 (기준 ≠ 1.00 이라 이름표는 '기준 이하')
    assert _decide(rows, M, fc)[:2] == ("보류", "기준 이하")
    set_cfg("decision.threshold", M - 0.0001)              # 기준보다 조금이라도 크면 투자
    assert _decide(rows, M, fc)[:2] == ("투자", None)

    set_cfg("decision.threshold", 1.30)
    d, h, reasons = _decide(rows, M, fc)
    assert (d, h) == ("보류", "기준 이하") and len(reasons) == 1        # 시나리오 기준(≠ 1.00)은 이름표도 기준으로
    assert reasons[0] == "동종 평균 대비 113.0(동종 평균 100, 기준 130) — 기준 130 이하 (미확인 문항 54%)"
    assert not any("동종 평균 130" in r for r in reasons)   # 기준을 '동종 평균'으로 부르지 않는다


def test_all_unknown_and_all_yes():
    """가상 기업(기준 집단 밖, 10곳 평균): 전부 미확인 0.867 보류, 전부 YES 1.367 투자.
    전부 미확인은 창업자 기준 0.888(동종 창업자 문항 평균 신호가 양수라 미확인이면 평균 미만)이라
    보류 유형 우선순위(Deal-killer > 창업자 점수 평균 미만 > 동종 평균 이하)에서 '창업자 점수 평균 미만'."""
    rows = _rows("UNKNOWN")
    M, fc, ref = _score(rows, exclude="가상 기업")
    assert (ref["n"], ref["loo"]) == (10, False)
    assert M == pytest.approx(0.867, abs=0.005) and fc == pytest.approx(0.888, abs=0.002)
    d, h, reasons = _decide(rows, M, fc)
    assert (d, h) == ("보류", "창업자 점수 평균 미만")
    assert reasons == ["창업자 기준 88.8(동종 평균 100) — 팀이 동종 평균 미만",
                       "동종 평균 대비 86.7(동종 평균 100) — 동종 평균 이하 (미확인 문항 100%)"]

    M, fc, _ = _score(_rows("YES"))
    assert M == pytest.approx(1.367, abs=0.005) and fc == pytest.approx(1.388, abs=0.002)
    assert _decide(_rows("YES"), M, fc) == ("투자", None, ["동종 평균 대비 136.7(동종 평균 100) — 동종 평균보다 높음",
                                                        "창업자 기준 138.8(동종 평균 100) — 동종 평균 이상", "Deal-killer 없음"])


def test_unknown_ratio_in_reason_not_label():
    """옛 '정보 부족'(미확인 ≥ 60%) 이름표는 없다. 미확인 비율은 사유 문장에만 적고, 보류 유형은
    Deal-killer > 창업자 점수 평균 미만 > 동종 평균 이하 우선순위를 따른다 (사유도 그 순서로 모두 적는다)."""
    assert dec.HOLD_TYPES == ("Deal-killer", "창업자 점수 평균 미만", "동종 평균 이하")
    # 창업자 F1 YES, 나머지 미확인(96%): 팀은 평균 이상(1.012)이지만 M ≤ 1.00 → '동종 평균 이하'
    rows = _rows({"F1": "YES"})
    M, fc, _ = _score(rows)
    assert M == pytest.approx(0.904, abs=0.002) and fc == pytest.approx(1.012, abs=0.002)
    d, h, reasons = _decide(rows, M, fc)
    assert (d, h) == ("보류", "동종 평균 이하")
    assert reasons == ["동종 평균 대비 90.4(동종 평균 100) — 동종 평균 이하 (미확인 문항 96%)"]
    # 전부 미확인 + P4 NO(K1): 세 조건을 모두 어겨도 이름표는 하나(Deal-killer), 사유는 우선순위 순서
    rows = _rows({"P4": "NO"})
    M, fc, _ = _score(rows)
    assert M <= 1.00 and fc < 1.00 and dec._killers(rows, load_rubric()) == ["K1"]
    d, h, reasons = _decide(rows, M, fc)
    assert (d, h) == ("보류", "Deal-killer") and len(reasons) == 3
    assert [r.split(" ")[0] for r in reasons] == ["Deal-killer", "창업자", "동종"] and "미확인 문항 96%" in reasons[2]
    assert not any(t in " ".join(reasons) for t in ("정보 부족", "창업자 근거 없음", "동종 대비 열위"))


def test_founder_gate_and_deal_killer(set_cfg, tmp_path):
    """M > 1.00 이어도 창업자 기준 c_founder < 1.00 이면 '창업자 점수 평균 미만'. c_founder = 1.00(평균과 같음)은 통과(≥).
    P4 = NO 면 K1 → 'Deal-killer'."""
    rows = _rows({q: "YES" for q in QIDS if not q.startswith("F")})   # 창업자 4문항만 미확인
    M, fc, _ = _score(rows)
    assert M == pytest.approx(1.217, abs=0.002) and fc == pytest.approx(0.888, abs=0.002)
    d, h, reasons = _decide(rows, M, fc)
    assert (d, h) == ("보류", "창업자 점수 평균 미만")
    assert reasons == ["창업자 기준 88.8(동종 평균 100) — 팀이 동종 평균 미만"]    # M 은 기준을 넘었으니 배수 사유 없음
    # 기준 집단이 없으면(평균 신호 0) 창업자 미확인은 c_founder = 1.00 정확히 동종 평균 → 창업자 조건 통과
    set_cfg("decision.reference_file", str(tmp_path / "없음.json"))
    M0, fc0, ref = _score(rows)
    assert ref["source"] == "fallback" and fc0 == 1.0 and M0 > 1.0
    assert _decide(rows, M0, fc0)[:2] == ("투자", None)
    set_cfg("decision.reference_file", REF)

    rows = _rows({**{q: "YES" for q in QIDS}, "P4": "NO"})
    M, fc, _ = _score(rows)
    assert M > 1.00 and fc >= 1.00 and dec._killers(rows, load_rubric()) == ["K1"]
    d, h, reasons = _decide(rows, M, fc)
    assert (d, h) == ("보류", "Deal-killer") and reasons == ["Deal-killer K1: 판매 중인데 필수 인허가·검정 미취득"]
    flip = dec.flip_conditions(rows, dec.load_reference()["mean"], load_rubric(), get_config(), True, ["K1"])
    assert flip["new_multiplier"] is None and "K1 해소 필요" in flip["note"] and flip["qids"] == ["P4"]


def test_reference_loo_and_fallback(set_cfg, tmp_path):
    """대상이 기준 집단에 있으면 n=9(자기 제외). 5곳 미만·파일 없음·옛 문항 목록이면 x̄=0, source 'fallback'."""
    assert dec.load_reference("메타파머스")["n"] == 9 and dec.load_reference(None)["n"] == 10
    data = json.loads((ROOT / REF).read_text(encoding="utf-8"))
    small = tmp_path / "small.json"
    small.write_text(json.dumps({**data, "members": data["members"][:4], "n": 4}, ensure_ascii=False), encoding="utf-8")
    set_cfg("decision.reference_file", str(small))
    ref = dec.load_reference()
    assert (ref["source"], ref["n"]) == ("fallback", 4) and set(ref["mean"].values()) == {0.0} and ref["note"]
    set_cfg("decision.reference_file", str(tmp_path / "없음.json"))
    assert dec.load_reference()["source"] == "fallback"
    old = tmp_path / "old.json"
    old.write_text(json.dumps({**data, "rubric_qids": QIDS[:-1]}, ensure_ascii=False), encoding="utf-8")
    set_cfg("decision.reference_file", str(old))
    assert dec.load_reference()["source"] == "fallback"
    # 기준 집단 없이(평균 0) 전부 미확인이면 모든 기준이 1.00
    assert dec.payne_multiplier(_rows("UNKNOWN"), dec.load_reference()["mean"], load_rubric(), 0.5, [0.5, 1.5])[0] == 1.0


def test_criteria_fields():
    rows = ROWS["메타파머스"]["rows"]
    ref = dec.load_reference("메타파머스")
    M, crit = dec.payne_multiplier(rows, ref["mean"], load_rubric(), 0.5, [0.5, 1.5])
    assert [c["dim"] for c in crit] == ["founder", "market", "product", "competition", "traction", "deal"]
    assert all(set(c) == {"dim", "name", "weight", "yes", "no", "unknown", "n", "pct", "peer_mean", "contribution"}
               for c in crit)
    assert sum(c["contribution"] for c in crit) == pytest.approx(M, abs=1e-3)
    assert crit[0]["pct"] == pytest.approx(129.2, abs=0.1) and crit[0]["yes"] == 3


def _flipped(rows: list[dict], qids: list[str]) -> list[dict]:
    return [{**r, "answer": "YES", "x": 1} if r["qid"] in qids else r for r in rows]


def test_flip_conditions(set_cfg):
    """보류 후보의 뒤집힘: 미확인 문항을 YES 로 바꿔 M' > 기준 ∧ c_founder ≥ 1.00 이 되는 문항 ≤ 3개.
    창업자 기준이 동종 평균 미만이면 창업자 문항부터. 뒤집힌 판정으로 다시 결정하면 실제로 '투자'여야 한다."""
    rubric, cfg = load_rubric(), get_config()
    # 바르카(0.956, 팀 1.014 · '동종 평균 이하'): F2·F3 가 YES 면 103.1 → 투자
    rows = ROWS["바르카"]["rows"]
    mean = dec.load_reference("바르카")["mean"]
    flip = dec.flip_conditions(rows, mean, rubric, cfg, True, [])
    assert flip["reached"] and flip["qids"] == ["F2", "F3"] and flip["new_multiplier"] == pytest.approx(1.031, abs=0.002)
    assert all(r["answer"] == "UNKNOWN" for r in rows if r["qid"] in flip["qids"])
    assert flip["note"] == "F2·F3 이(가) YES 로 확인되면 동종 평균 대비 103.1(동종 평균 100) → 투자"
    M, fc, _ = _score(_flipped(rows, flip["qids"]), "바르카")
    assert M == pytest.approx(flip["new_multiplier"]) and _decide(_flipped(rows, flip["qids"]), M, fc)[0] == "투자"

    # Nature Robots(0.935, 팀 0.875 · '창업자 점수 평균 미만'): 창업자 문항부터 채운다
    rows = ROWS["Nature Robots"]["rows"]
    mean = dec.load_reference("Nature Robots")["mean"]
    flip = dec.flip_conditions(rows, mean, rubric, cfg, False, [])
    assert flip["reached"] and flip["qids"] == ["F1", "F2"] and flip["new_multiplier"] == pytest.approx(1.010, abs=0.002)
    M, fc, _ = _score(_flipped(rows, flip["qids"]), "Nature Robots")
    assert fc >= 1.00 and M > 1.00 and _decide(_flipped(rows, flip["qids"]), M, fc)[0] == "투자"

    # 시나리오 기준 1.10: 3문항으로 모자라면 reached False, 문장에 동종 평균 100 과 기준 110 을 따로 적는다
    set_cfg("decision.threshold", 1.10)
    flip = dec.flip_conditions(rows, mean, rubric, get_config(), False, [])
    assert not flip["reached"] and flip["qids"] == ["F1", "F2", "F3"] and flip["new_multiplier"] < 1.10
    assert "(동종 평균 100, 기준 110) — 기준 미달(최대 3문항)" in flip["note"]

    # 창업자 문항에 미확인이 없고(모두 NO) 창업자 기준이 평균 미만이면 미확인 확인만으로는 뒤집을 수 없다
    rows = _rows({"F1": "NO", "F2": "NO", "F3": "NO", "F4": "NO"})
    flip = dec.flip_conditions(rows, dec.load_reference()["mean"], rubric, get_config(), False, [])
    assert (flip["qids"], flip["reached"]) == ([], False) and flip["note"].startswith("창업자 문항에 미확인이 없어")


def test_dd_items_and_bessemer():
    rubric, cfg = load_rubric(), get_config()
    rows = ROWS["메타파머스"]["rows"]
    dd = dec._dd_items(rows, dec.load_reference("메타파머스")["mean"], rubric, cfg, [])
    assert 1 <= len(dd) <= 5 and all(set(x) == {"qid", "item", "why"} for x in dd)
    assert all(next(r for r in rows if r["qid"] == x["qid"])["answer"] in ("UNKNOWN", "NO") for x in dd)

    panel = dec.bessemer_panel(rows, rubric)
    assert [p["q"] for p in panel] == list(range(1, 11))
    assert all(p["answer"] in ("YES", "NO", "미확인") for p in panel) and [p["q"] for p in panel if p["proxy"]] == [10]
    no = dec.bessemer_panel(_rows({"M3": "YES", "P1": "NO"}), rubric)
    assert no[1]["answer"] == "NO" and no[0]["answer"] == "미확인"   # NO 우선, 그다음 YES, 그 밖은 미확인


@pytest.mark.parametrize("raw, expected", [
    ("30억 원", {"krw": 3e9, "usd_m": None}), ("$12M", {"krw": None, "usd_m": 12}), ("12 million", {"krw": None, "usd_m": 12}),
    ("259억 9991만원", {"krw": 25_999_910_000, "usd_m": None}), ("1,500만 달러", {"krw": None, "usd_m": 15}),
    ("$4,013,044", {"krw": None, "usd_m": 4.013044}), ("비공개", None), ("", None), (None, None), ("€4 million", None),
])
def test_parse_amount(raw, expected):
    got = dec.parse_amount(raw)
    if expected is None:
        assert got is None
    else:
        assert got["krw"] == (pytest.approx(expected["krw"]) if expected["krw"] else None)
        assert got["usd_m"] == (pytest.approx(expected["usd_m"]) if expected["usd_m"] else None)


def test_roi():
    cfg = get_config()
    r = dec.roi({"round_amount": "30억 원", "stage": "Seed", "stage_evidence_ids": ["W1"]}, {}, cfg)
    assert r["computable"] and r["round_amount_usd_m"] == pytest.approx(2.14, abs=0.01)
    assert (r["stage_median_usd_m"], r["vs_stage_median"]) == (1, pytest.approx(2.14, abs=0.01))
    assert r["post_money_usd_m"] == [pytest.approx(10.71, abs=0.01), pytest.approx(21.43, abs=0.01)]
    assert r["required_exit_usd_m"] == [pytest.approx(107.1, abs=0.1), pytest.approx(214.3, abs=0.1)]
    assert r["source_ids"] == ["W1"] and all("가정" in a for a in r["assumptions"])
    assert "exit_share_of_market" not in r and "warning" not in r
    r = dec.roi({"round_amount": "$12M", "stage": "Series A"}, {}, cfg)
    assert r["vs_stage_median"] == pytest.approx(12 / 7, abs=0.01) and not any("환율" in a for a in r["assumptions"])
    for amount in ("", "비공개", None):
        r = dec.roi({"round_amount": amount, "stage": "Pre-A"}, {}, cfg)
        assert not r["computable"] and r["post_money_usd_m"] is None and r["required_exit_usd_m"] is None and r["note"]


def test_reference_class_and_threshold_sensitivity_files(tmp_path):
    """보정 실행 결과(evaluations)로 쓴 기준 집단 파일이 v1 픽스처와 같고, 기준 배수별 투자 수가 설계서 사전 점검과 같다."""
    evals = [{"name": n, "region": c["region"], "stage": c["stage"], "segment_id": c["segment_id"],
              "scorecard": {"rows": c["rows"]}} for n, c in ROWS.items()]
    ref = dec.write_reference_class(evals, str(tmp_path / "ref.json"), "2026-09-30")
    fx = json.loads((ROOT / REF).read_text(encoding="utf-8"))
    assert ref["n"] == 10 and ref["created_by"] == "app.py --calibrate" and ref["rubric_qids"] == QIDS
    assert ref["members"] == fx["members"] and ref["mean"] == fx["mean"]
    assert json.loads((tmp_path / "ref.json").read_text(encoding="utf-8")) == ref
    ts = dec.write_threshold_sensitivity(str(tmp_path / "ref.json"), str(tmp_path / "ts.json"))
    assert ts["invest_count_at"] == {"1.00": 5, "1.05": 2, "1.10": 1, "1.15": 0, "1.20": 0}
    assert ts["members"][0]["name"] == "메타파머스" and ts["members"][0]["multiplier_loo"] == pytest.approx(1.130, abs=0.002)
    assert set(ts["members"][0]["decision_at"]) == {"1.00", "1.05", "1.10", "1.15", "1.20"}


# ── 노드 (judge_dimension 은 가짜: v1 판정 행을 돌려준다)
def _criterion(dim: str, rows: list[dict]) -> dict:
    d = next(x for x in load_rubric()["dimensions"] if x["id"] == dim)
    rs = [r for r in rows if r["dim"] == dim]
    return {"dim": dim, "name": d["name"], "weight": d["weight"], "owner": d["owner"], "rows": rs,
            "yes": sum(r["answer"] == "YES" for r in rs), "no": sum(r["answer"] == "NO" for r in rs),
            "unknown": sum(r["answer"] == "UNKNOWN" for r in rs), "na": 0, "n": len(rs), "rejected_yes": [],
            "quote_retried": 0}


def _state(name: str, **kw) -> dict:
    c = ROWS[name]
    rows = c["rows"]
    current = {"name": name, "official_name": name, "region": c["region"], "stage": c["stage"],
               "segment_id": c["segment_id"], "round_date": "2025-08", "round_amount": "30억원",
               "stage_evidence_ids": ["W1"], "evidence_ids": ["W1"], "one_line": "농업 로봇"}
    st = {"current": current, "run_date": "2026-09-30", "registry": {"W1": {"id": "W1", "kind": "web"}}, "iterations": 1,
          "founder": {"criterion": _criterion("founder", rows), "pool_ids": []},
          "tech": {"criterion": _criterion("product", rows)}, "market": {"criterion": _criterion("market", rows)},
          "competition": {"criterion": _criterion("competition", rows)}}
    st.update(kw)
    return st


@pytest.fixture
def judged(monkeypatch):
    calls = []

    def fake(dim_id, company, pool_ids, reg, analysis, run_date):
        calls.append(dim_id)
        return _criterion(dim_id, ROWS[company["official_name"]]["rows"])

    monkeypatch.setattr(dec, "judge_dimension", fake)
    return calls


def test_node_invest(judged, set_cfg):
    out = dec.decision_node(_state("메타파머스"))
    assert judged == ["traction", "deal"]                       # 나머지 네 기준은 분석 에이전트의 criterion
    assert set(out) == {"scorecard", "decision", "evaluations", "log"}  # 기본: 투자 추천이어도 멈추지 않아 종료 사유 없음
    assert out["decision"] == "투자"
    sc, ev = out["scorecard"], out["evaluations"][0]
    assert set(ev) == C7_EVAL and C7_SCORECARD <= set(sc) and len(out["evaluations"]) == 1
    assert sc["multiplier"] == pytest.approx(1.130, abs=0.002) and sc["score100"] == pytest.approx(113.0, abs=0.2)
    assert sc["founder_c"] == pytest.approx(1.292, abs=0.002) and sc["hold_type"] is None
    assert sc["reference"]["n"] == 9 and sc["reference"]["loo"] and sc["flip"] is None
    assert list(sc["sensitivity"]) == ["1.00", "1.05", "1.10", "1.15", "1.20"] and sc["sensitivity"]["1.10"] == "투자"
    assert len(sc["bessemer"]) == 10 and len(sc["rows"]) == 24 and len(sc["criteria"]) == 6
    assert (sc["target_rank"], sc["peer_n"]) == (1, 10) and sum(r["is_target"] for r in sc["ranking"]) == 1
    assert sc["ranking"][0]["name"] == "메타파머스" and sc["roi"]["computable"]
    assert ev["multiplier"] == sc["multiplier"] and ev["profile"]["official_name"] == "메타파머스"
    set_cfg("workflow.stop_on_invest", True)                   # 노션 Graph(안) 그대로: 첫 투자 추천에서 멈춘다
    assert dec.decision_node(_state("메타파머스"))["end_reason"] == "invest_found"


def test_node_hold_backfills_missing_criterion(judged, set_cfg):
    """criterion 이 없는 기준(시장성)은 투자 판단이 보완 판정한다. 보류 + 평가 상한이면 end_reason 'max_evaluations'."""
    set_cfg("decision.threshold", 1.30)
    out = dec.decision_node(_state("퓨처커넥트", market={}, iterations=10))
    assert judged == ["market", "traction", "deal"] and out["scorecard"]["judged_in_decide"] == judged
    assert (out["decision"], out["end_reason"]) == ("보류", "max_evaluations")
    sc = out["scorecard"]
    # 퓨처커넥트 1.095 · 팀 1.153: 시나리오 기준 1.30 이하 → '기준 이하' (설계 기준 1.00 에서는 투자)
    assert sc["founder_c"] == pytest.approx(1.153, abs=0.002) and sc["hold_type"] == "기준 이하"
    assert sc["reasons"] == ["동종 평균 대비 109.5(동종 평균 100, 기준 130) — 기준 130 이하 (미확인 문항 62%)"]
    assert sc["flip"] and not sc["flip"]["reached"] and sc["sensitivity"]["1.00"] == "투자"
    assert sc["ranking"][sc["target_rank"] - 1]["name"] == "퓨처커넥트"

    judged.clear()
    out = dec.decision_node(_state("퓨처커넥트", iterations=3))
    assert "end_reason" not in out and judged == ["traction", "deal"]


def test_node_calibrate_makes_no_decision(judged, set_cfg):
    """보정 실행: 결정·배수 없이 판정·신호만 남긴다 (투자여도 멈추지 않음)."""
    set_cfg("workflow.calibrate", True)
    set_cfg("workflow.stop_on_invest", False)
    out = dec.decision_node(_state("메타파머스"))
    sc = out["scorecard"]
    assert out["decision"] is None and sc["decision"] is None and sc["multiplier"] is None and "end_reason" not in out
    assert len(sc["rows"]) == 24 and sc["founder_yes"] == 3 and sc["ranking"] == [] and sc["reference"] is None
    out = dec.decision_node(_state("메타파머스", iterations=10))
    assert out["end_reason"] == "max_evaluations"


def test_node_fallback_without_reference(judged, set_cfg, tmp_path):
    """기준 집단 파일이 없으면 x̄ = 0 으로 계산하고 순위는 만들지 않는다."""
    set_cfg("decision.reference_file", str(tmp_path / "없음.json"))
    sc = dec.decision_node(_state("메타파머스"))["scorecard"]
    assert sc["reference"]["source"] == "fallback" and sc["reference"]["note"]
    assert (sc["ranking"], sc["target_rank"], sc["peer_n"]) == ([], None, 0)


def test_rubric_decision_rule_matches_config():
    """rubric.yaml 의 decision_rule 설명이 config.yaml decision 수치와 같다 (설명과 코드가 따로 놀지 않게)."""
    from core.config import get_config

    d, rule = get_config().decision, load_rubric()["decision_rule"]
    for s in (f"M > {d.threshold:.2f}", f"c_founder ≥ {d.founder_min_c:.2f}", f"1 + {d.step} ×", f"{d.clip[0]}, {d.clip[1]})",
              f"{d.reference_min_n}곳 미만", f"{min(d.sensitivity):.2f}~{max(d.sensitivity):.2f}",
              " > ".join(dec.HOLD_TYPES), "동종 평균 = 1.00", "투자를 받은 동종 평균 기업 = 100%"):
        assert s in rule, s
    # 옛 규칙의 근거 없는 수치·이름표가 남아 있지 않다: v1 70점, 기준 1.10, 창업자 YES 개수, 미확인 60%(정보 부족)
    assert not re.search(r"(?<![\d.])(70|1\.10)(?![\d])", rule), "옛 기준 수치(70 · 1.10)"
    assert not re.search(r"[≥>]=?\s*(1\.1|110|70)", rule) and "YES ≥" not in rule and "미확인 ≥" not in rule
    assert not any(t in rule for t in ("정보 부족", "창업자 근거 없음", "동종 대비 열위", "설계 가정"))
