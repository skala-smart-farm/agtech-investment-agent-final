"""투자 판단(agents/decision.py) 검사: Payne 상대 배수·결정 규칙·뒤집힘·실사·Bessemer·ROI·민감도·순위·노드 반환 (계약 C7·C10).

기대값은 v1 제출 실행(outputs/run_log.json)의 판정 24문항×10곳으로 다시 계산한 값이다.
픽스처는 eval/build_reference_class.py 로 만들었다 (tests/fixtures/reference_v1.json, rows_v1.json).
LLM·네트워크 없이 돈다 (judge_dimension 은 가짜로 바꾼다).
"""
from __future__ import annotations

import json

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
C7_SCORECARD = {"criteria", "rows", "multiplier", "score100", "threshold", "reference", "founder_yes", "unknown_ratio",
                "deal_killers", "decision", "hold_type", "reasons", "flip", "dd_items", "bessemer", "roi", "sensitivity",
                "ranking", "target_rank", "peer_n"}


@pytest.fixture(autouse=True)
def _ref(set_cfg):
    set_cfg("decision.reference_file", REF)


def _rows(answers: dict[str, str] | str) -> list[dict]:
    """{qid: 판정} 또는 모든 문항에 같은 판정 → 판정 행."""
    x = {"YES": 1, "NO": -1, "UNKNOWN": 0, "N/A": None}
    get = (lambda q: answers) if isinstance(answers, str) else (lambda q: answers.get(q, "UNKNOWN"))
    return [{"dim": DIM_OF[q], "qid": q, "answer": get(q), "x": x[get(q)]} for q in QIDS]


def _score(rows: list[dict], exclude: str | None = None) -> tuple[float, dict]:
    ref = dec.load_reference(exclude=exclude)
    d = get_config().decision
    return dec.payne_multiplier(rows, ref["mean"], load_rubric(), d.step, d.clip)[0], ref


def _decide(rows: list[dict], M: float):
    fy = sum(r["answer"] == "YES" for r in rows if r["dim"] == "founder")
    judged = [r for r in rows if r["answer"] != "N/A"]
    unk = sum(r["answer"] == "UNKNOWN" for r in judged) / len(judged)
    return dec.decide_rule(M, fy, dec._killers(rows, load_rubric()), unk, get_config())


# ── 배수 재계산 (v1 판정 + 자기 제외 기준 집단)
@pytest.mark.parametrize("name, M, decision, hold", [
    ("메타파머스", 1.130, "투자", None),
    ("퓨처커넥트", 1.095, "보류", "정보 부족"),        # 미확인 15/24 = 62.5% ≥ info_gap_ratio 0.6
    ("에임비랩", 1.015, "보류", "정보 부족"),          # P4 N/A 는 분모에서 빠짐
    ("Nature Robots", 0.935, "보류", "창업자 근거 없음"),
])
def test_v1_recompute(name, M, decision, hold):
    rows = ROWS[name]["rows"]
    got, ref = _score(rows, exclude=name)
    assert (ref["n"], ref["loo"], ref["source"]) == (9, True, "calibration") and name not in ref["members"]
    assert got == pytest.approx(M, abs=0.002)
    d, h, reasons = _decide(rows, got)
    assert (d, h) == (decision, hold) and reasons


def test_peer_inferior_label_on_real_data(set_cfg):
    """미확인 60% 미만인데 기준 미달이면 '동종 대비 열위' (메타파머스 1.130, 미확인 54%, 기준 1.30)."""
    set_cfg("decision.threshold", 1.30)
    rows = ROWS["메타파머스"]["rows"]
    d, h, reasons = _decide(rows, _score(rows, "메타파머스")[0])
    assert (d, h) == ("보류", "동종 대비 열위") and "기준 130) — 기준 미달" in reasons[0]


def test_all_unknown_and_all_yes():
    """가상 기업(기준 집단 밖, 10곳 평균): 전부 미확인 0.867 보류, 전부 YES 1.367 투자.
    전부 미확인은 창업자 YES 가 0개라 보류 유형 우선순위(Deal-killer > 창업자 근거 없음 > 정보 부족)에서 '창업자 근거 없음'."""
    rows = _rows("UNKNOWN")
    M, ref = _score(rows, exclude="가상 기업")
    assert (ref["n"], ref["loo"]) == (10, False)
    assert M == pytest.approx(0.867, abs=0.005)
    d, h, reasons = _decide(rows, M)
    assert (d, h) == ("보류", "창업자 근거 없음") and any("미확인 문항 100%" in r for r in reasons)

    M, _ = _score(_rows("YES"))
    assert M == pytest.approx(1.367, abs=0.005)
    assert _decide(_rows("YES"), M)[:2] == ("투자", None)


def test_info_gap_label():
    """창업자 YES 1개, 나머지 미확인 → 기준 미달 + 미확인 ≥ 60% → '정보 부족'."""
    rows = _rows({"F1": "YES"})
    M, _ = _score(rows)
    assert M < 1.10
    assert _decide(rows, M)[:2] == ("보류", "정보 부족")


def test_founder_gate_and_deal_killer():
    """M ≥ 1.10 이어도 창업자 YES 0개면 '창업자 근거 없음'. P4 = NO 면 K1 → 'Deal-killer'."""
    rows = _rows({q: "YES" for q in QIDS if not q.startswith("F")})
    M, _ = _score(rows)
    assert M >= 1.10
    d, h, reasons = _decide(rows, M)
    assert (d, h) == ("보류", "창업자 근거 없음") and reasons[0].startswith("창업자 문항")

    rows = _rows({**{q: "YES" for q in QIDS}, "P4": "NO"})
    M, _ = _score(rows)
    assert M >= 1.10 and dec._killers(rows, load_rubric()) == ["K1"]
    d, h, reasons = _decide(rows, M)
    assert (d, h) == ("보류", "Deal-killer") and reasons[0].startswith("Deal-killer K1")
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


def test_flip_conditions():
    """보류 후보의 뒤집힘: M' ≥ 1.10 이고 문항 ≤ 3. 창업자 요건이 모자라면 창업자 문항부터."""
    rubric, cfg = load_rubric(), get_config()
    rows = ROWS["퓨처커넥트"]["rows"]
    mean = dec.load_reference("퓨처커넥트")["mean"]
    flip = dec.flip_conditions(rows, mean, rubric, cfg, True, [])
    assert flip["reached"] and flip["new_multiplier"] >= 1.10 and 1 <= len(flip["qids"]) <= 3
    assert all(r["answer"] == "UNKNOWN" for r in rows if r["qid"] in flip["qids"]) and "투자" in flip["note"]

    rows = ROWS["Nature Robots"]["rows"]
    flip = dec.flip_conditions(rows, dec.load_reference("Nature Robots")["mean"], rubric, cfg, False, [])
    assert flip["qids"][0].startswith("F") and len(flip["qids"]) <= 3
    assert flip["new_multiplier"] > 0.935


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


def test_node_invest(judged):
    out = dec.decision_node(_state("메타파머스"))
    assert judged == ["traction", "deal"]                       # 나머지 네 기준은 분석 에이전트의 criterion
    assert set(out) == {"scorecard", "decision", "evaluations", "end_reason", "log"}
    assert (out["decision"], out["end_reason"]) == ("투자", "invest_found")
    sc, ev = out["scorecard"], out["evaluations"][0]
    assert set(ev) == C7_EVAL and C7_SCORECARD <= set(sc) and len(out["evaluations"]) == 1
    assert sc["multiplier"] == pytest.approx(1.130, abs=0.002) and sc["score100"] == pytest.approx(113.0, abs=0.2)
    assert sc["reference"]["n"] == 9 and sc["reference"]["loo"] and sc["flip"] is None
    assert list(sc["sensitivity"]) == ["1.00", "1.05", "1.10", "1.15", "1.20"] and sc["sensitivity"]["1.10"] == "투자"
    assert len(sc["bessemer"]) == 10 and len(sc["rows"]) == 24 and len(sc["criteria"]) == 6
    assert (sc["target_rank"], sc["peer_n"]) == (1, 10) and sum(r["is_target"] for r in sc["ranking"]) == 1
    assert sc["ranking"][0]["name"] == "메타파머스" and sc["roi"]["computable"]
    assert ev["multiplier"] == sc["multiplier"] and ev["profile"]["official_name"] == "메타파머스"


def test_node_hold_backfills_missing_criterion(judged, set_cfg):
    """criterion 이 없는 기준(시장성)은 투자 판단이 보완 판정한다. 보류 + 평가 상한이면 end_reason 'max_evaluations'."""
    set_cfg("decision.threshold", 1.30)
    out = dec.decision_node(_state("퓨처커넥트", market={}, iterations=10))
    assert judged == ["market", "traction", "deal"] and out["scorecard"]["judged_in_decide"] == judged
    assert (out["decision"], out["end_reason"]) == ("보류", "max_evaluations")
    sc = out["scorecard"]
    assert sc["hold_type"] == "정보 부족" and sc["flip"] and sc["sensitivity"]["1.00"] == "투자"
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
    for s in (f"M ≥ {d.threshold:.2f}", f"1 + {d.step} ×", f"{d.clip[0]}, {d.clip[1]})", f"YES ≥ {d.min_founder_yes}",
              f"미확인 ≥ {d.info_gap_ratio:.0%}", f"{d.reference_min_n}곳 미만", f"{min(d.sensitivity):.2f}~{max(d.sensitivity):.2f}"):
        assert s in rule, s
    assert "70" not in rule  # v1 의 100점 만점 규칙이 남아 있지 않다
