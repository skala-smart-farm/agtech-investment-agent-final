"""보고서 대상 선정과 평가 범위 문장 — 팀원의 새 평가(--fresh)에서 나온 상황을 그대로 옮긴 회귀 테스트.

팀원 실행(평가 순서대로): 메타파머스 109.9 투자 → … → 퓨처커넥트 110.9 투자 → Octopusbot 101.1 투자 → … → 파미레세 100.5 투자.
옛 코드는 처음 나온 투자 후보(메타파머스)를 대상으로 삼고, 멈추지 않았는데도 '투자 추천 후보가 나와 평가를 멈춤'이라 적었다.
"""
from __future__ import annotations

import agents.decision as dec
import agents.report as rep
from core.config import get_config

TEAMMATE_RUN = [  # (이름, 배수, 판정) — 평가 순서
    ("메타파머스", 1.099, "투자"), ("Nature Robots", 0.979, "보류"), ("리비타", 0.900, "보류"),
    ("새팜", 1.131, "보류"), ("Quercus Biosolutions", 0.996, "보류"), ("루트릭스", 1.099, "투자"),
    ("퓨처커넥트", 1.109, "투자"), ("Octopusbot", 1.011, "투자"), ("트랙팜", 0.951, "보류"), ("파미레세", 1.005, "투자"),
]


def _evals(rows=TEAMMATE_RUN):
    return [{"name": n, "decision": d, "hold_type": None if d == "투자" else "창업자 점수 평균 미만",
             "stage": "Seed", "region": "KR", "scorecard": {"multiplier": m, "founder_c": 1.0}} for n, m, d in rows]


def test_target_is_top_invested_not_first_invested():
    invested, ranked, target = rep._pick_target(_evals())
    assert target["name"] == "퓨처커넥트"                       # 투자 5곳 중 배수 1위 (첫 투자 후보 메타파머스가 아님)
    assert ranked[0]["name"] == "새팜" and ranked[0]["decision"] == "보류"  # 전체 1위라도 보류면 대상이 아니다
    assert [e["name"] for e in invested] == ["메타파머스", "루트릭스", "퓨처커넥트", "Octopusbot", "파미레세"]


def test_all_hold_targets_top_multiplier():
    rows = [(n, m, "보류") for n, m, _ in TEAMMATE_RUN]
    invested, _, target = rep._pick_target(_evals(rows))
    assert not invested and target["name"] == "새팜"


def test_unevaluated_reason_is_cap_when_not_stopped():
    cfg = get_config()
    screened = [{"name": n, "eligible": True} for n, _, _ in TEAMMATE_RUN] + \
               [{"name": f"미평가{i}", "eligible": True} for i in range(4)]
    pool = rep._pool({"end_reason": "max_evaluations"}, _evals(), screened, cfg)
    assert pool["why"] == f"평가 상한({cfg.workflow.max_evaluations}곳, 비용 관리) 도달"
    assert [u["name"] for u in pool["unevaluated"]] == [f"미평가{i}" for i in range(4)]
    line = next(x for x in rep._limitations(cfg, pool, {}, [], "invest", []) if "평가하지 않았다" in x)
    assert "멈춤" not in line and "미평가0·미평가1·미평가2·미평가3" in line


def test_reference_gap_flags_fresh_run_without_calibration(monkeypatch):
    """기준 집단(보정 실행)에 없는 후보를 평가했으면 한계에 밝힌다. 제출 실행(평가 = 기준 집단)에서는 아무것도 바뀌지 않는다."""
    members = [{"name": n} for n in ("메타파머스", "리비타", "새팜", "퓨처커넥트", "트랙팜",
                                     "Nature Robots", "에임비랩", "바르카", "Bonsai Robotics", "Upside Robotics")]
    monkeypatch.setattr(dec, "_read_json", lambda p: {"members": members, "run_date": "2026-09-30"})
    gap = rep._reference_gap(_evals())
    assert gap["unseen"] == ["Quercus Biosolutions", "루트릭스", "Octopusbot", "파미레세"] and gap["n_ref"] == 10
    same = [(m["name"], 1.0, "보류") for m in members]
    assert rep._reference_gap(_evals(same)) is None


def test_submission_run_has_no_reference_gap():
    """커밋된 제출 실행: 평가한 10곳이 기준 집단 10곳과 같다."""
    import json

    run = json.loads((rep.ROOT / "outputs/run_log.json").read_text(encoding="utf-8")) if hasattr(rep, "ROOT") else None
    if run is None:
        from core.config import ROOT
        run = json.loads((ROOT / "outputs/run_log.json").read_text(encoding="utf-8"))
    assert rep._reference_gap(run["evaluations"]) is None
