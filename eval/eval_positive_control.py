"""양성 대조 실험: 공개 정보가 풍부한 후기(Series B~C) AgTech 기업에도 평가표가 "투자"를 줄 수 있는가.

본 실행의 후보가 모두 보류로 끝났을 때, 그것이 "무조건 보류하는 평가표" 때문인지
"초기 기업의 공개 정보 부족" 때문인지 가르기 위한 실험이다.
적격성 검증 → 창업자 → 기술 → 시장성 → 경쟁사 → 투자 판단을 그래프와 같은 순서로 실행한다 (v2: 동종 기준 집단 대비 배수).
결과는 기업마다 outputs/eval/positive_control.json 에 바로 쓴다 (중간에 실패해도 앞 기업 결과와 실패 사유가 남는다).
기업별: decision(투자/보류/적격성 탈락/실행 실패), multiplier(동종 평균 = 1.00), hold_type, founder_yes, unknown_ratio,
reasons, 뒤집힘 조건, 강등된 YES(rejected_yes), 미확인 문항, 기준별 점수(동종 평균 = 100), 기준 배수 민감도,
그 기업 평가 중 실패한 웹 검색 수. v1 결과(100점 만점 70점 기준: 0/3)는 v1-safe 태그에 남아 있다.

LLM·웹 검색 API 를 부른다 (유료). 키 없이 돌리면 캐시에 없는 호출에서 그 기업은 '실행 실패'로 기록된다.

    uv run python -m eval.eval_positive_control
"""
from __future__ import annotations

import json

import tools.web_search as ws
from agents.competition import competition_node
from agents.decision import decision_node
from agents.eligibility import _check_one
from agents.founder import founder_node
from agents.market import market_node
from agents.tech import tech_node
from core.config import get_config, path, run_date

# 정보가 풍부하고 조사 단계에서 적격(비상장·Series B~C)으로 확인된 기업
CONTROLS = [
    {"name": "Source.ag", "name_en": "Source.ag", "region": "GLOBAL", "segment_id": "greenhouse"},
    {"name": "Agtonomy", "name_en": "Agtonomy", "region": "GLOBAL", "segment_id": "robotics"},
    {"name": "아이오크롭스", "name_en": "IOCROPS", "region": "KR", "segment_id": "greenhouse"},
]


def _failed_searches() -> int | None:
    """tools/web_search.py 의 검색 실패 기록 수. 기록 기능이 없는 버전이면 None (0 으로 오해하지 않게)."""
    failed = getattr(ws, "FAILED_QUERIES", None)
    return None if failed is None else len(failed)


def _evaluate(c: dict) -> dict:
    rec, reg = _check_one({**c, "evidence_ids": [], "channels": []}, {})
    if not rec["eligible"]:
        return {"name": c["name"], "eligible": False, "stage": rec.get("stage"), "decision": "적격성 탈락",
                "multiplier": None, "unknown_ratio": None, "reasons": [rec["reason"]]}
    state = {"registry": reg, "current": rec, "run_date": run_date(), "market_cache": {}}
    for node in (founder_node, tech_node, market_node):
        out = node(state)
        state["registry"] = {**state["registry"], **out.get("registry", {})}
        state.update({k: v for k, v in out.items() if k in ("founder", "tech", "market", "market_cache")})
    out = competition_node(state)
    state["registry"] = {**state["registry"], **out.get("registry", {})}
    state["competition"] = out["competition"]
    sc = decision_node(state)["scorecard"]
    return {"name": c["name"], "eligible": True, "stage": rec["stage"], "decision": sc["decision"],
            "multiplier": sc["multiplier"], "threshold": sc["threshold"], "hold_type": sc["hold_type"],
            "founder_yes": sc["founder_yes"], "unknown_ratio": sc["unknown_ratio"], "reasons": sc["reasons"],
            "flip": (sc.get("flip") or {}).get("note"), "reference_n": (sc.get("reference") or {}).get("n"),
            "rejected_yes": [{"qid": r["qid"], "reason": r["reason"]} for r in sc["rejected_yes"]],
            "unknown_qids": [r["qid"] for r in sc["rows"] if r["answer"] == "UNKNOWN"],
            "criteria": {c["dim"]: c["pct"] for c in sc["criteria"]}, "sensitivity": sc["sensitivity"]}


def main() -> None:
    get_config()
    out = path("outputs/eval/positive_control.json")
    rows = []
    for c in CONTROLS:
        before = _failed_searches()
        try:
            row = _evaluate(c)
        except Exception as e:  # 한 기업이 실패해도 나머지를 평가하고 실패 사유를 남긴다
            row = {"name": c["name"], "eligible": None, "decision": "실행 실패", "multiplier": None, "unknown_ratio": None,
                   "reasons": [f"{type(e).__name__}: {str(e)[:200]}"]}
        after = _failed_searches()
        row["failed_searches"] = None if before is None else after - before
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    invest = [r["name"] for r in rows if r["decision"] == "투자"]
    print(f"투자 판정 {len(invest)}/{len(rows)}곳: {', '.join(invest) or '없음'} → {out}")


if __name__ == "__main__":
    main()
