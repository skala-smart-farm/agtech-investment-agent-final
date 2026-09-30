"""적격성 검증 에이전트 평가: 사람이 교차 검증한 20개사 정답과 비교.

정답은 조사 단계에서 투자 보도·거래소 목록·인수 발표를 사람이 직접 확인해 만든 라벨이다
(data/eval/eligibility_gold.jsonl 의 basis). 경계 사례(예: '프리IPO'를 '시리즈C급'으로 보도한 기업)는
정답을 하나로 정할 수 없어 제외했다.

    python -m eval.eval_eligibility            # 개선에 쓴 20개사
    python -m eval.eval_eligibility holdout    # 개선에 쓰지 않은 10개사
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from agents.eligibility import _check_one
from core.config import get_config, path


def main(split: str = "gold") -> None:
    """split=gold: 개선에 쓴 20개사 / split=holdout: 개선에 쓰지 않은 10개사 (과적합 확인)."""
    get_config()
    f = "data/eval/eligibility_gold.jsonl" if split == "gold" else "data/eval/eligibility_holdout.jsonl"
    gold = [json.loads(l) for l in path(f).read_text(encoding="utf-8").splitlines() if l.strip()]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda g: _check_one({**g, "evidence_ids": [], "channels": []}, {})[0], gold))
    rows = []
    for g, r in zip(gold, results):
        rows.append({"name": g["name"], "region": g["region"], "gold": g["gold"], "pred": r["eligible"],
                     "ok": g["gold"] == r["eligible"], "pred_stage": r.get("stage"), "raw_stage": r.get("raw_stage"),
                     "stage_quote": r.get("stage_quote"), "reason": r["reason"],
                     "basis": g["basis"]})
    tp = sum(r["gold"] and r["pred"] for r in rows)
    tn = sum(not r["gold"] and not r["pred"] for r in rows)
    fp = sum(not r["gold"] and r["pred"] for r in rows)
    fn = sum(r["gold"] and not r["pred"] for r in rows)
    summary = {"n": len(rows), "accuracy": round((tp + tn) / len(rows), 3),
               "precision": round(tp / (tp + fp), 3) if tp + fp else None,
               "recall": round(tp / (tp + fn), 3) if tp + fn else None,
               "false_positive": fp, "false_negative": fn}
    path(f"outputs/eval/eligibility_eval_{split}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{'✓' if r['ok'] else '✗'} {r['name']}: 정답 {'적격' if r['gold'] else '부적격'} / "
              f"예측 {'적격' if r['pred'] else '부적격'} ({r['pred_stage']}) — {r['reason']}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "gold")
