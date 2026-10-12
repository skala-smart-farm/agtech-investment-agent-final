"""평가표 문항 판정 정확도: 사람이 라벨한 정답(data/eval/judgment_gold.csv)과 실행 기록의 판정을 비교한다.

    python -m eval.eval_judgments build                       # 정답 템플릿 만들기 (제출 실행 10곳 × 24문항 = 240행)
    python -m eval.eval_judgments                             # 채점: 제출 실행 기록 vs 정답
    python -m eval.eval_judgments score outputs/xxx/run_log.json   # 다른 실행(규칙 변경 뒤 등)을 같은 정답으로 채점

라벨 방법: gold 칸에 YES / NO / UNKNOWN / N/A 중 하나. 기준은 "이 문항의 YES 요건(need)을 만족하는 공개 근거가 있는가"이다.
quote·url 은 모델이 고른 근거라 참고만 하고, 직접 검색해 더 나은 근거를 찾으면 gold_note 에 적는다. 비워 둔 행은 채점에서 빠진다.
build 는 이미 라벨한 gold 칸을 덮어쓰지 않도록 파일이 있으면 멈춘다.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import sys
from collections import Counter

from core.config import path
from core.judge import load_rubric

GOLD = "data/eval/judgment_gold.csv"
RUN_LOG = "outputs/run_log.json"
FIELDS = ["candidate", "qid", "short", "question", "need", "pred", "quote", "rationale", "url",
          "gold", "gold_note", "labeler"]
LABELS = {"YES", "NO", "UNKNOWN", "N/A"}


def _sha(prefix: str, key: str) -> str:
    return prefix + hashlib.sha1(key.encode()).hexdigest()[:5]   # tools/sources.py SourceRegistry._id 와 같은 규칙


def _url_index() -> dict:
    """근거 id → URL. 실행 기록에는 근거 저장소가 없어서, 재현용 캐시의 검색 결과·문서 조각으로 id 를 다시 계산한다."""
    idx = {}
    for f in glob.glob(str(path("replay/search/*.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("url"):
                idx[_sha("W", "web:" + r["url"])] = r["url"]
    for f in glob.glob(str(path("replay/chunks_*.json"))):
        for c in json.load(open(f, encoding="utf-8")):
            m = c["m"]
            key = f"doc:{m.get('doc_id')}:{m.get('page')}:{hashlib.md5(c['t'].encode()).hexdigest()[:8]}"
            idx[_sha("D", key)] = f"{m.get('title')} p.{m.get('page')} ({m.get('url')})"
    return idx


def build(run_log: str = RUN_LOG) -> None:
    out = path(GOLD)
    if out.exists():
        raise SystemExit(f"{GOLD} 가 이미 있습니다 (라벨 보호). 새로 만들려면 먼저 옮기거나 지우세요.")
    q = {x["id"]: x for d in load_rubric()["dimensions"] for x in d["questions"]}
    urls = _url_index()
    rows = []
    for e in json.load(open(path(run_log), encoding="utf-8"))["evaluations"]:
        for r in e["rows"]:
            rows.append({"candidate": e["name"], "qid": r["qid"], "short": q[r["qid"]].get("short", ""),
                         "question": q[r["qid"]]["text"], "need": q[r["qid"]]["need"], "pred": r["answer"],
                         "quote": r.get("quote", ""), "rationale": r.get("rationale", ""),
                         "url": " | ".join(urls.get(i, i) for i in r.get("evidence_ids", [])),
                         "gold": "", "gold_note": "", "labeler": ""})
    with open(out, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig: 엑셀에서 한글이 깨지지 않게
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}행 → {GOLD} (근거 URL 복원 {sum(bool(r['url']) and 'http' in r['url'] for r in rows)}행)")


def score(run_log: str = RUN_LOG, gold_csv: str = GOLD, out: str = "outputs/eval/judgment_eval.json") -> dict:
    with open(path(gold_csv), encoding="utf-8-sig") as f:
        gold = {(r["candidate"], r["qid"]): r["gold"].strip().upper() for r in csv.DictReader(f)}
    bad = {v for v in gold.values() if v and v not in LABELS}
    if bad:
        raise SystemExit(f"gold 칸에 알 수 없는 값: {sorted(bad)} (가능: {sorted(LABELS)})")
    pred = {(e["name"], r["qid"]): r["answer"]
            for e in json.load(open(path(run_log), encoding="utf-8"))["evaluations"] for r in e["rows"]}
    pairs = [(k, gold[k], pred[k]) for k in gold if gold[k] and k in pred]
    if not pairs:
        raise SystemExit("채점할 행이 없습니다 (gold 칸이 비었거나 실행 기록의 후보가 정답과 다름)")
    yes = [(g, p) for _, g, p in pairs if p == "YES"]
    gold_yes = [(g, p) for _, g, p in pairs if g == "YES"]
    by_q = Counter(k[1] for k, g, p in pairs if g != p)
    summary = {"run_log": run_log, "n": len(pairs), "accuracy": round(sum(g == p for _, g, p in pairs) / len(pairs), 3),
               "yes_precision": round(sum(g == "YES" for g, _ in yes) / len(yes), 3) if yes else None,  # YES 라고 한 것 중 맞은 비율
               "yes_recall": round(sum(p == "YES" for _, p in gold_yes) / len(gold_yes), 3) if gold_yes else None,
               "confusion": {f"{g}->{p}": n for (g, p), n in sorted(Counter((g, p) for _, g, p in pairs).items())},
               "wrong_by_qid": dict(by_q.most_common()),
               "wrong": [{"candidate": k[0], "qid": k[1], "gold": g, "pred": p} for k, g, p in pairs if g != p]}
    path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "wrong"}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["score"]
    {"build": build, "score": score}[cmd](*rest)
