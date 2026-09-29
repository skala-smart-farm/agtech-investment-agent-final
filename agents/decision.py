"""🧮 투자 판단 에이전트 (노드 decide).

1) 판정: 실적(R1~R4)·투자조건(D1~D4)은 여기서 core.judge.judge_dimension 으로 판정한다. 창업자·시장성·제품/기술력·경쟁 우위는
   각 분석 에이전트가 state[키]['criterion'] 에 둔 판정을 쓰고, 없거나 문항이 모자라면 여기서 최종 근거 풀로 보완 판정한다.
2) 점수(코드): Payne Scorecard 의 비교 원리(동종 투자 유치 기업 평균 = 1.00)로 배수 M 을 구한다.
   - 문항 신호 x: YES +1, NO −1, 미확인 0 (N/A 제외)
   - 동종 평균 x̄: 보정 실행(app.py --calibrate)이 만든 data/reference_class.json 의 문항별 평균.
     대상이 기준 집단에 있으면 자기 자신을 뺀 평균(LOO). 파일이 없거나 문항 목록이 옛것이거나 decision.reference_min_n 곳보다
     적으면 x̄ = 0(fallback, 설계 가정)
   - 기준 d 마다 c_d = clip(1 + step × 평균_q(x_q − x̄_q)), M = Σ (가중치/100) × c_d
3) 결정(코드): 투자 ⇔ M ≥ decision.threshold ∧ 창업자 YES ≥ decision.min_founder_yes ∧ Deal-killer 없음.
   보류 유형(보고서의 '왜 안 되는가' 이름표, 앞선 것 우선): Deal-killer > 창업자 근거 없음 > 정보 부족 > 동종 대비 열위.
4) 보고서 재료(코드): 뒤집힘 조건, 실사 항목, Bessemer 10문, ROI 참고치, 기준 배수 민감도, 동종 순위.
보정 실행(workflow.calibrate)에서는 기준 집단을 만드는 중이라 결정하지 않는다(decision None, 판정·신호만 기록).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from core.config import ROOT, get_config
from core.judge import SIGNAL, evidence_pool, judge_dimension
from core.judge import load_rubric  # 판정 공용 모듈로 옮김. agents.report 가 이 이름으로 import 하므로 재수출
from tools.grounding import norm
from tools.sources import SourceRegistry

AGENT = "decision"
HOLD_TYPES = ("Deal-killer", "창업자 근거 없음", "정보 부족", "동종 대비 열위")   # 앞선 것 우선
UNDISCLOSED = ("비공개", "미공개", "undisclosed", "비밀", "n/a")

__all__ = ["decision_node", "load_rubric", "load_reference", "write_reference_class", "write_threshold_sensitivity",
           "payne_multiplier", "decide_rule", "flip_conditions", "bessemer_panel", "roi", "parse_amount"]


# ── 공통 계산 도우미
def _x(r: dict) -> int | None:
    """문항 신호. 행에 x 가 있으면 그 값, 없으면 판정 값으로 정한다 (v1 행에는 x 가 없음)."""
    return r["x"] if "x" in r else SIGNAL.get(r.get("answer"), 0)


def _qids(rubric: dict) -> list[str]:
    return [q["id"] for d in rubric["dimensions"] for q in d["questions"]]


def _questions(rubric: dict) -> dict:
    """qid → 문항 정의 (dim 포함)."""
    return {q["id"]: {**q, "dim": d["id"]} for d in rubric["dimensions"] for q in d["questions"]}


def _founder_yes(rows: list[dict]) -> int:
    return sum(r["answer"] == "YES" for r in rows if r["dim"] == "founder")


def _killers(rows: list[dict], rubric: dict) -> list[str]:
    """rubric deal_killers 의 조건(예: P4 = NO)을 모두 만족하는 Deal-killer id."""
    v = {r["qid"]: r["answer"] for r in rows}
    return [k["id"] for k in rubric["deal_killers"] if all(v.get(q) == a for q, a in k["when"].items())]


def _decide(M: float, founder_yes: int, killers: list[str], threshold: float, min_founder_yes: int) -> str:
    return "투자" if M >= threshold and founder_yes >= min_founder_yes and not killers else "보류"


def _with_yes(rows: list[dict], i: int) -> list[dict]:
    """i 번째 문항을 YES 로 바꾼 사본 (뒤집힘·실사 영향 계산용)."""
    out = list(rows)
    out[i] = {**rows[i], "answer": "YES", "x": 1}
    return out


def _rows_from_signals(signals: dict, rubric: dict) -> list[dict]:
    """기준 집단 파일의 신호 {qid: 1|-1|0|None} → 판정 행 (배수·Deal-killer 계산용)."""
    answer = {1: "YES", -1: "NO", 0: "UNKNOWN", None: "N/A"}
    return [{"dim": d["id"], "qid": q["id"], "answer": answer[signals.get(q["id"], 0)], "x": signals.get(q["id"], 0)}
            for d in rubric["dimensions"] for q in d["questions"]]


# ── 동종 기준 집단 (계약 C10)
def _ref_path(p: str | None = None) -> Path:
    p = Path(p or get_config().decision.reference_file)
    return p if p.is_absolute() else ROOT / p


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mean(members: list[dict], qids: list[str]) -> dict:
    """문항별 평균 신호 (N/A 인 구성원은 그 문항 평균에서 뺀다. 모두 N/A 면 0)."""
    out = {}
    for q in qids:
        v = [m["signals"][q] for m in members if m["signals"].get(q) is not None]
        out[q] = sum(v) / len(v) if v else 0.0
    return out


def _reference(data: dict | None, exclude: str | None, rubric: dict, min_n: int) -> dict:
    """기준 집단 파일 내용 → 이번 평가에 쓸 동종 평균. exclude 와 이름이 같은 구성원은 뺀다(LOO)."""
    qids = _qids(rubric)
    members = list((data or {}).get("members") or [])
    note = None
    if data is None:
        note = "기준 집단 파일 없음"
    elif data.get("rubric_qids") != qids:
        note, members = "기준 집단 파일의 문항 목록이 현재 평가표와 다름(옛 파일)", []
    used = [m for m in members if not (exclude and norm(m["name"]) == norm(exclude))]
    ok = note is None and len(used) >= min_n
    if note is None and not ok:
        note = f"기준 집단 {len(used)}곳 < 최소 {min_n}곳"
    return {"mean": _mean(used, qids) if ok else {q: 0.0 for q in qids}, "n": len(used), "loo": len(used) < len(members),
            "source": "calibration" if ok else "fallback", "members": [m["name"] for m in used],
            "run_date": (data or {}).get("run_date"), "note": None if ok else f"{note} → 동종 평균 신호 0 (설계 가정)"}


def load_reference(exclude: str | None = None) -> dict:
    """동종 기준 집단(config decision.reference_file)의 문항별 평균 신호.
    exclude: 평가 대상 이름 — 기준 집단에 있으면 빼고 평균을 낸다(자기 제외, LOO).
    파일이 없거나, 문항 목록이 현재 rubric 과 다르거나, 빼고 남은 구성원이 decision.reference_min_n 보다 적으면
    평균 0(source 'fallback')이다.
    반환: {'mean': {qid: float}, 'n': int, 'loo': bool, 'source': 'calibration'|'fallback', 'members': [이름],
          'run_date': str|None, 'note': fallback 사유 또는 None}"""
    return _reference(_read_json(_ref_path()), exclude, load_rubric(), get_config().decision.reference_min_n)


def _reference_data(evaluations: list[dict], run_date: str, created_by: str) -> dict:
    rubric = load_rubric()
    qids = _qids(rubric)
    members: dict[str, dict] = {}
    for e in evaluations:
        rows = (e.get("scorecard") or {}).get("rows") or e.get("rows") or []
        if not rows:
            continue
        sig = {r["qid"]: _x(r) for r in rows}
        members[norm(e["name"])] = {"name": e["name"], "region": e.get("region"), "stage": e.get("stage"),
                                    "segment_id": e.get("segment_id"), "signals": {q: sig.get(q, 0) for q in qids}}
    ms = list(members.values())   # 같은 이름을 두 번 평가했으면 마지막 평가를 쓴다
    return {"version": 1, "run_date": run_date, "rubric_qids": qids, "created_by": created_by, "n": len(ms),
            "members": ms, "mean": {q: round(v, 4) for q, v in _mean(ms, qids).items()}}


def _write_json(p: str | Path, data: dict) -> None:
    p = Path(p) if Path(p).is_absolute() else ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_reference_class(evaluations: list[dict], path: str, run_date: str) -> dict:
    """보정 실행(app.py --calibrate)의 평가 결과로 동종 기준 집단 파일을 쓴다.
    evaluations 원소의 scorecard.rows(없으면 rows)에서 문항 신호를 읽는다. 같은 이름은 마지막 평가만 남긴다.
    파일 형식: {'version': 1, 'run_date', 'rubric_qids': [24개], 'created_by': 'app.py --calibrate', 'n',
               'members': [{'name','region','stage','segment_id','signals': {qid: 1|-1|0|None}}], 'mean': {qid: float}}
    반환: 쓴 내용(dict)."""
    data = _reference_data(evaluations, run_date, "app.py --calibrate")
    _write_json(path, data)
    return data


def _peer_scores(data: dict, rubric: dict) -> list[dict]:
    """기준 집단 구성원마다 자기 제외(LOO) 동종 평균으로 구한 배수·창업자 YES·Deal-killer."""
    d = get_config().decision
    out = []
    for m in data.get("members") or []:
        rows = _rows_from_signals(m["signals"], rubric)
        ref = _reference(data, m["name"], rubric, d.reference_min_n)
        M, _ = payne_multiplier(rows, ref["mean"], rubric, d.step, d.clip)
        out.append({"name": m["name"], "multiplier": M, "founder_yes": _founder_yes(rows),
                    "killers": _killers(rows, rubric)})
    return out


def write_threshold_sensitivity(ref_path: str, out_path: str) -> dict:
    """기준 집단 구성원마다 자기 제외 배수(multiplier_loo)를 구하고, config decision.sensitivity 의 각 기준 배수에서의 결정을 기록한다.
    파일 형식: {'thresholds': [...], 'threshold': 현재 기준, 'reference_n': int,
               'members': [{'name','multiplier_loo','founder_yes','killers','decision_at': {'1.10': '투자'|'보류', …}}],
               'invest_count_at': {'1.00': int, …}}   (members 는 배수 내림차순)
    반환: 쓴 내용(dict)."""
    d = get_config().decision
    data = _read_json(_ref_path(ref_path))
    if data is None:
        raise FileNotFoundError(f"기준 집단 파일을 읽을 수 없음: {ref_path}")
    rubric = load_rubric()
    keys = [(f"{t:.2f}", t) for t in d.sensitivity]
    members = [{"name": p["name"], "multiplier_loo": p["multiplier"], "founder_yes": p["founder_yes"],
                "killers": p["killers"],
                "decision_at": {k: _decide(p["multiplier"], p["founder_yes"], p["killers"], t, d.min_founder_yes)
                                for k, t in keys}}
               for p in sorted(_peer_scores(data, rubric), key=lambda p: -p["multiplier"])]
    out = {"thresholds": [t for _, t in keys], "threshold": d.threshold, "reference_n": len(members),
           "members": members,
           "invest_count_at": {k: sum(m["decision_at"][k] == "투자" for m in members) for k, _ in keys}}
    _write_json(out_path, out)
    return out


# ── 점수와 결정 (계약 C7)
def payne_multiplier(rows: list[dict], mean: dict, rubric: dict, step: float, clip: list) -> tuple[float, list[dict]]:
    """Payne Scorecard 의 비교 원리(동종 평균 = 1.00)로 배수 M 을 구한다.
    기준 d 마다 c_d = clip(1 + step × 평균_q(x_q − x̄_q)) (N/A 문항 제외, 문항이 없으면 1.0), M = Σ (weight/100) × c_d.
    반환: (M 소수 4자리, criteria) — criteria 원소 {dim, name, weight, yes, no, unknown, n, pct, peer_mean, contribution}
    pct = c_d × 100 (동종 평균 100 대비), peer_mean = 그 기준 문항들의 동종 평균 신호 x̄ 평균, contribution = weight/100 × c_d."""
    lo, hi = clip
    M, criteria = 0.0, []
    for d in rubric["dimensions"]:
        rs = [r for r in rows if r["dim"] == d["id"]]
        used = [(r["qid"], _x(r)) for r in rs if _x(r) is not None]
        diffs = [x - mean.get(q, 0.0) for q, x in used]
        c = min(hi, max(lo, 1 + step * sum(diffs) / len(diffs))) if diffs else 1.0
        contribution = d["weight"] / 100 * c
        M += contribution
        criteria.append({"dim": d["id"], "name": d["name"], "weight": d["weight"],
                         "yes": sum(r["answer"] == "YES" for r in rs), "no": sum(r["answer"] == "NO" for r in rs),
                         "unknown": sum(r["answer"] == "UNKNOWN" for r in rs), "n": len(used),
                         "pct": round(c * 100, 1),
                         "peer_mean": round(sum(mean.get(q, 0.0) for q, _ in used) / len(used), 3) if used else 0.0,
                         "contribution": round(contribution, 4)})
    return round(M, 4), criteria


def decide_rule(M: float, founder_yes: int, killers: list[str], unknown_ratio: float, cfg) -> tuple[str, str | None, list[str]]:
    """투자 ⇔ M ≥ decision.threshold ∧ founder_yes ≥ decision.min_founder_yes ∧ Deal-killer 없음.
    보류 유형 우선순위: 'Deal-killer' > '창업자 근거 없음' > '정보 부족'(unknown_ratio ≥ info_gap_ratio) > '동종 대비 열위'.
    '정보 부족'과 '동종 대비 열위'는 둘 다 M 이 기준에 못 미친 경우이고, 미확인 비율로 이름만 나눈다.
    반환: (결정 '투자'|'보류', 보류 유형 또는 None, 사람이 읽는 사유 문장 목록 — 보류면 보류 유형의 사유가 맨 앞)"""
    d = cfg.decision
    t, need, gap = d.threshold, d.min_founder_yes, d.info_gap_ratio
    score = f"동종 평균 대비 {M * 100:.0f}(평균 100, 기준 {t * 100:.0f})"   # 보고서 표기와 같은 형식
    if _decide(M, founder_yes, killers, t, need) == "투자":
        return "투자", None, [f"{score} — 기준 충족", f"창업자 문항(F1~F4) YES {founder_yes}개 (최소 {need}개)",
                            "Deal-killer 없음"]
    text = {k["id"]: k["text"] for k in load_rubric()["deal_killers"]}
    reasons = [f"Deal-killer {k}: {text.get(k, '')}" for k in killers]
    if founder_yes < need:
        reasons.append(f"창업자 문항(F1~F4) YES {founder_yes}개 — 최소 {need}개 필요")
    if M < t and unknown_ratio >= gap:
        reasons.append(f"미확인 문항 {unknown_ratio:.0%} (≥ {gap:.0%}) — 공개 정보로는 판단하기 어려움")
    if M < t:
        reasons.append(f"{score} — 기준 미달")
    hold = ("Deal-killer" if killers else "창업자 근거 없음" if founder_yes < need
            else "정보 부족" if unknown_ratio >= gap else "동종 대비 열위")
    return "보류", hold, reasons


def flip_conditions(rows: list[dict], mean: dict, rubric: dict, cfg, founder_ok: bool, killers: list[str]) -> dict | None:
    """보류 후보의 뒤집힘 조건: 미확인 문항 중 YES 로 확인되면 M 이 가장 많이 오르는 문항을 하나씩(탐욕적으로) 골라,
    투자 조건(M ≥ 기준, 창업자 요건)을 채우거나 decision.flip_max_items 개가 될 때까지 더하고 새 배수 M' 을 계산한다.
    창업자 요건이 모자라면(founder_ok False) 창업자 문항을 먼저 넣는다.
    Deal-killer 가 있으면 미확인 문항으로는 뒤집을 수 없어 new_multiplier None 과 'Kx 해소 필요'를 돌려준다.
    반환: {'qids': [str], 'items': [str], 'new_multiplier': float|None, 'reached': bool, 'note': str}
    (호출하는 decision_node 는 보류일 때만 부르고, 투자면 flip 을 None 으로 둔다)"""
    d = cfg.decision
    info = _questions(rubric)
    if killers:
        rule = {k["id"]: k for k in rubric["deal_killers"]}
        qids = list(dict.fromkeys(q for k in killers for q in rule[k]["when"]))
        return {"qids": qids, "items": [f"{q} {info[q]['short']}" for q in qids], "new_multiplier": None,
                "reached": False, "note": " · ".join(f"{k} 해소 필요({rule[k]['text']})" for k in killers)}
    cur = list(rows)
    M, _ = payne_multiplier(cur, mean, rubric, d.step, d.clip)
    fy, picked = _founder_yes(cur), []

    def done() -> bool:
        return M >= d.threshold and fy >= d.min_founder_yes

    while not done() and len(picked) < d.flip_max_items:
        cand = [i for i, r in enumerate(cur) if r["answer"] == "UNKNOWN"]
        if not founder_ok and fy < d.min_founder_yes:
            cand = [i for i in cand if cur[i]["dim"] == "founder"]   # 창업자 요건부터 채운다
        if not cand:
            break
        gain = {i: payne_multiplier(_with_yes(cur, i), mean, rubric, d.step, d.clip)[0] for i in cand}
        best = max(cand, key=lambda i: (gain[i], -i))                 # 같으면 평가표 순서
        cur, M = _with_yes(cur, best), gain[best]
        fy += cur[best]["dim"] == "founder"
        picked.append(cur[best]["qid"])
    t = f"{d.threshold * 100:.0f}"
    if not picked:
        note = ("창업자 문항에 미확인이 없어(모두 NO) 미확인 문항 확인만으로는 뒤집을 수 없음" if fy < d.min_founder_yes
                else "미확인 문항이 없어 뒤집힘 조건 없음")
    elif done():
        note = f"{'·'.join(picked)} 이(가) YES 로 확인되면 동종 평균 대비 {M * 100:.0f}(기준 {t}) → 투자"
    else:
        note = (f"미확인 {len(picked)}문항({'·'.join(picked)})이 YES 로 확인돼도 동종 평균 대비 {M * 100:.0f}"
                f"{f', 창업자 YES {fy}개' if fy < d.min_founder_yes else ''} — 기준 미달(최대 {d.flip_max_items}문항)")
    return {"qids": picked, "items": [f"{q} {info[q]['short']}" for q in picked], "new_multiplier": round(M, 4),
            "reached": done(), "note": note}


def _dd_items(rows: list[dict], mean: dict, rubric: dict, cfg, killers: list[str]) -> list[dict]:
    """실사 항목: 미확인·NO 문항 중 YES 로 확인될 때 배수가 크게 오르는 순서(Deal-killer 문항은 맨 앞)로
    decision.dd_max_items 개를 rubric dd_item 문구로 만든다. 반환 [{qid, item, why}]."""
    d = cfg.decision
    info = _questions(rubric)
    base, _ = payne_multiplier(rows, mean, rubric, d.step, d.clip)
    rule = {k["id"]: k for k in rubric["deal_killers"]}
    killer_q = {q: k for k in killers for q in rule[k]["when"]}
    cand = []
    for i, r in enumerate(rows):
        if r["answer"] not in ("UNKNOWN", "NO"):
            continue
        gain = payne_multiplier(_with_yes(rows, i), mean, rubric, d.step, d.clip)[0] - base
        why = f"{'NO(반대 근거 있음)' if r['answer'] == 'NO' else '미확인'} — YES 로 확인되면 배수 +{gain:.3f}"
        if r["qid"] in killer_q:
            why += f", Deal-killer {killer_q[r['qid']]} 관련"
        cand.append((r["qid"] not in killer_q, -gain, i, {"qid": r["qid"], "item": info[r["qid"]]["dd_item"], "why": why}))
    return [c[3] for c in sorted(cand)[:d.dd_max_items]]


def bessemer_panel(rows: list[dict], rubric: dict) -> list[dict]:
    """rubric.yaml bessemer 10문마다 via 문항의 판정으로 답을 정한다(NO 가 하나라도 있으면 NO, 그다음 YES, 그 밖은 미확인).
    결론은 Scorecard 규칙으로만 내고, 이 표는 점검용이다.
    반환: [{'q': int, 'text': str, 'answer': 'YES'|'NO'|'미확인', 'via': [qid], 'proxy': bool}] (10행)"""
    v = {r["qid"]: r["answer"] for r in rows}
    out = []
    for b in rubric["bessemer"]:
        ans = [v.get(q) for q in b["via"]]
        out.append({"q": b["q"], "text": b["text"], "answer": "NO" if "NO" in ans else "YES" if "YES" in ans else "미확인",
                    "via": list(b["via"]), "proxy": bool(b.get("proxy"))})
    return out


# ── ROI 참고치 (점수·결정에 쓰지 않음)
_NUM = r"\d[\d,]*(?:\.\d+)?"
_KR_PART = re.compile(rf"({_NUM})\s*(조|억|천만|백만|만)")
_EN_PART = re.compile(rf"({_NUM})\s*(billion|bn|million|mn|thousand|[bmk])(?![a-z])", re.I)
_KR_UNIT = {"조": 1e12, "억": 1e8, "천만": 1e7, "백만": 1e6, "만": 1e4}
_EN_UNIT = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
_USD = re.compile(r"\$|usd|달러|dollar", re.I)
_KRW = re.compile(r"₩|krw|원|won", re.I)
_OTHER = re.compile(r"€|£|¥|eur|euro|유로|gbp|jpy|엔화|cad|aud|chf|cny|위안", re.I)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_amount(s: str) -> dict | None:
    """라운드 금액 문자열 → {'krw': float|None, 'usd_m': float|None}. 예: '30억 원' → krw 3e9, '$12M'·'12 million' → usd_m 12.
    - 첫 금액 표현만 읽는다. 한국식 단위는 이어진 조각을 더한다('259억 9991만원' → 259.9991억, '1,500만 달러' → usd_m 15).
    - 통화: 금액 바로 앞뒤 4글자의 $·달러·USD / 원·KRW 표기. 표기가 없으면 조·억·만 단위는 원화, million·M 단위는 달러로 본다.
    - '비공개'·빈 문자열처럼 금액이 없거나, 단위·통화 표기 없는 숫자만 있거나, 달러·원화가 아닌 통화(€ 등)면 None."""
    t = str(s or "").strip()
    if not re.search(r"\d", t) or any(u in t.lower() for u in UNDISCLOSED):
        return None
    kr, en = _KR_PART.search(t), _EN_PART.search(t)
    if kr and (not en or kr.start() <= en.start()):
        value, start, end = 0.0, kr.start(), kr.start()
        for part in _KR_PART.finditer(t, kr.start()):
            if t[end:part.start()].strip():
                break
            value += _num(part.group(1)) * _KR_UNIT[part.group(2)]
            end = part.end()
        default = "KRW"
    elif en:
        value, start, end, default = _num(en.group(1)) * _EN_UNIT[en.group(2).lower()], en.start(), en.end(), "USD"
    else:
        m = re.search(_NUM, t)
        value, start, end, default = _num(m.group()), m.start(), m.end(), None
    near = t[max(0, start - 4):start] + " " + t[end:end + 4]   # 금액 바로 앞뒤 4글자의 통화 표기
    cur = ("USD" if _USD.search(near) else "KRW" if _KRW.search(near) else None if _OTHER.search(near) else default)
    if cur == "USD":
        return {"krw": None, "usd_m": value / 1e6}
    if cur == "KRW":
        return {"krw": value, "usd_m": None}
    return None


def roi(current: dict, market: dict, cfg) -> dict:
    """ROI 참고치 (점수에 쓰지 않음). 두 가지만 계산한다.
    1) 라운드 금액 대 단계별 중앙값: config roi.stage_median_usd_m (AgFunder 2026 p.13)
    2) VC Method 필요 Exit(가정): post-money = 금액 ÷ 지분율 가정(roi.stake_assumption), 필요 Exit = post-money × roi.target_multiple
    라운드 금액을 모르면 computable=False. 가정 값은 assumptions 에 '가정'을 붙인 문장으로 남긴다.
    market 은 계약 시그니처라 받지만 쓰지 않는다(세부 시장 규모 대비 회수 여력 경고는 넣지 않기로 함).
    반환: {'computable','round_amount_raw','round_amount_usd_m','stage','stage_median_usd_m','vs_stage_median',
           'stake_assumption','post_money_usd_m': [lo, hi]|None,'target_multiple','required_exit_usd_m': [lo, hi]|None,
           'assumptions': [str],'source_ids': [str],'stage_median_source': str,'note': str|None}"""
    r = cfg.roi
    raw, stage = str(current.get("round_amount") or ""), str(current.get("stage") or "")
    stakes, target = list(r.stake_assumption), r.target_multiple
    amt = parse_amount(raw)
    usd_m, assumptions = None, []
    if amt and amt["usd_m"] is not None:
        usd_m = amt["usd_m"]
    elif amt and amt["krw"] is not None:
        usd_m = amt["krw"] / r.fx_krw_per_usd / 1e6
        assumptions.append(f"환율 가정: 1달러 = {r.fx_krw_per_usd:,}원")
    median = r.stage_median_usd_m.get(stage)
    if median is not None and stage.startswith("Pre-"):
        assumptions.append(f"{stage} 중앙값은 앞 단계 값(${median}M)을 쓴 가정")
    note = None
    if usd_m is None:
        note = (f"라운드 금액 '{raw}' 을(를) 달러·원화로 읽을 수 없어 산정 불가 (실사 항목)" if raw.strip()
                else "라운드 금액 비공개 → 산정 불가 (실사 항목)")
    else:
        assumptions += [f"지분율 가정 {stakes[0]:.0%}~{stakes[-1]:.0%}: post-money = 라운드 금액 ÷ 지분율",
                        f"목표 회수 배수 가정 {target}배(VC Method): 필요 Exit = post-money × {target}"]
    post = [round(usd_m / max(stakes), 2), round(usd_m / min(stakes), 2)] if usd_m is not None else None
    return {"computable": usd_m is not None, "round_amount_raw": raw,
            "round_amount_usd_m": round(usd_m, 2) if usd_m is not None else None, "stage": stage,
            "stage_median_usd_m": median,
            "vs_stage_median": round(usd_m / median, 2) if usd_m is not None and median else None,
            "stake_assumption": stakes, "post_money_usd_m": post, "target_multiple": target,
            "required_exit_usd_m": [round(v * target, 1) for v in post] if post else None,
            "assumptions": assumptions, "source_ids": list(current.get("stage_evidence_ids") or []),
            "stage_median_source": r.get("stage_median_source", ""), "note": note}


# ── 노드
def _nps_line(n: dict) -> str:
    if n.get("status") != "matched":
        return "국민연금 가입 사업장 목록에서 확인 안 됨 (3인 미만 법인이거나 사명 다름)"
    return (f"국민연금 가입자 {n['members']}명({n['ym']}), 최초 가입 {n['first_date']}, "
            f"{'탈퇴' if n['withdrawn'] else '가입 중'}")


def _analysis_text(state: dict) -> str:
    """판정 프롬프트의 [분석 요약]. v1 형식을 따르되 창업자·팀은 👤 창업자 에이전트 결과(state['founder'])에서 읽는다."""
    c, f = state["current"], state.get("founder") or {}
    t, m, k = state.get("tech") or {}, state.get("market") or {}, state.get("competition") or {}
    people = "; ".join(f"{p['name']}({p.get('role')}): {p.get('background')}" for p in f.get("people", [])) or "확인 불가"
    t0 = f"창업 시점 {f['t0']} ({f.get('t0_source')})" if f.get("t0") else "창업 시점 확인 불가"
    return "\n".join([
        f"[후보] {c['official_name']} | 단계 {c.get('stage')} ({c.get('round_date') or '시점 미상'}, "
        f"{c.get('round_amount') or '금액 미상'}) | 설립 {c.get('founded_year') or '확인 불가'} | {c.get('one_line')}",
        f"[기술] 제품: {t.get('product')} / 핵심 기술: {t.get('core_technology')} / 성숙도: {t.get('maturity')} "
        f"({t.get('maturity_evidence')}) / 특허·인증: {t.get('ip_evidence')}",
        f"[팀] {f.get('team_assessment')} / 창업자: {people}",
        f"[고용·창업 시기] {_nps_line(c.get('nps') or {})} / {t0}",
        f"[시장] 규모: {m.get('market_size')} / 성장: {m.get('growth')} / 지불 의향: {m.get('willingness_to_pay')}",
        f"[경쟁] 차별성: {k.get('differentiation')} / 진입장벽: {k.get('entry_barriers')}",
    ])


def _complete(crit: dict | None, dim: dict) -> bool:
    """분석 에이전트의 criterion 이 이 차원의 문항을 빠짐없이 판정했는지."""
    rows = (crit or {}).get("rows") or []
    return [r.get("qid") for r in rows] == [q["id"] for q in dim["questions"]] and all("answer" in r for r in rows)


def _ranking(data: dict | None, ref: dict, name: str, me: dict, rubric: dict, cfg) -> tuple[list[dict], int | None, int]:
    """동종 순위: 기준 집단 구성원(자기 제외 배수)과 이번 대상을 배수 내림차순으로. fallback 이면 순위를 만들지 않는다."""
    if ref["source"] != "calibration":
        return [], None, 0
    d = cfg.decision
    out = [{**p, "decision": _decide(p["multiplier"], p["founder_yes"], p["killers"], d.threshold, d.min_founder_yes),
            "is_target": False}
           for p in _peer_scores(data, rubric) if norm(p["name"]) != norm(name)]
    out.append({**me, "is_target": True})
    out.sort(key=lambda e: -e["multiplier"])
    return out, next(i + 1 for i, e in enumerate(out) if e["is_target"]), len(out)


def decision_node(state: dict) -> dict:
    cfg = get_config()
    dc = cfg.decision
    rubric = load_rubric()
    reg = SourceRegistry(state.get("registry"))
    c = state["current"]
    name = c["official_name"]
    run_date = state.get("run_date") or datetime.now().strftime("%Y-%m-%d")
    pool = evidence_pool(state)
    analysis = _analysis_text(state)

    # 1) 판정: 실적·투자조건은 여기서, 나머지는 담당 에이전트의 criterion (없거나 문항이 모자라면 여기서 보완)
    crits, here = {}, []
    for d in rubric["dimensions"]:
        crit = None if d["owner"] == AGENT else (state.get(d["owner"]) or {}).get("criterion")
        if not _complete(crit, d):
            crit = judge_dimension(d["id"], c, pool, reg, analysis, run_date)
            here.append(d["id"])
        crits[d["id"]] = crit
    rows = [r for d in rubric["dimensions"] for r in crits[d["id"]]["rows"]]
    founder_yes, killers = _founder_yes(rows), _killers(rows, rubric)
    judged = [r for r in rows if r["answer"] != "N/A"]
    unknown_ratio = round(sum(r["answer"] == "UNKNOWN" for r in judged) / len(judged), 3) if judged else 1.0
    base = {"rows": rows, "threshold": dc.threshold, "founder_yes": founder_yes, "unknown_ratio": unknown_ratio,
            "deal_killers": killers, "bessemer": bessemer_panel(rows, rubric), "roi": roi(c, state.get("market") or {}, cfg),
            "rejected_yes": [x for cr in crits.values() for x in cr.get("rejected_yes", [])],
            "quote_retried": sum(cr.get("quote_retried", 0) for cr in crits.values()), "judged_in_decide": here}
    counts = f"YES {sum(r['answer'] == 'YES' for r in rows)} · NO {sum(r['answer'] == 'NO' for r in rows)} · 미확인 {unknown_ratio:.0%}"

    if cfg.workflow.get("calibrate"):
        # 2') 보정 실행: 기준 집단을 만드는 중이라 배수·결정을 내지 않고 판정·신호만 남긴다
        _, criteria = payne_multiplier(rows, {}, rubric, dc.step, dc.clip)
        criteria = [{**x, "pct": None, "peer_mean": None, "contribution": None} for x in criteria]
        decision = hold_type = flip = M = score100 = None
        reasons = ["보정 실행: 동종 기준 집단을 만드는 중이라 결정하지 않음"]
        scorecard = {**base, "criteria": criteria, "multiplier": None, "score100": None, "reference": None,
                     "decision": None, "hold_type": None, "reasons": reasons, "flip": None, "dd_items": [],
                     "sensitivity": {}, "ranking": [], "target_rank": None, "peer_n": 0}
        msg = f"[투자 판단·보정] {name}: 판정만 기록 ({counts}, 창업자 YES {founder_yes}, Deal-killer {killers or '없음'})"
    else:
        # 2) 점수: 동종 기준 집단 대비 배수 (대상이 기준 집단에 있으면 자기 제외)
        data = _read_json(_ref_path())
        ref = _reference(data, name, rubric, dc.reference_min_n)
        M, criteria = payne_multiplier(rows, ref["mean"], rubric, dc.step, dc.clip)
        score100 = round(M * 100, 1)
        # 3) 결정과 보고서 재료
        decision, hold_type, reasons = decide_rule(M, founder_yes, killers, unknown_ratio, cfg)
        flip = (flip_conditions(rows, ref["mean"], rubric, cfg, founder_yes >= dc.min_founder_yes, killers)
                if decision == "보류" else None)
        ranking, target_rank, peer_n = _ranking(
            data, ref, name, {"name": name, "multiplier": M, "founder_yes": founder_yes, "killers": killers,
                              "decision": decision}, rubric, cfg)
        scorecard = {**base, "criteria": criteria, "multiplier": M, "score100": score100,
                     "reference": {k: ref[k] for k in ("n", "loo", "source", "members", "run_date", "note")},
                     "decision": decision, "hold_type": hold_type, "reasons": reasons, "flip": flip,
                     "dd_items": _dd_items(rows, ref["mean"], rubric, cfg, killers),
                     "sensitivity": {f"{t:.2f}": _decide(M, founder_yes, killers, t, dc.min_founder_yes)
                                     for t in dc.sensitivity},
                     "ranking": ranking, "target_rank": target_rank, "peer_n": peer_n}
        msg = (f"[투자 판단] {name}: Scorecard {score100}점(동종 평균 100, 기준 {dc.threshold * 100:.0f}, "
               f"기준 집단 {ref['n']}곳{' 자기 제외' if ref['loo'] else ''}{' · fallback' if ref['source'] == 'fallback' else ''}) "
               f"· {counts} · 창업자 YES {founder_yes} · Deal-killer {killers or '없음'} → {decision}"
               + (f" ({hold_type})" if hold_type else ""))
    if here:
        msg += f" [여기서 판정한 기준: {', '.join(here)}]"
    print(msg)

    evaluation = {"name": name, "region": c.get("region"), "segment_id": c.get("segment_id"), "stage": c.get("stage"),
                  "round_date": c.get("round_date"), "round_amount": c.get("round_amount"), "decision": decision,
                  "hold_type": hold_type, "multiplier": M, "score100": score100, "reasons": reasons, "flip": flip,
                  "criteria": criteria, "scorecard": scorecard, "founder": state.get("founder"), "tech": state.get("tech"),
                  "market": state.get("market"), "competition": state.get("competition"), "profile": c}
    out = {"scorecard": scorecard, "decision": decision, "evaluations": [evaluation], "log": [msg]}
    from graph.routes import stops_on_invest  # 라우터와 같은 규칙으로 종료 사유를 적는다

    if decision == "투자" and stops_on_invest(cfg.workflow):
        out["end_reason"] = "invest_found"
    elif state.get("iterations", 0) >= cfg.workflow.max_evaluations:
        out["end_reason"] = "max_evaluations"
    return out
