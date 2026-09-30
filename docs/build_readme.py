"""README.md 생성: 발표 수치가 제출 실행 결과(outputs/run_log.json)·평가 결과(outputs/eval/)·문서 목록(data/)과
항상 같도록 템플릿(docs/README.md.j2)으로 만든다. 숫자는 여기서 파일을 읽어 채우고 템플릿에는 적지 않는다.
결론 모드(투자 추천 / 모두 보류 / 후보 없음)가 바뀌어도 다시 실행하면 된다 (API 호출 없음).

    uv run python -m docs.build_readme
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader

from core.config import ROOT, get_config, path
from rag.loader import load_manifest, total_pages

# 코퍼스 v2 에서 추가한 기술 기준선 문서 (data/manifest.yaml 원칙 5). 쪽 범위·라이선스는 manifest 에서 읽는다
ADDED_DOCS = {"frontiers_pd2024": "작물 병해 진단 AI 리뷰", "worldbank_ai2025": "농업 AI 보고서"}
END_REASON = {"invest_found": "투자 추천이 나와 평가를 멈춤", "max_evaluations": "평가 상한 도달(비용 관리)",
              "exhausted": "후보 소진", "no_eligible": "적격 후보 없음"}
REGION = {"KR": "국내", "GLOBAL": "해외"}
DOWNGRADE_SHORT = {"인용문이 근거 본문에서 확인되지 않음": "인용이 원문에 없음",
                   "인용 주변(±300자)에 회사명이 없음": "인용 주변에 회사명 없음",
                   "최근 24개월 이내 근거 아님": "24개월 밖 근거", "회사 측 발언·계획(제3자 근거 아님)": "회사 측 발언"}


# ── 공통 ─────────────────────────────────────────────────────────────────────
def _j(rel: str):
    f = path(rel)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _text(rel: str) -> str:
    f = path(rel)
    return f.read_text(encoding="utf-8") if f.exists() else ""


def _link(rel: str) -> str:
    """저장소 안 파일 링크 (한글·'+' 가 든 파일명도 GitHub 에서 열리게)."""
    return quote(rel, safe="/")


def _s100(m) -> str:
    """배수 → 보고서 표기(동종 평균 = 100), 소수 첫째 자리까지."""
    return "-" if m is None else f"{m * 100:.1f}"


def _cell(s) -> str:
    return str(s).replace("|", "\\|")


# ── 검색 공급자·검색기 ────────────────────────────────────────────────────────
def _search_providers() -> str:
    """재현용 캐시에 실제로 결과를 준 검색 공급자만 적는다 (키를 받지 못한 공급자를 쓴 것처럼 쓰지 않게)."""
    used = set()
    for f in path(f"{get_config().cache.dir}/search").glob("*.json"):
        used |= {r.get("provider", "tavily") for r in json.loads(f.read_text(encoding="utf-8"))}
    names = [n for p, n in (("serper", "Serper(구글)"), ("tavily", "Tavily")) if p in used]
    return "·".join(names) or "Tavily"


def _retrieval(cfg) -> dict:
    """실제 파이프라인 설정 측정표(outputs/eval/runtime_retriever.json): 지금 설정 행, 처음 고른 설정 행, 규칙 1위."""
    w = [float(x) for x in cfg.rag.ensemble_weights]
    label = "Dense 단독" if w[0] == 0 else f"하이브리드 BM25 {w[0]:g} : Dense {w[1]:g}"
    d = _j("outputs/eval/runtime_retriever.json")
    rows = d.get("rows", []) if isinstance(d, dict) else []
    row = next((x for x in rows if x.get("embedding") == cfg.embedding.model
                and [float(v) for v in x.get("weights[bm25,dense]", [])] == w), None)
    out = {"label": label, "hit4": "-", "mrr4": "-", "n": "-", "unit": "-", "ext": None, "base": None, "rule_top": None,
           "design": None, "k": cfg.rag.candidate_k, "top_k": cfg.rag.top_k}
    if not row:
        return out
    a = row["all"]
    out.update(hit4=f"{a['Hit@4']:.3f}", mrr4=f"{a['MRR@4']:.3f}", n=a.get("n", "-"),
               unit=f"{1 / a['n']:.3f}" if a.get("n") else "-")
    if ext := row.get("extended"):
        out["ext"] = {"n": ext["n"], "hit4": f"{ext['Hit@4']:.3f}", "mrr4": f"{ext['MRR@4']:.3f}"}
    conf = d.get("config", {})
    base_name = (conf.get("comparison_baseline") or {}).get("name")
    if base := next((x for x in rows if x["name"] == base_name and x is not row), None):
        emb = base["embedding"].split("/")[-1].replace("-embed-l-v2.0-ko", "-ko")
        bw = base.get("weights[bm25,dense]", [0, 1])
        out["base"] = {"name": f"{emb} 하이브리드 {bw[0]:g}:{bw[1]:g}", "hit4": f"{base['all']['Hit@4']:.3f}",
                       "mrr4": f"{base['all']['MRR@4']:.3f}"}
    rec = d.get("recommendation") or {}
    if rec.get("name") and rec["name"] != row["name"]:
        pretty = re.sub(r"\bdense$", "Dense 단독", re.sub(r"\bhybrid\b", "하이브리드", rec["name"]))
        out["rule_top"] = {"name": pretty, "hit4": f"{rec['Hit@4']:.3f}", "mrr4": f"{rec['MRR@4']:.3f}"}
    # 설계서(선정 당시) 수치: retrieval_decision.md 의 선정 결과 문장에서 읽는다
    m = re.search(r"Hit@4 ([\d.]+)\(\d+문항 중 \d+\), MRR@4 ([\d.]+)", _text("outputs/eval/retrieval_decision.md"))
    if m:
        out["design"] = {"hit4": m.group(1), "mrr4": m.group(2)}
    return out


# ── 평가 결과 ─────────────────────────────────────────────────────────────────
def _judge() -> dict | None:
    d = _j("outputs/eval/judge_eval.json")
    s = d.get("summary") if isinstance(d, dict) else None
    if not s:
        return None
    ko = {"docs": "문서", "web": "웹", "both": "문서+웹", "direct": "바로 답"}
    rate = lambda k: f"{s[k] * 100:.0f}%" if s.get(k) is not None else "-"  # noqa: E731
    return {"line": f"Relevance {s['relevance']:.2f} · Faithfulness {s['faithfulness']:.2f} · Correctness {s['correctness']:.2f}",
            "n": s.get("n", "-"), "unit": f"{1 / s['n']:.2f}" if s.get("n") else "-",
            "routes": " · ".join(f"{ko.get(k, k)} {v}" for k, v in (s.get("route_dist") or {}).items() if v),
            "rewrite": rate("rewrite_rate"), "web": rate("web_fallback_rate")}


def _eligibility() -> dict:
    out = {}
    for k in ("gold", "holdout"):
        s = _j(f"outputs/eval/eligibility_eval_{k}.json").get("summary") or {}
        out[k] = {"n": s.get("n", "-"), "acc": f"{s.get('accuracy', 0):.2f}", "fn": s.get("false_negative", 0),
                  "fp": s.get("false_positive", 0)} if s else None
    return out


def _positive_control() -> dict | None:
    """양성 대조(실제 투자를 받은 후기 기업): 기업별 결정·배수·보류 유형."""
    rows = _j("outputs/eval/positive_control.json")
    if not isinstance(rows, list) or not rows:
        return None
    parts = []
    for x in rows:
        if x.get("multiplier") is not None:
            tail = "투자 추천" if x.get("decision") == "투자" else (x.get("hold_type") or x.get("decision"))
            parts.append(f"{x['name']} {_s100(x['multiplier'])}({tail})")
        else:
            parts.append(f"{x['name']} {x.get('decision') or '-'}")
    return {"n": len(rows), "invest": sum(x.get("decision") == "투자" for x in rows), "parts": " · ".join(parts)}


def _sensitivity() -> dict | None:
    d = _j("outputs/eval/threshold_sensitivity.json")
    if not isinstance(d, dict) or not d.get("invest_count_at"):
        return None
    counts = " · ".join(f"{float(t) * 100:.0f} → {n}곳" for t, n in d["invest_count_at"].items())
    return {"n": d.get("reference_n") or len(d.get("members", [])), "counts": counts, "members": d.get("members", []),
            "invest_count_at": d["invest_count_at"]}


def _reference_mix() -> str:
    """동종 기준 집단의 단계·지역 구성 (Scorecard 의 '같은 지역·같은 단계' 비교를 어디까지 근사했는지)."""
    ms = _j("data/reference_class.json").get("members") or []
    if not ms:
        return ""
    st = Counter(m.get("stage") for m in ms)
    rg = Counter(REGION.get(m.get("region"), m.get("region")) for m in ms)
    return " · ".join(f"{k} {v}" for k, v in st.most_common()) + " / " + " · ".join(f"{k} {v}" for k, v in rg.most_common())


def _downgrades() -> dict | None:
    """코드가 LLM 판정을 미확인으로 내린 수 (보정 실행 = 평가 상한까지 평가한 실행 기준, 없으면 본 실행)."""
    for rel in ("outputs/calibration/run_log.json", "outputs/run_log.json"):
        run = _j(rel)
        evals = run.get("evaluations", []) if isinstance(run, dict) else []
        rows = [x for e in evals for x in e.get("rows") or []]
        if not rows:
            continue
        down = [x for x in rows if "UNKNOWN 강등" in (x.get("rationale") or "")]
        why = Counter(re.split(r" → UNKNOWN 강등| \(", x["rationale"])[0] for x in down)
        return {"companies": len(evals), "rows": len(rows), "down": len(down),
                "pct": f"{len(down) / len(rows) * 100:.0f}%",
                "top": " · ".join(f"{DOWNGRADE_SHORT.get(k, k)} {v}" for k, v in why.most_common(3)),
                "where": "보정 실행" if "calibration" in rel else "본 실행"}
    return None


def _cost(run: dict) -> dict:
    """비용: 보정 실행(평가 상한까지 평가)들의 실측 범위(outputs/cost_history.jsonl), 이 보고서를 만든 실행의 API 호출 수."""
    calib = []
    for line in _text("outputs/cost_history.jsonl").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("mode") == "calibrate" and rec.get("api_calls"):
            calib.append(rec)

    def span(vals: list[str]) -> str:
        return vals[0] if vals[0] == vals[-1] else f"{vals[0]}~{vals[-1]}"

    c = run.get("llm_cost") or {}
    return {"calib": {"usd": span([f"{x:.2f}" for x in sorted(r["usd"] for r in calib)]),
                      "min": span([f"{x / 60:.0f}" for x in sorted(r["elapsed_sec"] for r in calib)]), "n": len(calib)}
            if calib else None,
            "run_calls": c.get("api_calls"), "run_hits": c.get("cache_hits"),
            "run_usd": f"{c['usd']:.2f}" if c.get("usd") is not None else "-"}


# ── 코퍼스 ────────────────────────────────────────────────────────────────────
def _corpus() -> dict:
    docs = load_manifest()
    added = []
    for d in docs:
        if d["doc_id"] in ADDED_DOCS:
            s, e = d.get("page_range") or (1, d.get("pages_total"))
            lic = (d.get("license") or "").split(" (")[0]
            added.append(f"{d['publisher']}({d['year']}) {ADDED_DOCS[d['doc_id']]} pp.{s}–{e}" + (f" ({lic})" if lic else ""))
    trimmed = [d for d in docs if d.get("trim") and d["doc_id"] not in ADDED_DOCS]
    m = re.search(r"심층 평가 (\d+)곳 중 (\d+)곳", _text(get_config().rag.manifest))
    old = re.search(r"설계 단계 코퍼스\((\d+)종 (\d+)쪽", _text("outputs/eval/retrieval_decision.md"))
    return {"n": len(docs), "pages": total_pages(), "limit": get_config().rag.max_total_pages, "added": added,
            "trimmed": len(trimmed), "why": f"평가 {m.group(1)}곳 중 {m.group(2)}곳" if m else "",
            "old": f"{old.group(1)}종 {old.group(2)}쪽" if old else ""}


# ── 실행 결과 (투자 보고서 핵심) ─────────────────────────────────────────────
def _clean(s: str) -> str:
    """보고서 문장을 README 용으로: 인용 번호 [5, 6] 을 빼고 (→n장) 은 (보고서 n장) 으로."""
    s = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", s)
    s = re.sub(r"\s*\(→\s*(\d+)장\)", r" (보고서 \1장)", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _summary(md: str) -> list[dict]:
    """보고서 SUMMARY 줄 '- 칸: 내용'. 상황·결론은 README 가 수치로 따로 쓰므로 뺀다. 근거는 항목별로 나눈다."""
    m = re.search(r"^## SUMMARY\n(.*?)(?=^## )", md, re.S | re.M)
    out = []
    for line in (m.group(1) if m else "").splitlines():
        kv = re.match(r"- ([^:]+):\s*(.+)", line)
        if not kv or kv.group(1) in ("상황", "결론"):
            continue
        parts = [_clean(p) for p in kv.group(2).split(" · ")]
        split = kv.group(1) == "근거" and len(parts) > 1 and all(p.endswith("장)") for p in parts)
        out.append({"k": kv.group(1), "v": _clean(kv.group(2)), "parts": parts if split else []})
    return out


def _chapters(md: str) -> str:
    return " → ".join(re.sub(r"^\d+\.\s*", "", h) for h in re.findall(r"^## (.+)$", md, re.M))


def _dd_items(md: str) -> list[str]:
    """보고서 '실사 확인 항목' 목록의 항목 이름."""
    lines = md.splitlines()
    i = next((k for k, l in enumerate(lines) if l.startswith("실사 확인 항목")), None)
    items = []
    for l in lines[i + 1:] if i is not None else []:
        if l.startswith("- "):
            items.append(l[2:].split(" — ")[0].strip())
        elif items and l.strip():
            break
    return items


def _flip(f) -> str:
    if isinstance(f, str):
        return f
    if not f:
        return ""
    if f.get("items") and f.get("new_multiplier") is not None:
        return (f"{'·'.join(f['items'])} 확인 시 {_s100(f['new_multiplier'])} → "
                + ("투자 추천" if f.get("reached") else "그래도 기준 미달"))
    return f.get("note") or ""


def _pipeline(run: dict, evals: list[dict], cfg) -> dict:
    log = "\n".join(run.get("log", []))
    found = sum(int(x) for x in re.findall(r"\[발굴 \d+라운드\][^\n]*?후보 (\d+)곳", log))
    channels = re.search(r"\[발굴 \d+라운드\] 채널 (\d+)개", log)
    screened = run.get("screened", [])
    eligible = sum(1 for s in screened if s.get("eligible"))
    return {"found": found or "-", "channels": channels.group(1) if channels else "여러",
            "screened": len(screened), "eligible": eligible, "evaluated": len(evals),
            "unevaluated": max(0, eligible - len(evals)), "max_eval": cfg.workflow.max_evaluations,
            "end": END_REASON.get(run.get("end_reason"), run.get("end_reason") or "-")}


def _target(run: dict, r: dict, sens: dict | None, thr: float) -> dict | None:
    """투자 추천 대상: 기준별 표·동종 순위·민감도·미확인 비율·ROI 참고치."""
    e = next((x for x in run.get("evaluations", []) if x["name"] == r.get("target")), None)
    if not e or e.get("multiplier") is None:
        return None
    crit = [{"name": c["name"].split(" (")[0], "w": c["weight"], "y": c["yes"], "n": c["no"], "u": c["unknown"],
             "pct": f"{c['pct']:.1f}"} for c in e.get("criteria") or []]
    scored = [x for x in e.get("rows") or [] if x.get("answer") != "N/A"]
    unknown = sum(x.get("answer") == "UNKNOWN" for x in scored)
    members = sorted((sens or {}).get("members", []), key=lambda x: -x.get("multiplier_loo", 0))
    names = [x["name"] for x in members]
    me = next((x for x in members if x["name"] == e["name"]), {})
    at = me.get("decision_at") or {}
    first_hold = next((t for t, d in at.items() if d != "투자" and float(t) > thr), None)
    roi, roi_line = e.get("roi") or {}, ""
    if roi.get("computable"):
        st, ex = roi.get("stake_assumption") or [0, 0], roi.get("required_exit_usd_m") or [0, 0]
        roi_line = (f"라운드 {roi['round_amount_raw']} ≈ ${roi['round_amount_usd_m']:.1f}M ({roi['stage']} 단계 중앙값 "
                    f"${roi['stage_median_usd_m']:g}M 의 {roi['vs_stage_median']:.1f}배) → 지분 {st[0] * 100:.0f}~{st[1] * 100:.0f}%·"
                    f"회수 {roi['target_multiple']}배를 가정하면 필요 Exit ${ex[0]:.0f}M~${ex[1]:.0f}M")
    margin = (e["multiplier"] - thr) * 100
    return {"name": e["name"], "m": _s100(e["multiplier"]), "crit": crit,
            "yes": sum(c["y"] for c in crit), "no": sum(c["n"] for c in crit), "unk": sum(c["u"] for c in crit),
            "founder_yes": next((c["y"] for c in crit if c["name"].startswith("창업자")), "-"),
            # 결정 조건인 창업자 기준 비율(동종 평균 = 100). run_log 평가에는 criteria 의 pct 로 남는다
            "founder_c": next((f"{c['pct']:.1f}" for c in e.get("criteria") or [] if c.get("dim") == "founder"), "-"),
            "killers": me.get("killers") or [], "unknown_pct": f"{unknown / len(scored) * 100:.0f}%" if scored else "-",
            "rank": f"동종 {len(names)}곳 중 {names.index(e['name']) + 1}위" if e["name"] in names else "",
            "at": " · ".join(f"{float(t) * 100:.0f} {d}" for t, d in at.items()),
            "margin": f"{margin:.1f}", "borderline": False,
            "first_hold": f"{float(first_hold) * 100:.0f}" if first_hold else "", "roi": roi_line}


def _hold_rows(evals: list[dict]) -> list[dict]:
    rows = sorted((e for e in evals if e.get("multiplier") is not None), key=lambda e: -e["multiplier"])
    return [{"name": e["name"], "m": _s100(e["multiplier"]), "type": e.get("hold_type") or e.get("decision") or "-",
             "flip": _flip(e.get("flip"))} for e in rows[:3]]


def _hold_focus(evals: list[dict], sens: dict | None, thr: float) -> dict | None:
    """모두 보류일 때: 최고점 후보가 기준에서 얼마나 떨어졌는지, 기준을 낮추면 누가 투자로 바뀌는지(민감도),
    보류가 '반대 근거'(Deal-killer·동종 평균 이하) 때문인지 '근거 부족' 때문인지."""
    scored = [e for e in evals if e.get("multiplier") is not None]
    if not scored:
        return None
    top = max(scored, key=lambda e: e["multiplier"])
    margin = (top["multiplier"] - thr) * 100
    counts = (sens or {}).get("invest_count_at") or {}
    lower = max((t for t, n in counts.items() if float(t) < thr and n), key=float, default=None)
    names = [m["name"] for m in (sens or {}).get("members", []) if (m.get("decision_at") or {}).get(lower) == "투자"]
    types = Counter(e.get("hold_type") for e in evals if e.get("decision") == "보류")
    rows = [x for e in evals for x in e.get("rows") or []]
    no = [x["qid"] for x in rows if x.get("answer") == "NO"]
    # 반대 근거로 보류: Deal-killer, 또는 NO(반대 사실 인용) 판정
    negative = sum(v for k, v in types.items() if k == "Deal-killer") + len(no)
    return {"name": top["name"], "m": _s100(top["multiplier"]), "gap": f"{-margin:.1f}", "borderline": margin < 0,
            "lower": f"{float(lower) * 100:.0f}" if lower else "", "lower_names": "·".join(names),
            "lower_n": counts.get(lower, 0) if lower else 0, "negative": negative,
            "rows": len(rows), "no": len(no), "n": len(evals),
            "mix": ", ".join(f"{k} {v}곳" for k, v in types.most_common() if k)}


def _conclusion(r: dict, evals: list[dict], thr100: str) -> str:
    mode = r.get("mode")
    if mode == "invest":
        return f"{r.get('target')} 투자 추천(실사 조건부)"
    if mode == "hold":
        top = max((e for e in evals if e.get("multiplier") is not None), key=lambda e: e["multiplier"], default=None)
        tail = f" (최고점 {top['name']} {_s100(top['multiplier'])}, 기준 {thr100})" if top else ""
        return f"투자 추천 없음 — 평가한 {len(evals)}곳 모두 보류{tail}"
    return "적격 후보 없음 — 투자 추천 없음" if mode == "none" else "-"


def _rag_line(traces: list[dict]) -> str:
    """이번 실행의 Agentic RAG 기록(run_log.rag_traces) 요약: 도구 선택·재작성·재생성·점검 결과."""
    answers = [t for t in traces if t.get("route")]
    if not answers:
        return ""
    steps = [s for t in answers for s in t.get("trace") or []]
    route_ko = {"docs": "문서", "web": "웹", "both": "문서+웹", "direct": "바로 답"}
    routes = Counter(t["route"] for t in answers)
    grounded = sum(t.get("status") == "grounded" for t in answers)
    return (f"질문 {len(answers)}개 · 도구 선택 " + " / ".join(f"{route_ko.get(k, k)} {v}" for k, v in routes.items())
            + f" · 재작성 {sum(s.get('step') == 'rewrite' for s in steps)}회"
            + f" · 재생성 {sum(bool(s.get('regeneration')) for s in steps)}회"
            + f" · 점검 통과 {grounded}/{len(answers)}")


def _rubric() -> dict:
    """평가표(rubric.yaml): 기준별 비중·문항 수·Bessemer 문항 수."""
    import yaml

    rb = yaml.safe_load(_text("rubric.yaml")) or {}
    dims = rb.get("dimensions", [])
    return {"w": {d["id"]: d["weight"] for d in dims}, "n_q": sum(len(d.get("questions", [])) for d in dims),
            "n_dims": len(dims), "n_bessemer": len(rb.get("bessemer", []))}


def _contributors() -> str:
    f = path("docs/contributors.md")
    return f.read_text(encoding="utf-8").strip() if f.exists() else "(조원별 수행 역할 확인 중)"


# ── 조립 ─────────────────────────────────────────────────────────────────────
def context(run_file: str = "outputs/run_log.json") -> dict:
    from docs.build_design import _rag_mermaid  # 설계서 D.4 와 같은 Agentic RAG 서브그래프 정의

    cfg = get_config()
    run = _j(run_file)
    r = run.get("report") or {}
    evals = run.get("evaluations", [])
    thr = float(run.get("threshold") or cfg.decision.threshold)
    thr100 = f"{thr * 100:.0f}"
    sens = _sensitivity()
    md = _text(r["md"]) if r.get("md") else ""
    team = cfg.submission
    members = "+".join(sorted(team.members))
    design_pdf = f"docs/RAG-Design_{team.campus}-{team['class']}_{members}.pdf"
    mode = r.get("mode")
    return dict(
        cfg=cfg, run=run, r=r, checks=r.get("checks") or {}, mode=mode, thr100=thr100, cell=_cell,
        conclusion=_conclusion(r, evals, thr100), pipe=_pipeline(run, evals, cfg),
        target=_target(run, r, sens, thr) if mode == "invest" else None,
        hold_rows=_hold_rows(evals) if mode == "hold" else [],
        hold=_hold_focus(evals, sens, thr) if mode == "hold" else None,
        summary=_summary(md), chapters=_chapters(md), dd=_dd_items(md), summary_pct=f"{(r.get('summary_ratio') or 0) * 100:.0f}%",
        sens=sens, ref_mix=_reference_mix(), rt=_retrieval(cfg), judge=_judge(), elig=_eligibility(),
        pc=_positive_control(), down=_downgrades(), cost=_cost(run), corpus=_corpus(),
        search_providers=_search_providers(), rag_line=_rag_line(run.get("rag_traces", [])),
        report_pdf=_link(f"outputs/RAG-Output_{team.campus}-{team['class']}_{members}.pdf"),
        design_pdf=_link(design_pdf) if path(design_pdf).exists() else "",
        rag_mermaid=_rag_mermaid(cfg), contributors=_contributors(), rb=_rubric(),
    )


def build(run_file: str = "outputs/run_log.json", out: Path | None = None) -> str:
    """run_file·out 은 다른 결론 모드를 미리 보는 용도 (예: outputs/scenario_hold/run_log.json → 임시 파일)."""
    env = Environment(loader=FileSystemLoader(ROOT / "docs"), trim_blocks=True, lstrip_blocks=True)
    md = env.get_template("README.md.j2").render(**context(run_file))
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    out = out or ROOT / "README.md"
    out.write_text(md, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    print(build())
