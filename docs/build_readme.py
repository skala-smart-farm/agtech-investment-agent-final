"""README.md 생성: 발표 수치가 제출 실행 결과(outputs/run_log.json)·평가 결과(outputs/eval/)와 항상 같도록 템플릿으로 만든다.

    uv run python -m docs.build_readme
"""
from __future__ import annotations

import json
import re

from jinja2 import Environment, FileSystemLoader

from core.config import ROOT, get_config, path
from rag.loader import load_manifest, total_pages


def _search_providers() -> str:
    """재현용 캐시에 실제로 결과를 준 검색 공급자만 적는다 (키를 받지 못한 공급자를 쓴 것처럼 쓰지 않게)."""
    used = set()
    for f in path(f"{get_config().cache.dir}/search").glob("*.json"):
        used |= {r.get("provider", "tavily") for r in json.loads(f.read_text(encoding="utf-8"))}
    names = [n for p, n in (("serper", "Serper(구글)"), ("tavily", "Tavily")) if p in used]
    return "·".join(names) or "Tavily"


def _j(rel: str) -> dict:
    f = path(rel)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _runtime(cfg) -> tuple[str, str, str]:
    """지금 설정(임베딩·가중치)과 같은 행을 실제 파이프라인 설정 측정표(outputs/eval/runtime_retriever.json)에서 찾는다."""
    w = [float(x) for x in cfg.rag.ensemble_weights]
    label = "Dense 단독" if w[0] == 0 else f"하이브리드 BM25 {w[0]:g} : Dense {w[1]:g}"
    rows = _j("outputs/eval/runtime_retriever.json").get("rows", [])
    row = next((r for r in rows if r.get("embedding") == cfg.embedding.model
                and [float(x) for x in r.get("weights[bm25,dense]", [])] == w), None)
    if not row:
        return label, "-", "-"
    return label, f"{row['all']['Hit@4']:.3f}", f"{row['all']['MRR@4']:.3f}"


def _rag_line(traces: list[dict]) -> str:
    """이번 실행의 Agentic RAG 기록(run_log.rag_traces) 요약: 도구 선택·재작성·재생성·점검 결과."""
    answers = [t for t in traces if t.get("route")]
    if not answers:
        return "-"
    steps = [s for t in answers for s in t.get("trace") or []]
    routes = {k: sum(t["route"] == k for t in answers) for k in ("docs", "web", "both", "direct")}
    route_ko = {"docs": "문서", "web": "웹", "both": "문서+웹", "direct": "바로 답"}
    grounded = sum(t.get("status") == "grounded" for t in answers)
    return (f"질문 {len(answers)}개 · 도구 선택 " + " / ".join(f"{route_ko[k]} {v}" for k, v in routes.items() if v)
            + f" · 질의 재작성 {sum(s.get('step') == 'rewrite' for s in steps)}회"
            + f" · 답변 재생성 {sum(bool(s.get('regeneration')) for s in steps)}회"
            + f" · 점검 통과(grounded) {grounded}/{len(answers)}")


def _pc_line() -> str:
    """양성 대조(실제 투자를 받은 후기 기업 3곳) 결과: v2 판단으로 몇 곳이 투자 추천인지."""
    d = _j("outputs/eval/positive_control.json")
    rows = d if isinstance(d, list) else []
    if not rows or not all("multiplier" in r for r in rows):
        return "(v2 로 재측정 전 · v1 은 0/3)"
    inv = [r["name"] for r in rows if r.get("decision") == "투자"]
    parts = [f"{r['name']} {r['multiplier'] * 100:.0f}" + ("" if r.get("decision") == "투자" else f"({r.get('hold_type') or r['decision']})")
             if r.get("multiplier") is not None else f"{r['name']} {r['decision']}" for r in rows]
    return f"{len(rows)}곳 중 투자 추천 {len(inv)}곳 — " + " · ".join(parts) + " (v1 은 0/3)"


def _evals(run: dict) -> list[dict]:
    """평가 결과를 동종 대비 배수 순으로 (보고서 표기: 동종 평균 = 100)."""
    out = []
    for e in run.get("evaluations", []):
        m = e.get("multiplier")
        label = "투자 추천" if e.get("decision") == "투자" else f"보류({e.get('hold_type') or '-'})"
        out.append({"name": e["name"], "m": m if m is not None else -1, "s100": "-" if m is None else f"{m * 100:.0f}",
                    "label": label, "flip": (e.get("flip") or {}), "decision": e.get("decision"),
                    "hold_type": e.get("hold_type")})
    return sorted(out, key=lambda e: -e["m"])


def _conclusion(r: dict, evals: list[dict]) -> str:
    mode = r.get("mode")
    if mode == "invest":
        return f"{r.get('target') or next((e['name'] for e in evals if e['decision'] == '투자'), '')} 투자 추천(실사 조건부)"
    if mode == "hold":
        return "투자 추천 없음 (평가한 후보 모두 보류)"
    return "적격 후보 없음" if mode == "none" else "-"


def _flip_line(evals: list[dict]) -> str:
    """보류 결론이면 1위 후보의 뒤집힘 조건 한 줄."""
    top = next((e for e in evals if e["decision"] == "보류"), None)
    if not top or any(e["decision"] == "투자" for e in evals):
        return ""
    f = top["flip"]
    if f.get("items") and f.get("new_multiplier") is not None:
        tail = "투자 조건 충족" if f.get("reached") else "기준 미달"
        return (f"뒤집힘 조건({top['name']}): {'·'.join(f['items'])} 이(가) 확인되면 동종 평균 대비 "
                f"{f['new_multiplier'] * 100:.0f} → {tail}")
    return f"뒤집힘 조건({top['name']}): {f.get('note') or '-'}"


def _hold_mix(evals: list[dict]) -> str:
    from collections import Counter

    c = Counter(e["hold_type"] for e in evals if e["decision"] == "보류" and e["hold_type"])
    return ", ".join(f"{k} {v}곳" for k, v in c.most_common())


def _contributors() -> str:
    f = path("docs/contributors.md")
    return f.read_text(encoding="utf-8").strip() if f.exists() else "(조원별 수행 역할 확인 중)"


def build() -> str:
    cfg = get_config()
    run = _j("outputs/run_log.json")
    judge = _j("outputs/eval/judge_eval.json").get("summary", {})
    gold = _j("outputs/eval/eligibility_eval_gold.json").get("summary", {})
    hold = _j("outputs/eval/eligibility_eval_holdout.json").get("summary", {})
    m = re.search(r"후보 (\d+)곳", " ".join(run.get("log", [])))
    team = cfg.submission
    members = "+".join(sorted(team.members))
    evals = _evals(run)
    n_eligible = sum(1 for s in run.get("screened", []) if s["eligible"])
    label, hit4, mrr4 = _runtime(cfg)
    r = run.get("report", {})
    md = Environment(loader=FileSystemLoader(ROOT / "docs")).get_template("README.md.j2").render(
        cfg=cfg, run=run, r=r, evals=evals, search_providers=_search_providers(),
        disc={"candidates": m.group(1) if m else "-"}, n_eligible=n_eligible,
        n_unevaluated=max(0, n_eligible - len(evals)), thr100=f"{cfg.decision.threshold * 100:.0f}",
        conclusion=_conclusion(r, evals), flip_line=_flip_line(evals), hold_mix=_hold_mix(evals),
        pc_line=_pc_line(), retrieval_label=label, rt_hit4=hit4, rt_mrr4=mrr4, rag_line=_rag_line(run.get("rag_traces", [])),
        judge_line=(f"Relevance {judge['relevance']:.2f} · Faithfulness {judge['faithfulness']:.2f} · "
                    f"Correctness {judge['correctness']:.2f} ({judge.get('n', '-')}문항)") if judge else "-",
        elig_gold=f"{gold.get('accuracy', 0):.2f}", elig_holdout=f"{hold.get('accuracy', 0):.2f}",
        n_docs=len(load_manifest()), total_pages=total_pages(), contributors=_contributors(),
        report_pdf=f"RAG-Output_{team.campus}-{team['class']}_{members}.pdf")
    out = ROOT / "README.md"
    out.write_text(md, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    print(build())
