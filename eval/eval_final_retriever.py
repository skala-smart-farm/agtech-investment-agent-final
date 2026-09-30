"""런타임 검색기 성능과 대안 비교: 기존 70문항 전체·정답 문서 언어별·질문 세트별 Hit@1/3/4/5/8, MRR@4 (+ 추가 문항).

런타임 동작(rag/agentic_rag.py): get_hybrid_retriever().invoke(질의)[:candidate_k] → Judge 가 조각별 관련 O/X →
관련 조각을 검색 순서 그대로 두고 앞 top_k(4)개만 컨텍스트로 쓴다(재정렬 없음).
- Hit@4 · MRR@4 : 파이프라인 지표. 정답 조각이 검색 순서 4위 안이면 Judge 가 앞 조각을 모두 관련으로 판정해도 컨텍스트에 들어간다
- Hit@8 : Judge 가 정답만 골라낸다고 가정한 상한 (Judge 가 보는 후보 8개 안 적중)
모든 설정을 런타임과 같은 k=candidate_k 로 잰다 (BM25·Dense 가 각각 k개, 하이브리드는 RRF 로 합친 순서의 앞 k개).
질문은 모두 한국어이고, ko/en 은 정답 문서의 언어다.

비교표 구성: 선정 실험 때와 같은 표가 되도록 'final (hybrid)'·'dense only'·'hybrid a:b' 행은 선정 당시 런타임 설정
(DECIDED = snowflake-arctic-ko + 하이브리드 0.3:0.7, '처음 선택')이고, KURE-v1 행이 대안이다. 행 이름은 설계서
(docs/build_design._runtime_table)와 README(docs/build_readme._runtime)가 읽는다. 지금 config 의 임베딩·가중치와 같은 행은
런타임 검색기(get_hybrid_retriever) 그대로 재고 runtime=true 를 붙인다(격자에 없으면 'runtime (hybrid)' 행을 더한다).

질문 묶음: 표제 지표 'all'(과 ko/en·natural/keyword)은 기존 70문항이다. 코퍼스를 바꾸며 추가한 문항(질문 파일의 "added" 필드)은
build_qa·Judge 를 거치지 않고 원문을 읽고 만든 문항이라 지표를 부풀릴 수 있어 'added' 로 따로, 합친 전체를 'extended' 로 적는다.

추천 규칙 (측정 전에 정해 두고 코드가 기계적으로 적용, 판정은 기존 70문항 'all' 로만):
  1) 전체 70문항 Hit@4 가 가장 높은 설정을 고른다
  2) Hit@4 가 같으면 MRR@4 가 높은 설정 (완전 동률이면 현재 런타임 설정, 그다음 표의 앞쪽)
  3) 1위가 현재 설정 임베딩이 아니고, 현재 임베딩으로 되는 설정(재색인 불필요) 중 최고가
     Hit@4 · MRR@4 모두 1위보다 0.01 이하로만 낮으면 그 설정을 추천한다
  (판정은 결과 파일에 적힌 소수 셋째 자리 값으로 한다)

    uv run python -m eval.eval_final_retriever
(두 임베딩 모델의 FAISS 색인이 replay/ 에 없으면 모델로 만든다. 질의 임베딩은 replay/qemb/ 캐시를 쓴다)
"""
from __future__ import annotations

import json

from langchain_classic.retrievers import EnsembleRetriever

from core.config import get_config, path
from eval.eval_retrieval import SETS, _keys, _load, _metrics
from rag.index import get_bm25, get_chunks, get_hybrid_retriever, get_vectorstore
from rag.loader import load_manifest, total_pages

DECIDED = ("dragonkue/snowflake-arctic-embed-l-v2.0-ko", "query: ", "", [0.3, 0.7])  # 선정 당시 런타임(처음 선택)
ALT_EMB = ("nlpai-lab/KURE-v1", "", "")  # bake-off 상위 동률 후보 (eval_retrieval 과 같은 접두어)
PREFIX = {DECIDED[0]: DECIDED[1], ALT_EMB[0]: ALT_EMB[1]}
GRID = (0.1, 0.2, 0.3, 0.4, 0.5)          # 하이브리드 BM25 가중치 (Dense = 1 - BM25)
TOL = 0.01                                 # 규칙 3의 허용 차이
RULE = ("1) 전체(기존 70문항) Hit@4 최대 2) 동률이면 MRR@4 최대(완전 동률이면 현재 런타임 설정 우선) "
        f"3) 1위가 다른 임베딩이면, 현재 임베딩 설정 중 최고가 Hit@4·MRR@4 모두 {TOL} 이내일 때 그쪽(재색인 불필요)")


def _score(retriever, groups: dict[str, list[dict]], k: int) -> dict:
    ranked: dict[str, list] = {}  # 같은 질문이 여러 묶음에 들어가므로 질문마다 한 번만 검색
    out = {}
    for g, qs in groups.items():
        for q in qs:
            if q["question"] not in ranked:
                ranked[q["question"]] = _keys(retriever.invoke(q["question"])[:k])  # 런타임과 같이 앞 k개만
        out[g] = {"n": len(qs), **_metrics([ranked[q["question"]] for q in qs], [(q["doc_id"], q["page"]) for q in qs])}
    return out


def _key(r: dict) -> tuple[float, float]:
    return r["all"]["Hit@4"], r["all"]["MRR@4"]


def _config_change(r: dict, cfg) -> str:
    if r["runtime"]:
        return "변경 없음 (현재 런타임 설정)"
    ws = r["weights[bm25,dense]"]
    w = f"rag.ensemble_weights: [{ws[0]}, {ws[1]}]" + (" (가중치 0 인 쪽은 앞 k개 순서에 영향 없음)" if 0.0 in ws else "")
    redo = "검색 순서가 바뀌므로 본 실행과 재현용 LLM 캐시는 다시 만들어야 함"
    if r["embedding"] in (cfg.embedding.model, "-"):
        return f"{w} — 재색인 불필요(같은 FAISS 색인), {redo}"
    return (f"embedding.model: {r['embedding']} (query_prefix '{PREFIX.get(r['embedding'], '')}') + {w} — 임베딩 교체라 "
            f"본 실행 질의의 임베딩 캐시가 없어 모델이 필요하고, {redo}")


def _recommend(rows: list[dict], cfg) -> dict:
    tie = lambda r: (*_key(r), r["runtime"])  # 완전 동률이면 런타임 설정, 그다음은 max 가 앞쪽을 고른다
    best = max(rows, key=tie)
    cur = max((r for r in rows if r["embedding"] in (cfg.embedding.model, "-")), key=tie)
    kept = best is not cur and all(b - c <= TOL + 1e-9 for b, c in zip(_key(best), _key(cur)))
    pick = cur if kept else best
    base = next(r for r in rows if r["runtime"])
    return {"rule": RULE, "name": pick["name"], "embedding": pick["embedding"],
            "weights[bm25,dense]": pick["weights[bm25,dense]"],
            "Hit@4": pick["all"]["Hit@4"], "MRR@4": pick["all"]["MRR@4"],
            "delta_vs_runtime": {m: round(pick["all"][m] - base["all"][m], 3) for m in ("Hit@4", "MRR@4", "Hit@8")},
            "rule3_applied": kept, "overall_best": best["name"],
            # 참고(규칙과 별개): 재색인 없이 현재 임베딩으로 되는 최고 설정
            "best_without_reindex": {"name": cur["name"], "Hit@4": cur["all"]["Hit@4"], "MRR@4": cur["all"]["MRR@4"],
                                     "config_change": _config_change(cur, cfg)},
            "config_change": _config_change(pick, cfg),
            "ranking": [f"{r['name']}: Hit@4 {r['all']['Hit@4']:.3f} / MRR@4 {r['all']['MRR@4']:.3f}"
                        for r in sorted(rows, key=_key, reverse=True)]}


def main() -> None:
    cfg = get_config()
    k = cfg.rag.candidate_k
    qs = {s: _load(f) for s, f in SETS.items() if path(f).exists()}
    base = {s: [q for q in v if not q.get("added")] for s, v in qs.items()}  # 기존 문항 (표제 지표)
    allq = [q for v in base.values() for q in v]
    added = [q for v in qs.values() for q in v if q.get("added")]
    groups = {"all": allq, **{lang: [q for q in allq if q["lang"] == lang] for lang in ("ko", "en")}, **base}
    if added:
        groups |= {"extended": allq + added, "added": added}
    w = [float(x) for x in cfg.rag.ensemble_weights]
    bm25 = get_bm25(k)
    dw = [float(x) for x in DECIDED[3]]
    dec = get_vectorstore(*DECIDED[:3]).as_retriever(search_kwargs={"k": k})
    alt = get_vectorstore(*ALT_EMB).as_retriever(search_kwargs={"k": k})
    alt_short = ALT_EMB[0].split("/")[-1]

    # (이름, 검색기, 임베딩, [BM25, Dense] 가중치). 앞 5행과 격자는 선정 실험 때와 같은 구성
    specs = [("final (hybrid)", EnsembleRetriever(retrievers=[bm25, dec], weights=dw), DECIDED[0], dw),
             ("dense only", dec, DECIDED[0], [0.0, 1.0]),
             ("Kiwi BM25 only", bm25, "-", [1.0, 0.0]),
             (f"{alt_short} dense", alt, ALT_EMB[0], [0.0, 1.0]),
             (f"{alt_short} hybrid", EnsembleRetriever(retrievers=[bm25, alt], weights=dw), ALT_EMB[0], dw)]
    for emb, name, r in ((DECIDED[0], "hybrid", dec), (ALT_EMB[0], f"{alt_short} hybrid", alt)):
        for b in GRID:
            if abs(b - dw[0]) < 1e-9:
                continue  # 처음 선택의 가중치는 위 행과 같은 검색기
            gw = [b, round(1 - b, 1)]
            specs.append((f"{name} {gw[0]}:{gw[1]}", EnsembleRetriever(retrievers=[bm25, r], weights=gw), emb, gw))
    # 지금 런타임 설정과 같은 첫 행은 런타임 검색기 그대로 잰다
    now = next((i for i, (_, _, e, ws) in enumerate(specs) if e == cfg.embedding.model and ws == w), None)
    if now is None:
        specs.append(("runtime (hybrid)", get_hybrid_retriever(), cfg.embedding.model, w))
        now = len(specs) - 1
    else:
        n, _, e, ws = specs[now]
        specs[now] = (n, get_hybrid_retriever(), e, ws)

    rows = [{"name": n, "embedding": e, "weights[bm25,dense]": ws, "k": k, "runtime": i == now, **_score(r, groups, k)}
            for i, (n, r, e, ws) in enumerate(specs)]
    rec = _recommend(rows, cfg)
    docs = load_manifest()
    meta = {"embedding": cfg.embedding.model, "weights[bm25,dense]": w, "runtime_row": rows[now]["name"],
            "comparison_baseline": {"name": "final (hybrid)", "embedding": DECIDED[0], "weights[bm25,dense]": dw,
                                    "note": "선정 실험 당시 런타임 설정(처음 선택). 선정 당시와 같은 표가 되도록 비교 기준으로 둔다"},
            "corpus": {"docs": len(docs), "pages": total_pages(), "chunks": len(get_chunks())},
            "candidate_k": k, "top_k": cfg.rag.top_k,
            "measured_k": k, "questions": {g: len(v) for g, v in groups.items()},
            "hit_resolution": f"전체 1문항 = {1 / len(allq):.3f}",
            "groups": ("all=기존 문항 전체(표제 지표·추천 규칙), ko/en=정답 문서 언어(질문은 모두 한국어), "
                       "natural/keyword=질문 세트(기존 문항)"
                       + (", extended=기존 + 추가 문항, added=추가 문항만(\"added\" 필드, 원문을 읽고 만든 문항)" if added else "")),
            "pipeline_metric": "Hit@4·MRR@4 (관련 조각을 검색 순서대로 앞 top_k=4개만 사용), Hit@8 = Judge 상한"}
    runtime = {"config": meta, "rows": rows, "recommendation": rec}
    # 설계서·README 생성 스크립트가 읽는 이름별 형식 (all/ko/en)
    compat = {"config": meta, **{r["name"]: {g: r[g] for g in ("all", "ko", "en")} for r in rows[:5]}}
    path("outputs/eval/runtime_retriever.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2),
                                                           encoding="utf-8")
    path("outputs/eval/final_retriever.json").write_text(json.dumps(compat, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(f"k={k} (런타임 candidate_k), top_k={cfg.rag.top_k}, 기존 질문 {len(allq)}개 + 추가 {len(added)}개, "
          f"코퍼스 {meta['corpus']}")
    ext = " ext Hit@4 ext MRR@4" if added else ""
    print(f"{'설정':<28} {'Hit@1':>6} {'Hit@3':>6} {'Hit@4':>6} {'Hit@5':>6} {'Hit@8':>6} {'MRR@4':>6}  en Hit@4{ext}")
    for r in rows:
        a = r["all"]
        print(f"{r['name'] + (' *' if r['runtime'] else ''):<28} "
              + " ".join(f"{a[m]:>6.3f}" for m in ("Hit@1", "Hit@3", "Hit@4", "Hit@5", "Hit@8", "MRR@4"))
              + f"  {r['en']['Hit@4']:.3f}"
              + (f"    {r['extended']['Hit@4']:.3f}      {r['extended']['MRR@4']:.3f}" if added else ""))
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
