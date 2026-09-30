"""임베딩 후보 비교(bake-off)와 검색기 비교: Hit Rate@K, MRR.

- 질문 세트 2종: natural(자연어 질문형, 표현을 바꿔 물음) / keyword(고유명사·수치를 그대로 쓴 키워드형)
  에이전트의 실제 질의는 두 형태가 섞여 있어서 한쪽에만 유리한 검색기를 고르지 않도록 둘 다 잰다.
- 정답 판정: 검색된 조각이 정답과 같은 (문서, 페이지)에서 왔으면 적중
- 임베딩 후보마다 Dense 단독, 하이브리드(Kiwi BM25 + Dense, RRF) 가중치 0.5:0.5 / 0.3:0.7 을 잰다
- 런타임(rag/agentic_rag.py)과 같게 각 검색기는 후보 candidate_k(8)개를 가져오고, 합친 순서에서 앞 8개만 본다.
  런타임은 Judge 가 관련 판정한 조각을 이 순서대로 앞 top_k(4)개만 쓰므로 Hit@4·MRR@4 가 파이프라인 지표다
- 재현성·비용 판단용으로 색인 시간, 질의 지연도 기록한다 (캐시 적중 여부에 따라 달라지므로 참고용)
- 후보 모델을 하나라도 못 불러오면(다운로드 실패 등) 결과 파일을 덮어쓰지 않고 종료 코드 1로 끝낸다
  (replay/ 에 FAISS 색인·질의 임베딩 캐시가 없는 후보는 모델을 불러오므로 해당 임베딩 모델(수 GB)이 필요하다)

    uv run python -m eval.eval_retrieval
"""
from __future__ import annotations

import gc
import json
import sys
import time

import pandas as pd
from langchain_classic.retrievers import EnsembleRetriever

from core.config import get_config, path
from rag.embeddings import get_embeddings
from rag.index import get_bm25, get_chunks, get_vectorstore

# 후보: (모델, query 접두어, passage 접두어). 선정 이유는 README·설계서 참고
CANDIDATES = [
    ("BAAI/bge-m3", "", ""),
    ("nlpai-lab/KURE-v1", "", ""),
    ("intfloat/multilingual-e5-large", "query: ", "passage: "),
    ("dragonkue/snowflake-arctic-embed-l-v2.0-ko", "query: ", ""),
    # 모델 카드의 공식 지시문 형식 (Query: 뒤 공백 없음)
    ("Qwen/Qwen3-Embedding-0.6B", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", ""),
    ("intfloat/multilingual-e5-small", "query: ", "passage: "),  # 경량 대안: 저사양 노트북 재현성 보험
]
SETS = {"natural": "data/eval/qa_set.jsonl", "keyword": "data/eval/qa_keyword.jsonl"}
WEIGHTS = [(0.5, 0.5), (0.3, 0.7)]
KS = (1, 3, 4, 5, 8)
MRR_KS = (4, 5)


def _metrics(ranked: list[list[tuple[str, int]]], gold: list[tuple[str, int]]) -> dict:
    out = {}
    for k in KS:
        out[f"Hit@{k}"] = sum(g in r[:k] for r, g in zip(ranked, gold)) / len(gold)
    for k in MRR_KS:
        rr = [next((1 / (i + 1) for i, x in enumerate(r[:k]) if x == g), 0.0) for r, g in zip(ranked, gold)]
        out[f"MRR@{k}"] = sum(rr) / len(rr)
    return {k: round(v, 3) for k, v in out.items()}


def _keys(docs) -> list[tuple[str, int]]:
    return [(d.metadata["doc_id"], d.metadata["page"]) for d in docs]


def _load(f: str) -> list[dict]:
    return [json.loads(l) for l in path(f).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    k = get_config().rag.candidate_k
    # 선정 실험은 기존 문항만 쓴다. 코퍼스를 바꾸며 추가한 문항("added")은 eval_final_retriever 가 따로 잰다
    sets = {s: [q for q in _load(v) if not q.get("added")] for s, v in SETS.items() if path(v).exists()}
    chunks = get_chunks()
    bm25 = get_bm25(k)
    rows = []

    def run(name: str, emb: str, retriever, extra: dict | None = None) -> None:
        for s, qa in sets.items():
            t = time.time()
            ranked = [_keys(retriever.invoke(q["question"])[:k]) for q in qa]  # 런타임과 같이 앞 k개만
            gold = [(q["doc_id"], q["page"]) for q in qa]
            rows.append({"set": s, "retriever": name, "embedding": emb, **_metrics(ranked, gold),
                         "query_ms": round((time.time() - t) / len(qa) * 1000, 1), **(extra or {})})

    run("Kiwi BM25", "-", bm25)
    for model, qp, pp in CANDIDATES:
        t0 = time.time()
        try:
            vs = get_vectorstore(model, qp, pp)
            load = round(time.time() - t0, 1)
            dense = vs.as_retriever(search_kwargs={"k": k})
            run("Dense", model, dense, {"load_or_build_sec": load})
            for w in WEIGHTS:
                run(f"Hybrid {w[0]}:{w[1]}", model, EnsembleRetriever(retrievers=[bm25, dense], weights=list(w)))
        except Exception as e:  # 일부 후보가 빠진 표로 커밋된 결과를 덮어쓰지 않는다
            sys.exit(f"[중단] {model} 을 불러오거나 검색하지 못해 결과 파일을 쓰지 않습니다: {str(e)[:300]}")
        del vs, dense  # 다음 후보를 위해 메모리 해제
        get_vectorstore.cache_clear()
        get_embeddings.cache_clear()
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(path("outputs/eval/retrieval_eval.csv"), index=False)
    # 두 질문 세트의 평균으로 종합 순위 (파이프라인 지표 Hit@4 → MRR@4 순)
    cols = [f"Hit@{x}" for x in KS] + [f"MRR@{x}" for x in MRR_KS]
    piv = (df.groupby(["retriever", "embedding"])[cols].mean().round(3)
           .sort_values(["Hit@4", "MRR@4"], ascending=False).reset_index())
    n = {s: len(v) for s, v in sets.items()}
    md = (f"질문 세트 {n} (정답 = 같은 문서·페이지), 조각 {len(chunks)}개, 검색기마다 후보 k={k}(런타임 candidate_k), "
          f"하이브리드는 합친 순서의 앞 {k}개. 시간 열(query_ms, load_or_build_sec)은 색인·임베딩 캐시가 있으면 0에 가깝다\n\n"
          f"## 종합 (두 세트 평균, Hit@4 → MRR@4 순)\n\n{piv.to_markdown(index=False)}\n\n## 세트별\n\n"
          f"{df.fillna('').to_markdown(index=False)}\n")
    path("outputs/eval/retrieval_eval.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
