"""문서 로딩과 전처리.

실습에서 겪은 문제를 반영했다.
- 출처·각주 줄이 본문과 섞이면 모델이 엉뚱한 날짜/수치를 인용한다 → 본문에서 분리해 메타데이터로 보관.
- 매 페이지 반복되는 머리글·바닥글(보고서 제목, 쪽 번호)은 검색을 방해한다 → 제거.
- 2단 편집은 좌·우 단을 섞어 읽는다 → 블록을 단(column) 순서로 정렬.
- 표는 행·열이 흩어진다 → 표를 따로 마크다운으로 추출해 별도 조각으로 추가.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pymupdf
import yaml
from langchain_core.documents import Document

from core.config import ROOT, get_config

NOTE_PAT = re.compile(r"^\s*(출처|자료|주\s*[:)）]|주\d*\)|※|\*\s|Source[s]?\s*:|Note[s]?\s*:|\d+\)\s)", re.I)
PAGE_NUM = re.compile(r"^\s*[-–]?\s*\d{1,3}\s*[-–]?\s*$")


def load_manifest() -> list[dict]:
    cfg = get_config()
    with open(ROOT / cfg.rag.manifest, encoding="utf-8") as f:
        docs = yaml.safe_load(f)["documents"]
    return docs


def _page_blocks(page: pymupdf.Page) -> list[str]:
    """텍스트 블록을 읽는 순서대로 정렬. 2단 편집이면 왼쪽 단을 먼저 읽는다."""
    w = page.rect.width
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    narrow = [b for b in blocks if (b[2] - b[0]) < 0.55 * w]
    left = [b for b in narrow if b[2] <= w * 0.55]
    right = [b for b in narrow if b[0] >= w * 0.45]
    two_col = len(blocks) >= 6 and len(left) >= 3 and len(right) >= 3 and len(narrow) >= 0.6 * len(blocks)
    if two_col:
        key = lambda b: (0 if b[0] < w * 0.45 else 1, round(b[1]), b[0])
    else:
        key = lambda b: (round(b[1] / 3), b[0])
    return [b[4] for b in sorted(blocks, key=key)]


def _repeated_lines(pages: list[list[str]]) -> set[str]:
    """여러 페이지의 첫·마지막 줄에 반복되는 머리글·바닥글."""
    cnt: Counter = Counter()
    for lines in pages:
        edge = lines[:2] + lines[-2:]
        cnt.update({l.strip() for l in edge if l.strip()})
    th = max(3, int(len(pages) * 0.4))
    return {l for l, c in cnt.items() if c >= th and len(l) < 80}


def _is_bibliography(text: str) -> bool:
    refs = len(re.findall(r"doi[:.]|https?://|et al\.|\(\d{4}\)\.", text, re.I))
    return refs >= 10 or bool(re.match(r"\s*(references|참고문헌|bibliography)\b", text, re.I)) and refs >= 4


def load_documents() -> tuple[list[Document], list[dict]]:
    """본문 Document(페이지 단위)와 표 Document 를 돌려준다."""
    cfg = get_config()
    manifest = load_manifest()
    out: list[Document] = []
    stats = []
    for meta in manifest:
        fpath = ROOT / cfg.rag.corpus_dir / meta["file"]
        doc = pymupdf.open(fpath)
        page_range = meta.get("page_range")  # [start, end] 1-based, 문서 일부만 쓰는 경우
        start, end = (page_range or [1, doc.page_count])
        raw_pages = []
        for pno in range(start - 1, end):
            text = "\n".join(_page_blocks(doc[pno]))
            raw_pages.append([l for l in text.splitlines()])
        repeated = _repeated_lines(raw_pages)
        base = {k: meta.get(k) for k in ("doc_id", "title", "publisher", "year", "url", "type", "lang",
                                         "authors", "journal", "volume", "issue", "pages")}
        for i, lines in enumerate(raw_pages):
            pno = start + i
            body, notes = [], []
            for l in lines:
                s = l.strip()
                if not s or s in repeated or PAGE_NUM.match(s):
                    continue
                (notes if NOTE_PAT.match(s) else body).append(s)
            text = re.sub(r"[ \t]+", " ", "\n".join(body))
            if len(text) < 40:
                continue
            if _is_bibliography(text):  # 참고문헌 페이지는 검색 잡음이라 제외
                continue
            out.append(Document(page_content=text, metadata={**base, "page": pno, "kind": "text",
                                                             "notes": " / ".join(notes)[:500]}))
            # 표는 캡션(페이지 첫 줄)과 묶어 별도 조각으로
            try:
                for t in doc[pno - 1].find_tables().tables:
                    cells = [c for row in t.extract() for c in row if c]
                    # 디자인 박스를 표로 오인하는 경우 제외: 3행·2열 이상, 셀이 짧은(표다운) 경우만
                    if t.row_count < 3 or t.col_count < 2 or not cells:
                        continue
                    if sum(len(str(c)) for c in cells) / len(cells) > 60:
                        continue
                    if len(cells) / (t.row_count * t.col_count) < 0.5 or len(set(cells)) < 4:
                        continue  # 빈 칸투성이·같은 글자 반복(장식 요소)
                    md = t.to_markdown().replace("<br>", " ")
                    if md and len(md) > 80:
                        caption = next((b for b in body if len(b) > 5), "")[:80]
                        out.append(Document(page_content=f"[표] {caption}\n{md}",
                                            metadata={**base, "page": pno, "kind": "table", "notes": ""}))
            except Exception:
                pass
        stats.append({"doc_id": meta["doc_id"], "pages": end - start + 1})
        doc.close()
    return out, stats


def total_pages() -> int:
    cfg = get_config()
    total = 0
    for meta in load_manifest():
        if meta.get("page_range"):
            s, e = meta["page_range"]
            total += e - s + 1
        else:
            with pymupdf.open(ROOT / cfg.rag.corpus_dir / meta["file"]) as d:
                total += d.page_count
    return total


@lru_cache(maxsize=1)
def corpus_summary() -> str:
    """코퍼스 요약 '{문서 수}종 {사용 쪽수}쪽, {최초}~{최근}년 발행' (data/manifest.yaml 과 total_pages 로 계산).
    검색 도구 설명(tools/agent_tools.py)에 들어가므로 코퍼스를 바꾸면 설명도 저절로 바뀐다."""
    docs = load_manifest()
    years = [int(d["year"]) for d in docs]
    return f"{len(docs)}종 {total_pages()}쪽, {min(years)}~{max(years)}년 발행"


def corpus_files() -> list[Path]:
    cfg = get_config()
    return [ROOT / cfg.rag.corpus_dir / m["file"] for m in load_manifest()]
