"""설계 보고 발표 덱(reveal.js, docs/pitch/index.html) → 16:9 PDF.

    uv run python -m docs.build_pitch            # docs/RAG-Design-Pitch_*.pdf (+ PNG 미리보기는 인자로 폴더를 주면)
    uv run python -m docs.build_pitch /tmp/out   # 장마다 PNG 도 저장
"""
from __future__ import annotations

import sys
from pathlib import Path

from core.config import ROOT, get_config, path


def build(png_dir: str | None = None) -> str:
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader

    team = get_config().submission
    pdf = path(f"docs/RAG-Design-Pitch_{team.campus}-{team['class']}_{'+'.join(sorted(team.members))}.pdf")
    url = (ROOT / "docs/pitch/index.html").resolve().as_uri() + "?print-pdf"
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720})
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_function("() => window.Reveal && Reveal.isReady()", timeout=30000)
        pg.wait_for_timeout(800)
        pg.pdf(path=str(pdf), width="1280px", height="720px", print_background=True, prefer_css_page_size=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    n = len(PdfReader(str(pdf)).pages)
    if png_dir:
        import fitz

        out = Path(png_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(fitz.open(pdf), 1):
            page.get_pixmap(dpi=72).save(out / f"pitch_{i:02d}.png")
    print(f"{pdf} — {n}장")
    return str(pdf)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
