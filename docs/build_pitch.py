"""설계 산출물 발표 덱(reveal.js) → 16:9 PDF.

docs/pitch/design_deck.html.j2 를 설계서와 같은 값(docs.build_design.context)으로 렌더링해 docs/pitch/design.html 을 만들고,
reveal.js 의 print-pdf 모드로 PDF 를 만든다.

    uv run python -m docs.build_pitch            # docs/RAG-Design-Deck_*.pdf
    uv run python -m docs.build_pitch /tmp/out   # 장마다 PNG 도 저장
"""
from __future__ import annotations

import sys
from pathlib import Path

from core.config import ROOT, path
from docs.build_design import EMOJI, _html_env, context


def build(png_dir: str | None = None) -> str:
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader

    ctx = context()
    env = _html_env()
    env.filters["noemoji"] = lambda s: EMOJI.sub("", s or "").strip()
    html_path = ROOT / "docs/pitch/design.html"
    html_path.write_text(env.get_template("pitch/design_deck.html.j2").render(**ctx), encoding="utf-8")
    team = ctx["team"]
    pdf = path(f"docs/RAG-Design-Deck_{team.campus}-{team['class']}_{'+'.join(sorted(team.members))}.pdf")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720})
        pg.goto(html_path.resolve().as_uri() + "?print-pdf", wait_until="networkidle")
        pg.wait_for_function("() => window.Reveal && Reveal.isReady && Reveal.isReady()", timeout=60000)
        pg.wait_for_timeout(1000)
        pg.pdf(path=str(pdf), width="1280px", height="720px", print_background=True, prefer_css_page_size=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    n = len(PdfReader(str(pdf)).pages)
    if png_dir:
        import fitz

        out = Path(png_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(fitz.open(pdf), 1):
            page.get_pixmap(dpi=72).save(out / f"deck_{i:02d}.png")
    print(f"{pdf} — {n}장")
    return str(pdf)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
