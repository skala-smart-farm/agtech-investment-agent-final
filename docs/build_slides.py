"""설계 보고 발표용 슬라이드(16:9 PDF): 설계서와 같은 값(docs.build_design.context)으로 docs/slides/deck.html.j2 를 렌더링한다.

    uv run python -m docs.build_slides                      # docs/RAG-Design-Slides_*.pdf
    uv run python -m docs.build_slides s3_rag /tmp/out      # 조각 하나만 미리보기(PDF + 장마다 PNG)
"""
from __future__ import annotations

import sys
from pathlib import Path

from docs.build_design import _html_env, context
from core.config import path


def render(part: str | None = None) -> str:
    env, ctx = _html_env(), context()
    if not part:
        return env.get_template("slides/deck.html.j2").render(**ctx)
    base = (Path(__file__).parent / "slides/deck.html.j2").read_text(encoding="utf-8")
    head, _, rest = base.partition("<body>")
    tail = rest[rest.index("<script>\nmermaid.initialize"):]
    return env.from_string(f"{head}<body>\n{{% include 'slides/{part}.html.j2' %}}\n{tail}").render(**ctx)


def to_pdf(html: str, pdf: Path) -> int:
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader

    tmp = pdf.with_suffix(".html")
    tmp.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720})
        pg.goto(tmp.resolve().as_uri(), wait_until="networkidle")
        pg.wait_for_function("() => document.querySelectorAll('pre.mermaid svg').length === "
                             "document.querySelectorAll('pre.mermaid').length", timeout=30000)
        pg.pdf(path=str(pdf), width="1280px", height="720px", print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    return len(PdfReader(str(pdf)).pages)


def main() -> None:
    if len(sys.argv) >= 3:
        part, out = sys.argv[1], Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        pdf = out / f"{part}.pdf"
    else:
        part, team = None, context()["team"]
        pdf = path(f"docs/RAG-Design-Slides_{team.campus}-{team['class']}_{'+'.join(sorted(team.members))}.pdf")
        out = pdf.parent
    n = to_pdf(render(part if part != "all" else None), pdf)
    if part:
        import fitz

        for i, page in enumerate(fitz.open(pdf), 1):
            page.get_pixmap(dpi=72).save(out / f"{part}_s{i:02d}.png")
    print(f"{pdf} — {n}장")


if __name__ == "__main__":
    main()
