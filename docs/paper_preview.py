"""학회지 양식 설계서 미리보기: 장 하나(docs/design_md/<part>.md.j2) 또는 전체를 PDF·PNG 로 (제출 PDF 는 건드리지 않음).

    .venv/bin/python -m docs.paper_preview p4_c_criteria /tmp/out
    .venv/bin/python -m docs.paper_preview all /tmp/out
"""
from __future__ import annotations

import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from core.config import ROOT
from docs.build_design import _blank_before_lists, _html_env, _paper_body, _to_pdf, context


def main(part: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = context()
    name = "design.md.j2" if part == "all" else f"design_md/{part}.md.j2"
    md = _blank_before_lists(Environment(loader=FileSystemLoader(ROOT / "docs")).get_template(name).render(**ctx))
    html = _html_env().get_template("design_html/paper.html.j2").render(**ctx, body=Markup(_paper_body(md)))
    pdf = out_dir / f"{part}.pdf"
    _to_pdf(html, pdf)
    import fitz

    doc = fitz.open(pdf)
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=80).save(out_dir / f"{part}_p{i:02d}.png")
    print(f"{pdf} — {len(doc)}쪽 → {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
