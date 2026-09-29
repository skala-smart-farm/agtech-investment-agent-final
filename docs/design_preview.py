"""설계서 HTML 미리보기: 조각 하나(또는 전체)를 PDF 로 만들고 쪽마다 PNG 로 저장한다 (제출 PDF 는 건드리지 않음).

    .venv/bin/python -m docs.design_preview 20_b_agents_tools /tmp/out     # 조각 하나 (표지·다른 장 없이)
    .venv/bin/python -m docs.design_preview all /tmp/out                   # 전체
"""
from __future__ import annotations

import sys
from pathlib import Path

from docs.build_design import _html_env, _to_pdf, context


def main(part: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env, ctx = _html_env(), context()
    if part == "all":
        html = env.get_template("design_html/base.html.j2").render(**ctx)
    else:
        base = (Path(__file__).parent / "design_html/base.html.j2").read_text(encoding="utf-8")
        head, _, rest = base.partition("<body>")
        tail = rest[rest.index("<script>\nmermaid.initialize"):]
        html = env.from_string(f"{head}<body>\n{{% include 'design_html/{part}.html.j2' %}}\n{tail}").render(**ctx)
    pdf = out_dir / f"{part}.pdf"
    _to_pdf(html, pdf)
    import fitz

    doc = fitz.open(pdf)
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=110).save(out_dir / f"{part}_p{i:02d}.png")
    print(f"{pdf} — {len(doc)}쪽, PNG {len(doc)}장 → {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
