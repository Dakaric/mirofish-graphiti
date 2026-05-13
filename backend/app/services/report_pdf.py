"""
Markdown -> HTML -> PDF rendering for report downloads.

Kept dependency-light: `markdown` for MD parsing, `weasyprint` for layout.
The CSS is inline so we don't need any static-file plumbing.
"""

from __future__ import annotations

import io

import markdown as md_lib
from weasyprint import HTML, CSS


_PRINT_CSS = """
@page {
    size: A4;
    margin: 22mm 18mm 22mm 18mm;
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: "DejaVu Sans", sans-serif;
        font-size: 9pt;
        color: #6B7280;
    }
}

html { font-family: "DejaVu Sans", sans-serif; font-size: 10.5pt; color: #1F2937; }
body { line-height: 1.55; }

h1 { font-size: 22pt; margin: 0 0 6pt 0; }
h2 { font-size: 15pt; margin-top: 18pt; border-bottom: 1px solid #E5E7EB; padding-bottom: 3pt; }
h3 { font-size: 12pt; margin-top: 14pt; }
h4, h5, h6 { font-size: 11pt; margin-top: 10pt; }

p { margin: 6pt 0; }
ul, ol { margin: 6pt 0 6pt 18pt; }
li { margin: 2pt 0; }

code { font-family: "DejaVu Sans Mono", monospace; font-size: 9.5pt;
       background: #F3F4F6; padding: 1pt 3pt; border-radius: 2pt; }
pre { background: #F3F4F6; padding: 8pt; border-radius: 4pt;
      font-family: "DejaVu Sans Mono", monospace; font-size: 9pt;
      white-space: pre-wrap; word-wrap: break-word; }

table { border-collapse: collapse; width: 100%; margin: 8pt 0; }
th, td { border: 1px solid #E5E7EB; padding: 4pt 6pt; text-align: left; }
th { background: #F9FAFB; }

blockquote { border-left: 3px solid #D1D5DB; padding-left: 10pt;
             color: #4B5563; margin: 6pt 0; }

.report-header { margin-bottom: 16pt; padding-bottom: 8pt;
                 border-bottom: 2px solid #111827; }
.report-tag { font-size: 8.5pt; letter-spacing: 1pt; color: #6B7280;
              text-transform: uppercase; }
.report-id  { font-size: 8.5pt; color: #9CA3AF; }
"""


def render_pdf(markdown_text: str, title: str | None = None, report_id: str | None = None) -> bytes:
    """Convert Markdown to a styled PDF and return the raw bytes."""
    body_html = md_lib.markdown(
        markdown_text or "",
        extensions=["extra", "sane_lists", "tables", "fenced_code", "toc"],
    )

    header_parts: list[str] = []
    if report_id:
        header_parts.append(f'<span class="report-tag">Prediction Report</span> '
                            f'<span class="report-id">· ID: {report_id}</span>')
    header_html = (
        f'<div class="report-header">{"".join(header_parts)}</div>'
        if header_parts else ""
    )

    title_attr = f'<title>{title}</title>' if title else ""
    full_html = (
        f'<!doctype html><html lang="de"><head><meta charset="utf-8">'
        f'{title_attr}</head><body>{header_html}{body_html}</body></html>'
    )

    buf = io.BytesIO()
    HTML(string=full_html).write_pdf(buf, stylesheets=[CSS(string=_PRINT_CSS)])
    return buf.getvalue()
