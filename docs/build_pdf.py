"""Render docs/WRITEUP.md to docs/WRITEUP.pdf (pure Python: markdown + xhtml2pdf)."""
from pathlib import Path

import markdown
from xhtml2pdf import pisa

HERE = Path(__file__).parent
md = (HERE / "WRITEUP.md").read_text()
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
body = body.replace('src="figures/', f'src="{(HERE / "figures").as_posix()}/')
css = """
@page { size: A4; margin: 1.6cm 1.5cm 1.6cm 1.5cm;
        @frame footer { -pdf-frame-content: footer; bottom: 0.6cm; margin-left: 1.5cm; margin-right: 1.5cm; height: 0.6cm; } }
body { font-family: Helvetica; font-size: 9pt; line-height: 1.38; color: #1f2933; }
h1 { font-size: 17pt; color: #1e3a8a; margin-bottom: 2pt; }
h2 { font-size: 12.5pt; color: #1e3a8a; border-bottom: 0.6pt solid #cbd2d9; padding-bottom: 2pt; margin-top: 12pt; }
h3 { font-size: 10.5pt; color: #334e68; margin-top: 8pt; }
p { margin: 3pt 0 5pt 0; }
li { margin-bottom: 2pt; }
code { font-family: Courier; font-size: 8pt; background-color: #f0f4f8; }
pre { font-family: Courier; font-size: 7.5pt; background-color: #f0f4f8; padding: 5pt; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; width: 100%; }
th { background-color: #e4e7eb; font-weight: bold; padding: 3pt 4pt; border: 0.5pt solid #cbd2d9; font-size: 8pt; text-align: left; }
td { padding: 2.5pt 4pt; border: 0.5pt solid #cbd2d9; font-size: 8pt; vertical-align: top; }
img { width: 17.5cm; }
"""
html = f"""<html><head><meta charset="utf-8"><style>{css}</style></head><body>{body}
<div id="footer" style="text-align:right; font-size:7pt; color:#7b8794;">TermMap-DX · HiLabs Hackathon 2026 · page <pdf:pagenumber/></div>
</body></html>"""
out = HERE / "WRITEUP.pdf"
with open(out, "wb") as f:
    status = pisa.CreatePDF(html, dest=f, encoding="utf-8")
print("errors:", status.err, "->", out)
