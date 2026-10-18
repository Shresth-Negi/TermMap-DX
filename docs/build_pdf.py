"""Render docs/WRITEUP.md to docs/WRITEUP.pdf (pure Python: markdown + xhtml2pdf)."""
import os
from pathlib import Path

import markdown
import matplotlib
from xhtml2pdf import pisa

import sys

HERE = Path(__file__).parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "WRITEUP"
md = (HERE / f"{NAME}.md").read_text()
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
body = body.replace('src="figures/', f'src="{(HERE / "figures").as_posix()}/')
import shutil

_SRC = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
FONTS = HERE / ".fonts"  # xhtml2pdf only reads resources under the document directory
FONTS.mkdir(exist_ok=True)
for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"):
    shutil.copy(os.path.join(_SRC, f), FONTS / f)
FONTS = FONTS.as_posix()
css = f"""
@font-face {{ font-family: DejaVu; src: url('{FONTS}/DejaVuSans.ttf'); }}
@font-face {{ font-family: DejaVu; src: url('{FONTS}/DejaVuSans-Bold.ttf'); font-weight: bold; }}
@font-face {{ font-family: DejaVu; src: url('{FONTS}/DejaVuSans-Oblique.ttf'); font-style: italic; }}
""" + """
@page { size: A4; margin: 1.6cm 1.5cm 1.6cm 1.5cm;
        @frame footer { -pdf-frame-content: footer; bottom: 0.6cm; margin-left: 1.5cm; margin-right: 1.5cm; height: 0.6cm; } }
body { font-family: DejaVu; font-size: 9pt; line-height: 1.38; color: #1f2933; }
h1 { font-size: 17pt; color: #1e3a8a; margin-bottom: 2pt; }
h2 { font-size: 12.5pt; color: #1e3a8a; border-bottom: 0.6pt solid #cbd2d9; padding-bottom: 2pt; margin-top: 12pt; }
h3 { font-size: 10.5pt; color: #334e68; margin-top: 8pt; }
p { margin: 3pt 0 5pt 0; }
li { margin-bottom: 2pt; }
code { font-family: DejaVu; font-size: 8pt; background-color: #f0f4f8; }
pre { font-family: DejaVu; font-size: 7.5pt; background-color: #f0f4f8; padding: 5pt; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; width: 100%; }
th { background-color: #e4e7eb; font-weight: bold; padding: 3pt 4pt; border: 0.5pt solid #cbd2d9; font-size: 8pt; text-align: left; }
td { padding: 2.5pt 4pt; border: 0.5pt solid #cbd2d9; font-size: 8pt; vertical-align: top; }
img { width: 17.5cm; }
"""
html = f"""<html><head><meta charset="utf-8"><style>{css}</style></head><body>{body}
<div id="footer" style="text-align:right; font-size:7pt; color:#7b8794;">TermMap-DX · HiLabs Hackathon 2026 · page <pdf:pagenumber/></div>
</body></html>"""
out = HERE / f"{NAME}.pdf"
with open(out, "wb") as f:
    status = pisa.CreatePDF(html, dest=f, encoding="utf-8", path=str(HERE / f"{NAME}.md"))
print("errors:", status.err, "->", out)
