"""Structural checks on the generated report (used instead of a visual render,
which is unavailable on this machine -- the LibreOffice renderer crashes)."""
import re
import sys

from docx import Document
from docx.shared import Emu

PATH = sys.argv[1] if len(sys.argv) > 1 else "report/报告_对抗攻击与对抗防御.docx"

doc = Document(PATH)
paras = doc.paragraphs
texts = [p.text for p in paras]
n_img = len(doc.inline_shapes)
n_tab = len(doc.tables)
missing = [t for t in texts if "缺少图" in t or t.strip() == "--"]
print(f"paragraphs      : {len(paras)}")
print(f"tables          : {n_tab}")
print(f"inline images   : {n_img}")
print(f"chars (no space): {sum(len(t) for t in texts)}")
print(f"missing figures : {len(missing)}")
for m in missing[:10]:
    print("   !", m[:80])

# caption / figure pairing
caps = [t for t in texts if re.match(r"^图\s*\d", t.strip())]
print(f"figure captions : {len(caps)}")

# check the images are not enormous / zero sized
sizes = [(s.width, s.height) for s in doc.inline_shapes]
print("image sizes (cm):", sorted({(round(w / 360000, 1), round(h / 360000, 1)) for w, h in sizes}))

# placeholder scan
ph = [t for t in texts if "____" in t]
print(f"placeholders    : {len(ph)}")

headings = [t for t in texts if re.match(r"^(第 \d+ 章|参考文献|附录)", t.strip())]
print("top-level headings:")
for h in headings:
    print("   ", h)
