"""Small helpers that make python-docx produce a decent-looking Chinese report.

* 正文：宋体 / Times New Roman，小四（12 pt），1.5 倍行距，首行缩进 2 字符
* 标题：黑体
* 图：居中，图注在图下（五号、居中、加粗编号）
* 表：三线表风格（只用上下框线 + 表头下框线）
* 公式：用 matplotlib mathtext 渲染成透明 PNG 后插入，避免 OMML 兼容问题
"""
import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BODY_FONT_CN = "宋体"
BODY_FONT_EN = "Times New Roman"
HEAD_FONT_CN = "黑体"
HEAD_FONT_EN = "Times New Roman"
EQ_DIR = "figures/equations"


# --------------------------------------------------------------------------- #
def _set_run_font(run, cn=BODY_FONT_CN, en=BODY_FONT_EN, size=12, bold=False,
                  color=None, italic=False):
    run.font.name = en
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)


def new_document():
    doc = Document()
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(2.6)
        section.right_margin = Cm(2.6)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.4)

    style = doc.styles["Normal"]
    style.font.name = BODY_FONT_EN
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT_CN)
    style.paragraph_format.first_line_indent = Pt(24)
    style.paragraph_format.line_spacing = 1.5
    return doc


def heading(doc, text, level=1, space_before=12, space_after=6, page_break=False):
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True
    sizes = {0: 20, 1: 16, 2: 14, 3: 12.5}
    run = p.add_run(text)
    _set_run_font(run, cn=HEAD_FONT_CN, en=HEAD_FONT_EN, size=sizes.get(level, 12), bold=True)
    if level == 0:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def para(doc, text, indent=True, size=12, bold=False, align=None, space_after=3,
         first_line_indent=None, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.5
    if first_line_indent is not None:
        p.paragraph_format.first_line_indent = Pt(first_line_indent)
    elif indent:
        p.paragraph_format.first_line_indent = Pt(24)
    else:
        p.paragraph_format.first_line_indent = Pt(0)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # inline **bold** segments
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        run = p.add_run(part)
        _set_run_font(run, size=size, bold=bold or (i % 2 == 1), color=color)
    return p


def bullet(doc, text, size=12, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.line_spacing = 1.4
    p.paragraph_format.space_after = Pt(2)
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        run = p.add_run(part)
        _set_run_font(run, size=size, bold=(i % 2 == 1))
    return p


def caption(doc, text, size=10.5, space_after=10):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.first_line_indent = Pt(0)
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        run = p.add_run(part)
        _set_run_font(run, size=size, bold=(i % 2 == 1))
    return p


def figure(doc, png_path, cap, width_cm=15.0):
    if not os.path.exists(png_path):
        para(doc, f"[缺少图：{png_path}]", indent=False, align="center", color=RGBColor(0xC0, 0, 0))
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(png_path, width=Cm(width_cm))
    caption(doc, cap)


def _set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge in ("top", "bottom", "left", "right"):
        if edge in kwargs:
            tag = "w:{}".format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in kwargs[edge].items():
                element.set(qn("w:{}".format(key)), str(val))


def table(doc, header, rows, cap=None, font_size=9.5, col_widths=None, align="center"):
    """Three-line table: top/bottom thick, header rule below."""
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(str(h))
        _set_run_font(run, size=font_size, bold=True)
        _set_cell_border(hdr[i], top={"val": "single", "sz": 12}, bottom={"val": "single", "sz": 8})
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if align == "center" else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(str(v))
            _set_run_font(run, size=font_size)
    # bottom rule on the last row
    for c in t.rows[-1].cells:
        _set_cell_border(c, bottom={"val": "single", "sz": 12})
    if cap:
        caption(doc, cap, size=10.5, space_after=8)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def equation(doc, tex, number=None, fontsize=13, width_cm=None):
    """Render a mathtext equation to a transparent PNG and centre it."""
    os.makedirs(EQ_DIR, exist_ok=True)
    key = str(abs(hash(tex)) % (10 ** 10))
    path = os.path.join(EQ_DIR, f"eq_{key}.png")
    if not os.path.exists(path):
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=fontsize)
        fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True,
                    pad_inches=0.02)
        plt.close(fig)
    from PIL import Image
    w_px, _ = Image.open(path).size
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    if width_cm is None:
        width_cm = min(15.0, w_px / 300 * 2.54 * 1.0)
    p.add_run().add_picture(path, width=Cm(width_cm))
    if number:
        p.add_run(f"    ({number})")
    return p


def code_block(doc, text, size=9):
    for line in text.strip("\n").split("\n"):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        run = p.add_run(line if line else " ")
        _set_run_font(run, cn="Consolas", en="Consolas", size=size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_page_number_footer(doc, text_left=""):
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text_left)
    _set_run_font(run, size=9)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    r = p.add_run()
    _set_run_font(r, size=9)
    r._r.append(fld_begin)
    r._r.append(instr)
    r._r.append(fld_end)


def toc_entry(doc, text, level=0):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Pt(18 * level)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    _set_run_font(run, size=11.5, bold=(level == 0))
