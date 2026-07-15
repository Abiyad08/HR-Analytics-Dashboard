"""
Export helpers: turn matplotlib figures into downloadable JPG bytes,
and bundle a set of (title, figure, insight) tuples into a PPTX deck.
"""

import io

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BRAND_BLUE = RGBColor(0x1E, 0x3A, 0x8A)
BRAND_GREY = RGBColor(0x37, 0x41, 0x51)


def fig_to_jpg_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()


def build_pptx(selections: list, company_name: str = "HR Analytics Report") -> bytes:
    """
    selections: list of dicts with keys 'title', 'fig', 'insight'
    Returns raw pptx bytes.
    """
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Title slide
    slide = prs.slides.add_slide(blank_layout)
    bg = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BRAND_BLUE
    bg.line.fill.background()
    bg.shadow.inherit = False

    title_box = slide.shapes.add_textbox(Inches(1), Inches(2.8), Inches(11.3), Inches(1.5))
    tf = title_box.text_frame
    tf.text = company_name
    tf.paragraphs[0].font.size = Pt(44)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    sub_box = slide.shapes.add_textbox(Inches(1), Inches(4.0), Inches(11.3), Inches(0.8))
    sf = sub_box.text_frame
    sf.text = "Generated with the HR Analytics Dashboard"
    sf.paragraphs[0].font.size = Pt(18)
    sf.paragraphs[0].font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)

    # One slide per selected analytic
    for item in selections:
        slide = prs.slides.add_slide(blank_layout)

        # header bar
        bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.9))
        bar.fill.solid()
        bar.fill.fore_color.rgb = BRAND_BLUE
        bar.line.fill.background()
        bar.shadow.inherit = False

        title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.12), Inches(12.5), Inches(0.7))
        tf = title_box.text_frame
        tf.text = item["title"]
        tf.paragraphs[0].font.size = Pt(26)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        img_bytes = fig_to_jpg_bytes(item["fig"])
        img_stream = io.BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(0.7), Inches(1.15), height=Inches(5.3))

        if item.get("insight"):
            note_box = slide.shapes.add_textbox(Inches(0.7), Inches(6.6), Inches(11.9), Inches(0.7))
            nf = note_box.text_frame
            nf.word_wrap = True
            nf.text = f"Insight: {item['insight']}"
            nf.paragraphs[0].font.size = Pt(13)
            nf.paragraphs[0].font.italic = True
            nf.paragraphs[0].font.color.rgb = BRAND_GREY

    out = io.BytesIO()
    prs.save(out)
    out.seek(0)
    return out.getvalue()
