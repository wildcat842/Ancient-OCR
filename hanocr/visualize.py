"""검수용 오버레이 이미지."""
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COLORS = {"textlayer": (30, 110, 255), "tesseract": (255, 140, 0),
          "facsimile": (220, 20, 60), "facsimile_ocr": (200, 0, 200)}


def overlay(page, res, path, dpi=110, labels=True):
    pix = page.get_pixmap(dpi=dpi)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    dr = ImageDraw.Draw(im)
    s = dpi / 72.0
    ox, oy = page.rect.x0, page.rect.y0
    try:
        font = ImageFont.truetype(FONT, max(9, int(9 * dpi / 72)), index=1)
    except Exception:
        font = ImageFont.load_default()
    for rg in res.get("regions", []):
        b = [(rg["bbox"][0] - ox) * s, (rg["bbox"][1] - oy) * s, (rg["bbox"][2] - ox) * s, (rg["bbox"][3] - oy) * s]
        dr.rectangle(b, outline=(0, 170, 0), width=2)
    for c in res["chars"]:
        key = c["engine"]
        if key == "facsimile" and not str(c.get("text_source", "")).startswith("transcript"):
            key = "facsimile_ocr"
        col = COLORS[key]
        b = c.get("ink_bbox") or c["bbox"]
        b = [(b[0] - ox) * s, (b[1] - oy) * s, (b[2] - ox) * s, (b[3] - oy) * s]
        dr.rectangle(b, outline=col, width=1)
        if labels and c["engine"] == "facsimile" and c["text"]:
            dr.text((b[2] + 1, b[1]), c["text"], fill=col, font=font)
    im.save(path, quality=85)
