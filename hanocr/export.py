"""영인 이미지 단위 OCR·레이아웃 데이터셋 내보내기.

page_NNNN_R0.png          : PDF에 내장된 영인 원본 이미지(원 해상도)
page_NNNN_R0.json         : regions(영역 라벨 계층) + chars(글자 text/bbox) + paragraphs(원문↔현대어역)
page_NNNN_R0.csv          : 글자 목록
page_NNNN_R0_regions.csv  : 영역 목록
page_NNNN_R0_overlay.jpg  : 검수용
좌표: 원본 이미지 픽셀 [x0,y0,x1,y1], 좌상단 원점. bbox_pt 는 PDF 포인트.
"""
import csv
import json
import os

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from .labels import LABELS

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MARGIN_FIELD = {"book_title": "MARGIN_TITLE", "section_title": "MARGIN_SECTION", "page_label": "MARGIN_FOLIO",
                None: "MARGIN_UNKNOWN"}


def _union(bs):
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def export_region(doc, res, region_id, outdir):
    os.makedirs(outdir, exist_ok=True)
    reg = next(g for g in res["regions"] if g["id"] == region_id)
    stem = os.path.join(outdir, f"page_{res['page']:04d}_{region_id}")
    si = reg.get("source_image")
    if si:
        pix = fitz.Pixmap(doc, si["xref"])
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        bx = si["bbox_pt"]
    else:
        bx = reg["bbox"]
        pix = doc[res["page"] - 1].get_pixmap(clip=fitz.Rect(*bx), dpi=reg.get("dpi", 150))
    pix.save(stem + ".png")
    W, H = pix.width, pix.height
    kx, ky = W / (bx[2] - bx[0]), H / (bx[3] - bx[1])

    def P(b):
        return [int(round((b[0] - bx[0]) * kx)), int(round((b[1] - bx[1]) * ky)),
                int(round((b[2] - bx[0]) * kx)), int(round((b[3] - bx[1]) * ky))]

    chars = [c for c in res["chars"] if c["engine"] == "facsimile" and c["region"] == region_id]
    regions = []

    def add(label, bbox_pt, parent=None, **attrs):
        rid = f"{region_id}.{label.lower()}_{sum(1 for x in regions if x['label'] == label)}"
        rg = {"id": rid, "label": label, "label_ko": LABELS[label]["ko"], "bbox": P(bbox_pt),
              "bbox_pt": [round(v, 2) for v in bbox_pt], "parent": parent}
        rg.update({k: v for k, v in attrs.items() if v is not None})
        regions.append(rg)
        return rid

    char_region = {}
    for blk in reg.get("blocks", []):
        inner = blk.get("inner_bbox", blk["bbox"])
        double = blk.get("inner_bbox") is not None and blk["inner_bbox"] != blk["bbox"]
        fr = add("FRAME", blk["bbox"], None, block=blk.get("id"), frame_lines="double" if double else "single/unknown")
        if blk["kind"] == "text":
            body = [c for c in chars if not c.get("panshim") and c.get("role") != "figure_text"
                    and (c.get("block") in (None, blk.get("id")))]
            if not body:
                continue
            tb = add("TEXT_BLOCK", _union([c["bbox"] for c in body]), fr)
            cols = {}
            for c in body:
                cols.setdefault(c["column"], []).append(c)
            for ci in sorted(cols):
                cs = sorted(cols[ci], key=lambda c: c["order"])
                main = [c for c in cs if not c.get("subline")]
                note = [c for c in cs if c.get("subline")]
                paras_in = {c.get("para") for c in cs}
                role = None
                if len(paras_in) == 1 and None not in paras_in:
                    pt = next((p["original"] for p in res.get("paragraphs", []) if p["para"] in paras_in), "")
                    if pt and len(pt.replace(" ", "")) <= 8 and len(cs) == len(pt.replace(" ", "")):
                        role = "heading"  # 소제목 행 (예: 쉿스란드국)
                colid = add("TEXT_COLUMN", _union([c["bbox"] for c in cs]), tb, column=ci, role=role,
                            text="".join(c["text"] for c in cs))
                for c in main:
                    char_region[id(c)] = (colid, "NOTE_CHAR" if c.get("small") else "BODY_CHAR")
                if note:
                    r_ = [c for c in note if c["subline"] == "R"]
                    l_ = [c for c in note if c["subline"] == "L"]
                    nid = add("DOUBLE_LINE_NOTE", _union([c["bbox"] for c in note]), colid,
                              text="".join(c["text"] for c in r_ + l_),
                              reading_order="right sub-line → left sub-line")
                    for sub, grp in (("R", r_), ("L", l_)):
                        if grp:
                            sid = add("NOTE_LINE", _union([c["bbox"] for c in grp]), nid, subline=sub,
                                      text="".join(c["text"] for c in grp))
                            for c in grp:
                                char_region[id(c)] = (sid, "NOTE_CHAR")
        else:
            info = blk.get("figure") or {}
            ftype = info.get("figure_type")
            fg = add("FIGURE", inner, fr, subtype="MAP", figure_id=info.get("caption_original"),
                     figure_id_modern=info.get("caption_modern"), figure_type=ftype,
                     description=info.get("commentary"), transcript_page=info.get("transcript_page"))
            ent = "CELESTIAL_BODY" if ftype == "천체도" else "PLACE_NAME"
            fchars = [c for c in chars if c.get("role") == "figure_text" and c.get("block") == blk.get("id")]
            for lab in blk.get("labels", []):
                if not lab.get("matched"):
                    continue
                cs = sorted([c for c in fchars if c["figure_label"] == lab["label_index"]], key=lambda c: c["order"])
                lid = add("FIGURE_LABEL", lab["bbox"], fg, text=lab["original"], modern=lab.get("modern"),
                          modern_source=lab.get("modern_source"), entity_type=ent,
                          writing_direction="vertical", verification=lab.get("verification"),
                          ocr_raw=lab.get("rec"),
                          marker_bbox=P(lab["marker_bbox"]) if lab.get("marker_bbox") else None)
                for c in cs:
                    char_region[id(c)] = (lid, "FIGURE_CHAR")
            for sy in blk.get("symbols", []):
                add("FIGURE_SYMBOL", sy["bbox"], fg, symbol_type="planet_marker" if sy["type"] == "marker"
                    else "satellite_dot")
            for cid, typ in (blk.get("symbol_candidates") or {}).items():
                c = next((x for x in blk.get("candidates", []) if x["id"] == int(cid)), None)
                if c:
                    add("FIGURE_SYMBOL", c["bbox"], fg, symbol_type=typ, verification="manual")
    # 광곽 바깥 난외 표기(판심)
    pan = [c for c in chars if c.get("panshim")]
    if pan:
        mt = add("MARGIN_TEXT", _union([c["bbox"] for c in pan]), None,
                 side=pan[0].get("margin_side"), note="광곽 바깥 난외 표기 (판심 상당)")
        segs = {}
        for c in pan:
            segs.setdefault((c.get("margin_side"), c.get("panshim_segment")), []).append(c)
        for key, cs in sorted(segs.items(), key=lambda kv: min(c["bbox"][1] for c in kv[1])):
            cs.sort(key=lambda c: c["bbox"][1])
            field = cs[0].get("panshim_field")
            sid = add(MARGIN_FIELD.get(field, "MARGIN_UNKNOWN"), _union([c["bbox"] for c in cs]), mt,
                      text="".join(c["text"] for c in cs), source=cs[0].get("text_source"))
            for c in cs:
                char_region[id(c)] = (sid, "MARGIN_CHAR")

    # 글자 목록 (읽기 순서: 블록 → 행 → 행 내 / 도판 표기 / 난외)
    def okey(c):
        if c.get("panshim"):
            return (3, 0, c["bbox"][1])
        if c.get("role") == "figure_text":
            return (1, c["figure_label"], c["order"])
        return (2 if any(b["kind"] == "figure" for b in reg.get("blocks", [])) else 0, c["column"], c["order"])
    items = []
    for n, c in enumerate(sorted([c for c in chars if id(c) in char_region or not c.get("panshim")], key=okey)):
        rid, lab = char_region.get(id(c), (None, "BODY_CHAR"))
        items.append({"id": c["id"], "reading_order": n, "label": lab, "region_id": rid, "text": c["text"],
                      "modern": c.get("modern"), "rec": c.get("rec", ""), "text_source": c.get("text_source"),
                      "bbox": P(c["bbox"]), "bbox_pt": c["bbox"], "bbox_estimated": c.get("bbox_estimated", False),
                      "paragraph": c.get("para"), "kind": c.get("kind")})
    idmap = {it["id"]: it for it in items}
    paras = []
    for p in res.get("paragraphs", []):
        its = [idmap[i] for i in p["char_ids"] if i in idmap]
        paras.append({"paragraph": p["para"], "header": p["header"], "original": p["original"],
                      "modern": p["modern"], "n_chars": len(its), "char_ids": p["char_ids"]})
    sec = res.get("sections") or {}
    ann = {"source_pdf_page": res["page"], "region": region_id, "image": os.path.basename(stem + ".png"),
           "image_size": [W, H], "coordinate": "image pixel [x0,y0,x1,y1], top-left origin",
           "writing_direction": "vertical, columns right-to-left",
           "margin_header_from_transcript": sec.get("header"), "alignment": res.get("alignment"),
           "label_schema": "hanocr.labels (see LABELS.md)",
           "regions": regions, "chars": items, "paragraphs": paras}
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=1)
    with open(stem + ".csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reading_order", "id", "label", "region_id", "text", "modern", "rec", "x0", "y0", "x1", "y1",
                    "bbox_estimated", "paragraph", "text_source"])
        for it in items:
            w.writerow([it["reading_order"], it["id"], it["label"], it["region_id"], it["text"], it["modern"] or "",
                        it["rec"], *it["bbox"], it["bbox_estimated"], "" if it["paragraph"] is None else it["paragraph"],
                        it["text_source"]])
    with open(stem + "_regions.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "label_ko", "parent", "x0", "y0", "x1", "y1", "text", "modern", "attributes"])
        for rg in regions:
            extra = {k: v for k, v in rg.items() if k not in ("id", "label", "label_ko", "parent", "bbox", "bbox_pt",
                                                               "text", "modern", "description")}
            w.writerow([rg["id"], rg["label"], rg["label_ko"], rg["parent"] or "", *rg["bbox"], rg.get("text", ""),
                        rg.get("modern", "") or "", json.dumps(extra, ensure_ascii=False)])
    _overlay(stem, items, regions)
    return ann


def _overlay(stem, items, regions):
    im = Image.open(stem + ".png").convert("RGB")
    sc = 1800 / max(im.size)
    im = im.resize((int(im.width * sc), int(im.height * sc)))
    dr = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, 17, index=1)
    fb = ImageFont.truetype(FONT, 20, index=1)
    for rg in sorted(regions, key=lambda r: LABELS[r["label"]]["z"]):
        L = LABELS[rg["label"]]
        if not L.get("draw", True):
            continue
        b = [v * sc for v in rg["bbox"]]
        dr.rectangle(b, outline=L["color"] + (255,), width=L.get("width", 2))
        if L.get("tag"):
            t = rg["label"] + (f" {rg['text']}" if rg["label"] in ("FIGURE_LABEL", "FIGURE_TITLE") or rg["label"].startswith("MARGIN_") and rg.get("text") else "")
            if rg["label"] == "FIGURE":
                t = f"FIGURE/MAP · {rg.get('figure_id', '')} · {rg.get('figure_type', '')}"
            if rg["label"] in ("FIGURE_LABEL", "FIGURE_TITLE") and rg.get("modern"):
                t += f"={rg['modern']}"
            tw = dr.textlength(t, font=fb)
            x, y = (b[2] + 3, b[1]) if rg["label"] in ("FIGURE_LABEL",) or rg["label"].startswith("MARGIN_") else (b[0], b[1] - 24)
            if rg["label"] == "FIGURE_PANEL":
                t = f"FIGURE_PANEL {rg.get('panel_title', '')}"
            if x + tw > im.width:
                x = b[0] - tw - 3
            dr.rectangle([x - 1, y, x + tw + 2, y + 24], fill=(255, 255, 255, 215))
            dr.text((x, y), t, fill=L["color"] + (255,), font=fb)
    for it in items:
        b = [v * sc for v in it["bbox"]]
        col = LABELS[it["label"]]["color"]
        dr.rectangle(b, outline=col + (220,), width=1)
        if it["label"] not in ("FIGURE_CHAR",):
            dr.text((b[2] + 1, b[1] - 2), it["text"], fill=col + (255,), font=f)
    im.save(stem + "_overlay.jpg", quality=88)
