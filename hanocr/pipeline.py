"""페이지/문서 단위 통합 파이프라인.

엔진
  textlayer : PDF 텍스트 레이어 (정확 좌표, 신뢰도 1.0)
  tesseract : 래스터 인쇄체(가로쓰기) OCR + 잉크 투영 경계 보정
  facsimile : 고문헌 영인 세로쓰기 글자 검출 + 일괄 인식 + 판독문 정렬
좌표: bbox 는 PDF 포인트(좌상단 원점, PyMuPDF 좌표계). bbox_px 는 render_dpi 기준 픽셀.
"""
import json
import os

import cv2
import numpy as np
import pymupdf as fitz

from .align import align, transcript_units, sim
from .facsimile import detect_chars, detect_margins
from .figure import detect_labels, split_k, trim_dots
from .layout import detect_blocks
from .normalize import char_kind
from .recognize import recognize
from .render import render, native_dpi, pdf_image_regions, detect_facsimile_regions
from .tess import ocr_chars, refine_boxes
from .textlayer import extract_chars, tighten
from .transcript import find_transcript
from .sections import find_sections, strip_for_lines


def _inside(b, r, frac=0.6):
    ix = max(0, min(b[2], r[2]) - max(b[0], r[0]))
    iy = max(0, min(b[3], r[3]) - max(b[1], r[1]))
    a = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / a >= frac


class PageResult(dict):
    pass


def facsimile_regions(page):
    """영인 후보 영역(pt). 전면 스캔 페이지면 색상 기반 검출."""
    area = page.rect.width * page.rect.height
    infos = page.get_image_info()
    full = [b for b in infos if (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) > 0.9 * area]
    if full:
        d = native_dpi(page) or 150
        img = render(page, dpi=d)
        s = 72.0 / d
        return [[r[0] * s + page.rect.x0, r[1] * s + page.rect.y0, r[2] * s + page.rect.x0, r[3] * s + page.rect.y0]
                for r in detect_facsimile_regions(img)], True
    return pdf_image_regions(page), False


def process_page(doc, pno, dpi=300, fac_min_chars=15, tight=True, lang="kor+chi_tra"):
    page = doc[pno]
    W, H = page.rect.width, page.rect.height
    res = PageResult(page=pno + 1, width_pt=round(W, 2), height_pt=round(H, 2), render_dpi=dpi,
                     regions=[], chars=[])
    tl = extract_chars(page)
    has_text = len(tl) > 20
    regs, fullscan = facsimile_regions(page)
    imginfo = page.get_image_info(xrefs=True)
    res["page_type"] = "born-digital" if has_text else ("full-page-scan" if fullscan else "outlined-vector+images")

    # ---- 1. 영인(고문헌 사진) 영역 ----
    fac_chars = []
    for ri, r in enumerate(regs):
        rect = fitz.Rect(*r)
        nd = native_dpi(page) or dpi
        fdpi = int(min(400, max(150, nd)))
        pix = page.get_pixmap(clip=rect, dpi=fdpi)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        s = 72.0 / fdpi
        to_pt = lambda b: [round(r[0] + b[0] * s, 2), round(r[1] + b[1] * s, 2),
                           round(r[0] + b[2] * s, 2), round(r[1] + b[3] * s, 2)]
        reg = {"id": f"R{ri}", "bbox": [round(v, 2) for v in r], "type": "image", "dpi": fdpi}
        blocks, bink, bstroke = detect_blocks(img)
        has_fig = any(b["kind"] == "figure" for b in blocks)
        units = []   # (char dict, ink image, bbox_px) 인식 대기열
        if not has_fig:
            det = detect_chars(img)
            reg.update(n_columns=len(det["columns"]), n_chars=len(det["chars"]))
            if len(det["chars"]) >= fac_min_chars and len(det["columns"]) >= 2:
                reg["type"] = "facsimile"
                frame = det["frame"]
                reg["blocks"] = [{"kind": "text", "bbox": to_pt(frame), "columns": [
                    dict(c, bbox=to_pt([c["x0"], c["y0"], c["x1"], c["y1"]])) for c in det["columns"]]}]
                for c in det["chars"] + det.get("margin_chars", []):
                    units.append((c, det["ink"], "text" if not c.get("margin") else "margin", 0))
        else:
            reg["type"] = "facsimile"
            reg["blocks"] = []
            col_off, heights, pitch = 0, [], None
            for bi, b in enumerate(blocks):
                ob = b["outer"]
                entry = {"id": f"B{bi}", "kind": b["kind"], "bbox": to_pt(ob), "inner_bbox": to_pt(b["inner"])}
                if b["kind"] == "text":
                    m = int(0.01 * max(img.shape[:2]))
                    X0, Y0 = max(0, ob[0] - m), max(0, ob[1] - m)
                    crop = img[Y0:min(img.shape[0], ob[3] + m), X0:min(img.shape[1], ob[2] + m)]
                    det = detect_chars(crop)
                    pitch = det["pitch"]
                    ink_full = np.zeros(img.shape[:2], np.uint8)
                    ink_full[Y0:Y0 + crop.shape[0], X0:X0 + crop.shape[1]] = det["ink"]
                    entry["columns"] = [dict(c, id=c["id"] + col_off,
                                             bbox=to_pt([c["x0"] + X0, c["y0"] + Y0, c["x1"] + X0, c["y1"] + Y0]))
                                        for c in det["columns"]]
                    for c in det["chars"]:
                        c = dict(c, bbox=[c["bbox"][0] + X0, c["bbox"][1] + Y0, c["bbox"][2] + X0, c["bbox"][3] + Y0],
                                 column=c["column"] + col_off, block=entry["id"])
                        heights.append(c["bbox"][3] - c["bbox"][1])
                        units.append((c, ink_full, "text", 0))
                    col_off += len(det["columns"])
                reg["blocks"].append(entry)
            H_body = float(np.median(heights)) if heights else 0.05 * img.shape[0]
            # 도판 안 표기
            for bi, b in enumerate(blocks):
                if b["kind"] != "figure":
                    continue
                entry = reg["blocks"][bi]
                ch = 0.9 * H_body  # 표기 글자 상한 추정(크게 잡아야 글자가 선으로 오분류되지 않음)
                cands, symbols, fig_ink = detect_labels(bink, b["inner"], ch, bstroke)
                entry["label_char_size_px"] = round(ch, 1)
                entry["symbols"] = [dict(sy, bbox=to_pt(sy["bbox"])) for sy in symbols]
                entry["candidates"] = []
                for cd in cands:
                    comps = trim_dots(cd["components"], ch)
                    bx = [min(c[0] for c in comps), min(c[1] for c in comps), max(c[2] for c in comps), max(c[3] for c in comps)]
                    w, h = bx[2] - bx[0], bx[3] - bx[1]
                    k = max(1, int(round(h / max(1.0, 0.95 * min(w, ch * 1.2)))))
                    pieces = split_k(comps, k)
                    ce = {"id": cd["id"], "parts": cd["parts"], "bbox": to_pt(bx),
                          "components": [to_pt(c) for c in comps], "rec": []}
                    entry["candidates"].append(ce)
                    for pc in pieces:
                        units.append(({"bbox": pc, "cand": ce}, fig_ink, "figure", 0))
            # 난외(광곽 바깥) 표기
            ux0 = min(b["outer"][0] for b in blocks); ux1 = max(b["outer"][2] for b in blocks)
            uy0 = min(b["outer"][1] for b in blocks); uy1 = max(b["outer"][3] for b in blocks)
            mchars, mink = detect_margins(img, [ux0, uy0, ux1, uy1], H_body, pitch or H_body * 1.3)
            for c in mchars:
                units.append((c, mink, "margin", 0))
            reg["n_chars"] = len(units)
        # ---- 일괄 인식 ----
        if units:
            by_ink = {}
            for u in units:
                by_ink.setdefault(id(u[1]), []).append(u)
            for group in by_ink.values():
                recs = recognize(group[0][1], [u[0]["bbox"] for u in group], lang=lang)
                for (c, _, kind, _), (t, cf) in zip(group, recs):
                    if kind == "figure":
                        c["cand"]["rec"].append({"bbox": to_pt(c["bbox"]), "rec": t, "conf": round(float(cf), 3)})
                        continue
                    fac_chars.append({"panshim": kind == "margin", "panshim_segment": c.get("segment"),
                        "margin_side": c.get("margin"),
                        "text": t, "rec": t, "conf": round(float(cf), 3), "bbox": to_pt(c["bbox"]),
                        "engine": "facsimile", "region": reg["id"], "block": c.get("block"),
                        "column": c.get("column", -1), "order": c.get("order_in_column", 0),
                        "small": bool(c.get("small")), "subline": c.get("subline"), "direction": "ttb",
                        "text_source": "ocr"})
        reg["n_panshim"] = sum(1 for c in fac_chars if c.get("panshim") and c["region"] == reg["id"])
        # 원본(내장) 이미지 픽셀 좌표 환산 정보
        for b in imginfo:
            bb = b["bbox"]
            if _inside(r, bb, 0.95) and b.get("xref"):
                reg["source_image"] = {"xref": b["xref"], "width": b["width"], "height": b["height"],
                                       "bbox_pt": [round(v, 3) for v in bb]}
                break
        res["regions"].append(reg)
    fac_rects = [rg["bbox"] for rg in res["regions"]]

    # ---- 2. 인쇄체(가로·세로 활자) ----
    typeset = []
    if has_text:
        if tight:
            tighten(tl, page, dpi=dpi)
        for c in tl:
            if any(_inside(c["bbox"], fr) for fr in fac_rects):
                continue
            typeset.append({"text": c["text"], "bbox": [round(v, 2) for v in c["bbox"]],
                            "ink_bbox": c.get("ink_bbox"), "conf": 1.0, "engine": "textlayer",
                            "line": c["line"], "direction": c["direction"], "font": c["font"],
                            "size": c["size"], "raw_pua": c["raw"] if c["raw"] != c["text"] else None})
    else:
        img = render(page, dpi=dpi, gray=True)
        sc = dpi / 72.0
        for fr in fac_rects:  # 영인 영역은 가림
            x0, y0, x1, y1 = [int(v * sc) for v in (fr[0] - page.rect.x0, fr[1] - page.rect.y0,
                                                     fr[2] - page.rect.x0, fr[3] - page.rect.y0)]
            img[max(0, y0):y1, max(0, x0):x1] = 255
        cs = refine_boxes(ocr_chars(img, lang=lang), img)
        for c in cs:
            b = [round(page.rect.x0 + v / sc, 2) if i % 2 == 0 else round(page.rect.y0 + v / sc, 2)
                 for i, v in enumerate(c["bbox"])]
            typeset.append({"text": c["text"], "bbox": b, "conf": c["conf"], "engine": "tesseract",
                            "line": c["line"], "word": c["word"], "direction": "ltr"})

    # ---- 3. 판독문(정렬용 정답 텍스트) 찾기 ----
    seq, tbox = find_transcript(typeset, W + page.rect.x0, H + page.rect.y0)
    res["transcript"] = None
    if seq:
        res["transcript"] = {"bbox": [round(v, 2) for v in tbox],
                             "text": "".join(c["text"] for c in seq),
                             "source": typeset[0]["engine"] if typeset else None}
    if has_text:  # 띄어쓰기 보존본으로 문단 구성
        sec_chars = [dict(c, text=c["text"]) for c in extract_chars(page, keep_space=True)]
    else:
        sec_chars = typeset
    sec = find_sections(sec_chars, W + page.rect.x0, H + page.rect.y0) if sec_chars else None
    if sec:
        res["sections"] = {"header": sec["header"], "pairs": sec["pairs"], "gloss": sec.get("gloss"),
                           "commentary": sec.get("commentary"),
                           "original": [strip_for_lines(p) for p in sec["original"]],
                           "modern": [strip_for_lines(p) for p in sec["modern"]]}
    res["chars"] = typeset + fac_chars
    return res


def _gold_for(src):
    """정렬용 정답 글자열과 글자별 문단 번호. 머리줄(〈쪽〉 ※ 제목)은 판심용이므로 제외."""
    sec = src.get("sections")
    if sec and sec["original"]:
        gold, para = [], []
        for i, p in enumerate(sec["original"]):
            if p["header"] or p.get("role") in ("figure_caption", "figure_labels"):
                continue
            u = transcript_units(p["text"])
            gold += u
            para += [i] * len(u)
        return gold, para, sec["header"]
    if src.get("transcript"):
        g = transcript_units(src["transcript"]["text"])
        return g, [None] * len(g), None
    return [], [], None


def _label_panshim(chars, header, lexicon):
    """판심 글자 묶음(segment)에 판독문 머리줄의 장차·제목 및 서명 사전을 배정."""
    segs = {}
    for c in chars:
        segs.setdefault((c["region"], c.get("margin_side"), c.get("panshim_segment")), []).append(c)
    cands = []
    if header:
        if header.get("page_label"):
            cands.append(("page_label", header["page_label"]))
        if header.get("section_title"):
            cands.append(("section_title", header["section_title"]))
    cands += [("book_title", t) for t in (lexicon or [])]
    used = set()
    for key, cs in sorted(segs.items(), key=lambda kv: kv[1][0]["bbox"][1]):
        cs.sort(key=lambda c: c["bbox"][1])
        best = None
        for ci, (kind, text) in enumerate(cands):
            if ci in used:
                continue
            u = transcript_units(text)
            if len(u) != len(cs):
                continue
            sc = sum(1.0 if c["rec"] == g else 0.3 if c["rec"] and _first(c["rec"]) == _first(g) else 0
                     for c, g in zip(cs, u))
            if best is None or sc > best[0]:
                best = (sc, ci, kind, u)
        for c in cs:
            c["panshim_field"] = None
        if best:
            used.add(best[1])
            for c, g in zip(cs, best[3]):
                c["text"], c["text_source"], c["panshim_field"] = g, "panshim-header", best[2]


def _first(s):
    import unicodedata
    return unicodedata.normalize("NFD", s)[:1]


def align_document(results, panshim_lexicon=None):
    """영인이 있는 쪽마다 같은 쪽·앞뒤 쪽 판독문 중 가장 잘 맞는 것과 정렬."""
    by_page = {r["page"]: r for r in results}
    for r in results:
        r["chars"] = [c for c in r["chars"] if c.get("role") != "figure_text"]
        _assign_figures(r, by_page)
        allfac = [c for c in r["chars"] if c["engine"] == "facsimile" and c.get("role") != "figure_text"]
        pan = [c for c in allfac if c.get("panshim")]
        fac = [c for c in allfac if not c.get("panshim")]
        for c in allfac:
            for k in ("gold_index", "para", "panshim_field"):
                c.pop(k, None)
        if not fac:
            continue
        fac.sort(key=lambda c: (c["region"], c["column"], c["order"]))
        recs = [c["rec"] for c in fac]
        best = None
        for q in (r["page"], r["page"] - 1, r["page"] + 1):
            src = by_page.get(q)
            if not src:
                continue
            gold, para, header = _gold_for(src)
            if len(gold) < 5:
                continue
            asg, st = align(recs, gold)
            score = st["exact_rec_matches"] - 0.2 * abs(len(gold) - len(recs))
            if best is None or score > best[0]:
                best = (score, q, gold, asg, st, para, header)
        ok = best is not None and best[4]["exact_rec_matches"] >= max(3, 0.1 * len(fac))
        method = "anchor"
        if not ok:
            cands = []
            for q in (r["page"], r["page"] - 1, r["page"] + 1):
                src = by_page.get(q)
                if src and src.get("transcript") and src["transcript"].get("source") == "textlayer":
                    gold, para, header = _gold_for(src)
                    if len(gold) >= 20 and abs(len(gold) - len(fac)) <= 0.12 * len(gold):
                        cands.append((abs(len(gold) - len(fac)), q, gold, para, header))
            if cands:
                _, q, gold, para, header = min(cands, key=lambda x: x[0])
                asg, st = align(recs, gold)
                best, ok, method = (0, q, gold, asg, st, para, header), True, "count"
        for rg in r.get("regions", []):
            if rg["type"] == "facsimile":
                rg["aligned"] = bool(ok)
        if pan:
            hdr = best[6] if best else (r.get("sections") or {}).get("header")
            _label_panshim(pan, hdr, panshim_lexicon)
        if not ok:
            r["alignment"] = {"status": "no-transcript-match"}
            for c in fac:
                c["text"], c["text_source"] = c["rec"], "ocr"
            continue
        _, q, gold, asg, st, para, header = best
        for c, a in zip(fac, asg):
            if a is not None:
                c["text"] = gold[a]
                c["text_source"] = "transcript-aligned" if method == "anchor" else "transcript-aligned-by-count"
                c["gold_index"] = a
                if para[a] is not None:
                    c["para"] = para[a]
            else:
                c["text"], c["text_source"] = c["rec"], "ocr"
        st.update(status="aligned" if method == "anchor" else "aligned-by-count", transcript_page=q)
        r["alignment"] = st
        r["paragraphs"] = _build_paragraphs(fac, by_page[q].get("sections"), q)


def _label_sources(r, by_page):
    """도판 표기 목록(원문·현대어)과 도판 번호를 같은 쪽 → 앞뒤 쪽 판독문에서 찾음."""
    for q in (r["page"], r["page"] - 1, r["page"] + 1):
        sec = (by_page.get(q) or {}).get("sections")
        if not sec:
            continue
        o = [i for i, p in enumerate(sec["original"]) if p.get("role") == "figure_labels"]
        if not o:
            continue
        i = o[0]
        cap = [p["text"] for p in sec["original"][:i] if p.get("role") == "figure_caption"]
        mods = [sec["modern"][j] for (a, j) in sec["pairs"] if a == i]
        mcap = [sec["modern"][j]["text"] for (a, j) in sec["pairs"]
                if sec["original"][a].get("role") == "figure_caption"]
        otoks = sec["original"][i]["text"].split()
        mtoks = mods[0]["text"].split() if mods else []
        gloss = sec.get("gloss") or {}
        labels = []
        for k, t in enumerate(otoks):
            m = gloss.get(t) or (mtoks[k] if len(mtoks) == len(otoks) else None)
            labels.append({"original": t, "modern": m,
                           "modern_source": "commentary_gloss" if t in gloss else ("label_list_order" if m else None)})
        com = sec.get("commentary") or {}
        ftype = None
        for kw in ("천체도", "세계지도", "지도", "도해", "그림"):
            if kw in (com.get("text") or ""):
                ftype = kw
                break
        return {"transcript_page": q, "caption_original": cap[0] if cap else None,
                "caption_modern": mcap[0] if mcap else None, "labels": labels,
                "figure_type": ftype, "commentary": com.get("text")}
    return None


def _jamo(s):
    import unicodedata
    return [c for c in unicodedata.normalize("NFD", s) if not c.isspace()]


def _lev(a, b):
    D = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, D[0] = D[0], i
        for j in range(1, len(b) + 1):
            cur = min(D[j] + 1, D[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, D[j] = D[j], cur
    return D[len(b)]


def _cand_score(recs, units):
    """자모 단위 편집거리 유사도 (옛한글·인식 오류에 너그러움) − 글자 수 차이 벌점."""
    a, b = _jamo("".join(recs)), _jamo("".join(units))
    if not a or not b:
        return 0.0
    simj = 1 - _lev(a, b) / max(len(a), len(b))
    n_rec = sum(1 for x in recs if x)
    return simj - 0.08 * abs(len(units) - max(len(recs), 1)) + (0.05 if n_rec else -0.2)


CORRECTIONS = {}   # {page: {block_id: {"labels": {원문표기: [후보id,...]}, "symbols": {후보id: 유형}}}}


def load_corrections(path):
    """검수 보정 파일(JSON) 읽기. 자동 배정이 틀린 도판 표기를 사람이 확인해 고정할 때 사용."""
    import json as _j
    with open(path, encoding="utf-8") as f:
        d = _j.load(f)
    CORRECTIONS.clear()
    CORRECTIONS.update({int(k): v for k, v in d.get("pages", {}).items()})


def _union4(bs):
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _tighten(box, comps):
    """box 와 겹치는 성분들의 x 범위로 좁힘 (y 는 box 유지)."""
    xs = [c for c in comps if min(c[3], box[3]) - max(c[1], box[1]) > 0]
    if not xs:
        return box
    return [min(c[0] for c in xs), box[1], max(c[2] for c in xs), box[3]]


def _assign_figures(r, by_page):
    from .align import transcript_units
    src = None
    for reg in r.get("regions", []):
        for b in reg.get("blocks", []):
            if b["kind"] != "figure":
                continue
            src = src or _label_sources(r, by_page)
            b["figure"] = {k: v for k, v in (src or {}).items() if k != "labels"} if src else None
            if not src:
                continue
            cands = b.get("candidates", [])
            labs = src["labels"]
            units = [transcript_units(l["original"]) for l in labs]
            scores = []
            for ci, c in enumerate(cands):
                recs = [x["rec"] for x in c["rec"]]
                for li, u in enumerate(units):
                    scores.append((_cand_score(recs, u), ci, li))
            scores.sort(reverse=True)
            used_parts, used_labels, result = set(), set(), {}
            corr = CORRECTIONS.get(r["page"], {}).get(b["id"], {})
            b["symbol_candidates"] = {}
            for cid, typ in corr.get("symbols", {}).items():
                c = next((x for x in cands if x["id"] == int(cid)), None)
                if c:
                    used_parts |= set(c["parts"])
                    b["symbol_candidates"][int(cid)] = typ
            for li, lab in enumerate(labs):
                ids = corr.get("labels", {}).get(lab["original"])
                if ids is None:
                    continue
                sel = [x for x in cands if x["id"] in ids]
                if not sel:
                    continue
                if len(sel) == 1:
                    ci = cands.index(sel[0])
                else:  # 여러 후보를 합친 새 후보
                    comps = [c for x in sel for c in x["components"]]
                    merged = {"id": len(cands), "parts": sorted({p for x in sel for p in x["parts"]}),
                              "components": comps, "rec": [q for x in sel for q in x["rec"]],
                              "bbox": [min(c[0] for c in comps), min(c[1] for c in comps),
                                       max(c[2] for c in comps), max(c[3] for c in comps)]}
                    cands.append(merged)
                    ci = len(cands) - 1
                used_labels.add(li); used_parts |= set(cands[ci]["parts"]); result[li] = (ci, "manual")
            for sc, ci, li in scores:
                if sc < 0.2:
                    break
                parts = set(cands[ci]["parts"])
                if li in used_labels or parts & used_parts:
                    continue
                used_labels.add(li); used_parts |= parts; result[li] = (ci, sc)
            b["labels"] = []
            for li, lab in enumerate(labs):
                entry = dict(lab, label_index=li, matched=li in result)
                if li in result:
                    ci, sc = result[li]
                    c = cands[ci]
                    u = units[li]
                    pieces = split_k(c["components"], len(u))
                    est = False
                    ub = _union4(c["components"])
                    uw, uh = ub[2] - ub[0], ub[3] - ub[1]
                    if len(pieces) < len(u) and uh >= 0.8 * len(u) * uw:
                        # 글자가 붙어 덩어리 수가 부족하지만 높이는 충분 → 높이를 k 등분
                        hh = uh / len(u)
                        pieces = [[ub[0], ub[1] + i * hh, ub[2], ub[1] + (i + 1) * hh] for i in range(len(u))]
                        pieces = [_tighten(pc, c["components"]) for pc in pieces]
                    if len(pieces) < len(u):  # 선에 가려 누락된 글자 → 같은 폭·평균 높이로 추정 박스
                        recs = [x["rec"] for x in c["rec"]]
                        asg, _ = align(recs[:len(pieces)], u)
                        hbar = np.mean([p[3] - p[1] for p in pieces])
                        gapv = hbar * 0.15
                        first = next((a for a in asg if a is not None), 0)
                        full = [None] * len(u)
                        for pi, a in enumerate(asg):
                            if a is not None:
                                full[a] = pieces[pi]
                        rest = [p for p, a in zip(pieces, asg) if a is None]
                        for gi in range(len(u)):
                            if full[gi] is None and rest:
                                full[gi] = rest.pop(0)
                        for gi in range(len(u)):
                            if full[gi] is None:
                                ref = next((full[j] for j in range(gi - 1, -1, -1) if full[j]), None)
                                if ref:
                                    full[gi] = [ref[0], ref[3] + gapv, ref[2], ref[3] + gapv + hbar]
                                else:
                                    nxt = next(full[j] for j in range(gi + 1, len(u)) if full[j])
                                    full[gi] = [nxt[0], nxt[1] - gapv - hbar, nxt[2], nxt[1] - gapv]
                        pieces, est = full, True
                    ids = []
                    for k, (pc, g) in enumerate(zip(pieces, u)):
                        ch = {"text": g, "rec": "", "conf": None, "bbox": [round(v, 2) for v in pc],
                              "engine": "facsimile", "role": "figure_text", "region": reg["id"], "block": b["id"],
                              "figure_label": li, "order": k, "direction": "ttb",
                              "text_source": "figure-label-list" + ("+manual" if sc == "manual" else ""),
                              "modern": lab.get("modern")}
                        if est and not any(abs(pc[1] - x[1]) < 1 for x in c["components"]):
                            ch["bbox_estimated"] = True
                        r["chars"].append(ch)
                        ids.append(len(r["chars"]) - 1)
                    bx = [min(p[0] for p in pieces), min(p[1] for p in pieces),
                          max(p[2] for p in pieces), max(p[3] for p in pieces)]
                    mk = None
                    best = 1e9
                    for sy in b.get("symbols", []):
                        if sy["type"] != "marker":
                            continue
                        m = sy["bbox"]
                        mx, my = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
                        dx = max(bx[0] - mx, 0, mx - bx[2]); dy = max(bx[1] - my, 0, my - bx[3])
                        d = np.hypot(dx, dy)
                        if d < best:
                            best, mk = d, m
                    entry.update(bbox=[round(v, 2) for v in bx],
                                 score=sc if sc == "manual" else round(float(sc), 3),
                                 verification="manual" if sc == "manual" else "auto",
                                 rec="".join(x["rec"] for x in c["rec"]), candidate=ci,
                                 marker_bbox=mk if best < (bx[3] - bx[1]) else None)
                b["labels"].append(entry)
            b["unmatched_candidates"] = [c["id"] for i, c in enumerate(cands)
                                         if not (set(c["parts"]) & used_parts)]


def _build_paragraphs(fac, sec, q):
    """원문 문단 ↔ 현대어역 문단 ↔ 영인 글자 박스 연결."""
    if not sec:
        return []
    out = []
    for i, p in enumerate(sec["original"]):
        mods = [sec["modern"][j] for (a, j) in sec["pairs"] if a == i]
        cs = [c for c in fac if c.get("para") == i]
        cols = {}
        for c in cs:
            b = cols.setdefault(c["column"], list(c["bbox"]))
            cols[c["column"]] = [min(b[0], c["bbox"][0]), min(b[1], c["bbox"][1]),
                                 max(b[2], c["bbox"][2]), max(b[3], c["bbox"][3])]
        out.append({"para": i, "header": p["header"], "original": p["text"],
                    "modern": " ".join(m["text"] for m in mods) or None,
                    "transcript_page": q, "n_chars": len(cs),
                    "char_ids": [],  # finalize 에서 채움
                    "facsimile_boxes": [[round(v, 2) for v in b] for _, b in sorted(cols.items())]})
    return out


def finalize(results, dpi=None):
    for r in results:
        s = r["render_dpi"] / 72.0
        for i, c in enumerate(r["chars"]):
            c["id"] = f"p{r['page']:04d}_c{i:05d}"
            c["kind"] = char_kind(c["text"]) if c["text"] else "unknown"
            c["bbox_px"] = [int(round(v * s)) for v in c["bbox"]]
            if c["engine"] == "facsimile":
                reg = next((g for g in r.get("regions", []) if g["id"] == c["region"]), None)
                si = (reg or {}).get("source_image")
                if si:
                    bx = si["bbox_pt"]
                    sx = si["width"] / (bx[2] - bx[0])
                    sy = si["height"] / (bx[3] - bx[1])
                    c["bbox_img"] = [int(round((c["bbox"][0] - bx[0]) * sx)), int(round((c["bbox"][1] - bx[1]) * sy)),
                                     int(round((c["bbox"][2] - bx[0]) * sx)), int(round((c["bbox"][3] - bx[1]) * sy))]
            for k in [k for k, v in c.items() if v is None]:
                del c[k]
        for pg in r.get("paragraphs", []):
            pg["char_ids"] = [c["id"] for c in r["chars"] if c["engine"] == "facsimile"
                              and not c.get("panshim") and c.get("para") == pg["para"]]
    return results


def save_page(res, outdir, doc=None, overlay_dpi=110):
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"page_{res['page']:04d}.json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    if doc is not None:
        from .visualize import overlay
        overlay(doc[res["page"] - 1], res, os.path.join(outdir, f"page_{res['page']:04d}_overlay.jpg"), overlay_dpi)
    return fn
