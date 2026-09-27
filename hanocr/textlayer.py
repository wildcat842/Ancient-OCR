"""PDF 텍스트 레이어(born-digital)에서 글자 단위 bbox 추출.

- PyMuPDF rawdict의 글자별 좌표 사용 (PDF 포인트, 좌상단 원점)
- 한양 PUA 옛한글 → 첫가끝 자모로 변환, 조합형 자모열은 하나의 글자마디로 병합(bbox 합집합)
- (선택) 렌더링 이미지에서 잉크 영역만 남긴 tight bbox 계산
"""
import numpy as np
import pymupdf as fitz

from .normalize import pua_to_jamo, jamo_class, attaches, nfc, char_kind, is_pua


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def extract_chars(page, keep_space=False):
    """페이지 텍스트 레이어의 글자마디 목록 반환."""
    rd = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)
    chars = []
    line_id = 0
    for b in rd["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            ldir = l.get("dir", (1, 0))
            direction = "ttb" if abs(ldir[1]) > abs(ldir[0]) else "ltr"
            cur, prev_cls = None, None
            for s in l["spans"]:
                for c in s["chars"]:
                    raw = c["c"]
                    txt = pua_to_jamo(raw) if is_pua(raw) else raw
                    bbox = list(c["bbox"])
                    cls = jamo_class(txt[0]) if txt else None
                    if cur is not None and cls and attaches(prev_cls, cls) and not is_pua(raw):
                        cur["text"] += txt
                        # 조합용 자모는 폭 0 글리프인 경우가 많음 → 유효 bbox만 합침
                        if bbox[2] - bbox[0] > 0.1:
                            cur["bbox"] = _union(cur["bbox"], bbox)
                        cur["raw"] += raw
                    else:
                        cur = {"text": txt, "raw": raw, "bbox": bbox, "font": s["font"], "ldir": [round(v, 2) for v in ldir],
                               "size": round(s["size"], 2), "color": "#%06x" % s["color"],
                               "block": b["number"], "line": line_id, "direction": direction}
                        chars.append(cur)
                    last = jamo_class(txt[-1]) if txt else None
                    prev_cls = last if last != "M" else prev_cls
            line_id += 1
    out = []
    for c in chars:
        c["text"] = nfc(c["text"])
        c["kind"] = char_kind(c["text"])
        if c["kind"] == "space" and not keep_space:
            continue
        if c["bbox"][2] - c["bbox"][0] < 0.2 * c["size"]:
            # 폭 0 으로 기록된 글리프(PUA 옛한글 등) → 글꼴 크기로 폭 보정
            c["bbox"][2] = c["bbox"][0] + 0.95 * c["size"]
            c["bbox_estimated"] = True
        if c["bbox"][3] <= c["bbox"][1]:
            continue
        out.append(c)
    # 세로쓰기 판정: 한 줄에 한 글자씩 세로로 쌓인 경우 (김씨부인 판독문 등)
    _mark_vertical_stacks(out)
    return out


def _mark_vertical_stacks(chars):
    by_block = {}
    for c in chars:
        by_block.setdefault(c["block"], []).append(c)
    for cs in by_block.values():
        if len(cs) < 4:
            continue
        lines = {}
        for c in cs:
            lines.setdefault(c["line"], []).append(c)
        if np.mean([len(v) for v in lines.values()]) <= 1.3:
            xs = np.array([(c["bbox"][0] + c["bbox"][2]) / 2 for c in cs])
            ys = np.array([(c["bbox"][1] + c["bbox"][3]) / 2 for c in cs])
            if np.ptp(ys) > 2 * np.ptp(xs) or len(set(np.round(xs))) < len(cs) / 3:
                for c in cs:
                    c["direction"] = "ttb"


def tighten(chars, page, dpi=300, thr=160):
    """렌더링 이미지에서 각 글자 bbox 내부의 잉크 픽셀 경계로 bbox를 좁힘 (glyph box)."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, annots=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    s = dpi / 72.0
    ox, oy = page.rect.x0, page.rect.y0
    for c in chars:
        x0, y0, x1, y1 = c["bbox"]
        X0, Y0 = max(int((x0 - ox) * s), 0), max(int((y0 - oy) * s), 0)
        X1, Y1 = min(int(np.ceil((x1 - ox) * s)), pix.width), min(int(np.ceil((y1 - oy) * s)), pix.height)
        if X1 <= X0 or Y1 <= Y0:
            continue
        crop = img[Y0:Y1, X0:X1]
        # 배경 밝기 대비 상대 임계 (컬러 글자 대응)
        bg = np.percentile(crop, 95)
        ink = crop < min(thr, bg - 40)
        ys, xs = np.nonzero(ink)
        if len(xs) == 0:
            c["ink_bbox"] = None
            continue
        c["ink_bbox"] = [float(x) for x in [round(ox + (X0 + xs.min()) / s, 2), round(oy + (Y0 + ys.min()) / s, 2),
                         round(ox + (X0 + xs.max() + 1) / s, 2), round(oy + (Y0 + ys.max() + 1) / s, 2)]]
    return chars
