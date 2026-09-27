"""사람이 지정한 도판 주석(검수 보정 파일의 manual_figure)을 라벨 데이터로 만든다.

회전 인쇄된 지도(예: 사민필지 지도 2)처럼 자동 검출이 어려운 도판에 쓴다.
- 영인을 upright(글·지도가 바로 선 방향)로 회전해 저장하고, 좌표는 그 이미지 기준
- 사람이 준 대략의 글자 상자는 글자 획 마스크(곡선·경위선 제외)로 자동으로 조임
- 표기 텍스트는 판독문의 패널별 표기 목록과 대조, 현대어는 해설의 원문(현대어) 쌍 → 목록 순서
- 경위도 눈금 숫자 등 목록 밖 작은 글씨는 GRATICULE_TEXT 영역으로 자동 검출(텍스트 미판독)
"""
import json
import os

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image

from .facsimile import binarize
from .figure import _classify
from .labels import LABELS
from .sections import find_sections
from .textlayer import extract_chars
from .export import _overlay
from .normalize import clusters


def _u(bs):
    bs = [b for b in bs if b]
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _fit_circles(gray, n=2):
    small = cv2.resize(gray, None, fx=0.25, fy=0.25)
    small = cv2.medianBlur(small, 5)
    h, w = small.shape
    cs = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w * 0.3, param1=120, param2=40,
                          minRadius=int(min(h, w) * 0.3), maxRadius=int(min(h, w) * 0.5))
    if cs is None:
        return []
    cs = sorted(cs[0][:n], key=lambda c: c[0])
    return [(float(x) * 4, float(y) * 4, float(r) * 4) for x, y, r in cs]


def _panel_lists(sec):
    """판독문·현대어역에서 패널 제목별 표기 목록(줄 단위 토큰화)과 범례를 모음."""
    from .sections import split_label_tokens

    def collect(paras):
        out, cur = [], None
        for p in paras:
            r = p.get("role")
            if r == "figure_panel_title":
                cur = {"title": p["text"].strip(), "labels": [], "legend": []}
                out.append(cur)
            elif cur is not None and r == "figure_labels":
                lines = [l["text"] for l in p.get("lines", [])] or [p["text"]]
                for t in lines:
                    cur["labels"] += split_label_tokens(t.strip())
            elif cur is not None and r == "figure_legend":
                cur["legend"] += [x.strip() for x in p["text"].split("●") if x.strip()]
        return out
    o, m = collect(sec["original"]), collect(sec["modern"])
    res = {}
    for i, po in enumerate(o):
        pm = m[i] if i < len(m) else {"title": None, "labels": [], "legend": []}
        res[po["title"]] = {"labels": po["labels"], "modern_title": pm["title"], "modern_labels": pm["labels"],
                            "legend": po["legend"], "modern_legend": pm["legend"]}
    return res


def _split_line(ccs, line, w, k):
    """중심선 [xa,ya,xb,yb] 띠 안의 글자 성분을 선 방향으로 투영해 k 글자로 나눔 (회전·호 배치 대응)."""
    xa, ya, xb, yb = line
    d = np.array([xb - xa, yb - ya], float)
    L = max(np.hypot(*d), 1.0)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    sel = []
    for b in ccs:
        c = np.array([(b[0] + b[2]) / 2 - xa, (b[1] + b[3]) / 2 - ya])
        t, q = c @ u, c @ nrm
        if -w / 2 <= t <= L + w / 2 and abs(q) <= w / 2:
            corners = np.array([[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]], float) - [xa, ya]
            ts = corners @ u
            sel.append((ts.min(), ts.max(), b))
    if not sel:
        sel = []
    sel.sort(key=lambda x: x[0])
    if len(sel) == 0 or len(_merge_blobs(sel)) < k:
        return _equal_split(sel, xa, ya, u, nrm, L, w, k)
    blobs = []
    for t0, t1, b in sel:
        if blobs and t0 <= blobs[-1][1] + 1:
            blobs[-1][1] = max(blobs[-1][1], t1); blobs[-1][2].append(b)
        else:
            blobs.append([t0, t1, [b]])
    n = len(blobs)
    if n <= k:
        groups = [[bl] for bl in blobs]
    else:
        m = (blobs[-1][1] - blobs[0][0]) / k
        INF = 1e18
        D = np.full((n + 1, k + 1), INF); B = np.zeros((n + 1, k + 1), int); D[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for a in range(j - 1, i):
                    if D[a, j - 1] >= INF:
                        continue
                    span = blobs[i - 1][1] - blobs[a][0]
                    gap = (blobs[a][0] - blobs[a - 1][1]) if a > 0 else 0
                    c = ((span - 0.8 * m) / m) ** 2 * 0.5 - 1.5 * gap / m
                    if D[a, j - 1] + c < D[i, j]:
                        D[i, j] = D[a, j - 1] + c; B[i, j] = a
        groups, i, j = [], n, k
        while j > 0:
            a = B[i, j]; groups.append(blobs[a:i]); i, j = a, j - 1
        groups = groups[::-1]
    out = []
    for g in groups:
        bs = [b for bl in g for b in bl[2]]
        out.append([min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)])
    return out


def _cc_features(lab, st, n):
    feats = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 6:
            continue
        pts = np.column_stack(np.nonzero(lab[y:y + h, x:x + w] == i))[:, ::-1].astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        L, T = max(rw, rh), max(1.0, min(rw, rh))
        diag = min(abs(ang % 90), 90 - abs(ang % 90)) > 18
        feats.append(([int(x), int(y), int(x + w), int(y + h)], int(a), float(L), float(T), diag))
    return feats


def _merge_blobs(sel):
    blobs = []
    for t0, t1, b in sel:
        if blobs and t0 <= blobs[-1][1] + 1:
            blobs[-1][1] = max(blobs[-1][1], t1); blobs[-1][2].append(b)
        else:
            blobs.append([t0, t1, [b]])
    return blobs


def _equal_split(sel, xa, ya, u, nrm, L, w, k):
    """성분이 붙어 있거나(글자 수보다 덩어리가 적음) 선에 가려 획이 빠진 경우:
    중심선 길이를 k 등분하고, 각 구간의 성분 합집합(없으면 기하 추정 상자)을 글자 상자로."""
    a = np.array([xa, ya], float)
    lo = min([t0 for t0, _, _ in sel] + [-0.4 * w]) if sel else -0.4 * w
    hi = max([t1 for _, t1, _ in sel] + [L + 0.4 * w]) if sel else L + 0.4 * w
    seg = (hi - lo) / k
    out = []
    for i in range(k):
        s0, s1 = lo + i * seg, lo + (i + 1) * seg
        bs = [b for t0, t1, b in sel if s0 <= (t0 + t1) / 2 < s1]
        if bs:
            out.append([min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)])
        else:  # 기하 추정
            c = a + u * (s0 + s1) / 2
            half = np.abs(u) * seg / 2 + np.abs(nrm) * w / 2 * 0.8
            out.append([int(c[0] - half[0]), int(c[1] - half[1]), int(c[0] + half[0]), int(c[1] + half[1]), "est"])
    return out


def _color_symbols(rgb, ch):
    """색 점 기호: 파란 점(도성·수도), 갈색·붉은 점(읍·도시)."""
    a = rgb.astype(int)
    R, G, Bc = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    V = a.max(2)
    masks = {"capital_marker": ((Bc - R) > 25) & ((Bc - G) > 5) & (V < 200),
             "city_marker": ((R - Bc) > 35) & (V < 150) & ((R - G) > 20)}
    out = []
    for typ, m in masks.items():
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y, w, h, ar = st[i]
            if not (0.15 * ch < max(w, h) < 0.8 * ch and 0.6 < w / float(h) < 1.6 and ar > 0.55 * w * h):
                continue
            out.append({"type": typ, "bbox": [int(x), int(y), int(x + w), int(y + h)]})
    return out


def build(doc, pno, spec, outdir, region_bbox_pt=None, stem_name=None):
    os.makedirs(outdir, exist_ok=True)
    page = doc[pno]
    info = page.get_image_info(xrefs=True)
    bb = region_bbox_pt or max(info, key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))["bbox"]
    dpi = spec.get("dpi", 300)
    pix = page.get_pixmap(clip=fitz.Rect(*bb), dpi=dpi)
    disp = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    rot = spec.get("rotate_cw", 0)
    up = disp.rotate(-rot, expand=True) if rot else disp
    stem = os.path.join(outdir, stem_name or f"page_{pno + 1:04d}_R0")
    up.save(stem + ".png")
    Wd, Hd = disp.size
    kx, ky = Wd / (bb[2] - bb[0]), Hd / (bb[3] - bb[1])

    def to_pt(b):  # upright px → page pt
        pts = []
        for (u, v) in ((b[0], b[1]), (b[2], b[3])):
            if rot == 90:
                x, y = v, Hd - u
            elif rot == 270:
                x, y = Wd - v, u
            elif rot == 180:
                x, y = Wd - u, Hd - v
            else:
                x, y = u, v
            pts.append((bb[0] + x / kx, bb[1] + y / ky))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]

    gray = cv2.cvtColor(np.array(up), cv2.COLOR_RGB2GRAY)
    ink = binarize(gray)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = 2 * np.percentile(dist[dist > 0], 75)
    # 글자 크기: 사람이 준 상자들의 중앙값
    sizes = [max(c[2] - c[0], c[3] - c[1]) for l in spec["labels"] for c in l.get("chars", []) if c] + \
            [l.get("w", 60) * 0.9 for l in spec["labels"] if "line" in l]
    ch = float(np.median(sizes)) * 0.9 if sizes else 50
    # 색 선·색 점(국경선·도시 기호)은 글자가 아님 → 채도 높은 픽셀을 잉크에서 제외
    hsv = cv2.cvtColor(np.array(up), cv2.COLOR_RGB2HSV)
    Hh, Ss, Vv = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    red = ((Hh < 12) | (Hh > 165)) & (Ss > 100) & (Vv > 130)      # 붉은 국경선(글자는 갈색·어두움)
    blue = (Hh > 95) & (Hh < 135) & (Ss > 60)                        # 파란 도성 기호
    ink = (ink & ~(red | blue)).astype(np.uint8)
    n, lab, st, cen, cls = _classify(ink, ch, stroke)
    textmask = ((cls[lab] == 1) | (cls[lab] == 4)).astype(np.uint8)
    feats = _cc_features(lab, st, n)

    def text_ccs_for(chl):
        """표기마다 글자 크기(chl)에 맞춰 글자 성분을 다시 고름: 큰 글자는 선으로, 작은 획·빗금은 잡티로 오분류되지 않게."""
        out = []
        for (b, a, L_, T_, diag) in feats:
            w_, h_ = b[2] - b[0], b[3] - b[1]
            if a < 6 or max(w_, h_) < 0.12 * chl:
                continue
            if L_ > 1.5 * chl or max(w_, h_) > 2.3 * chl:
                continue
            if L_ > 0.8 * chl and T_ < 2.2 * stroke and diag:
                continue
            if T_ < 1.6 * stroke and L_ / T_ > 3.5 and L_ < 0.55 * chl:  # 해안 빗금
                continue
            out.append(b)
        return out
    text_ccs = text_ccs_for(ch)

    def tighten_cc(box):
        """사람이 준 글자 상자 안에 70% 이상 들어오는 잉크 성분(선·잡티 제외)의 합집합으로 조임."""
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        sel = []
        for (b, a, L_, T_, diag) in feats:
            ix = max(0, min(b[2], x1) - max(b[0], x0)); iy = max(0, min(b[3], y1) - max(b[1], y0))
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area == 0 or ix * iy < 0.7 * area:
                continue
            if max(b[2] - b[0], b[3] - b[1]) < 0.12 * min(bw, bh):
                continue
            if T_ < 1.6 * stroke and L_ / T_ > 3.5 and L_ < 0.3 * min(bw, bh):
                continue
            sel.append(b)
        if not sel:
            return [int(v) for v in box], False
        u = _u(sel)
        if (u[2] - u[0]) * (u[3] - u[1]) < 0.12 * bw * bh:
            return [int(v) for v in box], False
        return u, True

    def tighten(box, mask=textmask, grow=0):
        x0, y0, x1, y1 = [int(v) for v in box]
        sub = mask[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        if len(xs) < 8:
            return [x0, y0, x1, y1], False
        t = [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]
        # 글자 획이 경위선·해안선과 붙어 선으로 분류되면 마스크가 일부만 남음 → 사람이 준 상자 유지
        if mask is textmask and (t[2] - t[0]) * (t[3] - t[1]) < 0.3 * (x1 - x0) * (y1 - y0):
            return [x0, y0, x1, y1], False
        return t, True

    # 판독문 구조
    sec = find_sections(extract_chars(page, keep_space=True), page.rect.width, page.rect.height)
    gloss = sec.get("gloss") or {}
    fig_caption = next((p["text"] for p in sec["original"] if p.get("role") == "figure_caption"), None)
    fig_caption_m = next((sec["modern"][j]["text"] for (i, j) in sec["pairs"]
                          if sec["original"][i].get("role") == "figure_caption"), None)
    panel_lists = _panel_lists(sec)
    com = (sec.get("commentary") or {}).get("text")
    ftype = next((k for k in ("천체도", "세계지도", "반구도", "지도") if k in (com or "")), None)

    regions, items, used = [], [], []

    def add(label, bbox, parent=None, **attrs):
        rid = f"R0.{label.lower()}_{sum(1 for x in regions if x['label'] == label)}"
        rg = {"id": rid, "label": label, "label_ko": LABELS[label]["ko"], "bbox": [int(v) for v in bbox],
              "bbox_pt": to_pt(bbox), "parent": parent}
        rg.update({k: v for k, v in attrs.items() if v is not None})
        regions.append(rg)
        return rid

    def add_char(text, bbox, rid, label, **attrs):
        it = {"id": f"p{pno + 1:04d}_m{len(items):05d}", "reading_order": len(items), "label": label,
              "region_id": rid, "text": text, "bbox": [int(v) for v in bbox] if bbox else None,
              "bbox_pt": to_pt(bbox) if bbox else None}
        it.update({k: v for k, v in attrs.items() if v is not None})
        items.append(it)
        return it

    H, W = gray.shape

    def add_title(t, parent, modern):
        fb, _ = tighten(t["frame"], ink)
        tid = add("FIGURE_TITLE", fb, parent, text=t["text"], modern=gloss.get(t["text"]) or modern,
                  writing_direction="vertical_ttb", note="제목 카르투시(테두리 상자)")
        units = [u for u in clusters(t["text"]) if u.strip()]
        cbs = [tighten(b)[0] for b in t["chars"]] if "chars" in t else \
            [b[:4] for b in _split_line(text_ccs, t["line"], t["w"], len(units))]
        for g, b in zip(units, cbs):
            add_char(g, b, tid, "FIGURE_CHAR", text_source="manual", bbox_tightened=True)
            used.append(b)
        return tid

    fig = add("FIGURE", [0, 0, W, H], None, subtype="MAP", figure_id=fig_caption, figure_id_modern=fig_caption_m,
              figure_type=ftype, description=com, image_orientation=f"rotated {rot}° clockwise from PDF page (north up)")
    circles = _fit_circles(gray) if spec.get("panels") else []
    pid = {}
    for k, pn in enumerate(spec.get("panels", [])):
        # 패널 원: 좌/우 위치로 매칭
        c = None
        if circles:
            c = circles[0] if pn.get("side") == "left" else circles[-1]
        pb = [c[0] - c[2], c[1] - c[2], c[0] + c[2], c[1] + c[2]] if c else [0, 0, W, H]
        pb = [max(0, pb[0]), max(0, pb[1]), min(W, pb[2]), min(H, pb[3])]
        lst = panel_lists.get(pn["transcript_title"], {})
        t = pn["title"]
        pid[pn["id"]] = add("FIGURE_PANEL", pb, fig, panel=pn["id"], panel_title=t["text"],
                            panel_title_modern=gloss.get(t["text"]) or lst.get("modern_title"),
                            circle=[round(v, 1) for v in c] if c else None, transcript_label_list=lst.get("labels"))
        add_title(t, pid[pn["id"]], gloss.get(t["text"]) or (lst.get("modern_title") or "").strip("[]") or None)
    # 단일 도판(패널 없음): 도판 제목 카르투시
    single = None
    if spec.get("title"):
        t = spec["title"]
        single = panel_lists.get(t["transcript_title"], {})
        pid[None] = fig
        add_title(t, fig, (single.get("modern_title") or "").strip("[]") or None)
        regions[0]["transcript_label_list"] = single.get("labels")
    # 범례
    if spec.get("legend"):
        lg = spec["legend"]
        lst = single or {}
        legs, mlegs = lst.get("legend", []), lst.get("modern_legend", [])
        items_b = [it["symbol"] for it in lg["items"]]
        lgid = add("FIGURE_LEGEND", lg.get("box") or _u(items_b), fig)
        legend_ids = {}
        for n_, it in enumerate(lg["items"]):
            text = legs[n_] if n_ < len(legs) else it.get("text")
            units = [u for u in clusters(text) if u.strip()]
            cbs = [b[:4] for b in _split_line(text_ccs, it["line"], it["w"], len(units))]
            li = add("LEGEND_ITEM", _u(cbs + [it["symbol"]]), lgid, text=text,
                     modern=mlegs[n_] if n_ < len(mlegs) else None, symbol_type=it["symbol_type"],
                     symbol_bbox=it["symbol"], writing_direction="vertical_ttb")
            legend_ids[it["symbol_type"]] = li
            add("FIGURE_SYMBOL", it["symbol"], li, symbol_type=it["symbol_type"], role="legend_key")
            for g, b in zip(units, cbs):
                add_char(g, b, li, "FIGURE_CHAR", text_source="transcript-legend+manual", bbox_tightened=True)
                used.append(b)
            used.append(it["symbol"])
    else:
        legend_ids = {}
    # 축척
    if spec.get("scale"):
        sc_ = spec["scale"]
        cbs = [tighten(b)[0] for b in sc_["chars"]]
        sid = add("FIGURE_SCALE", _u(cbs + [sc_["bar"]]), fig, text=sc_["text"], verification="manual-reading",
                  uncertain=sc_.get("uncertain"), note=sc_.get("note"), bar_bbox=sc_["bar"])
        for g, b in zip([u for u in clusters(sc_["text"]) if u.strip()], cbs):
            add_char(g, b, sid, "FIGURE_CHAR", text_source="manual-reading", bbox_tightened=True)
            used.append(b)
        used.append(sc_["bar"])
    for L in spec["labels"]:
        pnl = L.get("panel")
        lst = single if pnl is None else panel_lists.get(
            next(p["transcript_title"] for p in spec["panels"] if p["id"] == pnl), {})
        labs = lst.get("labels", [])
        in_list = L.get("in_list", L["text"] in labs)
        modern, msrc = L.get("modern"), L.get("modern_source")
        if in_list and modern is None:
            if L["text"] in gloss:
                modern, msrc = gloss[L["text"]], "commentary_gloss"
            elif len(lst.get("modern_labels", [])) == len(labs):
                modern, msrc = lst["modern_labels"][labs.index(L["text"])], "label_list_order"
        glyphs = [u for u in clusters(L["text"]) if u.strip()]
        est_flags = []
        if "line" in L:
            cbs = _split_line(text_ccs_for(L.get("ch", L["w"] * 0.85)), L["line"], L["w"], len(glyphs))
            est_flags = [len(b) == 5 for b in cbs]
            cbs = [b[:4] for b in cbs]
            boxes = [(b, True) for b in cbs] + [(None, False)] * (len(glyphs) - len(cbs))
        else:
            boxes = [(None, False) if chb is None else tighten_cc(chb) for chb in L["chars"]]
        vis = [b for b, _ in boxes if b]
        if not vis:
            continue
        ub = _u(vis)
        ent = L.get("entity_type") or ("CELESTIAL_BODY" if ftype == "천체도" else "PLACE_NAME")
        lid = add("FIGURE_LABEL", ub, pid.get(pnl, fig), text=L["text"], modern=modern, modern_source=msrc,
                  entity_type=ent, writing_direction=L["dir"], in_transcript_list=in_list,
                  verification="manual", uncertain=L.get("uncertain"), note=L.get("note") or L.get("glyph_note"),
                  n_chars_visible=len(vis), n_chars=len(glyphs), centerline=L.get("line"))
        for gi, (g, (b, ok)) in enumerate(zip(glyphs, boxes)):
            est = est_flags[gi] if gi < len(est_flags) else False
            add_char(g, b, lid, "FIGURE_CHAR", modern=modern,
                     text_source="manual" if not in_list else "figure-label-list+manual",
                     bbox_tightened=ok and not est, bbox_estimated=est or None,
                     visibility="visible" if b else "occluded")
            if b:
                used.append(b)
    # 지도 위 색 점 기호 (범례와 연결)
    for sy in _color_symbols(np.array(up), ch):
        if any(min(sy["bbox"][2], b[2]) > max(sy["bbox"][0], b[0]) and min(sy["bbox"][3], b[3]) > max(sy["bbox"][1], b[1])
               for b in used):
            continue
        add("FIGURE_SYMBOL", sy["bbox"], fig, symbol_type=sy["type"], legend_item=legend_ids.get(sy["type"]),
            verification="auto")
    # 목록에 있으나 주석이 없는 표기 보고
    missing = {}
    groups = [(p["id"], p["transcript_title"]) for p in spec.get("panels", [])]
    if spec.get("title"):
        groups.append((None, spec["title"]["transcript_title"]))
    for gid, tt in groups:
        labs = panel_lists.get(tt, {}).get("labels", [])
        done = [L["text"] for L in spec["labels"] if L.get("panel") == gid]
        miss = [x for x in labs if x not in done]
        if miss:
            missing[str(gid)] = miss
    # 경위도 눈금 등 작은 글씨 (자동, 미판독)
    small = ((cls[lab] == 1) | (cls[lab] == 4)).astype(np.uint8)
    for b in used:
        small[b[1]:b[3], b[0]:b[2]] = 0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.5 * ch), int(0.5 * ch)))
    ng, gl, gs, _ = cv2.connectedComponentsWithStats(cv2.dilate(small, k), 8)
    n_grat = 0
    for i in range(1, ng):
        x, y, w, h, a = gs[i]
        inner = small[y:y + h, x:x + w].sum()
        if inner < 0.02 * ch * ch or max(w, h) < 0.35 * ch or max(w, h) > 4 * ch:
            continue
        cx, cy = x + w / 2, y + h / 2
        # 눈금 표기는 적도선 부근 또는 테두리 원 둘레에만 있음 → 그 밖(섬·해안선 조각)은 제외
        near = False
        for c in circles:
            dd = np.hypot(cx - c[0], cy - c[1])
            if dd < c[2] * 1.03 and (abs(cy - c[1]) < 0.06 * c[2] or dd > 0.9 * c[2]):
                near = True
        if circles and not near:
            continue
        if not circles and spec.get("graticule_bands"):
            if not any(bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3] for bx in spec["graticule_bands"]):
                continue
        par = next((pid[p["id"]] for p, c in zip(spec.get("panels", []), [circles[0] if p.get("side") == "left" else circles[-1]
                    for p in spec.get("panels", [])] if circles else [None] * 2)
                    if c and np.hypot(cx - c[0], cy - c[1]) < c[2] * 1.02), fig)
        add("GRATICULE_TEXT", [x, y, x + w, y + h], par, verification="auto", text=None,
            note="경위도 눈금 숫자 등 작은 표기(자동 검출, 미판독)")
        n_grat += 1
    ann = {"source_pdf_page": pno + 1, "region": "R0", "image": os.path.basename(stem + ".png"),
           "image_size": [W, H], "coordinate": "upright image pixel [x0,y0,x1,y1], top-left origin",
           "image_orientation": f"PDF 쪽에서 시계방향 {rot}° 회전(북쪽이 위)", "label_schema": "LABELS.md",
           "annotation_method": "사람 지정 글자 상자(검수 보정 파일) + 잉크 기반 자동 조임, 텍스트·현대어는 판독문·해설에서",
           "transcript_labels_not_annotated": missing, "n_graticule_regions": n_grat,
           "regions": regions, "chars": items, "paragraphs": []}
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=1)
    import csv
    with open(stem + ".csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reading_order", "id", "label", "region_id", "text", "modern", "x0", "y0", "x1", "y1", "visibility"])
        for it in items:
            w.writerow([it["reading_order"], it["id"], it["label"], it["region_id"], it["text"], it.get("modern") or "",
                        *(it["bbox"] or ["", "", "", ""]), it.get("visibility", "visible")])
    with open(stem + "_regions.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "label_ko", "parent", "x0", "y0", "x1", "y1", "text", "modern"])
        for rg in regions:
            w.writerow([rg["id"], rg["label"], rg["label_ko"], rg["parent"] or "", *rg["bbox"], rg.get("text", "") or "",
                        rg.get("modern", "") or ""])
    _overlay(stem, [dict(it, bbox=it["bbox"]) for it in items if it["bbox"]], regions)
    return ann
