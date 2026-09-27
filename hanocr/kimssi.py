"""『김씨부인 한글상언·정조어필 한글편지첩·곤전어필』(소장자료총서 1) 전용 처리.

자료 유형별 처리
 - DOCUMENT (1부 상언): 판독문(세로쓰기)이 원본 행 배열을 그대로 따름 → 행 단위 정렬, 행 번호 01~40
   주석문·현대어역도 행 번호가 매겨져 있어 행마다 원문/주석문/현대어역을 연결
 - LETTER (2부 편지): 판독문이 원본 행 배열을 따름 + 봉투(피봉) 글씨는 90° 회전, 삽입 글자·삭제 표시
   이미지 아래 캡션 '<편지 3> 1755(?)~1759.2. 이전' → figure_id / date
 - BOOK_PAGE (3부 곤전어필): 쪽마다 여러 면(1a, 1b …)의 사진, 판독문은 줄바꿈이 원본과 다름 → 면 단위 정렬
"""
import json
import os
import re

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image

from .align import align, transcript_units
from .facsimile import detect_chars, binarize, remove_lines
from .figure_manual import _split_line, _u, _cc_features
from .labels import LABELS
from .normalize import clusters, char_kind
from .recognize import recognize
from .textlayer import extract_chars
from .export import _overlay

CAPTION_PAT = re.compile(r"<\s*([^>]+?)\s*>\s*(.*)")


# ---------------------------------------------------------------- 텍스트 레이어
def page_lines(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            t = "".join(s["text"] for s in l["spans"])
            if t.strip():
                out.append({"bbox": list(l["bbox"]), "text": t, "dir": l["dir"]})
    return out


def captions(page):
    """이미지 아래 캡션: <편지 3> 1755(?)~1759.2. 이전 / <1a>"""
    caps = []
    for l in page_lines(page):
        m = CAPTION_PAT.match(l["text"].strip())
        if m and len(l["text"]) < 60:
            caps.append({"bbox": l["bbox"], "text": l["text"].strip(), "id": m.group(1).strip(),
                         "rest": m.group(2).strip() or None})
    return caps


def vertical_transcript(page, clip, col_tol=6.0):
    """세로쓰기 판독문 → 행(列) 목록 (오른쪽→왼쪽). 회전 글자(봉투)와 행 사이 삽입 글자를 분리."""
    x0, y0, x1, y1 = clip
    allc = extract_chars(page)
    per_line = {}
    for c in allc:
        per_line[c["line"]] = per_line.get(c["line"], 0) + 1
    # 세로쓰기 판독문은 글자마다 한 줄(line) → 여러 글자짜리 가로 줄(주석·각주 등)은 제외
    chars = [c for c in allc if x0 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= x1
             and y0 <= (c["bbox"][1] + c["bbox"][3]) / 2 <= y1 and c["kind"] != "digit" and per_line[c["line"]] <= 2]
    if not chars:
        return [], [], []
    size = np.median([c["bbox"][3] - c["bbox"][1] for c in chars])
    rotated = [c for c in chars if abs(c.get("ldir", [1, 0])[0] + 1) < 0.1]   # 방향 (-1,0): 90° 회전 글자
    rest = [c for c in chars if c not in rotated]
    cols = []
    for c in sorted(rest, key=lambda c: -(c["bbox"][0] + c["bbox"][2]) / 2):
        cx = (c["bbox"][0] + c["bbox"][2]) / 2
        if cols and abs(cols[-1]["x"] - cx) < col_tol:
            cols[-1]["chars"].append(c)
            cols[-1]["x"] = np.mean([(q["bbox"][0] + q["bbox"][2]) / 2 for q in cols[-1]["chars"]])
        else:
            cols.append({"x": cx, "chars": [c]})
    # 행 사이 한 글자 = 삽입 글자 → 가까운 행에 y 순서로 넣고 표시
    pitch = np.median(np.diff(sorted([c["x"] for c in cols if len(c["chars"]) > 1]))) if len(cols) > 2 else 20
    main, inserted = [], []
    for i, c in enumerate(cols):
        if len(c["chars"]) == 1 and 0 < i < len(cols) - 1 and \
                min(abs(c["x"] - cols[i - 1]["x"]), abs(c["x"] - cols[i + 1]["x"])) < 0.75 * abs(pitch):
            inserted.append(c["chars"][0])
        else:
            main.append(c)
    for q in inserted:
        cx = (q["bbox"][0] + q["bbox"][2]) / 2
        host = min(main, key=lambda c: abs(c["x"] - cx))
        q["inserted"] = True
        host["chars"].append(q)
    out = []
    for c in main:
        cs = sorted(c["chars"], key=lambda q: q["bbox"][1])
        units = []
        for q in cs:
            units.append({"text": q["text"], "bbox_pt": [round(v, 2) for v in q["bbox"]],
                          "inserted": q.get("inserted", False), "deleted": q["text"] in ("∎", "■"),
                          "repeat_mark": q["text"] in ("〃", "″")})
        top = min(q["bbox"][1] for q in cs)
        out.append({"x": float(c["x"]), "top": top, "units": units})
    # 대두(擡頭): 다른 행보다 위로 올려 쓴 행
    tops = np.array([c["top"] for c in out])
    base = np.median(tops)
    for c in out:
        c["elevated"] = bool(c["top"] < base - 0.4 * size)
    rot = sorted(rotated, key=lambda q: q["bbox"][0])
    return out, rot, inserted


def numbered_lines(page, clip):
    """주석문·현대어역의 행 번호(01, 02 …) 붙은 줄 → {번호: 텍스트} (번호 없는 이어진 줄은 앞 번호에 합침)."""
    x0, y0, x1, y1 = clip
    ls = [l for l in page_lines(page) if x0 <= l["bbox"][0] <= x1 and y0 <= l["bbox"][1] <= y1]
    nums = [l for l in ls if re.fullmatch(r"\d{2}\s*", l["text"])]
    body = [l for l in ls if l not in nums]
    body.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    out, cur = {}, None
    for l in body:
        n = next((m for m in nums if abs(m["bbox"][1] - l["bbox"][1]) < 5 and m["bbox"][0] < l["bbox"][0]), None)
        if n:
            cur = int(n["text"])
            out[cur] = l["text"].strip()
        elif cur is not None and l["bbox"][0] - x0 < 60:
            out[cur] += " " + l["text"].strip()
    return out


# ---------------------------------------------------------------- 영인 처리
def render_region(page, bbox, dpi=300, rotate_cw=0):
    pix = page.get_pixmap(clip=fitz.Rect(*bbox), dpi=dpi)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if rotate_cw:
        im = im.rotate(-rotate_cw, expand=True)
    return im


def _ink_ccs(ink):
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    return [[int(st[i][0]), int(st[i][1]), int(st[i][0] + st[i][2]), int(st[i][1] + st[i][3])]
            for i in range(1, n) if st[i][4] >= 12]


def align_columns(det_cols, trans_cols, skip=0.8):
    """검출 행과 판독문 행의 순서 정렬(DP). 비용: 행 길이(잉크 높이)와 글자 수의 비율 차."""
    hs = np.array([c["y1"] - c["y0"] for c in det_cols], float)
    ks = np.array([len(c["units"]) for c in trans_cols], float)
    pitch = np.median(hs) / max(np.median(ks), 1)
    n, m = len(det_cols), len(trans_cols)
    D = np.full((n + 1, m + 1), 1e9); B = np.zeros((n + 1, m + 1), int); D[0, :] = np.arange(m + 1) * skip
    D[:, 0] = np.arange(n + 1) * skip
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(np.log((hs[i - 1] + 0.5 * pitch) / ((ks[j - 1] + 0.5) * pitch)))
            opts = [(D[i - 1, j - 1] + c, 0), (D[i - 1, j] + skip, 1), (D[i, j - 1] + skip, 2)]
            if j >= 2:  # 판독문 두 행이 원본에서는 한 행에 위아래로 이어진 경우(예: '다' + 서명)
                c2 = abs(np.log((hs[i - 1] + 0.5 * pitch) / ((ks[j - 1] + ks[j - 2] + 1) * pitch))) + 0.3
                opts.append((D[i - 1, j - 2] + c2, 3))
            D[i, j], B[i, j] = min(opts)
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        b = B[i, j]
        if b == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif b == 3:
            pairs.append((i - 1, (j - 2, j - 1))); i -= 1; j -= 2
        elif b == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def detect_columns(img_bgr, mask_boxes=(), split_wide=False):
    im = img_bgr.copy()
    for b in mask_boxes:
        im[b[1]:b[3], b[0]:b[2]] = 255
    det = detect_chars(im)
    cols = []
    for c in det["columns"]:
        cs = [q for q in det["chars"] if q["column"] == c["id"]]
        if not cs:
            continue
        cols.append({"x0": c["x0"], "x1": c["x1"], "y0": min(q["bbox"][1] for q in cs),
                     "y1": max(q["bbox"][3] for q in cs), "n": len(cs)})
    if split_wide and len(cols) >= 3:
        cols = _split_wide_columns(cols, det["ink"])
        cols = [_trim_column(c, det["ink"]) for c in cols]
    return cols, det["ink"]


def _trim_column(c, ink, gap_ratio=0.45, mass_ratio=0.12):
    """행 띠의 위·아래 끝에 떨어져 붙은 잉크 덩어리(봉투 띠 끝, 이웃 행 획, 종이 가장자리)를 행 범위에서 뺀다 (r01).
    편지 3 검수에서, 행이 이미지 맨 위부터 시작한 것으로 잡혀 글자 상자가 수백 px 위로 밀린 오류의 원인."""
    x0, x1, y0, y1 = int(c["x0"]), int(c["x1"]), int(c["y0"]), int(c["y1"])
    band = ink[y0:y1, x0:x1]
    if band.size == 0:
        return c
    rp = band.sum(1).astype(float)
    on = rp > 0.02 * max(rp.max(), 1)
    gap = gap_ratio * (x1 - x0)
    segs, y = [], 0
    while y < len(on):
        if on[y]:
            e = y
            while e < len(on) and (on[e] or (e + 1 < len(on) and not on[e] and np.any(on[e:e + int(gap)]))):
                e += 1
            segs.append([y, e, rp[y:e].sum()]); y = e
        else:
            y += 1
    if len(segs) <= 1:
        return c
    tot = sum(sg[2] for sg in segs)
    while len(segs) > 1 and segs[0][2] < mass_ratio * tot:
        segs.pop(0)
    while len(segs) > 1 and segs[-1][2] < mass_ratio * tot:
        segs.pop()
    return dict(c, y0=y0 + segs[0][0], y1=y0 + segs[-1][1])


def _split_wide_columns(cols, ink, ratio=1.4):
    """흘림 편지에서 이웃한 두 행이 겹쳐 한 행으로 잡힌 경우: 행 폭이 중앙값의 ratio 배를 넘으면
    가운데 근처 세로 잉크 투영이 가장 적은 곳에서 둘로 나눈다 (r01, 편지 3 검수에서 발견한 오류)."""
    wmed = float(np.median([c["x1"] - c["x0"] for c in cols]))
    out = []
    for c in cols:
        w = c["x1"] - c["x0"]
        if w > ratio * wmed:
            band = ink[int(c["y0"]):int(c["y1"]), int(c["x0"]):int(c["x1"])]
            vp = np.convolve(band.sum(0).astype(float), np.ones(9) / 9, mode="same")
            a, b = int(0.3 * w), int(0.7 * w)
            cut = c["x0"] + a + int(np.argmin(vp[a:b]))
            for x0, x1 in ((cut, c["x1"]), (c["x0"], cut)):      # 오른쪽 행이 먼저
                sub = ink[int(c["y0"]):int(c["y1"]), x0:x1]
                rows = np.flatnonzero(sub.sum(1) > 0)
                if len(rows):
                    out.append({"x0": x0, "x1": x1, "y0": int(c["y0"] + rows.min()), "y1": int(c["y0"] + rows.max() + 1),
                                "n": max(1, c["n"] // 2)})
        else:
            out.append(c)
    return out


def decorations(rgb):
    """시전지 문양·인장 등 색 있는 비문자 요소 (채도 높은 큰 덩어리)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    H, W = m.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.0015 * H * W or w > 0.6 * W or h > 0.6 * H:
            continue
        hue = float(np.median(hsv[y:y + h, x:x + w, 0][m[y:y + h, x:x + w] > 0]))
        kind = "seal" if (hue < 12 or hue > 165) and max(w, h) < 0.12 * max(W, H) else "decoration"
        out.append({"bbox": [int(x), int(y), int(x + w), int(y + h)], "kind": kind, "hue": round(hue, 1)})
    return out


class Builder:
    """regions / chars 누적 + 저장 (내보내기 형식은 export.py 와 같음)."""

    def __init__(self, pno, rid, img, to_pt):
        self.pno, self.rid, self.img, self.to_pt = pno, rid, img, to_pt
        self.regions, self.items = [], []

    def add(self, label, bbox, parent=None, **attrs):
        rid = f"{self.rid}.{label.lower()}_{sum(1 for x in self.regions if x['label'] == label)}"
        rg = {"id": rid, "label": label, "label_ko": LABELS[label]["ko"], "bbox": [int(v) for v in bbox] if bbox else None,
              "bbox_pt": self.to_pt(bbox) if bbox else None, "parent": parent}
        rg.update({k: v for k, v in attrs.items() if v is not None})
        self.regions.append(rg)
        return rid

    def char(self, text, bbox, rid, label, **attrs):
        it = {"id": f"p{self.pno + 1:04d}_{self.rid}_{len(self.items):05d}", "reading_order": len(self.items),
              "label": label, "region_id": rid, "text": text, "kind": char_kind(text),
              "bbox": [int(v) for v in bbox] if bbox else None, "bbox_pt": self.to_pt(bbox) if bbox else None}
        it.update({k: v for k, v in attrs.items() if v is not None and v is not False})
        self.items.append(it)
        return it

    def save(self, outdir, meta):
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.join(outdir, f"page_{self.pno + 1:04d}_{self.rid}")
        self.img.save(stem + ".png")
        ann = dict(meta, source_pdf_page=self.pno + 1, region=self.rid, image=os.path.basename(stem + ".png"),
                   image_size=list(self.img.size), coordinate="image pixel [x0,y0,x1,y1], top-left origin",
                   label_schema="LABELS.md", regions=self.regions, chars=self.items)
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=1)
        import csv
        with open(stem + ".csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reading_order", "id", "label", "region_id", "text", "x0", "y0", "x1", "y1", "flags"])
            for it in self.items:
                fl = ",".join(k for k in ("inserted", "deleted", "rotated", "bbox_estimated", "repeat_mark") if it.get(k))
                w.writerow([it["reading_order"], it["id"], it["label"], it["region_id"], it["text"],
                            *(it["bbox"] or ["", "", "", ""]), fl])
        with open(stem + "_regions.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label", "label_ko", "parent", "x0", "y0", "x1", "y1", "text", "attributes"])
            for rg in self.regions:
                extra = {k: v for k, v in rg.items() if k not in ("id", "label", "label_ko", "parent", "bbox", "bbox_pt", "text")}
                w.writerow([rg["id"], rg["label"], rg["label_ko"], rg["parent"] or "", *(rg["bbox"] or ["", "", "", ""]),
                            rg.get("text", "") or "", json.dumps(extra, ensure_ascii=False)[:500]])
        _overlay(stem, [it for it in self.items if it["bbox"]], [r for r in self.regions if r["bbox"]])
        return ann


def make_to_pt(bbox_pt, size, rotate_cw):
    """이미지 픽셀(회전 후) → PDF pt."""
    W, H = size
    Wd, Hd = (H, W) if rotate_cw in (90, 270) else (W, H)
    kx, ky = Wd / (bbox_pt[2] - bbox_pt[0]), Hd / (bbox_pt[3] - bbox_pt[1])

    def f(b):
        pts = []
        for (u, v) in ((b[0], b[1]), (b[2], b[3])):
            if rotate_cw == 90:
                x, y = v, Hd - u
            elif rotate_cw == 270:
                x, y = Wd - v, u
            else:
                x, y = u, v
            pts.append((bbox_pt[0] + x / kx, bbox_pt[1] + y / ky))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]
    return f


LETTER_PRIOR = None   # 검수 정답에서 배운 글자 모양 사전값 (set_letter_prior 로 설정)
EXEMPLAR_BANK = None  # 정답 글자 예시 은행 (set_exemplar_bank 로 설정)


def set_exemplar_bank(b):
    global EXEMPLAR_BANK
    EXEMPLAR_BANK = b


def set_letter_prior(p):
    global LETTER_PRIOR
    LETTER_PRIOR = p


def split_column_dp(ink, x0, x1, y0, y1, k, prior=None):
    """행(列) 띠를 k 글자로 자르는 DP (흘림 편지용, r01).
    잘리는 줄의 잉크 양 + 글자 높이가 사전값(검수 정답의 높이/행폭 분포)에서 벗어난 정도를 최소화하고,
    각 조각은 띠 안 잉크 경계로 조인다. 이전의 '글자 수만큼 등분'을 대체."""
    H, W = ink.shape
    x0, x1 = max(0, int(x0)), min(W, int(x1)); y0, y1 = max(0, int(y0)), min(H, int(y1))
    band = ink[y0:y1, x0:x1]
    L = y1 - y0
    if k <= 0 or L <= 2:
        return []
    prof = band.sum(1).astype(float)
    prof = np.convolve(prof, np.ones(7) / 7, mode="same")
    mx = max(prof.max(), 1e-6)
    exp = L / k
    sd = (prior or {}).get("h_cv", 0.35)
    step = max(1, int(L / 400))
    cand = list(range(0, L + 1, step))
    if cand[-1] != L:
        cand.append(L)
    idx = {c: i for i, c in enumerate(cand)}
    INF = 1e18
    D = np.full((k + 1, len(cand)), INF); Bk = np.zeros((k + 1, len(cand)), int); D[0, 0] = 0
    for j in range(1, k + 1):
        for bi, b in enumerate(cand):
            if b < j:           # 최소 1px
                continue
            lo = max(0, b - 2.2 * exp); hi = b - 0.35 * exp
            for ai in range(bi):
                a = cand[ai]
                if a < lo or a > hi or D[j - 1, ai] >= INF:
                    continue
                cut = 0.0 if b == L else prof[min(b, L - 1)] / mx
                c = ((b - a - exp) / (sd * exp)) ** 2 * 0.5 + 3.0 * cut
                v = D[j - 1, ai] + c
                if v < D[j, bi]:
                    D[j, bi], Bk[j, bi] = v, ai
    if D[k, len(cand) - 1] >= INF:
        return None
    segs, bi = [], len(cand) - 1
    for j in range(k, 0, -1):
        ai = Bk[j, bi]; segs.append((cand[ai], cand[bi])); bi = ai
    segs = segs[::-1]
    out = []
    for a, b in segs:
        sub = band[a:b]
        ys, xs = np.nonzero(sub)
        if len(xs) < 5:
            out.append([x0, y0 + a, x1, y0 + b, "est"])
            continue
        out.append([x0 + int(xs.min()), y0 + a + int(ys.min()), x0 + int(xs.max()) + 1, y0 + a + int(ys.max()) + 1])
    return out


def fill_columns(B, parent, det_cols, ink, trans_cols, line_meta=None, char_label="BODY_CHAR"):
    """정렬된 검출 행마다 판독문 글자 수(k)만큼 중심선 분할해 글자 상자를 만든다."""
    ccs = _ink_ccs(ink)
    raw = align_columns(det_cols, trans_cols)
    pairs = []
    for i, j in raw:
        if isinstance(j, tuple):  # 한 검출 행을 판독문 두 행으로 위아래 비율 분할
            ka, kb = len(trans_cols[j[0]]["units"]), len(trans_cols[j[1]]["units"])
            dc = det_cols[i]
            ym = dc["y0"] + (dc["y1"] - dc["y0"]) * ka / (ka + kb)
            det_cols = det_cols + [dict(dc, y1=ym), dict(dc, y0=ym)]
            pairs += [(len(det_cols) - 2, j[0]), (len(det_cols) - 1, j[1])]
        else:
            pairs.append((i, j))
    matched = {j for _, j in pairs}
    stats = {"det_columns": len(det_cols), "transcript_columns": len(trans_cols), "paired": len(pairs)}
    for i, j in pairs:
        dc, tc = det_cols[i], trans_cols[j]
        xc = (dc["x0"] + dc["x1"]) / 2
        units = [u for u in tc["units"] if not u["inserted"]]
        ins = [u for u in tc["units"] if u["inserted"]]
        boxes = None
        sims = None
        if EXEMPLAR_BANK is not None:
            from .exemplar import split_column_rag
            pad = (LETTER_PRIOR or {}).get("x_pad", 0.08) * (dc["x1"] - dc["x0"])
            boxes, sims = split_column_rag(ink, dc["x0"] - pad, dc["x1"] + pad, dc["y0"], dc["y1"],
                                           [u["text"] for u in units], EXEMPLAR_BANK, LETTER_PRIOR)
        if not boxes and LETTER_PRIOR is not None:
            pad = (LETTER_PRIOR or {}).get("x_pad", 0.08) * (dc["x1"] - dc["x0"])
            boxes = split_column_dp(ink, dc["x0"] - pad, dc["x1"] + pad, dc["y0"], dc["y1"], len(units), LETTER_PRIOR)
        if not boxes:
            boxes = _split_line(ccs, [xc, dc["y0"], xc, dc["y1"]], (dc["x1"] - dc["x0"]) * 1.05, len(units))
        meta = (line_meta or {}).get(j, {})
        role = meta.pop("role", None) if meta else None
        cid = B.add("TEXT_COLUMN", _u([b[:4] for b in boxes]) if boxes else [dc["x0"], dc["y0"], dc["x1"], dc["y1"]],
                    parent, text="".join(u["text"] for u in units), elevated=tc.get("elevated") or None,
                    inserted_text="".join(u["text"] for u in ins) or None,
                    role=role, **meta)
        for n_, (u, b) in enumerate(zip(units, boxes)):
            sm = sims[n_] if sims and n_ < len(sims) else None
            B.char(u["text"], b[:4], cid, "DELETED_MARK" if u["deleted"] else char_label,
                   bbox_estimated=len(b) == 5, repeat_mark=u["repeat_mark"], transcript_bbox_pt=u["bbox_pt"],
                   exemplar_sim=round(sm, 3) if sm is not None else None)
        for u in ins:  # 삽입 글자: 행 오른쪽 옆, 판독문 y 비율 위치로 추정
            B.char(u["text"], None, cid, "INSERTED_CHAR", inserted=True, transcript_bbox_pt=u["bbox_pt"],
                   note="행 옆에 작게 끼워 쓴 글자. 위치는 검수 필요")
    stats["transcript_columns_unmatched"] = [j for j in range(len(trans_cols)) if j not in matched]
    return stats


# ---------------------------------------------------------------- 1부: 상언
def process_document(doc, outdir, img_page=6, trans_page=7, annot_page=8, modern_page=9,
                     part="제1부 김씨부인 한글상언", title="김씨부인 한글상언", rotate_cw=0, dpi=300):
    p = doc[img_page - 1]
    infos = sorted([b for b in p.get_image_info(xrefs=True) if b["width"] > 1000], key=lambda b: -b["bbox"][0])
    tp = doc[trans_page - 1]
    halves = [(tp.rect.width / 2, 150, tp.rect.width, 720), (0, 150, tp.rect.width / 2, 720)]  # 오른쪽 반 = 앞부분
    markers = [(int(l["text"]), (l["bbox"][0] + l["bbox"][2]) / 2) for l in page_lines(tp)
               if re.fullmatch(r"\d{2}\s*", l["text"]) and 80 < l["bbox"][1] < 150]
    annot = {}
    for clip in ((30, 100, 600, 800), (600, 100, 1219, 800)):
        annot.update(numbered_lines(doc[annot_page - 1], clip))
    modern = {}
    for clip in ((30, 120, 600, 800), (740, 150, 1219, 470)):
        modern.update(numbered_lines(doc[modern_page - 1], clip))
    # 머리 행·연기 행 (주석문의 번호 없는 첫 줄·끝 줄)
    ap = page_lines(doc[annot_page - 1])
    head_annot = next((l["text"].strip() for l in sorted(ap, key=lambda l: l["bbox"][1])
                       if 100 < l["bbox"][1] < 140 and l["bbox"][0] < 600), None)
    date_annot = next((l["text"].strip() for l in sorted(ap, key=lambda l: -l["bbox"][1])
                       if l["bbox"][0] < 600 and 740 < l["bbox"][1] < 790), None)
    mp = page_lines(doc[modern_page - 1])
    head_modern = next((l["text"].strip() for l in mp if 120 < l["bbox"][1] < 150 and l["bbox"][0] < 600), None)
    date_modern = next((l["text"].strip() for l in mp if 440 < l["bbox"][1] < 470 and l["bbox"][0] > 740), None)

    def line_meta(tc, first, last):
        m = [n for n, x in markers if abs(x - tc["x"]) < 8]
        if m:
            n = m[0]
            return {"line_no": f"{n:02d}", "annotated_text": annot.get(n), "modern": modern.get(n)}
        if first:
            return {"role": "header", "annotated_text": head_annot, "modern": head_modern}
        if last:
            return {"role": "date", "annotated_text": date_annot, "modern": date_modern}
        return {}
    tparts = []
    for h in halves:
        tc, _, _ = vertical_transcript(tp, h)
        tparts.append(tc)
    ntot = sum(len(t) for t in tparts)
    imgs = []
    for k, b in enumerate(infos):
        im = render_region(p, b["bbox"], dpi, rotate_cw)
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        cols, ink = detect_columns(bgr)
        tc = tparts[k] if k < len(tparts) else []
        meta = {j: line_meta(c, k == 0 and j == 0, k == len(infos) - 1 and j == len(tc) - 1) for j, c in enumerate(tc)}
        imgs.append((im, b, ink, cols, tc, meta))
    results = []
    for k, (im, b, ink, dcols, tsel, meta) in enumerate(imgs):
        to_pt = make_to_pt(b["bbox"], im.size, rotate_cw)
        B = Builder(img_page - 1, f"R{k}", im, to_pt)
        W, H = im.size
        top = B.add("DOCUMENT", [0, 0, W, H], None, part=part, doc_title=title, image_segment=k + 1,
                    image_segments=len(imgs), image_orientation=("PDF 쪽 방향 그대로" if not rotate_cw else f"PDF 쪽에서 시계방향 {rotate_cw}° 회전"),
                    transcript_pages=[trans_page, annot_page, modern_page])
        pairs_ = align_columns(dcols, tsel)
        msel = meta
        # role → 라벨
        st = fill_columns(B, top, dcols, ink, tsel, {j: dict(v) for j, v in msel.items()})
        for rg in B.regions:
            if rg["label"] == "TEXT_COLUMN" and rg.get("role") in ("header", "date"):
                rg["label"] = "DOC_HEADER" if rg["role"] == "header" else "DATE_LINE"
                rg["label_ko"] = LABELS[rg["label"]]["ko"]
        for it in B.items:
            if it["label"] == "BODY_CHAR":
                par = next(r for r in B.regions if r["id"] == it["region_id"])
                it["line_no"] = par.get("line_no")
        header_modern = None
        ann = B.save(outdir, {"part": part, "alignment": dict(st, transcript_lines_total=ntot)})
        results.append(ann)
    return results


# ---------------------------------------------------------------- 2부: 편지
def section_texts(page, half, labels=("주석문", "현대어역", "해설")):
    """오른쪽 반쪽의 '주석문/현대어역/해설' 구역 텍스트."""
    x0, x1 = half
    ls = sorted([l for l in page_lines(page) if x0 <= l["bbox"][0] <= x1 and l["bbox"][1] > 60],
                key=lambda l: (l["bbox"][1], l["bbox"][0]))
    heads = [l for l in ls if l["text"].strip() in labels]
    out = {}
    for i, h in enumerate(heads):
        y_end = heads[i + 1]["bbox"][1] if i + 1 < len(heads) else 1e9
        body = [l for l in ls if h["bbox"][1] + 2 < l["bbox"][1] < y_end - 1 and l["bbox"][0] >= h["bbox"][0] - 5
                and not re.match(r"^\d+\s", l["text"])]
        out[h["text"].strip()] = " ".join(l["text"].strip() for l in body)
    return out


def _fit(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0:
        return lambda v: v
    k, c = np.polyfit(a, b, 1)
    return lambda v: k * v + c


def find_envelope(bgr):
    """편지 위쪽에 가로로 오려 붙인 봉투 글씨: 윗부분의 큰 글자 성분을 가로로 이어 붙인 사슬 중 가장 넓은 것."""
    H, W = bgr.shape[:2]
    ink = binarize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ink[(hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)] = 0
    ink[:, : int(0.02 * W)] = 0; ink[:, -int(0.02 * W):] = 0
    allc = _ink_ccs(ink)
    big = [b for b in allc if max(b[2] - b[0], b[3] - b[1]) > 0.03 * max(H, W) and b[3] < 0.45 * H]
    n = len(big)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a in range(n):
        for b in range(a + 1, n):
            A, Bb = big[a], big[b]
            ov = min(A[3], Bb[3]) - max(A[1], Bb[1])
            gap = max(Bb[0] - A[2], A[0] - Bb[2], 0)
            if ov >= 0.4 * min(A[3] - A[1], Bb[3] - Bb[1]) and gap < 0.06 * W:
                parent[find(a)] = find(b)
    groups = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(big[a])
    best = None
    for g in groups.values():
        box = _u(g)
        if len(g) >= 3 and (box[2] - box[0]) > 0.25 * W and (box[2] - box[0]) > 1.5 * (box[3] - box[1]):
            if best is None or (box[1] + box[3]) < (best[1] + best[3]):  # 가장 위쪽 사슬
                best = box
    if best is None:
        return None
    inside = [b for b in allc if b[0] >= best[0] - 10 and b[2] <= best[2] + 10 and b[1] >= best[1] - 10 and b[3] <= best[3] + 10]
    return _u(inside + [best])


def columns_from_transcript(ink, tcols, masks=()):
    """판독문 행의 x 배치(원본 행 배열을 따름)를 이미지에 옮겨 행 띠를 정한다 (r01).
    흘림 편지는 행끼리 겹치고 기울어서 투영만으로 행을 나누면 한 칸씩 밀린다 → 판독문 행 간격을 뼈대로 쓰고,
    배율·위치는 '행 중심선이 잉크 많은 곳을 지나도록' 격자 탐색으로 맞춘다."""
    H, W = ink.shape
    m = ink.copy()
    for b in masks:
        m[max(0, int(b[1])):int(b[3]), max(0, int(b[0])):int(b[2])] = 0
    vp = np.convolve(m.sum(0).astype(float), np.ones(15) / 15, mode="same")
    xs = np.flatnonzero(vp > 0.05 * vp.max())
    if len(xs) < 10 or len(tcols) < 2:
        return None
    tx = np.array([c["x"] for c in tcols], float)            # 판독문 x (오른쪽→왼쪽 순)
    ix0, ix1 = xs.min(), xs.max()
    k0 = (ix1 - ix0) / max(tx.max() - tx.min(), 1e-6)
    best = None
    for k in np.linspace(0.8 * k0, 1.15 * k0, 36):
        span = k * (tx.max() - tx.min())
        for c in np.linspace(ix0 - 0.1 * span, ix1 - 0.9 * span, 40):
            px = c + k * (tx - tx.min())
            if px.min() < 0 or px.max() >= W:
                continue
            sc = vp[px.astype(int)].sum()
            if best is None or sc > best[0]:
                best = (sc, k, c)
    if best is None:
        return None
    _, k, c = best
    px = c + k * (tx - tx.min())
    order = np.argsort(-px)
    cols = []
    for n, j in enumerate(range(len(tcols))):
        x = px[j]
        left = px[j + 1] if j + 1 < len(px) else x - (px[j - 1] - x)
        right = px[j - 1] if j > 0 else x + (x - px[j + 1])
        x0 = int((x + left) / 2); x1 = int((x + right) / 2)
        band = m[:, max(0, x0):min(W, x1)]
        rows = np.flatnonzero(band.sum(1) > 0)
        if len(rows) == 0:
            cols.append({"x0": x0, "x1": x1, "y0": 0, "y1": H, "n": 0}); continue
        cols.append({"x0": max(0, x0), "x1": min(W, x1), "y0": int(rows.min()), "y1": int(rows.max()) + 1, "n": 0})
    return cols


def process_letter(doc, pno, outdir, part="제2부 정조어필한글편지첩", dpi=300, use_transcript_grid=False, split_wide=False):
    page = doc[pno - 1]
    W2 = page.rect.width / 2
    caps = captions(page)
    imgs = [b for b in page.get_image_info(xrefs=True) if b["width"] > 800]
    results = []
    for k, b in enumerate(sorted(imgs, key=lambda b: b["bbox"][0])):
        bb = b["bbox"]
        half = (0, W2) if (bb[0] + bb[2]) / 2 < W2 else (W2, page.rect.width)
        cap = next((c for c in caps if 0 <= c["bbox"][1] - bb[3] < 30 and c["bbox"][0] < bb[2]), None)
        lab = next((l for l in page_lines(page) if l["text"].strip() == "판독문" and half[0] <= l["bbox"][0] < half[1]), None)
        ty0 = lab["bbox"][1] if lab else bb[3] + 30
        tcols, rot, ins = vertical_transcript(page, (half[0], ty0, half[1], page.rect.height - 20))
        other = (W2, page.rect.width) if half[0] == 0 else (0, W2)
        secs = section_texts(page, (other[0] + 40, other[1]))
        im = render_region(page, bb, dpi)
        rgb = np.array(im)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        to_pt = make_to_pt(bb, im.size, 0)
        B = Builder(pno - 1, f"R{k}", im, to_pt)
        Wi, Hi = im.size
        fig_id, date = (cap["id"], cap["rest"]) if cap else (None, None)
        top = B.add("LETTER", [0, 0, Wi, Hi], None, part=part, figure_id=fig_id, date=date,
                    caption={"text": cap["text"], "bbox_pt": [round(v, 2) for v in cap["bbox"]]} if cap else None,
                    annotated_text=secs.get("주석문"), modern=secs.get("현대어역"), commentary=secs.get("해설"))
        # 1차 행 검출 → 판독문 좌표와 이미지 좌표의 선형 대응 추정 → 봉투(회전 글자) 위치 추정
        cols, ink = detect_columns(bgr)
        env_box = None
        env_text = None
        m = re.search(r"\(봉투\)\s*(\S+)", secs.get("주석문") or "")
        if m:
            env_text = re.sub(r"\d+$", "", m.group(1))
        if rot or env_text:
            env_box = find_envelope(bgr)
        decos = decorations(rgb)
        masks = [d_["bbox"] for d_ in decos]
        if env_box:
            masks.append([env_box[0] - 10, env_box[1] - 10, env_box[2] + 10, env_box[3] + 10])
        if masks or split_wide:
            cols, ink = detect_columns(bgr, masks, split_wide=split_wide)
        if use_transcript_grid:
            grid = columns_from_transcript(ink, tcols, masks)
            if grid:
                cols = grid
        # 서명: 마지막 짧은 한자 행
        meta = {}
        if tcols and len(tcols[-1]["units"]) <= 3 and all(char_kind(u["text"]) == "hanja" for u in tcols[-1]["units"]):
            meta[len(tcols) - 1] = {"role": "signature"}
        st = fill_columns(B, top, cols, ink, tcols, meta)
        for rg in B.regions:
            if rg.get("role") == "signature":
                rg["label"], rg["label_ko"] = "SIGNATURE", LABELS["SIGNATURE"]["ko"]
            if rg["label"] == "TEXT_COLUMN" and rg.get("elevated"):
                rg["note"] = "대두(擡頭): 높임 대상 앞에서 행을 바꾸어 위로 올려 씀"
        if env_box:
            text = env_text or "".join(q["text"] for q in rot)
            units = [u for u in clusters(text) if u.strip()]
            # 판독문에 회전 글자가 원본과 같은 배치로 놓임: x 오름차순이 읽기 순서의 역순이면 오른쪽→왼쪽(글자 시계 90°)
            spatial = "".join(q["text"] for q in sorted(rot, key=lambda q: q["bbox"][0]))
            rtl = bool(rot) and spatial == "".join(reversed([q["text"] for q in rot])) or \
                (bool(rot) and spatial[::-1].startswith(text[:len(spatial)]))
            rotation = 90 if rtl else -90
            eid = B.add("ENVELOPE_TEXT", env_box, top, text=text, rotation=rotation,
                        reading_direction="right_to_left (글자가 시계방향 90° 돌아감)" if rtl else
                        "left_to_right (글자가 반시계방향 90° 돌아감)",
                        modern=(re.search(r"\(봉투\)\s*([^(]+?)\s*(서릿|$)", secs.get("현대어역") or "") or [None, None])[1],
                        verification="auto-estimated")
            cy = (env_box[1] + env_box[3]) / 2
            boxes = _split_line(_ink_ccs(ink) if False else [bx for bx in _ink_ccs(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 0)],
                                [env_box[0], cy, env_box[2], cy], env_box[3] - env_box[1], len(units)) if False else None
            ink0 = binarize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            eccs = [bx for bx in _ink_ccs(ink0) if bx[0] >= env_box[0] - 5 and bx[2] <= env_box[2] + 5
                    and bx[1] >= env_box[1] - 5 and bx[3] <= env_box[3] + 5]
            line = [env_box[2], cy, env_box[0], cy] if rtl else [env_box[0], cy, env_box[2], cy]
            boxes = _split_line(eccs, line, env_box[3] - env_box[1] + 10, len(units))
            for g, bx in zip(units, boxes):
                B.char(g, bx[:4], eid, "ENVELOPE_CHAR", rotated=True, rotation=rotation, bbox_estimated=len(bx) == 5)
        for d_ in decos:
            B.add("PAPER_DECORATION", d_["bbox"], top, kind=d_["kind"], hue=d_["hue"], verification="auto")
        results.append(B.save(outdir, {"part": part, "alignment": st, "figure_id": fig_id, "date": date}))
    return results


# ---------------------------------------------------------------- 3부: 곤전어필 (책 면)
def book_blocks(page, img_bb, labels=("판독 및 주석문", "판독문", "현대어역")):
    """이미지 오른쪽, 같은 높이 띠의 '판독 및 주석문 / 현대어역' 구역."""
    ls = [l for l in page_lines(page) if l["bbox"][0] > img_bb[2] - 2 and img_bb[1] - 15 <= l["bbox"][1] <= img_bb[3] + 5
          and l["bbox"][0] < img_bb[2] + 300]
    ls.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    heads = [l for l in ls if l["text"].strip() in labels]
    out = {}
    for i, h in enumerate(heads):
        y_end = heads[i + 1]["bbox"][1] if i + 1 < len(heads) else 1e9
        body = [l for l in ls if h["bbox"][1] + 2 < l["bbox"][1] < y_end - 1]
        key = "original" if h["text"].strip() != "현대어역" else "modern"
        out[key] = [l["text"].strip() for l in body]
    return out


def allocate_counts(est, total):
    """연속 추정치 est 를 합이 total 인 정수로 (최대 나머지 방식)."""
    est = np.asarray(est, float)
    if est.sum() <= 0:
        return [0] * len(est)
    est = est * total / est.sum()
    base = np.floor(est).astype(int)
    rem = total - base.sum()
    for idx in np.argsort(-(est - base))[:rem]:
        base[idx] += 1
    return base.tolist()


GLYPH_W = np.array([0.82, 0.28, 0.2, 0.3, 0.23, -0.32])   # [기본, 가로모음, 섞인모음, 받침, 한자, 부호]


def glyph_features(u):
    import unicodedata
    f = np.zeros(6)
    if not u or not u.strip():
        return f
    f[0] = 1
    k = char_kind(u)
    if k == "punct" or u in ("〃", "″", "ㅅ"):
        f[5] = 1
        return f
    if k == "hanja":
        f[4] = 1
        return f
    j = unicodedata.normalize("NFD", u)
    V = [c for c in j if 0x1161 <= ord(c) <= 0x11A7 or 0xD7B0 <= ord(c) <= 0xD7C6]
    T = [c for c in j if 0x11A8 <= ord(c) <= 0x11FF or 0xD7CB <= ord(c) <= 0xD7FB]
    if any(v in set("\u1169\u116d\u116e\u1172\u1173\u119e\u11a1") for v in V):
        f[1] = 1
    elif any(v in set("\u116a\u116b\u116c\u116f\u1170\u1171\u1174") for v in V):
        f[2] = 1
    if T:
        f[3] = 1
    return f


def glyph_height_old(u):
    """글자 모양으로 예측한 세로 길이(보통 글자 = 1.0).
    받침이 있거나 가로 모음(ㅗ ㅛ ㅜ ㅠ ㅡ ㆍ)이 아래에 붙으면 길고, 세로 모음(ㅏ ㅓ ㅣ …)만 있으면 짧다."""
    import unicodedata
    if not u or not u.strip():
        return 0.0
    k = char_kind(u)
    if k in ("punct",) or u in ("〃", "″", "ㅅ"):
        return 0.5
    if k == "hanja":
        return 1.05
    j = unicodedata.normalize("NFD", u)
    V = [c for c in j if 0x1161 <= ord(c) <= 0x11A7 or 0xD7B0 <= ord(c) <= 0xD7C6]
    T = [c for c in j if 0x11A8 <= ord(c) <= 0x11FF or 0xD7CB <= ord(c) <= 0xD7FB]
    horiz = set("\u1169\u116d\u116e\u1172\u1173\u119e\u11a1")  # ㅗ ㅛ ㅜ ㅠ ㅡ ㆍ ㆎ
    mixed = set("\u116a\u116b\u116c\u116f\u1170\u1171\u1174")  # ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ
    h = 0.82
    if any(v in horiz for v in V):
        h += 0.28
    elif any(v in mixed for v in V):
        h += 0.2
    if T:
        h += 0.3
    return h


def column_fit(prof, k):
    """행 잉크 세로 투영(prof)을 k 글자로 나눌 때의 적합도(낮을수록 좋음):
    등간격 기대 위치 근처에서 잉크가 가장 적은 곳을 자르고, 그 잉크 양과 등간격에서 벗어난 정도를 벌점."""
    L = len(prof)
    if k <= 1:
        return 0.0
    pitch = L / k
    mx = max(prof.max(), 1e-6)
    cost = 0.0
    for i in range(1, k):
        ideal = i * pitch
        a, b = int(max(0, ideal - 0.35 * pitch)), int(min(L - 1, ideal + 0.35 * pitch))
        seg = prof[a:b + 1]
        j = a + int(np.argmin(seg))
        cost += seg.min() / mx + 0.6 * ((j - ideal) / pitch) ** 2
    return cost / (k - 1)


def partition_by_height(gold, spans, counts=None, w_count=0.15, fits=None, w_fit=3.0, weights=None, anchors=None,
                        fixed=None):
    """판독문 글자열을 행(列) 수만큼 연속 구간으로 나눔 (weights: 글자 모양 특징별 높이 가중치):
    각 행의 잉크 길이 ≈ 배율 × (그 구간 글자들의 예측 높이 합) 이 되도록 DP.
    상자 개수(counts)는 약한 보조 신호로만 씀."""
    N, C = len(gold), len(spans)
    weights = GLYPH_W if weights is None else weights
    h = np.array([max(0.2, glyph_features(u) @ weights) for u in gold])
    cum = np.concatenate([[0], np.cumsum(h)])
    scale = sum(spans) / max(cum[-1], 1e-6)
    avg = sum(spans) / max(C, 1)
    INF = 1e18
    D = np.full((C + 1, N + 1), INF); Bk = np.zeros((C + 1, N + 1), int); D[0, 0] = 0
    for c in range(1, C + 1):
        for p in range(c, N - (C - c) + 1):
            best, arg = INF, 0
            for q in range(c - 1, p):
                if D[c - 1, q] >= INF:
                    continue
                if fixed and (c - 1) in fixed:       # 사람이 확인한 행 전체(검수 보정) → 글자열 고정
                    if gold[q:p] != fixed[c - 1]:
                        continue
                if anchors and (c - 1) in anchors:   # 사람이 확인한 행 첫머리(검수 보정)
                    pre = anchors[c - 1]
                    if gold[q:q + len(pre)] != pre:
                        continue
                if anchors and c in anchors and p < N:   # 다음 행의 첫머리도 맞아야 함
                    pre = anchors[c]
                    if gold[p:p + len(pre)] != pre:
                        continue
                pred = scale * (cum[p] - cum[q])
                pitch = avg / max(N / C, 1)
                cost = ((spans[c - 1] - pred) / pitch) ** 2          # 1글자 어긋남 ≈ 1.0
                if fits is not None:
                    cost += w_fit * fits[c - 1](p - q)
                if counts is not None:
                    cost += w_count * (counts[c - 1] - (p - q)) ** 2 / max(np.mean(counts), 1)
                v = D[c - 1, q] + cost
                if v < best:
                    best, arg = v, q
            D[c, p], Bk[c, p] = best, arg
    bounds, p = [], N
    for c in range(C, 0, -1):
        q = Bk[c, p]; bounds.append((q, p)); p = q
    return bounds[::-1]


def learn_glyph_weights(faces, iters=3):
    """여러 면(face)의 행 길이로 글자 모양별 높이 가중치를 학습 (EM식 반복):
    ① 현재 가중치로 판독문을 행에 나눔 → ② 행마다 [특징 합] 과 [행 길이/면 배율] 로 최소제곱 → 반복."""
    w = GLYPH_W.copy()
    for _ in range(iters):
        X, y = [], []
        for f in faces:
            b = partition_by_height(f["gold"], f["spans"], f["ns"], fits=f["fits"], weights=w)
            F = np.array([glyph_features(u) for u in f["gold"]])
            pred = np.array([F[a:bb].sum(0) @ w for a, bb in b])
            scale = f["spans"].sum() / max(pred.sum(), 1e-6)
            for (a, bb), sp in zip(b, f["spans"]):
                X.append(F[a:bb].sum(0)); y.append(sp / scale)
        X, y = np.array(X), np.array(y)
        lam = 5.0   # 사전값(GLYPH_W)으로 당기는 정규화 — 행 수가 적어도 안정
        A = X.T @ X + lam * np.eye(6)
        w = np.linalg.solve(A, X.T @ y + lam * GLYPH_W)
    return w


def assign_reflowed(B, top, det, title, gold, w_span=0.5, weights=None, prepare_only=False, forced_bounds=None):
    """줄바꿈이 원본과 다른 판독문을 행(列)에 나눠 담기.

    글자 상자 개수만으로 순서대로 대응하면, 흘림 궁서체에서 한 글자가 옆 글자와 붙어 상자 하나가 되거나
    (예: '티시고'의 '고' 누락) 한 글자가 여러 조각으로 갈라질 때(예: '하'가 두 상자) 오류가 뒤 행으로 계속 밀린다.
    그래서 ① 행마다 글자 수를 '잉크 길이/글자 간격'과 '상자 수'를 섞어 추정하고
    ② 전체 합이 판독문 글자 수와 같아지도록 정수로 배분한 뒤
    ③ 행 안에서는 그 글자 수만큼 중심선 분할(_split_line)로 다시 나눈다 → 오류가 행 밖으로 번지지 않음."""
    cols = {}
    for c in det["chars"]:
        cols.setdefault(c["column"], []).append(c)
    order = sorted(cols)                      # 0 = 오른쪽 첫 행
    info = []
    Hh, Ww = det["ink"].shape
    hs_all = [c["bbox"][3] - c["bbox"][1] for c in det["chars"]]
    hmed = float(np.median(hs_all)) if hs_all else 50
    amed = float(np.median([(c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]) for c in det["chars"]])) if det["chars"] else 1
    for ci in order:
        cs = sorted(cols[ci], key=lambda c: c["bbox"][1])
        # 행 끝 잡티 제거: 이미지 가장자리에 닿거나, 작고 떨어져 있는 첫/끝 상자는 행 길이에서 제외
        def junk(c, nb):
            b = c["bbox"]
            area = (b[2] - b[0]) * (b[3] - b[1])
            edge = b[1] <= 3 or b[3] >= Hh - 3
            gap = abs((nb["bbox"][1] - b[3]) if nb["bbox"][1] > b[1] else (b[1] - nb["bbox"][3])) if nb else 0
            return edge or (area < 0.25 * amed and gap > 0.5 * hmed)
        while len(cs) > 2 and junk(cs[-1], cs[-2]):
            cs.pop()
        while len(cs) > 2 and junk(cs[0], cs[1]):
            cs.pop(0)
        cols[ci] = cs
        info.append({"ci": ci, "x0": min(c["bbox"][0] for c in cs), "x1": max(c["bbox"][2] for c in cs),
                     "y0": min(c["bbox"][1] for c in cs), "y1": max(c["bbox"][3] for c in cs), "n": len(cs)})
    tunits = [u for u in clusters(title)] if title else []
    tunits = [u for u in tunits if u.strip()]
    body_cols = info
    if title and info and abs(info[0]["n"] - len(tunits)) <= 1:
        tcol, body_cols = info[0], info[1:]
    else:
        tcol = None
    N = len(gold)
    spans = np.array([c["y1"] - c["y0"] for c in body_cols], float)
    ns = np.array([c["n"] for c in body_cols], float)
    ink = det["ink"]
    fits = []
    for c in body_cols:
        prof = ink[int(c["y0"]):int(c["y1"]), int(c["x0"]):int(c["x1"])].sum(1).astype(float)
        prof = np.convolve(prof, np.ones(5) / 5, mode="same")
        cache = {}
        def fit_k(k, prof=prof, cache=cache):
            if k not in cache:
                cache[k] = column_fit(prof, k)
            return cache[k]
        fits.append(fit_k)
    ccs_all = _ink_ccs(det["ink"])

    def split(c, k):
        if k <= 0:
            return []
        xc = (c["x0"] + c["x1"]) / 2
        return _split_line(ccs_all, [xc, c["y0"], xc, c["y1"]], (c["x1"] - c["x0"]) * 1.1, k)
    if prepare_only:
        return {"gold": gold, "spans": spans, "ns": ns, "fits": fits, "cols": body_cols, "split": split,
                "ink": det["ink"]}
    bounds = forced_bounds if forced_bounds is not None else partition_by_height(gold, spans, ns, fits=fits, weights=weights)
    ks = [int(b - a) for a, b in bounds]
    ccs = _ink_ccs(det["ink"])
    st = {"method": "column-partition-by-glyph-height", "gold_chars": N, "columns": len(body_cols),
          "detected_boxes": int(ns.sum()), "allocated": ks}
    if tcol:
        xc = (tcol["x0"] + tcol["x1"]) / 2
        bx = _split_line(ccs, [xc, tcol["y0"], xc, tcol["y1"]], (tcol["x1"] - tcol["x0"]) * 1.1, len(tunits))
        rid = B.add("SECTION_TITLE", _u([b[:4] for b in bx]), top, text=title)
        for g, b in zip(tunits, bx):
            B.char(g, b[:4], rid, "BODY_CHAR", text_source="transcript-title", bbox_estimated=len(b) == 5)
    pos = 0
    for c, k in zip(body_cols, ks):
        seg = gold[pos:pos + k]; pos += k
        xc = (c["x0"] + c["x1"]) / 2
        bx = _split_line(ccs, [xc, c["y0"], xc, c["y1"]], (c["x1"] - c["x0"]) * 1.1, k) if k else []
        rid = B.add("TEXT_COLUMN", _u([b[:4] for b in bx]) if bx else [c["x0"], c["y0"], c["x1"], c["y1"]], top,
                    column=c["ci"], text="".join(seg), detected_boxes=c["n"])
        for g, b in zip(seg, bx):
            B.char(g, b[:4], rid, "BODY_CHAR", text_source="transcript-column-allocated", bbox_estimated=len(b) == 5)
    return st


def _glyph_vec(ink, box, S=32):
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    sub = ink[max(0, y0):y1, max(0, x0):x1].astype(np.float32)
    if sub.size == 0 or sub.sum() == 0:
        return None
    h, w = sub.shape
    sc = S / max(h, w)
    sub = cv2.resize(sub, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    out = np.zeros((S, S), np.float32)
    oy, ox = (S - sub.shape[0]) // 2, (S - sub.shape[1]) // 2
    out[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
    out = cv2.GaussianBlur(out, (5, 5), 1.2).ravel()
    out -= out.mean()
    n = np.linalg.norm(out)
    return out / n if n > 0 else None


def refine_with_prototypes(faces, iters=2, radius=2, w_fit=1.0):
    """글자 모양 자기일관성으로 행 경계 보정.

    같은 쪽(여러 면)에서 같은 글자(예: '고', '이', 'ᄒᆞ')는 비슷한 모양으로 쓰인다.
    현재 배정으로 글자별 평균 모양(프로토타입)을 만들고, 각 행 경계를 ±radius 글자 옮겨 보면서
    행을 다시 나눈 글자 조각들이 자기 글자의 프로토타입과 가장 닮는 경계를 고른다.
    (궁서체에서 '티시고'의 '고'처럼 앞 글자와 붙어 상자가 하나로 잡혀도, 경계를 글자 모양으로 바로잡음)"""
    for _ in range(iters):
        protos = {}
        for f in faces:
            for c, (a, b) in zip(f["cols"], f["bounds"]):
                bx = f["split"](c, b - a)
                for u, box in zip(f["gold"][a:b], bx):
                    v = _glyph_vec(f["ink"], box)
                    if v is not None:
                        protos.setdefault(u, []).append(v)
        P = {u: np.mean(vs, 0) for u, vs in protos.items() if len(vs) >= 2}
        for f in faces:
            C, N = len(f["cols"]), len(f["gold"])
            init = [0] + [b for _, b in f["bounds"]]
            cands = [sorted({min(N, max(0, init[c] + d)) for d in range(-radius, radius + 1)}) if 0 < c < C else [init[c]]
                     for c in range(C + 1)]
            memo = {}

            def vis(c, a, b):
                key = (c, a, b)
                if key in memo:
                    return memo[key]
                if b <= a:
                    memo[key] = 1e6
                    return 1e6
                bx = f["split"](f["cols"][c], b - a)
                sims = []
                for u, box in zip(f["gold"][a:b], bx):
                    if u in P:
                        v = _glyph_vec(f["ink"], box)
                        sims.append(float(v @ P[u]) if v is not None else 0.0)
                val = (1 - np.mean(sims)) * (b - a) if sims else 0.5 * (b - a)
                val += w_fit * f["fits"][c](b - a)
                memo[key] = val
                return val
            D = [{init[0]: (0.0, None)}]
            for c in range(1, C + 1):
                row = {}
                for bnd in cands[c]:
                    best = None
                    for prev, (cost, _) in D[c - 1].items():
                        v = cost + vis(c - 1, prev, bnd)
                        if best is None or v < best[0]:
                            best = (v, prev)
                    row[bnd] = best
                D.append(row)
            path, bnd = [], N
            for c in range(C, 0, -1):
                prev = D[c][bnd][1]
                path.append((prev, bnd)); bnd = prev
            f["bounds"] = path[::-1]
    return faces


def process_book_page(doc, pno, outdir, part="제3부 곤전어필", dpi=300, lang="kor+chi_tra", corrections=None):
    page = doc[pno - 1]
    caps = captions(page)
    imgs = sorted([b for b in page.get_image_info(xrefs=True) if b["width"] > 500],
                  key=lambda b: (b["bbox"][0] > page.rect.width / 2, b["bbox"][1]))
    results, faces = [], []
    for k, b in enumerate(imgs):
        bb = b["bbox"]
        cap = next((c for c in caps if 0 <= c["bbox"][1] - bb[3] < 25 and bb[0] - 5 <= c["bbox"][0] <= bb[2]), None)
        folio = cap["id"] if cap else None
        blk = book_blocks(page, bb)
        orig_lines = blk.get("original", [])
        title = None
        if orig_lines and len(orig_lines[0].replace(" ", "")) <= 8 and not re.search(r"\d", orig_lines[0]):
            title = orig_lines[0]
        orig_text = " ".join(orig_lines[1:] if title else orig_lines)
        gold = transcript_units(re.sub(r"(?<=[^\s\d])\d+", "", orig_text))  # 각주 번호 제거
        im = render_region(page, bb, dpi)
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        to_pt = make_to_pt(bb, im.size, 0)
        B = Builder(pno - 1, f"R{k}", im, to_pt)
        Wi, Hi = im.size
        top = B.add("BOOK_PAGE", [0, 0, Wi, Hi], None, part=part, folio=folio, book_title="곤전어필",
                    caption={"text": cap["text"], "bbox_pt": [round(v, 2) for v in cap["bbox"]]} if cap else None,
                    transcript=" ".join(orig_lines), modern=" ".join(blk.get("modern", [])))
        det = detect_chars(bgr)
        faces.append({"B": B, "top": top, "det": det, "title": title, "gold": gold, "folio": folio})
        continue
    # 같은 쪽의 여러 면으로 글자 높이 가중치 학습 → 면마다 행 배분
    prep = []
    for f in faces:
        B = f["B"]
        p_ = assign_reflowed(B, f["top"], f["det"], f["title"], f["gold"], prepare_only=True)
        prep.append(p_)
    w = learn_glyph_weights(prep) if prep else GLYPH_W
    for f, p_ in zip(faces, prep):
        anc = {}
        for ln, pre in (corrections or {}).get(str(pno), {}).get(f["folio"] or "", {}).get("line_starts", {}).items():
            anc[int(ln) - 1] = [u for u in clusters(pre.replace(" ", "")) if u.strip()]
        fixed = {}
        for ln, txt in (corrections or {}).get(str(pno), {}).get(f["folio"] or "", {}).get("lines", {}).items():
            units = [u for u in transcript_units(re.sub(r"\d+", "", txt))]
            fixed[int(ln) - 1] = units
            anc[int(ln) - 1] = units[:3]
        f["anchors"] = anc
        p_["bounds"] = partition_by_height(p_["gold"], p_["spans"], p_["ns"], fits=p_["fits"], weights=w,
                                           anchors=anc or None, fixed=fixed or None)
    init_bounds = [list(p_["bounds"]) for p_ in prep]
    for f, p_, ib in zip(faces, prep, init_bounds):
        st = assign_reflowed(f["B"], f["top"], f["det"], f["title"], f["gold"], weights=w,
                             forced_bounds=[(int(a), int(b)) for a, b in p_["bounds"]])
        st["anchored_lines"] = sorted(k + 1 for k in f["anchors"])
        # 검수 권고: 상자 수·잘림 적합도가 배정 글자 수와 크게 어긋나는 행
        review = []
        for c_, ((a, b), col) in enumerate(zip(p_["bounds"], p_["cols"])):
            k = b - a
            if c_ in f["anchors"]:
                continue
            best_k = min(range(max(1, k - 2), k + 3), key=lambda kk: p_["fits"][c_](kk))
            if best_k != k and p_["fits"][c_](best_k) < 0.5 * p_["fits"][c_](k) and abs(col["n"] - best_k) < abs(col["n"] - k):
                review.append(c_ + 1)
        st["lines_to_review"] = review
        for rg in f["B"].regions:
            if rg["label"] == "TEXT_COLUMN":
                ln = [r for r in f["B"].regions if r["label"] == "TEXT_COLUMN"].index(rg) + 1
                rg["line_no"] = ln
                if ln in review:
                    rg["needs_review"] = True
                if (ln - 1) in f["anchors"]:
                    rg["verification"] = "line-start-anchored"
        st["glyph_weights"] = [round(float(v), 3) for v in w]
        results.append(f["B"].save(outdir, {"part": part, "folio": f["folio"], "alignment": st}))
    return results
