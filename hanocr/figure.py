"""도판(지도·천체도·삽화) 안의 작은 표기 글자 검출.

1) 선 성분 분류 : 긴 곡선·직선(궤도, 해안선, 경계) / 채워진 점(행성·지명 기호) / 작은 점(위성) / 글자 획
2) 글자 획을 세로쓰기 방향으로 묶어 표기(label) 후보를 만들고, 폭이 넓으면 세로 띠로 다시 나눔
3) 후보마다 글자 조각(성분 bbox)을 보관 → 판독문의 표기 목록과 짝지은 뒤 글자 수에 맞춰 분할
"""
import cv2
import numpy as np


def _classify(ink, ch, stroke):
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    cls = np.zeros(n, np.int8)  # 0 무시 1 글자획 2 기호점 3 선 4 작은점
    for i in range(1, n):
        x, y, w, h, a = st[i]
        fill = a / float(w * h)
        if a < 6:
            continue
        pts = np.column_stack(np.nonzero(lab[y:y + h, x:x + w] == i))[:, ::-1].astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        L, T = max(rw, rh), max(1.0, min(rw, rh))
        if L > 1.4 * ch or max(w, h) > 2.2 * ch:
            cls[i] = 3
            continue
        # 대각선 방향의 가늘고 긴 조각 = 곡선 파편 (한글 획은 대부분 수평·수직)
        diag = min(abs(ang % 90), 90 - abs(ang % 90)) > 18
        if L > 0.8 * ch and T < 2.2 * stroke and diag:
            cls[i] = 3
            continue
        if fill > 0.55 and 0.7 < w / float(h) < 1.4 and max(w, h) > 0.35 * ch:
            m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            per = sum(cv2.arcLength(c, True) for c in cs) + 1e-6
            circ = 4 * np.pi * a / per ** 2
            if circ > 0.72 and fill > 0.65:  # 원형 기호점(행성·지명 표시)
                cls[i] = 2
                continue
        if max(w, h) < 0.28 * ch and fill > 0.45:
            cls[i] = 4
            continue
        cls[i] = 1
    return n, lab, st, cen, cls


def _split_columns(boxes, ch):
    """세로 표기 후보의 폭이 넓으면 x 투영 공백으로 다시 나눔."""
    if not boxes:
        return []
    x0 = min(b[0] for b in boxes); x1 = max(b[2] for b in boxes)
    if x1 - x0 < 1.5 * ch:
        return [boxes]
    prof = np.zeros(x1 - x0 + 1)
    for b in boxes:
        prof[b[0] - x0:b[2] - x0] += 1
    gaps = np.flatnonzero(prof == 0)
    if len(gaps) == 0:
        return [boxes]
    # 연속 공백의 가운데를 경계로
    cuts, run = [], [gaps[0]]
    for g in gaps[1:]:
        if g == run[-1] + 1:
            run.append(g)
        else:
            cuts.append((run[0] + run[-1]) // 2 + x0); run = [g]
    cuts.append((run[0] + run[-1]) // 2 + x0)
    groups = {}
    for b in boxes:
        k = sum(((b[0] + b[2]) / 2) > c for c in cuts)
        groups.setdefault(k, []).append(b)
    return [v for _, v in sorted(groups.items(), key=lambda kv: -kv[0])]  # 오른쪽 띠부터


def detect_labels(bgr_or_ink, box, ch, stroke=4.0, is_ink=True):
    """box 안 도판의 표기 후보. ch = 표기 글자 크기(px) 추정치."""
    x0, y0, x1, y1 = box
    ink = bgr_or_ink[y0:y1, x0:x1]
    n, lab, st, cen, cls = _classify(ink, ch, stroke)
    markers = [[int(st[i][0]), int(st[i][1]), int(st[i][0] + st[i][2]), int(st[i][1] + st[i][3])]
               for i in range(1, n) if cls[i] == 2]
    # 선에 붙은 기호점: 굵은 원반만 남기는 opening
    r = max(2, int(round(1.6 * stroke)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    blobs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, disk)
    nb, bl, bs, _ = cv2.connectedComponentsWithStats(blobs, 8)
    for i in range(1, nb):
        x, y, w, h, a = bs[i]
        if 0.25 * ch < max(w, h) < 1.2 * ch and 0.6 < w / float(h) < 1.6 and a / float(w * h) > 0.6:
            b = [int(x), int(y), int(x + w), int(y + h)]
            if not any(abs(b[0] - m[0]) < 5 and abs(b[1] - m[1]) < 5 for m in markers):
                markers.append(b)
                # 이 기호점을 포함하는 성분이 글자로 분류됐으면 기호로 변경
                cy, cx = y + h // 2, x + w // 2
                j = lab[cy, cx]
                if j and cls[j] == 1 and st[j][2] < 1.3 * w and st[j][3] < 1.3 * h:
                    cls[j] = 2
    # 기호점 주변 작은 점 = 위성 등 부속 기호
    for i in range(1, n):
        if cls[i] == 4:
            c = cen[i]
            if any(np.hypot(c[0] - (m[0] + m[2]) / 2, c[1] - (m[1] + m[3]) / 2) < 1.6 * ch for m in markers):
                cls[i] = 5
            else:
                cls[i] = 1  # 아래아(ㆍ) 등 글자의 점일 수 있음
    textmask = (cls[lab] == 1).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.3 * ch)), max(5, int(0.75 * ch))))
    grp = cv2.dilate(textmask, k)
    ng, gl, gs, _ = cv2.connectedComponentsWithStats(grp, 8)
    comp_group = {}
    for i in range(1, n):
        if cls[i] == 1:
            cy, cx = int(cen[i][1]), int(cen[i][0])
            g = gl[min(cy, gl.shape[0] - 1), min(cx, gl.shape[1] - 1)]
            if g == 0:
                ys, xs = np.nonzero(lab == i)
                g = gl[ys[0], xs[0]]
            comp_group.setdefault(g, []).append([int(st[i][0]), int(st[i][1]),
                                                 int(st[i][0] + st[i][2]), int(st[i][1] + st[i][3])])
    cands = []
    for g, comps in comp_group.items():
        for sub in _split_columns(comps, ch):
            bx = [min(b[0] for b in sub), min(b[1] for b in sub), max(b[2] for b in sub), max(b[3] for b in sub)]
            w, h = bx[2] - bx[0], bx[3] - bx[1]
            thin = all(min(b[2] - b[0], b[3] - b[1]) < 2.2 * stroke for b in sub)
            if (w < 0.45 * ch and h < 0.6 * ch) or (thin and len(sub) < 3) or w * h < (0.35 * ch) ** 2:
                continue
            cands.append({"bbox": [bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0],
                          "components": [[b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0] for b in sub]})
    symbols = [{"type": "marker", "bbox": [m[0] + x0, m[1] + y0, m[2] + x0, m[3] + y0]} for m in markers]
    for i in range(1, n):
        if cls[i] == 5:
            symbols.append({"type": "satellite_dot",
                            "bbox": [int(st[i][0] + x0), int(st[i][1] + y0),
                                     int(st[i][0] + st[i][2] + x0), int(st[i][1] + st[i][3] + y0)]})
    # 세로로 이어진 후보를 합친 '결합 후보' 추가 (표기가 선에 끊겨 두 조각이 된 경우)
    base = list(cands)
    for i, a in enumerate(base):
        for j, b in enumerate(base):
            if i == j or b["bbox"][1] < a["bbox"][1]:
                continue
            ov = min(a["bbox"][2], b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0])
            gap = b["bbox"][1] - a["bbox"][3]
            if ov > 0.3 * min(a["bbox"][2] - a["bbox"][0], b["bbox"][2] - b["bbox"][0]) and -0.2 * ch < gap < 1.0 * ch:
                cands.append({"bbox": [min(a["bbox"][0], b["bbox"][0]), a["bbox"][1],
                                       max(a["bbox"][2], b["bbox"][2]), max(a["bbox"][3], b["bbox"][3])],
                              "components": a["components"] + b["components"], "parts": [i, j]})
    for i, c in enumerate(cands):
        c.setdefault("parts", [i])
        c["id"] = i
    full = np.zeros(bgr_or_ink.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = ((cls[lab] == 1) | (cls[lab] == 4)).astype(np.uint8)
    return cands, symbols, full


def trim_dots(components, ch):
    """후보 위·아래 끝의 고립된 작은 점(위성 점 등) 제거."""
    comps = sorted(components, key=lambda b: b[1])
    def tiny(b):
        return max(b[2] - b[0], b[3] - b[1]) < 0.3 * ch
    while len(comps) > 1 and tiny(comps[0]) and comps[1][1] - comps[0][3] > 0.1 * ch:
        comps.pop(0)
    while len(comps) > 1 and tiny(comps[-1]) and comps[-1][1] - comps[-2][3] > 0.1 * ch:
        comps.pop()
    return comps


def split_k(components, k):
    """성분 bbox 들을 세로 순서로 k 글자로 분할 (행 겹침 → 덩어리 → DP)."""
    comps = sorted(components, key=lambda b: b[1])
    blobs = []
    for b in comps:  # y 범위가 겹치면 같은 덩어리 (좌우로 놓인 자모)
        if blobs and b[1] < blobs[-1][3] - 0.25 * (b[3] - b[1]):
            o = blobs[-1]
            blobs[-1] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
        else:
            blobs.append(list(b))
    n = len(blobs)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return blobs
    span = blobs[-1][3] - blobs[0][1]
    hbar = span / k
    INF = 1e18
    D = np.full((n + 1, k + 1), INF); B = np.zeros((n + 1, k + 1), int); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for a in range(j - 1, i):
                if D[a, j - 1] >= INF:
                    continue
                h = blobs[i - 1][3] - blobs[a][1]
                gap_in = max([blobs[t][1] - blobs[t - 1][3] for t in range(a + 1, i)] or [0])
                c = ((h - hbar) / hbar) ** 2 + 2 * max(0, gap_in) / hbar
                if D[a, j - 1] + c < D[i, j]:
                    D[i, j] = D[a, j - 1] + c; B[i, j] = a
    out, i, j = [], n, k
    while j > 0:
        a = B[i, j]
        grp = blobs[a:i]
        out.append([min(b[0] for b in grp), min(b[1] for b in grp), max(b[2] for b in grp), max(b[3] for b in grp)])
        i, j = a, j - 1
    return out[::-1]
