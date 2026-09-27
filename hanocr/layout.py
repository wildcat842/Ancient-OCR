"""영인 이미지의 판면 구조 분석: 광곽(테두리) 블록 검출과 블록 유형(본문/도판) 판별."""
import cv2
import numpy as np

from .facsimile import binarize


def _rects_from_lines(ink):
    H, W = ink.shape
    k = int(max(40, 0.06 * min(H, W)))
    vc = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    hc = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    v = cv2.morphologyEx(vc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    h = cv2.morphologyEx(hc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    lines = cv2.dilate(v | h, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        x, y, w, hh = cv2.boundingRect(c)
        if w * hh < 0.03 * H * W or w < 0.15 * W or hh < 0.08 * H:
            continue
        # 사각형 윤곽인지: 테두리 네 변에 선 픽셀이 충분한가
        band = 12
        sides = [lines[y:y + band, x:x + w], lines[y + hh - band:y + hh, x:x + w],
                 lines[y:y + hh, x:x + band], lines[y:y + hh, x + w - band:x + w]]
        cov = [s.any(axis=0).mean() if i < 2 else s.any(axis=1).mean() for i, s in enumerate(sides)]
        if min(cov) < 0.7:
            continue
        rects.append([x, y, x + w, y + hh])
    return rects


def _group_nested(rects, tol=0.06):
    """이중 광곽(바깥선·안선)을 하나의 블록으로 묶음."""
    rects = sorted(rects, key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    blocks = []
    for r in rects:
        placed = False
        for b in blocks:
            o = b["outer"]
            inside = r[0] >= o[0] - 5 and r[1] >= o[1] - 5 and r[2] <= o[2] + 5 and r[3] <= o[3] + 5
            near = max(r[0] - o[0], r[1] - o[1], o[2] - r[2], o[3] - r[3]) < tol * min(o[2] - o[0], o[3] - o[1])
            if inside and near:
                b["inner"] = [max(b["inner"][0], r[0]), max(b["inner"][1], r[1]),
                              min(b["inner"][2], r[2]), min(b["inner"][3], r[3])]
                placed = True
                break
            if inside:  # 블록 안의 다른 블록(표 칸 등)은 무시
                placed = True
                break
        if not placed:
            blocks.append({"outer": list(r), "inner": list(r)})
    return blocks


def classify_block(ink, box, stroke):
    """본문(text) / 도판(figure) 판별: 긴 곡선(궤도·경계선 등) 성분이 많으면 도판."""
    x0, y0, x1, y1 = box
    pad = int(0.02 * min(x1 - x0, y1 - y0)) + 6
    sub = ink[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
    if sub.size == 0:
        return "text", {}
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    size = min(sub.shape)
    curve_ink = 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) > 0.25 * size and a / float(w * h) < 0.12:
            curve_ink += a
    total = sub.sum() + 1
    frac = curve_ink / total
    return ("figure" if frac > 0.15 else "text"), {"curve_ink_frac": round(float(frac), 3)}


def detect_blocks(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    ink = binarize(gray)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = 2 * np.percentile(dist[dist > 0], 75) if np.any(dist > 0) else 3
    blocks = _group_nested(_rects_from_lines(ink))
    for b in blocks:
        b["kind"], b["stats"] = classify_block(ink, b["inner"], stroke)
    # 읽기 순서: 위→아래, 같은 높이면 오른쪽→왼쪽
    blocks.sort(key=lambda b: (round(b["outer"][1] / (0.1 * gray.shape[0])), -b["outer"][0]))
    return blocks, ink, stroke
