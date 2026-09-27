"""고문헌 영인(사진) 영역의 세로쓰기 글자 검출.

단계
 1) 이진화 (Sauvola, 얼룩·배접 종이 대응)
 2) 광곽(테두리)·계선(행 구분선) 검출 및 제거 → 계선 위치는 행 경계 힌트로 사용
 3) 행(列) 분할 : 세로 투영 + 자기상관으로 행 간격(pitch) 추정
 4) 글자 분할 : 행 내부 가로 투영 blob 을 동적계획법(DP)으로 묶어 정사각형에 가까운 글자로 분할
 5) 쌍행 협주(작은 글씨 두 줄) 구간 검출 → 좌우 반행으로 다시 분할
 6) 읽기 순서 : 오른쪽 행 → 왼쪽 행, 위 → 아래 (협주는 오른쪽 반행 → 왼쪽 반행)
"""
import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def binarize(gray):
    h, w = gray.shape
    win = int(max(15, (min(h, w) // 25) | 1))
    t = threshold_sauvola(gray, window_size=win, k=0.2)
    ink = gray < t
    # 전체적으로 밝은(종이) 영역에서 옅은 얼룩 제거
    glob = gray < np.percentile(gray, 50) - 10
    ink &= glob | (gray < np.percentile(gray, 20))
    # 굵은 선·획 내부가 Sauvola에서 비는 현상 보완: 전역 Otsu 기준으로 확실히 어두운 픽셀 추가
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink |= gray < 0.85 * otsu
    return ink.astype(np.uint8)


def remove_lines(ink, vfrac=0.12, hfrac=0.12):
    h, w = ink.shape
    vk = max(15, int(h * vfrac))
    hk = max(15, int(w * hfrac))
    # 끊긴 선 연결 후 긴 선만 남김
    vc = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    hc = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    vert = cv2.morphologyEx(vc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    hori = cv2.morphologyEx(hc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    lines = cv2.dilate(vert | hori, np.ones((5, 5), np.uint8))
    clean = ink & (1 - lines)
    vx = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(vert, 8)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if hh > 0.3 * h and ww < 0.03 * w:
            vx.append((x + ww / 2.0, y, y + hh, ww))
    hy = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(hori, 8)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if ww > 0.3 * w and hh < 0.03 * h:
            hy.append((y + hh / 2.0, x, x + ww, hh))
    return clean, sorted(vx), sorted(hy)


def remove_line_remnants(ink, stroke):
    """광곽·계선 제거 후 남은 가늘고 긴 조각 제거."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.ones(n, bool)
    keep[0] = False
    for i in range(1, n):
        x, y, w, h, a = st[i]
        long_, short = max(w, h), max(1, min(w, h))
        if long_ / short > 8 and short <= 2.2 * stroke and long_ > 12 * stroke:
            keep[i] = False
    return keep[lab].astype(np.uint8)


def remove_specks(ink, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab].astype(np.uint8)


def _smooth(p, k):
    k = max(1, int(k))
    return np.convolve(p, np.ones(k) / k, mode="same")


def estimate_pitch(prof, lo, hi):
    p = prof - prof.mean()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    lo, hi = max(2, int(lo)), min(len(ac) - 1, int(hi))
    if hi <= lo:
        return None
    seg = ac[lo:hi]
    gmax = seg.max()
    if gmax <= 0:
        return None
    # 배수(harmonic) 대신 첫 번째 강한 국소 최대값 선택
    for i in range(1, len(seg) - 1):
        if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] >= 0.5 * gmax:
            return lo + i
    return lo + int(np.argmax(seg))


def find_columns(ink, vlines, frame):
    """행(세로줄) x 범위 목록 (오른쪽→왼쪽)."""
    x0f, y0f, x1f, y1f = frame
    sub = ink[y0f:y1f, x0f:x1f]
    W = sub.shape[1]
    prof = sub.sum(0).astype(float)
    pitch = estimate_pitch(_smooth(prof, 3), W * 0.03, W * 0.3) or W / 10
    # 1) 계선 기반: 내부 세로선이 충분히 규칙적이면 그대로 사용
    inner = [v[0] - x0f for v in vlines if x0f + 3 < v[0] < x1f - 3]
    cols = []
    if len(inner) >= 3:
        bounds = [0] + inner + [W]
        gaps = np.diff(bounds)
        med = np.median(gaps)
        if np.all(gaps > 0.4 * med):
            cols = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)
                    if prof[int(bounds[i]):int(bounds[i + 1])].sum() > 0]
            # 계선 간격이 한 행보다 넓으면(계선 누락) 아래 투영 방식으로 재분할
            cols2 = []
            for a, b in cols:
                if b - a > 1.7 * med:
                    cols2 += _split_by_profile(prof, a, b, med)
                else:
                    cols2.append((a, b))
            cols = cols2
            pitch = med
    # 2) 투영 기반
    if not cols:
        cols = _split_by_profile(prof, 0, W, pitch)
    cols = [(a + x0f, b + x0f) for a, b in cols]
    return sorted(cols, key=lambda c: -c[0]), pitch


def _split_by_profile(prof, a, b, pitch):
    seg = _smooth(prof[a:b], max(1, pitch * 0.12))
    if seg.max() <= 0:
        return []
    thr = 0.12 * np.median(seg[seg > 0]) if np.any(seg > 0) else 0
    on = seg > thr
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    # 너무 가까운 run 병합 (한 글자의 좌우 부분이 떨어진 경우)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.12 * pitch and (r[1] - merged[-1][0]) < 1.25 * pitch:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for s, e in merged:
        w = e - s
        if w < 0.2 * pitch:
            continue
        k = int(round(w / pitch))
        if k >= 2:  # 붙은 행 분리
            cuts = [s]
            for m in range(1, k):
                g = s + m * w / k
                lo, hi = int(g - 0.3 * pitch), int(g + 0.3 * pitch)
                lo, hi = max(lo, s + 1), min(hi, e - 1)
                c = lo + int(np.argmin(seg[lo:hi])) if hi > lo else int(g)
                cuts.append(c)
            cuts.append(e)
            out += [(cuts[m], cuts[m + 1]) for m in range(k)]
        else:
            out.append((s, e))
    return [(a + s, a + e) for s, e in out]


def _blobs(rowprof, min_gap=1):
    on = rowprof > 0
    runs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    out = []
    for r in runs:
        if out and r[0] - out[-1][1] < min_gap:
            out[-1][1] = r[1]
        else:
            out.append(r)
    return out


def segment_chars_1d(rowprof, H, allow_small=True, max_ratio=1.45):
    """세로 1차원 잉크 투영에서 글자 구간 [(y0,y1)] 을 DP로 결정."""
    blobs = _blobs(rowprof)
    if not blobs:
        return []
    # 너무 긴 blob(붙은 글자) 은 투영 최소점에서 미리 분리
    sm = _smooth(rowprof.astype(float), max(1, H * 0.08))
    pieces = []
    for s, e in blobs:
        h = e - s
        k = int(round(h / H))
        if h > max_ratio * H and k >= 2:
            cuts = [s]
            for m in range(1, k):
                g = s + m * h / k
                lo, hi = int(max(s + 1, g - 0.3 * H)), int(min(e - 1, g + 0.3 * H))
                cuts.append(lo + int(np.argmin(sm[lo:hi])) if hi > lo else int(g))
            cuts.append(e)
            pieces += [[cuts[m], cuts[m + 1]] for m in range(k)]
        else:
            pieces.append([s, e])
    n = len(pieces)
    INF = 1e18
    dp = [0.0] + [INF] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j, 0, -1):
            top, bot = pieces[i - 1][0], pieces[j - 1][1]
            h = bot - top
            if h > max_ratio * H and i < j:
                break
            cost_h = ((h - H) / H) ** 2
            if allow_small:
                cost_h = min(cost_h, ((h - 0.5 * H) / (0.5 * H)) ** 2 * 0.6 + 0.25)
            # 내부 공백이 크면 병합 억제
            gmax = max([pieces[k][0] - pieces[k - 1][1] for k in range(i, j)] or [0])
            cost = cost_h + 2.0 * (gmax / H) ** 2 + 0.05
            if dp[i - 1] + cost < dp[j]:
                dp[j] = dp[i - 1] + cost
                back[j] = i - 1
    segs, j = [], n
    while j > 0:
        i = back[j]
        segs.append((pieces[i][0], pieces[j - 1][1]))
        j = i
    return segs[::-1]


def _tight(ink, x0, y0, x1, y1):
    sub = ink[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return None
    return [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]


def _is_double_line(ink, box, colw):
    """글자 상자가 쌍행 협주(좌·우 작은 글씨 두 줄)인지: 가운데 세로 공백 + 양쪽 잉크."""
    x0, y0, x1, y1 = box
    sub = ink[y0:y1, x0:x1]
    if sub.size == 0 or (x1 - x0) < 0.5 * colw:
        return None
    vp = sub.sum(0).astype(float)
    w = len(vp)
    mid = vp[int(w * 0.3):int(w * 0.7)]
    if len(mid) == 0:
        return None
    k = int(np.argmin(mid)) + int(w * 0.3)
    left, right = vp[:k].sum(), vp[k:].sum()
    if vp[k] <= 0.02 * vp.max() and left > 0.2 * vp.sum() and right > 0.2 * vp.sum():
        # 공백 폭 확인
        gap = np.flatnonzero(vp[int(w * 0.2):int(w * 0.8)] <= 0.02 * vp.max())
        if len(gap) >= max(1, 0.04 * w):
            return x0 + k
    return None


def detect_chars(bgr, frame=None, char_h=None):
    """영인 이미지(BGR)에서 글자 박스 검출. 반환: dict(chars=[...], columns=[...], frame=..., pitch=...)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    H, W = gray.shape
    ink0 = binarize(gray)
    ink, vlines, hlines = remove_lines(ink0)
    # 광곽(테두리) 추정: 가장 바깥 긴 선
    if frame is None:
        long_v = [v for v in vlines if (v[2] - v[1]) > 0.25 * H]
        long_h = [h for h in hlines if (h[2] - h[1]) > 0.25 * W]
        fx0 = int(min(v[0] for v in long_v)) if len(long_v) >= 2 else 0
        fx1 = int(max(v[0] for v in long_v)) if len(long_v) >= 2 else W
        fy0 = int(min(h[0] for h in long_h)) if len(long_h) >= 2 else 0
        fy1 = int(max(h[0] for h in long_h)) if len(long_h) >= 2 else H
        if fx1 - fx0 < 0.3 * W:
            fx0, fx1 = 0, W
        if fy1 - fy0 < 0.3 * H:
            fy0, fy1 = 0, H
        frame = [fx0, fy0, fx1, fy1]
    # 잡티 제거 (획 두께 기준)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = max(1.0, 2 * np.percentile(dist[dist > 0], 75)) if np.any(dist > 0) else 2
    ink = remove_specks(ink, min_area=max(4, int(stroke * stroke * 1.2)))
    ink_all = remove_line_remnants(ink.copy(), stroke)
    mask = np.zeros_like(ink)
    fx0, fy0, fx1, fy1 = frame
    pad = int(max(4, 0.012 * min(H, W), 2.5 * stroke))
    mask[fy0 + pad:fy1 - pad, fx0 + pad:fx1 - pad] = 1
    ink &= mask
    ink = remove_line_remnants(ink, stroke)
    cols, pitch = find_columns(ink, vlines, frame)
    # 1차: 행별 글자 높이 추정 → 쪽 전체 중앙값 (협주·좁은 행 때문에 흔들리는 추정 보정)
    hs = []
    for (a, b) in cols:
        vp = ink[fy0:fy1, a:b].sum(0)
        nz = np.flatnonzero(vp)
        if len(nz):
            hs.append(max(0.75 * (b - a), min(nz[-1] - nz[0] + 1, b - a) * 0.95))
    H_page = float(np.median(hs)) if hs else None
    chars, columns = [], []
    for ci, (a, b) in enumerate(cols):
        strip = ink[fy0:fy1, a:b]
        vp = strip.sum(0)
        nz = np.flatnonzero(vp)
        if len(nz) == 0:
            continue
        textw = nz[-1] - nz[0] + 1
        colw = b - a
        Hc = char_h or max(0.75 * colw, min(textw, colw) * 0.95)
        if H_page and not char_h and abs(Hc - H_page) > 0.12 * H_page:
            Hc = H_page
        rp = strip.sum(1)
        segs = segment_chars_1d(rp, Hc)
        col_chars = []
        for (s, e) in segs:
            box = _tight(ink, a, fy0 + s, b, fy0 + e)
            if box is None:
                continue
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw * bh < (0.08 * Hc) ** 2:  # 점 · 잡티
                continue
            split = _is_double_line(ink, box, colw)
            if split is not None:
                col_chars.append({"bbox": box, "split": split, "small": True})
            else:
                col_chars.append({"bbox": box, "small": bool(bh < 0.55 * Hc and bw < 0.75 * colw)})
        # 연속한 쌍행 구간 → 좌·우 반행을 각각 작은 글씨로 재분할
        out, i = [], 0
        while i < len(col_chars):
            c = col_chars[i]
            if "split" not in c:
                out.append(c)
                i += 1
                continue
            j = i
            while j < len(col_chars) and "split" in col_chars[j]:
                j += 1
            zy0 = min(x["bbox"][1] for x in col_chars[i:j])
            zy1 = max(x["bbox"][3] for x in col_chars[i:j])
            if (zy1 - zy0) < 1.6 * Hc or (j - i) < 2:
                # 좌우 자모가 떨어진 한 글자(예: 이, 라) → 협주 아님
                for x in col_chars[i:j]:
                    x.pop("split")
                    bh = x["bbox"][3] - x["bbox"][1]
                    x["small"] = bool(bh < 0.55 * Hc)
                    out.append(x)
                i = j
                continue
            mid = int(np.median([x["split"] for x in col_chars[i:j]]))
            for (xa, xb, side) in ((mid, b, "R"), (a, mid, "L")):
                sub = ink[zy0:zy1, xa:xb]
                segs2 = segment_chars_1d(sub.sum(1), Hc * 0.5, allow_small=False)
                for (s, e) in segs2:
                    bx = _tight(ink, xa, zy0 + s, xb, zy0 + e)
                    if bx and (bx[2] - bx[0]) * (bx[3] - bx[1]) >= (0.05 * Hc) ** 2:
                        out.append({"bbox": bx, "small": True, "subline": side})
            i = j
        for k, c in enumerate(out):
            c["column"] = len(columns)
            c["order_in_column"] = k
            chars.append(c)
        columns.append({"id": len(columns), "x0": int(a), "x1": int(b), "y0": int(fy0), "y1": int(fy1),
                        "n_chars": len(out)})
    # ---- 판심(광곽 바깥 여백의 서명·권차·장차) ----
    main_h = np.median([c["bbox"][3] - c["bbox"][1] for c in chars if not c.get("small")] or [pitch])
    margin = []
    for side, (xa, xb) in (("L", (0, fx0)), ("R", (fx1, W))):
        margin += _margin_chars(ink_all, xa, xb, pad, main_h, pitch, side)
    ink_out = ink | ink_all * _margin_mask(ink.shape, margin)
    return {"chars": chars, "columns": columns, "frame": [int(v) for v in frame], "pitch": float(pitch),
            "ink": ink_out, "margin_chars": margin}


def _margin_mask(shape, margin):
    m = np.zeros(shape, np.uint8)
    for c in margin:
        x0, y0, x1, y1 = c["bbox"]
        m[y0:y1, x0:x1] = 1
    return m


def _margin_chars(ink, xa, xb, pad, main_h, pitch, side):
    """광곽 바깥 세로 띠에서 글자열(판심 제목·장차 등) 검출."""
    m = max(3, pad // 3)
    xa, xb = (xa, xb - m) if side == "L" else (xa + m, xb)
    if xb - xa < 0.3 * pitch:
        return []
    strip = ink[:, xa:xb]
    vp = _smooth(strip.sum(0).astype(float), 3)
    if vp.max() < 2 * main_h * 0.05:
        return []
    # 가장 잉크가 많은 세로 띠(한 행 폭)를 선택
    w = int(max(3, min(pitch, xb - xa)))
    acc = np.convolve(vp, np.ones(w), mode="valid")
    c0 = int(np.argmax(acc))
    lo, hi = c0, c0 + w
    # 좌우로 잉크가 이어지면 확장
    while lo > 0 and vp[lo - 1] > 0.05 * vp.max():
        lo -= 1
    while hi < len(vp) and vp[hi] > 0.05 * vp.max():
        hi += 1
    x0, x1 = xa + lo, xa + hi
    rp = ink[:, x0:x1].sum(1)
    textw = x1 - x0
    Hc = max(0.6 * main_h, min(main_h, textw * 0.95))
    segs = segment_chars_1d(rp, Hc, allow_small=False)
    out = []
    for (s0, e0) in segs:
        box = _tight(ink, x0, s0, x1, e0)
        if box is None:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw * bh < (0.2 * Hc) ** 2 or bh > 2.2 * Hc or bw > 1.6 * Hc:
            continue
        out.append({"bbox": box, "small": False, "margin": side})
    # 글자 사이 큰 공백으로 묶음(segment) 구분: 서명 / 권차 / 장차
    seg = 0
    for i, c in enumerate(out):
        if i and c["bbox"][1] - out[i - 1]["bbox"][3] > 1.2 * Hc:
            seg += 1
        c["segment"] = seg
        c["order_in_column"] = i
    # 글자가 1~2개뿐이고 고립된 조각은 얼룩일 가능성 → 그래도 남기되 표시
    return out


def detect_margins(bgr, frame, main_h, pitch):
    """주어진 광곽(frame) 바깥 좌우 여백의 글자(판심·난외 표기) 검출."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    H, W = gray.shape
    ink0 = binarize(gray)
    ink, _, _ = remove_lines(ink0)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = max(1.0, 2 * np.percentile(dist[dist > 0], 75)) if np.any(dist > 0) else 2
    ink = remove_specks(ink, min_area=max(4, int(stroke * stroke * 1.2)))
    ink = remove_line_remnants(ink, stroke)
    fx0, fy0, fx1, fy1 = frame
    pad = int(max(4, 0.012 * min(H, W), 2.5 * stroke))
    out = []
    for side, (xa, xb) in (("L", (0, fx0)), ("R", (fx1, W))):
        out += _margin_chars(ink, xa, xb, pad, main_h, pitch, side)
    return out, ink
