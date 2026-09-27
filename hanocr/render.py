"""페이지 렌더링 및 영역(region) 검출."""
import cv2
import numpy as np
import pymupdf as fitz


def render(page, dpi=300, gray=False):
    pix = page.get_pixmap(dpi=int(round(dpi)), colorspace=fitz.csGRAY if gray else fitz.csRGB, annots=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if gray:
        return img[:, :, 0].copy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def native_dpi(page):
    """이미지 전용 페이지(스캔본)의 원본 해상도 추정 → 과도한 업샘플링 방지."""
    infos = page.get_image_info()
    best = 0
    for b in infos:
        w_pt = b["bbox"][2] - b["bbox"][0]
        if w_pt > 0:
            best = max(best, b["width"] / w_pt * 72)
    return best


def pdf_image_regions(page, min_frac=0.08):
    """PDF 안에 배치된 큰 래스터 이미지의 영역(pt) — 사진/영인본 후보."""
    area = page.rect.width * page.rect.height
    regs = []
    for b in page.get_image_info():
        x0, y0, x1, y1 = b["bbox"]
        x0, y0 = max(x0, page.rect.x0), max(y0, page.rect.y0)
        x1, y1 = min(x1, page.rect.x1), min(y1, page.rect.y1)
        if (x1 - x0) * (y1 - y0) >= min_frac * area:
            regs.append([x0, y0, x1, y1])
    # 겹치는 박스 병합
    regs.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    out = []
    for r in regs:
        if not any(r[0] >= o[0] - 2 and r[1] >= o[1] - 2 and r[2] <= o[2] + 2 and r[3] <= o[3] + 2 for o in out):
            out.append(r)
    return out


def detect_facsimile_regions(img, min_frac=0.06):
    """전면 스캔 페이지에서 고문헌 사진(영인) 영역 검출 — 흰 지면 대비 '종이색/채도' 영역.
    반환: 픽셀 bbox 목록."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    # 지면(흰색/아주 옅은 톤) 이 아닌 픽셀 = 종이·사진 톤
    bg_v = np.percentile(v, 90)
    paper = ((s > 28) & (v > 60)) | (v < bg_v - 60)
    k = max(3, int(min(H, W) * 0.004))
    m = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    k2 = max(5, int(min(H, W) * 0.02))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k2, k2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    regs = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w * h < min_frac * H * W:
            continue
        fill = a / float(w * h)
        if fill < 0.55 or w < 0.12 * W or h < 0.12 * H:
            continue
        regs.append([int(x), int(y), int(x + w), int(y + h)])
    return regs
