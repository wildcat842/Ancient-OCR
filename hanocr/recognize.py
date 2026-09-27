"""영인 글자 박스 일괄 인식: 글자 조각을 격자로 붙여 Tesseract를 한 번만 호출하고,
hOCR 글자 좌표를 원래 조각에 되돌려 매핑한다."""
import cv2
import numpy as np

from .tess import run_hocr, parse_hocr_chars

CELL = 64      # 정규화 글자 높이(px)
GAP = 44       # 조각 사이 여백
PER_LINE = 24


def _norm_crop(ink, box):
    x0, y0, x1, y1 = box
    sub = ink[y0:y1, x0:x1]
    if sub.size == 0:
        return np.full((CELL, CELL), 255, np.uint8)
    img = np.where(sub > 0, 0, 255).astype(np.uint8)
    h, w = img.shape
    s = CELL / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((CELL, CELL), 255, np.uint8)
    oy, ox = (CELL - img.shape[0]) // 2, (CELL - img.shape[1]) // 2
    out[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    return out


def recognize(ink, boxes, lang="kor+chi_tra", engine="tesseract", hrcnet_model=None):
    """글자 조각 일괄 인식.
    engine: "tesseract" | "hrcnet" | "none"
      - tesseract: Tesseract 5 (기본)
      - hrcnet: HRCNet 모델 객체(hrcnet_model)에 위임 — GPU 환경에서만 사용
      - none: 인식 없이 빈 문자열 반환 (상자 검출만 필요할 때)
    """
    if engine == "none":
        return [("", 0.0)] * len(boxes)
    if engine == "hrcnet":
        if hrcnet_model is None:
            raise ValueError("hrcnet 엔진을 쓰려면 hrcnet_model 인자가 필요합니다.")
        return hrcnet_model.predict_batch(ink, boxes)
    # engine == "tesseract" (기본)
    """boxes: [[x0,y0,x1,y1],...] (ink 이미지 좌표) → [(text, conf)] 같은 순서."""
    if not boxes:
        return []
    crops = [_norm_crop(ink, b) for b in boxes]
    nlines = (len(crops) + PER_LINE - 1) // PER_LINE
    Wd = GAP + PER_LINE * (CELL + GAP)
    Hd = GAP + nlines * (CELL + 2 * GAP)
    sheet = np.full((Hd, Wd), 255, np.uint8)
    slots = []
    for i, c in enumerate(crops):
        r, k = divmod(i, PER_LINE)
        x, y = GAP + k * (CELL + GAP), GAP + r * (CELL + 2 * GAP)
        sheet[y:y + CELL, x:x + CELL] = c
        slots.append((x, y, x + CELL, y + CELL))
    chars = parse_hocr_chars(run_hocr(sheet, lang=lang, psm=6))
    res = [("", 0.0)] * len(boxes)
    # LSTM 글자 박스는 옆 조각으로 밀리는 일이 있어, 정확한 '단어' 박스 중심으로 조각을 찾는다
    words = {}
    for ch in chars:
        words.setdefault(ch["word"], []).append(ch)
    for w, chs in words.items():
        wb = chs[0].get("word_bbox") or chs[0]["bbox"]
        # 단어가 여러 조각에 걸치면 글자 박스 중심으로 개별 배정
        span = (wb[2] - wb[0]) > 1.6 * CELL
        for ch in chs:
            b = ch["bbox"] if span else wb
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            r = int((cy - GAP) // (CELL + 2 * GAP))
            k = int((cx - GAP / 2) // (CELL + GAP))
            i = r * PER_LINE + k
            if not (0 <= i < len(boxes) and 0 <= k < PER_LINE):
                continue
            conf = ch["conf"] or 0.0
            if not res[i][0] or conf > res[i][1]:
                res[i] = (ch["text"], conf)
    return res
