"""Tesseract 5 (LSTM) 래퍼: hOCR의 글자 단위 박스(ocrx_cinfo) 파싱."""
import os
import re
import subprocess
import tempfile
from html import unescape

import cv2
import numpy as np

TESSDATA = os.environ.get("HANOCR_TESSDATA") or os.path.join(os.path.dirname(__file__), "..", "tessdata")

_bbox_re = re.compile(r"(?:x_)?bboxe?s? (\d+) (\d+) (\d+) (\d+)")
_conf_re = re.compile(r"x_conf ([\d.]+)")
_span_re = re.compile(r"<span class='(ocr_line|ocr_textfloat|ocr_header|ocr_caption|ocrx_word|ocrx_cinfo)'[^>]*title=[\"']([^\"']*)[\"'][^>]*>([^<]*)")


def run_hocr(img, lang="kor+chi_tra", psm=3, extra=()):
    """img(np.ndarray) → hOCR 문자열."""
    with tempfile.TemporaryDirectory() as td:
        ip = os.path.join(td, "in.png")
        cv2.imwrite(ip, img)
        cmd = ["tesseract", ip, os.path.join(td, "out"), "-l", lang, "--psm", str(psm),
               "--tessdata-dir", os.path.abspath(TESSDATA),
               "-c", "hocr_char_boxes=1", "-c", "preserve_interword_spaces=1", *extra, "hocr"]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(os.path.join(td, "out.hocr"), encoding="utf-8") as f:
            return f.read()


def parse_hocr_chars(hocr):
    """hOCR → [{'text','bbox':[x0,y0,x1,y1],'conf','line','word'}] (픽셀 좌표)."""
    chars, line, word, wbox = [], -1, -1, None
    for m in _span_re.finditer(hocr):
        cls, title, text = m.group(1), m.group(2), unescape(m.group(3))
        if cls in ("ocr_line", "ocr_textfloat", "ocr_header", "ocr_caption"):
            line += 1
        elif cls == "ocrx_word":
            word += 1
            wb = _bbox_re.search(title)
            wbox = list(map(int, wb.groups())) if wb else None
        elif cls == "ocrx_cinfo":
            bb = _bbox_re.search(title)
            cf = _conf_re.search(title)
            if not bb or not text.strip():
                continue
            x0, y0, x1, y1 = map(int, bb.groups())
            chars.append({"text": text, "bbox": [x0, y0, x1, y1],
                          "conf": round(float(cf.group(1)) / 100.0, 3) if cf else None,
                          "line": line, "word": word, "word_bbox": wbox})
    return chars


def ocr_chars(img, lang="kor+chi_tra", psm=3, scale=1.0):
    """이미지 OCR → 글자 박스 목록 (원본 img 좌표로 환산)."""
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    chars = parse_hocr_chars(run_hocr(img, lang=lang, psm=psm))
    if scale != 1.0:
        for c in chars:
            c["bbox"] = [v / scale for v in c["bbox"]]
    return chars


def refine_boxes(chars, img, dark=None):
    """LSTM 글자 경계는 부정확하므로 단어 박스 내부의 세로 투영(잉크) 최소점으로 경계를 재조정하고
    각 글자의 잉크 영역으로 bbox를 조인다 (가로쓰기 인쇄체용)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if dark is None:
        dark = img < min(170, int(np.percentile(img, 60)) - 30)
    by_word = {}
    for c in chars:
        by_word.setdefault(c["word"], []).append(c)
    for w, cs in by_word.items():
        wb = cs[0].get("word_bbox")
        if not wb:
            continue
        x0, y0, x1, y1 = wb
        sub = dark[y0:y1 + 1, x0:x1 + 1]
        if sub.size == 0:
            continue
        prof = sub.sum(0).astype(float)
        n = len(cs)
        # 초기 경계: tesseract 글자 시작점
        cuts = [x0]
        for i in range(1, n):
            guess = int(round((cs[i - 1]["bbox"][2] + cs[i]["bbox"][0]) / 2))
            guess = int(round(cs[i]["bbox"][0])) if cs[i]["bbox"][0] > cs[i-1]["bbox"][0] else guess
            pitch = (x1 - x0) / n
            lo = max(cuts[-1] + 1, int(guess - 0.4 * pitch))
            hi = min(x1, int(guess + 0.4 * pitch))
            if hi <= lo:
                cuts.append(min(max(guess, cuts[-1] + 1), x1))
                continue
            seg = prof[lo - x0:hi - x0 + 1]
            # 동률일 때 guess에 가까운 쪽
            cand = np.flatnonzero(seg == seg.min()) + lo
            cuts.append(int(cand[np.argmin(np.abs(cand - guess))]))
        cuts.append(x1 + 1)
        for i, c in enumerate(cs):
            a, b = cuts[i], cuts[i + 1]
            box = dark[y0:y1 + 1, a:b]
            ys, xs = np.nonzero(box)
            if len(xs):
                c["bbox"] = [a + int(xs.min()), y0 + int(ys.min()), a + int(xs.max()) + 1, y0 + int(ys.max()) + 1]
            else:
                c["bbox"] = [a, y0, b, y1 + 1]
    return chars
