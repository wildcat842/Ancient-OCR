"""텍스트 레이어를 정답(Ground Truth)으로 삼아 래스터 OCR(tesseract 엔진)의 글자 박스 정확도를 측정.

born-digital 쪽을 이미지로 렌더링 → OCR → 정답 글자의 잉크 bbox 와 IoU 매칭.
지표: 검출 재현율(recall@IoU≥0.5), 정밀도, 매칭 글자 중 문자 일치율, 평균 IoU, 글자 종류별 정확도."""
import collections
import json
import sys

import numpy as np
import pymupdf as fitz

from .render import render, pdf_image_regions
from .tess import ocr_chars, refine_boxes
from .textlayer import extract_chars, tighten
from .pipeline import _inside


def iou_matrix(A, B):
    A, B = np.asarray(A, float)[:, None, :], np.asarray(B, float)[None, :, :]
    ix = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]), 0, None)
    iy = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]), 0, None)
    inter = ix * iy
    ua = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1]) + (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def eval_page(doc, pno, dpi=300, lang="kor+chi_tra"):
    page = doc[pno]
    regs = pdf_image_regions(page)
    gt = [c for c in tighten(extract_chars(page), page, dpi=dpi)
          if c.get("ink_bbox") and not any(_inside(c["bbox"], r) for r in regs)]
    img = render(page, dpi=dpi, gray=True)
    s = dpi / 72.0
    for r in regs:
        x0, y0, x1, y1 = [int(v * s) for v in (r[0] - page.rect.x0, r[1] - page.rect.y0,
                                                 r[2] - page.rect.x0, r[3] - page.rect.y0)]
        img[max(0, y0):y1, max(0, x0):x1] = 255
    pred = refine_boxes(ocr_chars(img, lang=lang), img)
    for p in pred:
        p["bbox_pt"] = [page.rect.x0 + p["bbox"][0] / s, page.rect.y0 + p["bbox"][1] / s,
                        page.rect.x0 + p["bbox"][2] / s, page.rect.y0 + p["bbox"][3] / s]
    if not gt or not pred:
        return None
    M = iou_matrix([g["ink_bbox"] for g in gt], [p["bbox_pt"] for p in pred])
    # 탐욕적 1:1 매칭
    pairs, used_g, used_p = [], set(), set()
    for idx in np.argsort(-M, axis=None):
        gi, pi = divmod(int(idx), M.shape[1])
        if M[gi, pi] < 0.5:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi); pairs.append((gi, pi, M[gi, pi]))
    by_kind = collections.defaultdict(lambda: [0, 0, 0])  # total, detected, correct
    for g in gt:
        by_kind[g["kind"]][0] += 1
    correct = 0
    for gi, pi, _ in pairs:
        k = gt[gi]["kind"]
        by_kind[k][1] += 1
        ok = gt[gi]["text"] == pred[pi]["text"]
        by_kind[k][2] += ok
        correct += ok
    return {"page": pno + 1, "gt_chars": len(gt), "pred_chars": len(pred),
            "recall@0.5": round(len(pairs) / len(gt), 4), "precision@0.5": round(len(pairs) / len(pred), 4),
            "mean_iou": round(float(np.mean([p[2] for p in pairs])) if pairs else 0, 4),
            "char_acc_on_matched": round(correct / len(pairs), 4) if pairs else 0,
            "end_to_end_acc": round(correct / len(gt), 4),
            "by_kind": {k: {"total": v[0], "detected": v[1], "correct": v[2]} for k, v in by_kind.items()}}


def main():
    pdf, pages = sys.argv[1], [int(x) - 1 for x in sys.argv[2].split(",")]
    doc = fitz.open(pdf)
    out = [eval_page(doc, p) for p in pages]
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
