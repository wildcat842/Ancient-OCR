"""정답 글자 예시 은행(OCR RAG)과 예시 검색 기반 행 안 글자 위치 찾기 (라운드 1 → 2).

- 은행: 사람이 확정·수정한 글자 조각(gold)을 필체·자료별로 모아 특징 벡터로 저장
- 검색: 판독문에서 '이 행에 어떤 글자들이 어떤 순서로 오는지' 이미 알므로,
  행 띠를 잘라 보며 각 조각이 그 자리 글자의 예시들과 얼마나 닮았는지를 DP 비용에 더한다.
  예시가 없는 글자는 높이 사전값과 잉크 잘림만으로 판단(기존 방식).
"""
import cv2
import numpy as np

S = 32


def glyph_feature(ink_crop):
    """이진 잉크 조각 → 32×32 정규화 벡터 (잉크 경계로 조인 뒤 비율 유지 축소, 흐림, 평균 0·길이 1)."""
    ys, xs = np.nonzero(ink_crop)
    if len(xs) < 8:
        return None
    sub = ink_crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = sub.shape
    sc = S / max(h, w)
    sub = cv2.resize(sub, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
    out = np.zeros((S, S), np.float32)
    oy, ox = (S - sub.shape[0]) // 2, (S - sub.shape[1]) // 2
    out[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
    out = cv2.GaussianBlur(out, (5, 5), 1.3).ravel()
    out -= out.mean()
    n = np.linalg.norm(out)
    return out / n if n > 0 else None


class ExemplarBank:
    def __init__(self):
        self.vecs = {}      # 글자 → [벡터]
        self.meta = {}

    def add(self, text, vec, meta=None):
        if vec is None or not text:
            return
        self.vecs.setdefault(text, []).append(vec)
        self.meta.setdefault(text, []).append(meta)

    def add_from_image(self, ink, chars):
        """chars: [{'text','bbox'}] (ink 이미지 좌표). 약간의 변형(±2px 이동, ±6% 크기)으로 예시를 늘림."""
        H, W = ink.shape
        for c in chars:
            x0, y0, x1, y1 = [int(round(v)) for v in c["bbox"]]
            for dx, dy, s in ((0, 0, 1.0), (2, 0, 1.0), (-2, 0, 1.0), (0, 2, 1.0), (0, -2, 1.0), (0, 0, 1.06), (0, 0, 0.94)):
                cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
                hw, hh = (x1 - x0) * s / 2, (y1 - y0) * s / 2
                a0, b0 = max(0, int(cx - hw)), max(0, int(cy - hh))
                a1, b1 = min(W, int(cx + hw)), min(H, int(cy + hh))
                self.add(c["text"], glyph_feature(ink[b0:b1, a0:a1]), c.get("id"))
        return self

    def sim(self, text, vec):
        vs = self.vecs.get(text)
        if not vs or vec is None:
            return None
        return float(np.max(np.array(vs) @ vec))

    def __contains__(self, t):
        return t in self.vecs

    def stats(self):
        return {"glyphs": len(self.vecs), "exemplars": sum(len(v) for v in self.vecs.values())}


def split_column_rag(ink, x0, x1, y0, y1, texts, bank, prior=None, w_sim=2.5, steps=90):
    """행 띠 [x0,x1]×[y0,y1] 를 판독문 글자열 texts 로 나눔.
    비용 = 높이 사전값 + 잉크 잘림 + w_sim·(1 − 예시 유사도)  (예시 있는 글자만)."""
    H, W = ink.shape
    x0, x1 = max(0, int(x0)), min(W, int(x1)); y0, y1 = max(0, int(y0)), min(H, int(y1))
    band = ink[y0:y1, x0:x1]
    L, k = y1 - y0, len(texts)
    if k == 0 or L < 4:
        return [], []
    prof = np.convolve(band.sum(1).astype(float), np.ones(7) / 7, mode="same")
    mx = max(prof.max(), 1e-6)
    exp = L / k
    sd = (prior or {}).get("h_cv", 0.3)
    step = max(1, L // steps)
    cand = list(range(0, L, step)) + [L]
    feat = {}

    def F(a, b):
        if (a, b) not in feat:
            feat[(a, b)] = glyph_feature(band[a:b])
        return feat[(a, b)]
    INF = 1e18
    n = len(cand)
    D = np.full((k + 1, n), INF); Bk = np.zeros((k + 1, n), int); D[0, 0] = 0
    SIM = {}
    for j in range(1, k + 1):
        t = texts[j - 1]
        has = t in bank
        for bi in range(1, n):
            b = cand[bi]
            for ai in range(bi):
                a = cand[ai]
                if b - a < 0.3 * exp or b - a > 2.4 * exp or D[j - 1, ai] >= INF:
                    continue
                cut = 0.0 if b == L else prof[min(b, L - 1)] / mx
                c = 0.5 * ((b - a - exp) / (sd * exp)) ** 2 + 3.0 * cut
                if has:
                    s = bank.sim(t, F(a, b))
                    c += w_sim * (1 - (s if s is not None else 0))
                v = D[j - 1, ai] + c
                if v < D[j, bi]:
                    D[j, bi], Bk[j, bi] = v, ai
    if D[k, n - 1] >= INF:
        return None, None
    segs, bi = [], n - 1
    for j in range(k, 0, -1):
        ai = Bk[j, bi]; segs.append((cand[ai], cand[bi])); bi = ai
    segs = segs[::-1]
    boxes, sims = [], []
    for (a, b), t in zip(segs, texts):
        sub = band[a:b]
        ys, xs = np.nonzero(sub)
        s = bank.sim(t, F(a, b)) if t in bank else None
        sims.append(s)
        if len(xs) < 5:
            boxes.append([x0, y0 + a, x1, y0 + b, "est"])
        else:
            boxes.append([x0 + int(xs.min()), y0 + a + int(ys.min()), x0 + int(xs.max()) + 1, y0 + a + int(ys.max()) + 1])
    return boxes, sims
