"""판독문(정답 텍스트)과 영인 검출 박스열의 정렬 (Needleman-Wunsch).

인식 결과가 부정확해도(특히 옛한글) 글자 수·순서와 일부 일치 글자(anchor)만으로
판독문 글자를 각 박스에 배정한다. 결과 박스에는 text(판독문 글자)와 rec(원 인식값)가 함께 남는다."""
import unicodedata

import numpy as np

from .normalize import clusters, pua_to_jamo, nfc, char_kind

_DROP = set("【】〔〕［］[]()（）〈〉《》「」『』“”‘’\"'.,·ㆍ:;!?/~-–—…　 \t\r\n∎■□○●◎※*")


def transcript_units(text):
    """판독문 문자열 → 원문에 실제로 새겨진 글자 단위 목록."""
    out = []
    for c in clusters(pua_to_jamo(text)):
        c = nfc(c)
        if not c.strip() or c in _DROP:
            continue
        k = char_kind(c)
        if k in ("punct", "space") and c not in ("〃", "ㅣ"):
            continue
        if k == "digit" or (k == "latin"):
            continue  # 각주 번호 등
        out.append(c)
    return out


def _decomp(s):
    return unicodedata.normalize("NFD", s)


def sim(rec, gold):
    if not rec:
        return 0.0
    if rec == gold:
        return 1.0
    a, b = _decomp(rec), _decomp(gold)
    if a and b and a[0] == b[0]:  # 초성 일치(옛한글 ↔ 현대한글 근사)
        return 0.45
    return 0.0


def align(recs, gold, match=3.0, mismatch=-0.3, gap_box=-1.2, gap_gold=-1.2):
    """recs: 박스별 인식 문자열 목록, gold: 판독문 글자 목록.
    반환: 박스별 배정 판독문 index (없으면 None), 정렬 통계."""
    n, m = len(recs), len(gold)
    S = np.zeros((n + 1, m + 1))
    T = np.zeros((n + 1, m + 1), np.int8)  # 0 diag, 1 up(box gap), 2 left(gold gap)
    S[1:, 0] = gap_box * np.arange(1, n + 1)
    S[0, 1:] = gap_gold * np.arange(1, m + 1)
    T[1:, 0], T[0, 1:] = 1, 2
    for i in range(1, n + 1):
        r = recs[i - 1]
        for j in range(1, m + 1):
            s = sim(r, gold[j - 1])
            d = S[i - 1, j - 1] + (match * s if s > 0 else mismatch)
            u = S[i - 1, j] + gap_box
            l = S[i, j - 1] + gap_gold
            if d >= u and d >= l:
                S[i, j], T[i, j] = d, 0
            elif u >= l:
                S[i, j], T[i, j] = u, 1
            else:
                S[i, j], T[i, j] = l, 2
    assign = [None] * n
    i, j = n, m
    exact = 0
    while i > 0 or j > 0:
        t = T[i, j]
        if i > 0 and j > 0 and t == 0:
            assign[i - 1] = j - 1
            exact += recs[i - 1] == gold[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and (t == 1 or j == 0):
            i -= 1
        else:
            j -= 1
    paired = sum(a is not None for a in assign)
    stats = {"boxes": n, "gold_chars": m, "paired": paired,
             "unpaired_boxes": n - paired, "unpaired_gold": m - paired,
             "exact_rec_matches": int(exact),
             "anchor_rate": round(exact / paired, 3) if paired else 0.0}
    return assign, stats
