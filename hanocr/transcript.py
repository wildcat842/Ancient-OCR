"""쪽 안의 '판독문(원문 전사)' 영역 찾기 — 텍스트 레이어 / OCR 결과 공통."""
import re

import numpy as np

START_PAT = re.compile(r"^(판독|원문(?![가-힣]))")
END_PAT = re.compile(r"^(현대어역|한문원문|해설|주해|번역)")


def _nospace(s):
    return re.sub(r"\s+", "", s)


def _group_lines(chars, key="line"):
    lines = {}
    for c in chars:
        lines.setdefault(c[key], []).append(c)
    out = []
    for lid, cs in lines.items():
        cs = sorted(cs, key=lambda c: (c["bbox"][0]) if cs[0].get("direction") != "ttb" else c["bbox"][1])
        xs0 = min(c["bbox"][0] for c in cs); ys0 = min(c["bbox"][1] for c in cs)
        xs1 = max(c["bbox"][2] for c in cs); ys1 = max(c["bbox"][3] for c in cs)
        txt = ""
        for k, c in enumerate(cs):
            if k and "word" in c and c["word"] != cs[k - 1].get("word"):
                txt += " "
            txt += c["text"]
        out.append({"id": lid, "chars": cs, "text": txt,
                    "bbox": [xs0, ys0, xs1, ys1], "direction": cs[0].get("direction", "ltr")})
    return out


def find_transcript(chars, page_w, page_h):
    """chars: 페이지 좌표(pt 또는 px 통일)의 글자 목록 (text, bbox, line, direction).
    반환: (판독문 글자 목록(읽기 순서), 판독문 영역 bbox) 또는 (None, None)."""
    lines = _group_lines(chars)
    starts = [l for l in lines if START_PAT.match(_nospace(l["text"])) and len(_nospace(l["text"])) < 30]
    if not starts:
        return None, None
    spread = page_w > page_h * 1.1
    best = None
    for L in sorted(starts, key=lambda l: l["bbox"][1]):
        lx0, ly0, lx1, ly1 = L["bbox"]
        if spread:
            hx0, hx1 = (0, page_w / 2) if (lx0 + lx1) / 2 < page_w / 2 else (page_w / 2, page_w)
        else:
            hx0, hx1 = 0, page_w
        ends = [l for l in lines if END_PAT.match(_nospace(l["text"])) and l["bbox"][1] > ly0 + 1
                and hx0 <= (l["bbox"][0] + l["bbox"][2]) / 2 <= hx1]
        yend = min([l["bbox"][1] for l in ends] + [page_h * 0.93])
        label_lines = {l["id"] for l in lines if _nospace(l["text"]) in ("주석문", "주석", "및주석문")}
        cand = [c for c in chars if c["line"] != L["id"] and c["line"] not in label_lines and hx0 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= hx1
                and ly0 - 1 <= (c["bbox"][1] + c["bbox"][3]) / 2 < yend]
        if not cand:
            continue
        n_ttb = sum(c.get("direction") == "ttb" for c in cand)
        direction = "ttb" if n_ttb > len(cand) / 2 else "ltr"
        cand = [c for c in cand if (c.get("direction") == "ttb") == (direction == "ttb")]
        if direction == "ltr":
            # 옆 라벨(라벨 오른쪽에 본문 시작) vs 머리 라벨(라벨 아래로 본문)
            side = any(l["id"] != L["id"] and l["bbox"][0] > lx1 and
                       min(l["bbox"][3], ly1) - max(l["bbox"][1], ly0) > 0.3 * (ly1 - ly0) for l in lines)
            xmin = lx1 - 1 if side else lx0 - 0.03 * (hx1 - hx0)
            cand = [c for c in cand if c["bbox"][0] >= xmin]
            ls = _group_lines(cand)
            ls.sort(key=lambda l: (round(l["bbox"][1] / 3), l["bbox"][0]))
            seq = [c for l in ls for c in l["chars"]]
        else:
            size = np.median([c["bbox"][3] - c["bbox"][1] for c in cand])
            cols = []
            for c in sorted(cand, key=lambda c: -(c["bbox"][0] + c["bbox"][2]) / 2):
                cx = (c["bbox"][0] + c["bbox"][2]) / 2
                if cols and abs(cols[-1][0] - cx) < 0.6 * size:
                    cols[-1][1].append(c)
                else:
                    cols.append([cx, [c]])
            seq = [c for _, cs in cols for c in sorted(cs, key=lambda c: c["bbox"][1])]
        if len(seq) > (len(best[0]) if best else 0):
            box = [min(c["bbox"][0] for c in seq), min(c["bbox"][1] for c in seq),
                   max(c["bbox"][2] for c in seq), max(c["bbox"][3] for c in seq)]
            best = (seq, box)
    return best if best else (None, None)
