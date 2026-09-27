"""판독문(원문)과 현대어역(해석)을 문단 단위로 구조화하고 서로 대응시킨다.

- 판독 라벨 ~ 현대어역 라벨 : 원문 판독문
- 현대어역 라벨 ~ 다음 라벨/쪽 끝 : 해석
- 문단 경계 : 앞 줄이 오른쪽 끝까지 차지 않고 끝난 경우 (들여쓰기 없는 조판)
- 머리줄 '〈ᄉᆞ십칠〉 ※ 포츄갈국' : 판심의 장차(쪽 번호)와 권차 제목 → 판심 글자의 정답으로 사용
"""
import re

import numpy as np

from .transcript import _group_lines, _nospace, START_PAT

MODERN_PAT = re.compile(r"^현대어역")
OTHER_PAT = re.compile(r"^(한문원문|해설|지도해설|주해|각주)")
HEADER_PAT = re.compile(r"^\s*[〈<]([^〉>]+)[〉>]\s*(※\s*(.+))?$")


def _collect(lines, label, y_end, half, side_label):
    lx0, ly0, lx1, ly1 = label["bbox"]
    hx0, hx1 = half
    out = []
    for l in lines:
        if l["id"] == label["id"]:
            continue
        cx = (l["bbox"][0] + l["bbox"][2]) / 2
        cy = (l["bbox"][1] + l["bbox"][3]) / 2
        if not (hx0 <= cx <= hx1 and ly0 - 1 <= cy < y_end):
            continue
        if l["direction"] == "ttb" or re.fullmatch(r"[\d\s.,]+", l["text"]):
            continue  # 세로 머리글·쪽 번호·각주 번호 줄 제외
        if side_label and l["bbox"][0] < lx1 - 1:
            continue
        if _nospace(l["text"]) in ("주석문", "주석", "및주석문"):
            continue
        out.append(l)
    # 본문 열 오른쪽 바깥의 한 글자 줄(세로 탭 머리글 조각) 제거
    if out:
        x0s = np.array([l["bbox"][0] for l in out])
        x1s = np.array([l["bbox"][2] for l in out])
        colx0, colx1 = np.median(x0s), np.percentile(x1s, 90)
        out = [l for l in out if not (len(l["chars"]) <= 2 and l["bbox"][0] > colx0 + 0.8 * (colx1 - colx0))]
    # 각주 번호(작은 위첨자 숫자) 글자 제거
    for l in out:
        l["chars"] = [c for c in l["chars"] if not (c["text"].isdigit() and c.get("size", 9) < 7)]
        l["text"] = "".join(c["text"] if c.get("direction") != "ttb" else c["text"] for c in l["chars"]) \
            if any(c["text"].isdigit() for c in l["chars"]) else l["text"]
    # 같은 y 의 조각 줄(한 줄이 span 여러 개로 나뉜 경우) 병합
    out.sort(key=lambda l: (round(l["bbox"][1] / 3), l["bbox"][0]))
    merged = []
    for l in out:
        if merged and abs(merged[-1]["bbox"][1] - l["bbox"][1]) < 3:
            m = merged[-1]
            m["text"] += l["text"]
            m["chars"] += l["chars"]
            m["bbox"] = [min(m["bbox"][0], l["bbox"][0]), min(m["bbox"][1], l["bbox"][1]),
                         max(m["bbox"][2], l["bbox"][2]), max(m["bbox"][3], l["bbox"][3])]
        else:
            merged.append(dict(l, chars=list(l["chars"])))
    return merged


def _paragraphs(lines):
    if not lines:
        return []
    right = np.percentile([l["bbox"][2] for l in lines], 90)
    cw = np.median([c["bbox"][2] - c["bbox"][0] for l in lines for c in l["chars"]] or [8])
    paras, cur = [], None
    for i, l in enumerate(lines):
        txt = l["text"].strip()
        header = HEADER_PAT.match(txt)
        if cur is None or header or cur.get("closed"):
            cur = {"lines": [], "text": "", "header": bool(header)}
            paras.append(cur)
        cur["lines"].append(l)
        # 조판 줄바꿈은 낱말 경계가 아닐 수 있음: 원문 줄 끝의 공백 유무를 그대로 따름
        raw = l["text"].replace("\x07", "")
        cur["text"] += raw if any(c.get("kind") == "space" for c in l["chars"]) else \
            ((" " if cur["text"] and not cur["text"].endswith(" ") else "") + txt)
        # 줄이 오른쪽 끝보다 확실히 짧게 끝나면 문단 끝
        if l["bbox"][2] < right - 2.0 * cw or header:
            cur["closed"] = True
    for p in paras:
        p.pop("closed", None)
        p["bbox"] = [min(l["bbox"][0] for l in p["lines"]), min(l["bbox"][1] for l in p["lines"]),
                     max(l["bbox"][2] for l in p["lines"]), max(l["bbox"][3] for l in p["lines"])]
        p["text"] = re.sub(r"\s+", " ", p["text"]).strip()
    return paras


def find_sections(chars, page_w, page_h):
    """반환: {'original': [문단], 'modern': [문단], 'header': {...}, 'pairs': [(i,j),...]} 또는 None"""
    lines = _group_lines(chars)
    starts = sorted([l for l in lines if START_PAT.match(_nospace(l["text"])) and len(_nospace(l["text"])) < 30],
                    key=lambda l: l["bbox"][1])
    if not starts:
        return None
    L = starts[0]
    spread = page_w > page_h * 1.1
    cxl = (L["bbox"][0] + L["bbox"][2]) / 2
    half = ((0, page_w / 2) if cxl < page_w / 2 else (page_w / 2, page_w)) if spread else (0, page_w)
    in_half = [l for l in lines if half[0] <= (l["bbox"][0] + l["bbox"][2]) / 2 <= half[1]]
    mods = sorted([l for l in in_half if MODERN_PAT.match(_nospace(l["text"])) and l["bbox"][1] > L["bbox"][1]],
                  key=lambda l: l["bbox"][1])
    others = [l for l in in_half if OTHER_PAT.match(_nospace(l["text"]))]
    side = any(l["id"] != L["id"] and l["bbox"][0] > L["bbox"][2] and
               min(l["bbox"][3], L["bbox"][3]) - max(l["bbox"][1], L["bbox"][1]) > 0.3 * (L["bbox"][3] - L["bbox"][1])
               for l in lines)
    y_mod = mods[0]["bbox"][1] if mods else min([o["bbox"][1] for o in others if o["bbox"][1] > L["bbox"][1]]
                                               + [page_h * 0.93])
    orig = _paragraphs(_collect(lines, L, y_mod, half, side))
    if not orig:  # 세로쓰기 판독문(김씨부인 편지 등): 행(列) 단위 줄, 전체를 한 문단으로
        from .transcript import find_transcript
        seq, box = find_transcript(chars, page_w, page_h)
        if seq:
            orig = [{"lines": [], "text": "".join(c["text"] for c in seq), "header": False, "bbox": box}]
    if not mods:  # 현대어역이 다른 반쪽에 있는 배치
        mods = sorted([l for l in lines if MODERN_PAT.match(_nospace(l["text"]))], key=lambda l: l["bbox"][1])
        if mods:
            cxm = (mods[0]["bbox"][0] + mods[0]["bbox"][2]) / 2
            half = ((0, page_w / 2) if cxm < page_w / 2 else (page_w / 2, page_w)) if spread else (0, page_w)
            others = [l for l in lines if OTHER_PAT.match(_nospace(l["text"]))
                      and half[0] <= (l["bbox"][0] + l["bbox"][2]) / 2 <= half[1]]
            side = False
    modern = []
    if mods:
        M = mods[0]
        y_end = min([o["bbox"][1] for o in others + starts if o["bbox"][1] > M["bbox"][1] + 1] + [page_h * 0.93])
        # 현대어역 라벨 줄 자체에 본문 머리줄이 붙어 있는 경우(같은 y) 포함
        modern = _paragraphs(_collect(lines, M, y_end, half, side))
    _assign_roles(orig)
    _assign_roles(modern)
    # 해설(지도해설 등) 본문
    commentary = None
    cl = sorted([l for l in in_half if OTHER_PAT.match(_nospace(l["text"]))], key=lambda l: l["bbox"][1])
    if cl:
        C = cl[0]
        side_c = any(l["id"] != C["id"] and l["bbox"][0] > C["bbox"][2] and
                     min(l["bbox"][3], C["bbox"][3]) - max(l["bbox"][1], C["bbox"][1]) > 0.3 * (C["bbox"][3] - C["bbox"][1])
                     for l in lines)
        cps = _paragraphs(_collect(lines, C, page_h * 0.93, half, side_c))
        commentary = {"label": re.sub(r"\s+", "", C["text"]), "text": " ".join(p["text"] for p in cps)}
    gloss = {}
    if commentary:
        for m in re.finditer(r"([\u1100-\u11ff\u3130-\u318e\uac00-\ud7a3]+)\(([\uac00-\ud7a3]+)\)",
                             commentary["text"]):
            gloss[m.group(1)] = m.group(2)
    header = None
    if orig and orig[0]["header"]:
        m = HEADER_PAT.match(orig[0]["text"])
        header = {"page_label": m.group(1).strip(), "section_title": (m.group(3) or "").strip() or None}
    return {"original": orig, "modern": modern, "header": header, "pairs": pair_paragraphs(orig, modern),
            "commentary": commentary, "gloss": gloss}


FIG_PAT = re.compile(r"^(지도|그림|도판|도|표)\s*\d+\s*$")


def split_label_tokens(t):
    """표기 목록 토큰화: [아시아 서쪽] 처럼 대괄호 안은 한 토큰."""
    return [m.group(0).strip("[]") if m.group(0).startswith("[") else m.group(0)
            for m in re.finditer(r"\[[^\]]*\]|\S+", t)]


PANEL_PAT = re.compile(r"^\[[^\]]+\]$")


def _assign_roles(paras):
    """문단 역할: header / figure_caption(지도 1) / figure_panel_title([디구 셔편]) / figure_labels / body"""
    prev = None
    for p in paras:
        t = p["text"].strip()
        if p["header"]:
            p["role"] = "header"
        elif FIG_PAT.match(t):
            p["role"] = "figure_caption"
        elif PANEL_PAT.match(t) and prev in ("figure_caption", "figure_labels", "figure_legend"):
            p["role"] = "figure_panel_title"
        elif t.startswith("●") and prev in ("figure_caption", "figure_panel_title", "figure_legend"):
            p["role"] = "figure_legend"   # 범례: ● 각 읍ᄂᆡㅅ 표ㅣ니라
        elif prev in ("figure_caption", "figure_panel_title", "figure_legend", "figure_labels"):
            toks = split_label_tokens(t)
            if len(toks) >= 1 and max(len(x) for x in toks) <= 12 and not re.search(r"(니라|이라|다\.|\.)$", t) \
                    and (prev != "figure_labels" or len(t) < 120):
                p["role"] = "figure_labels"
            else:
                p["role"] = "body"
        else:
            p["role"] = "body"
        prev = p["role"]


def pair_paragraphs(orig, modern):
    """문단 대응 DP: 1:1, 2:1, 1:2 병합과 건너뛰기를 허용 (길이 비율 기준)."""
    n, m = len(orig), len(modern)
    if n == 0 or m == 0:
        return []
    if n == 1:
        return [(0, j) for j in range(m)]
    lo = [max(1, len(_nospace(p["text"]))) for p in orig]
    lm = [max(1, len(_nospace(p["text"]))) for p in modern]
    r = sum(lm) / sum(lo)
    moves = [(1, 1, 0.0), (2, 1, 0.35), (1, 2, 0.35), (1, 0, 1.5), (0, 1, 1.5)]
    INF = 1e9
    D = np.full((n + 1, m + 1), INF)
    B = {}
    D[0, 0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            for di, dj, pen in moves:
                a, b = i - di, j - dj
                if a < 0 or b < 0 or D[a, b] >= INF:
                    continue
                if di and dj:
                    so, sm = sum(lo[a:i]) * r, sum(lm[b:j])
                    c = abs(so - sm) / max(so, sm) * 2 + pen
                    # 머리줄(〈…〉)끼리는 반드시 대응
                    if (orig[a]["header"]) != (modern[b]["header"]):
                        c += 3
                else:
                    c = pen
                if D[a, b] + c < D[i, j]:
                    D[i, j] = D[a, b] + c
                    B[(i, j)] = (a, b)
    pairs, i, j = [], n, m
    while (i, j) != (0, 0):
        a, b = B[(i, j)]
        if i > a and j > b:
            for x in range(a, i):
                for y in range(b, j):
                    pairs.append((x, y))
        i, j = a, b
    return sorted(pairs)


def strip_for_lines(p):
    return {"text": p["text"], "header": p["header"], "role": p.get("role", "body"),
            "bbox": [round(v, 2) for v in p["bbox"]]}
