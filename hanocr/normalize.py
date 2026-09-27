"""유니코드 정규화: 한양 PUA → 첫가끝(조합형 자모), 옛한글 글자마디(grapheme cluster) 분할."""
import unicodedata

try:
    from hypua2jamo import translate as _pua2jamo
except ImportError:  # pragma: no cover
    _pua2jamo = None


def is_pua(ch):
    return 0xE000 <= ord(ch) <= 0xF8FF


def pua_to_jamo(s):
    """한양 PUA(한컴·나눔 옛한글 폰트) 코드를 표준 첫가끝 자모열로 변환."""
    if _pua2jamo is None or not any(is_pua(c) for c in s):
        return s
    return _pua2jamo(s)


def jamo_class(ch):
    c = ord(ch)
    if 0x1100 <= c <= 0x115F or 0xA960 <= c <= 0xA97C:
        return "L"
    if 0x1160 <= c <= 0x11A7 or 0xD7B0 <= c <= 0xD7C6:
        return "V"
    if 0x11A8 <= c <= 0x11FF or 0xD7CB <= c <= 0xD7FB:
        return "T"
    if c in (0x302E, 0x302F):  # 방점
        return "M"
    if 0xAC00 <= c <= 0xD7A3:
        return "S"
    return None


def attaches(prev_cls, cls):
    """cls 문자가 앞 글자마디에 붙는가 (UAX#29 한글 규칙 간소화)."""
    if cls == "M":
        return prev_cls is not None
    if cls == "V":
        return prev_cls in ("L", "V", "S")
    if cls == "T":
        return prev_cls in ("V", "T", "S")
    if cls == "L":
        return prev_cls == "L"
    return False


def clusters(s):
    """문자열을 글자마디 단위로 분할 (현대 음절·옛한글 첫가끝·방점 포함)."""
    out, prev = [], None
    for ch in s:
        cls = jamo_class(ch)
        if out and cls and attaches(prev, cls):
            out[-1] += ch
        else:
            out.append(ch)
        prev = cls if cls != "M" else prev
    return out


def nfc(s):
    return unicodedata.normalize("NFC", s)


def char_kind(s):
    """글자 종류 분류: hangul / old_hangul / hanja / latin / digit / punct / space / other"""
    if not s or s.isspace():
        return "space"
    c = ord(s[0])
    if len(s) > 1 or jamo_class(s[0]) in ("L", "V", "T"):
        return "old_hangul" if any(jamo_class(x) in ("L", "V", "T", "M") for x in s) and not (
            len(s) == 1 and 0xAC00 <= c <= 0xD7A3) else "hangul"
    if 0xAC00 <= c <= 0xD7A3 or 0x3131 <= c <= 0x318E:
        return "old_hangul" if 0x3165 <= c <= 0x318E else "hangul"
    if 0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or 0xF900 <= c <= 0xFAFF or 0x20000 <= c <= 0x2FFFF:
        return "hanja"
    if s.isdigit():
        return "digit"
    if s.isalpha():
        return "latin"
    cat = unicodedata.category(s[0])
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    return "other"


def normalize_text(s):
    """PUA 변환 + 글자마디별 NFC (현대 자모열은 완성형으로, 옛한글은 첫가끝 유지)."""
    return "".join(nfc(c) for c in clusters(pua_to_jamo(s)))
