"""명령행 실행: python -m hanocr.cli INPUT.pdf -o OUT --pages 1-5,19"""
import argparse
import csv
import json
import os
import time

import pymupdf as fitz

from .pipeline import process_page, align_document, finalize, save_page


def parse_pages(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a) - 1, min(n, int(b))))
        else:
            out.append(int(part) - 1)
    return sorted(set(p for p in out if 0 <= p < n))


def _worker(args):
    pdf, p, dpi, tight = args
    import os as _os
    _os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    d = fitz.open(pdf)
    try:
        return p, process_page(d, p, dpi=dpi, tight=tight)
    except Exception as e:  # 한 쪽의 실패가 전체를 멈추지 않도록
        return p, {"page": p + 1, "error": repr(e), "chars": [], "regions": [], "render_dpi": dpi,
                   "page_type": "error"}


def run(pdf, outdir, pages=None, dpi=300, neighbors=True, overlay=True, tight=True, log=print, workers=1,
        panshim_lexicon=None):
    doc = fitz.open(pdf)
    pnos = parse_pages(pages, doc.page_count)
    results = {}
    t0 = time.time()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for p, r in pool.imap_unordered(_worker, [(pdf, p, dpi, tight) for p in pnos]):
                results[p] = r
                log(f"  p{p+1}: {len(r['chars'])} chars ({time.time()-t0:.0f}s)")
    else:
        raw = os.path.join(outdir, "_raw")
        os.makedirs(raw, exist_ok=True)
        for p in pnos:
            fn = os.path.join(raw, f"p{p+1:04d}.json")
            if os.path.exists(fn):  # 중단 후 재시작 시 이어서 처리
                with open(fn, encoding="utf-8") as f:
                    results[p] = json.load(f)
                continue
            results[p] = _worker((pdf, p, dpi, tight))[1]
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(results[p], f, ensure_ascii=False)
            log(f"  p{p+1}: {len(results[p]['chars'])} chars, regions={[r['type'] for r in results[p]['regions']]}"
                f" ({time.time()-t0:.0f}s)")
    # 영인만 있고 판독문이 없는 쪽 → 앞뒤 쪽도 처리해 판독문 확보
    if neighbors:
        for p in list(results):
            r = results[p]
            if any(rg["type"] == "facsimile" for rg in r.get("regions", [])) and not r.get("transcript"):
                for q in (p - 1, p + 1):
                    if 0 <= q < doc.page_count and q not in results:
                        results[q] = _worker((pdf, q, dpi, tight))[1]
                        log(f"  p{q+1} (판독문 확보용 인접 쪽)")
    res = [results[k] for k in sorted(results)]
    align_document(res, panshim_lexicon)
    finalize(res, dpi)
    summary = write_outputs(res, outdir, pdf, doc if overlay else None, dpi, time.time() - t0)
    return res, summary


def write_outputs(res, outdir, pdf, doc=None, dpi=300, seconds=None, overlay_pages=None):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for r in res:
        use_doc = doc if (doc is not None and (overlay_pages is None or r["page"] in overlay_pages)) else None
        save_page(r, outdir, use_doc)
        for c in r["chars"]:
            rows.append([r["page"], c["id"], c["text"], c.get("kind"), c["engine"], c.get("text_source", ""),
                         c.get("rec", ""), c.get("conf", ""), *c["bbox"], *(c.get("ink_bbox") or ["", "", "", ""]),
                         *c["bbox_px"]])
    with open(os.path.join(outdir, "chars.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page", "id", "text", "kind", "engine", "text_source", "rec", "conf",
                    "x0_pt", "y0_pt", "x1_pt", "y1_pt", "ink_x0_pt", "ink_y0_pt", "ink_x1_pt", "ink_y1_pt",
                    "x0_px", "y0_px", "x1_px", "y1_px"])
        w.writerows(rows)
    summary = {"pdf": os.path.basename(pdf), "pages": [r["page"] for r in res], "render_dpi": dpi,
               "n_chars": len(rows), "seconds": round(seconds, 1) if seconds else None,
               "per_page": [{"page": r["page"], "type": r.get("page_type"), "chars": len(r["chars"]),
                             "regions": [(g["type"], g["n_chars"], g.get("aligned")) for g in r.get("regions", [])],
                             "alignment": r.get("alignment"), "error": r.get("error")} for r in res]}
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


def postprocess(outdir, pdf, overlay=False, overlay_pages=None, panshim_lexicon=None):
    """저장된 page_*.json 을 다시 읽어 판독문 정렬만 재수행 (OCR 재실행 불필요)."""
    import glob as _g
    res = []
    for fn in sorted(_g.glob(os.path.join(outdir, "page_*.json"))):
        with open(fn, encoding="utf-8") as f:
            res.append(json.load(f))
    for r in res:
        for c in r["chars"]:
            if c["engine"] == "facsimile":
                c["text"] = c.get("rec", "")
    align_document(res, panshim_lexicon)
    finalize(res, res[0]["render_dpi"] if res else 300)
    doc = fitz.open(pdf) if overlay else None
    return write_outputs(res, outdir, pdf, doc, overlay_pages=overlay_pages)


def main():
    ap = argparse.ArgumentParser(description="HanOCR: 글자 단위 bounding box OCR")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--pages", default=None, help="예: 1-10,35")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-tight", action="store_true")
    ap.add_argument("-j", "--workers", type=int, default=1)
    ap.add_argument("--panshim", nargs="*", default=None, help="판심 서명 사전 (예: ᄉᆞ민필지)")
    ap.add_argument("--corrections", default=None, help="검수 보정 JSON")
    a = ap.parse_args()
    if a.corrections:
        from .pipeline import load_corrections
        load_corrections(a.corrections)
    run(a.pdf, a.out, a.pages, a.dpi, overlay=not a.no_overlay, tight=not a.no_tight, workers=a.workers,
        panshim_lexicon=a.panshim)


if __name__ == "__main__":
    main()
