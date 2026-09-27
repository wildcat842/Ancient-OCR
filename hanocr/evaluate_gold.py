"""고정 평가셋(gold)과 모델 예측을 비교해 성능 보드용 JSON을 출력합니다.

사용법:
  python -m hanocr.evaluate_gold \\
    --gold out/kimssi/gold-r02/chars.tsv \\
    --eval-set eval-v1 \\
    --pred predictions.tsv \\
    --out result_hrcnet-ft01.json

predictions.tsv 형식 (탭 구분):
  path  text  conf
  (chars.tsv 의 path 열과 일치해야 합니다)
"""
import argparse
import csv
import json
import pathlib
from datetime import datetime

import numpy as np


def load_tsv(path, key="path"):
    rows = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows[row[key]] = row
    return rows


def old_hangul(text):
    import unicodedata
    for ch in unicodedata.normalize("NFD", text):
        cp = ord(ch)
        if 0x1113 <= cp <= 0x115F or 0x11C3 <= cp <= 0x11FF or 0x119E <= cp <= 0x11A2 or \
           0xD7CB <= cp <= 0xD7FB:
            return True
    return False


def evaluate(gold_path, pred_path, eval_set="eval-v1"):
    gold = load_tsv(gold_path)
    pred = load_tsv(pred_path)
    ev = {k: v for k, v in gold.items() if v.get("split") == eval_set.replace("eval-v", "eval"[-1]) or v.get("split") == "eval"}
    if not ev:
        # fallback: split 열 값이 "eval" 이면
        ev = {k: v for k, v in gold.items() if v.get("split", "").startswith("eval")}
    if not ev:
        raise ValueError(f"평가셋 '{eval_set}'에 해당하는 글자가 없습니다. chars.tsv의 split 열을 확인하세요.")

    total = len(ev)
    correct = 0
    old_total = 0
    old_correct = 0
    per_item = {}

    for path, grow in ev.items():
        gt = grow["text"]
        prow = pred.get(path)
        pr = prow["text"] if prow else ""
        match = pr == gt
        if match:
            correct += 1
        item = grow.get("item", "unknown")
        d = per_item.setdefault(item, {"total": 0, "correct": 0})
        d["total"] += 1
        if match:
            d["correct"] += 1
        if old_hangul(gt):
            old_total += 1
            if match:
                old_correct += 1

    return {
        "total": total,
        "correct": correct,
        "char_acc": round(correct / total, 4) if total else 0,
        "old_hangul_total": old_total,
        "old_hangul_correct": old_correct,
        "old_hangul_acc": round(old_correct / old_total, 4) if old_total else None,
        "per_item": [{"item": k, "char_acc": round(v["correct"] / v["total"], 4), "n": v["total"]} for k, v in per_item.items()],
    }


def main():
    ap = argparse.ArgumentParser(description="HanOCR: 고정 평가셋 기준 성능 측정")
    ap.add_argument("--gold", required=True, help="chars.tsv 경로")
    ap.add_argument("--eval-set", default="eval-v1", help="평가셋 이름 (split 열 값)")
    ap.add_argument("--pred", required=True, help="예측 TSV (path, text, conf 열)")
    ap.add_argument("--run-id", required=True, help="실행 식별자 (예: hrcnet-r03-ft01)")
    ap.add_argument("--model", default="", help="모델 이름 (예: HRCNet-ResNet50)")
    ap.add_argument("--mode", default="finetune", choices=["zero-shot", "finetune"])
    ap.add_argument("--train-snapshot", default="", help="학습에 쓴 스냅샷 이름")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", required=True, help="결과 JSON 출력 경로")
    args = ap.parse_args()

    m = evaluate(args.gold, args.pred, args.eval_set)
    result = {
        "run_id": args.run_id,
        "model": args.model,
        "mode": args.mode,
        "train_snapshot": args.train_snapshot,
        "eval_set": args.eval_set,
        "metrics": {
            "char_acc": m["char_acc"],
            "old_hangul_acc": m["old_hangul_acc"],
        },
        "per_item": m["per_item"],
        "detail": {k: v for k, v in m.items() if k != "per_item"},
        "finished_at": datetime.utcnow().isoformat() + "Z",
        "notes": args.notes,
    }
    pathlib.Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"글자 정확도: {m['char_acc']:.1%}  ({m['correct']}/{m['total']})")
    if m["old_hangul_total"]:
        print(f"옛한글 정확도: {m['old_hangul_acc']:.1%}  ({m['old_hangul_correct']}/{m['old_hangul_total']})")
    print(f"결과 저장: {args.out}")


if __name__ == "__main__":
    main()
