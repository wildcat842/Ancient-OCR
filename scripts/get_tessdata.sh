#!/usr/bin/env bash
# Tesseract 학습 데이터(tessdata_best) 내려받기 — hanocr/../tessdata 에 저장
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/tessdata"
mkdir -p "$DIR"
for l in kor kor_vert chi_tra chi_tra_vert eng osd; do
  [ -f "$DIR/$l.traineddata" ] || curl -sL -o "$DIR/$l.traineddata" \
    "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/$l.traineddata"
done
# hOCR 출력용 configs
SYS=$(dirname "$(tesseract --list-langs 2>&1 | head -1 | sed -E 's/.*"(.*)".*/\1/')/x)
cp -rn /usr/share/tesseract-ocr/5/tessdata/configs "$DIR/" 2>/dev/null || cp -rn "$SYS/configs" "$DIR/" || true
echo "tessdata ready: $DIR"
