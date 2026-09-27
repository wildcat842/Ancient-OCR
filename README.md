# HanOCR — 국립한글박물관 소장자료총서 글자 단위 OCR 파이프라인

국립한글박물관 소장자료총서 4종을 대상으로 글자 단위 바운딩 박스(bounding box)와 텍스트를 구축합니다.
판독문이 있는 자료는 판독문 정렬(Needleman–Wunsch)로 정확한 글자 값을 붙이고,
판독문 없는 자료는 HRCNet(ResNet 백본 옛한글 사전학습 모델)을 통해 인식합니다.

---

## 대상 자료와 처리 방식

| 총서 | 자료명 | 쪽수 | 텍스트 레이어 | 주요 특성 | 처리 엔진 |
|------|--------|------|--------------|----------|----------|
| 1 | 김씨부인 한글상언·정조어필한글편지첩·곤전어필 | 83 | ✅ (판독문) | 흘림 궁체·정자 필사, 봉투 90° 회전, 삽입·삭제 글자, 복수 이미지/쪽 | `kimssi`, `exemplar` |
| 9 | 사민필지 | 179 | ✅ (해설+판독문) | 지도 8종(90° 회전·경위선 겹침), 영인 본문 163쪽 | `textlayer`, `facsimile`, `figure_manual` |
| 10 | 간이벽온방언해 | 205 | ❌ (벡터 윤곽선) | 한자 대자+소자 독음 쌍, 방점, 판독문 OCR 필요 | `facsimile`, `tesseract`, HRCNet |
| 11 | 삼강행실도언해 | 348 | ❌ (96dpi 스캔) | 삽화+언해+한문 혼재, 해상도 보강 필요 | `facsimile` (S2 편집기 우선) |

---

## 아키텍처 개요

```
PDF 입력
 ├─ 텍스트 레이어 있음?
 │   ├─ Yes → textlayer.extract_chars()        # 좌표·글자 그대로 추출 (신뢰도 1.0)
 │   │         영인 영역 → facsimile 엔진
 │   └─ No  → tesseract / HRCNet               # 인쇄체 or 필사체 OCR
 │
 ├─ 영인 영역(facsimile 엔진)
 │   ├─ 글자 검출 (binarize → connected_components → column 분할)
 │   ├─ 일괄 인식 (recognize.py: Tesseract or HRCNet)    → rec 필드
 │   └─ 판독문 정렬 (align.py: NW, 자모 초성 부분 일치)  → text 필드
 │       status: aligned / aligned-by-count / no-transcript-match
 │
 ├─ 지도·도판 (figure_manual.py)
 │   ├─ 사람이 지정한 중심선 + 상자 사양 (corrections/samin_pilji.json)
 │   └─ 잉크 기반 자동 조임 → FIGURE_LABEL, FIGURE_TITLE, GRATICULE_TEXT
 │
 └─ 필사 편지 (kimssi.py)
     ├─ 행 단위 정렬 + 넓은 행 분리 + 행 끝 정리 (v0.9)
     ├─ 정답 예시 검색 (exemplar.py: ExemplarBank, RAG split)
     └─ 봉투 글씨 자동 검출 (find_envelope → ENVELOPE_TEXT)

chars.csv 출력 컬럼:
  page, id, text, kind, engine, text_source, rec, conf,
  x0_pt, y0_pt, x1_pt, y1_pt,          # PDF 포인트 (영인 위치 or 활자 위치)
  ink_x0_pt … ink_y1_pt,               # 잉크 경계로 조인 상자 (facsimile만)
  x0_px, y0_px, x1_px, y1_px           # render_dpi 기준 픽셀
```

---

## 설치

```bash
pip install pymupdf opencv-python pillow numpy pytesseract
# Tesseract 5 + kor 학습 데이터 필요
# HRCNet 체크포인트는 별도 환경(GPU)에서 사용
```

---

## 빠른 시작

### 사민필지 — 텍스트 레이어 있는 자료

```python
from hanocr.cli import run, postprocess

# 쪽 처리 (OCR + 판독문 정렬)
res, summary = run("사민필지.pdf", "out/samin", pages="19,60")

# 판독문 수정 후 OCR 재실행 없이 정렬만 다시
postprocess("out/samin", "사민필지.pdf")
```

> **`run()`의 동작**: 텍스트 레이어가 있는 쪽은 레이어를 직접 읽습니다(Tesseract 미사용).
> 같은 쪽의 영인 영역은 글자 검출 → Tesseract 일괄 인식 → 판독문 NW 정렬 순으로 처리합니다.
> 정렬에 성공한 글자는 `text_source = "transcript-aligned"`, 최종 값은 판독문에서 옵니다.
> Tesseract 원래 인식값은 `rec` 필드에 따로 남습니다.

### 사민필지 — 지도 수동 주석 (90° 회전 인쇄)

```python
import json, pymupdf as fitz
from hanocr.figure_manual import build

spec = json.load(open("corrections/samin_pilji.json"))["pages"]["25"]["manual_figure"]
build(fitz.open("사민필지.pdf"), 24, spec, "out/samin/p25")
# → page_0025_R0.png, page_0025_R0.json, page_0025_R0.csv
```

### 김씨부인 — 필사 편지

```python
import json, pymupdf as fitz, pickle
from hanocr.kimssi import process_letter, set_letter_prior, set_exemplar_bank

# 검수 정답으로 학습한 사전값과 예시 은행 로드
set_letter_prior(json.load(open("out/kimssi/letter_prior.json")))
set_exemplar_bank(pickle.load(open("out/kimssi/exemplar_bank.pkl", "rb")))

doc = fitz.open("김씨부인.pdf")
res = process_letter(doc, 14, "out/kimssi/p14", split_wide=True)
# 결과: {"figure_id": "편지 3", "date": "1755(?)~1759.2. 이전", "chars": [...]}
```

---

## 평가

### 텍스트 레이어 쪽 (현대 활자 OCR)

```bash
python -m hanocr.evaluate 사민필지.pdf 19,60
# → 텍스트 레이어를 정답으로 Tesseract 결과 평가 (영인 옛한글은 측정하지 않음)
```

### 고정 평가셋 (gold) 기준

```bash
python -m hanocr.evaluate_gold \
  --gold out/kimssi/gold-r02/chars.tsv \
  --eval-set eval-v1 \
  --pred predictions.tsv \
  --out result_hrcnet-ft01.json
# → 성능 보드 등록용 JSON 출력 (char_acc, old_hangul_acc, cer, iou50)
```

> `evaluate_gold`는 편집기(S2)에서 사람이 확인한 정답(`split=eval`)과 모델 예측을 비교합니다.
> `evaluate`는 텍스트 레이어 현대 활자만 평가합니다. 두 명령의 범위가 다릅니다.

---

## 출력 파일 구조

```
out/samin/
  chars.csv              # 모든 쪽 글자 통합 (engine 열로 구분)
  summary.json           # 쪽별 처리 결과·정렬 상태
  page_0019.json         # 쪽별 상세 (chars, regions, paragraphs, alignment)
  page_0019_overlay.jpg  # 검수용 오버레이 이미지
  _raw/p0019.json        # 중간 결과 (재시작 시 재사용)

out/kimssi/
  gold-r02/
    chars.tsv            # 학습·평가 정답 (split=train/eval)
    lines.tsv            # 완전한 행 단위 정답
    chars/               # 글자 조각 이미지 (HRCNet 학습용)
    manifest.json        # 내용 해시·계보
  exemplar_bank.pkl      # 정답 예시 은행 (ExemplarBank)
  letter_prior.json      # 필체 사전값 (h_cv, aspect_med)
```

---

## 라벨 체계 (`LABELS.md` 참고)

| 계층 | 주요 라벨 |
|------|----------|
| 최상위 | `DOCUMENT`, `LETTER`, `BOOK_PAGE`, `FIGURE` |
| 영역 | `TEXT_BLOCK`, `TEXT_COLUMN`, `MARGIN_TEXT`, `ENVELOPE_TEXT`, `FIGURE_PANEL` |
| 도판 | `FIGURE_LABEL`, `FIGURE_TITLE`, `GRATICULE_TEXT`, `FIGURE_LEGEND`, `FIGURE_SCALE` |
| 글자 | `BODY_CHAR`, `NOTE_CHAR`, `INSERTED_CHAR`, `DELETED_MARK`, `ENVELOPE_CHAR`, `FIGURE_CHAR` |

---

## 모듈 구조

| 모듈 | 역할 |
|------|------|
| `cli.py` | `run()`, `postprocess()`, CLI 진입점 |
| `pipeline.py` | 쪽 단위 통합: textlayer → facsimile → tesseract → 판독문 정렬 |
| `align.py` | NW 정렬 (자모 초성 부분 일치·자모 편집거리) |
| `facsimile.py` | 영인 글자 검출 (이진화, 연결 요소, 열 분할) |
| `recognize.py` | 글자 조각 일괄 인식 (`--ocr-engine tesseract\|hrcnet\|none`) |
| `figure_manual.py` | 지도 수동 주석 (회전·중심선 기반 분할, `FIGURE_LABEL` 생성) |
| `kimssi.py` | 필사 편지·상언·책 전용 (행 정렬, 봉투 검출, 곤전어필 면 배정) |
| `exemplar.py` | 정답 예시 은행 (`ExemplarBank`) + RAG 기반 글자 위치 탐색 |
| `sections.py` | 판독문·해설 문단 구조 분석 (역할 분류, 면 번호 캡션) |
| `textlayer.py` | PDF 텍스트 레이어 글자 좌표 추출 |
| `normalize.py` | 한양 PUA → 첫가끝, NFC, 자모 분해, 글자마디 분할 |
| `export.py` | 검수용 오버레이 이미지, CSV 내보내기 |
| `evaluate.py` | 텍스트 레이어 기준 Tesseract 평가 (현대 활자만) |
| `labels.py` | 라벨 정의 (색·계층·설명) |

---

## 버전 이력

| 버전 | 주요 변경 |
|------|----------|
| v0.7 | 사민필지 지도 수동 주석 (`figure_manual`), 라벨 v0.4 (FIGURE_PANEL·GRATICULE_TEXT) |
| v0.8 | 넓은 행 분리 + DP 열 나누기 (흘림 편지 v1) |
| v0.9 | 행 끝 정리 + 정답 예시 검색 RAG (흘림 편지 v2) |
| v1.0 | 예정: HRCNet 엔진 연결, `evaluate_gold`, 회전 글자 검출기 |

---

## 다음 과제 (`ROADMAP.md` 참고)

1. **회전 글자 검출기** (사민필지 지도 4~8, 사람 수동 라벨 →학습)
2. **HRCNet 엔진 연결** (`recognize.py --ocr-engine hrcnet`)
3. **`evaluate_gold` 명령** (고정 평가셋 기준 자동 리포트)
4. **간이벽온 판독문 OCR** (벡터 윤곽선 → 고해상도 래스터 변환)
5. **삼강행실도 해상도 보강** (업스케일 후 영역 분리)
