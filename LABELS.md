# 고문헌 영인 OCR 라벨 체계 (HanOCR v0.6)

좌표는 모두 **영인 원본 이미지 픽셀** `[x0, y0, x1, y1]`(좌상단 원점)이며, `bbox_pt`에 PDF 포인트 좌표를 함께 둡니다. 영역(region)은 `parent`로 계층을 이룹니다.

```
FRAME 광곽
├─ TEXT_BLOCK 본문 영역
│   └─ TEXT_COLUMN 본문 행 (role=heading 이면 소제목 행)
│        └─ DOUBLE_LINE_NOTE 협주 ─ NOTE_LINE(R/L)
└─ FIGURE 도판(지도)  subtype=MAP, figure_id='지도 1', figure_type='천체도'
     ├─ FIGURE_PANEL 도판 패널 (지도 2의 서반구·동반구 등)
     │    ├─ FIGURE_TITLE 도판 내 제목  text=디구셔편, modern=지구서편
     │    ├─ FIGURE_LABEL …
     │    └─ GRATICULE_TEXT 경위도 눈금 표기 (자동 검출, 미판독)
     ├─ FIGURE_LABEL 도판 내 표기  text=넵츈, modern=해왕성, entity_type, marker_bbox
     ├─ FIGURE_LEGEND 범례 ─ LEGEND_ITEM (기호 + 설명문)
     ├─ FIGURE_SCALE 축척 표시
     └─ FIGURE_SYMBOL 도판 기호  symbol_type=planet_marker | satellite_dot | sun | city_marker | capital_marker
MARGIN_TEXT 난외 표기(판심 상당, 광곽 바깥)
├─ MARGIN_TITLE 서명      예: ᄉᆞ민필지
├─ MARGIN_SECTION 편명·권차 예: 포츄갈국
└─ MARGIN_FOLIO 장차     예: ᄉᆞ십칠
```

## 영역 라벨

| 라벨 | 한국어 | 설명 |
|---|---|---|
| `FRAME` | 광곽(匡郭) | 인쇄면을 둘러싼 테두리선. 이중선이면 frame_lines=double |
| `TEXT_BLOCK` | 본문 영역 | 광곽 안의 본문 글자 영역 |
| `TEXT_COLUMN` | 본문 행(行) | 세로쓰기 한 줄. text 에 행 전사 |
| `DOUBLE_LINE_NOTE` | 협주(夾註, 쌍행 주석) | 본문 행 안의 작은 글씨 두 줄 주석 |
| `NOTE_LINE` | 협주 반행 | 협주의 오른쪽(R)/왼쪽(L) 작은 줄. R→L 순으로 읽음 |
| `FIGURE` | 도판(지도) | 지도·천체도·삽화 영역. subtype=MAP, figure_id=판독문의 '지도 N', figure_type=해설에서 추출(천체도 등) |
| `FIGURE_PANEL` | 도판 패널(반구도 등) | 한 도판 안의 하위 그림. panel_title, circle=[cx,cy,r], transcript_label_list |
| `FIGURE_TITLE` | 도판 내 제목 | 도판 안 제목 카르투시. 예: 디구셔편(현대어 지구서편) |
| `FIGURE_LEGEND` | 범례(凡例) | 지도 기호 설명 묶음 |
| `LEGEND_ITEM` | 범례 항목 | 기호 1개와 설명문. symbol_type, symbol_bbox, text(원문), modern(현대어역) |
| `FIGURE_SCALE` | 축척 표시 | 축척 막대(bar_bbox)와 설명문. 판독문에 없으면 제가 읽은 값(verification=manual-reading) |
| `GRATICULE_TEXT` | 경위도 눈금 표기 | 적도선·테두리 원 둘레의 도수 숫자 영역. 자동 검출이며 text=null |
| `FIGURE_LABEL` | 도판 내 표기 | 지도 안의 지명·천체명 표기 한 단위. text=원문, modern=현대어, entity_type, marker_bbox=가리키는 기호 |
| `FIGURE_SYMBOL` | 도판 기호 | 행성·지점 표시 점, 위성 점, 태양 등 (symbol_type) |
| `MARGIN_TEXT` | 난외 표기(판심 상당) | 광곽 바깥 여백의 서명·편명·장차 묶음 |
| `MARGIN_TITLE` | 난외 서명(書名) | 예: ᄉᆞ민필지 |
| `MARGIN_SECTION` | 난외 편명·권차 | 예: 포츄갈국 (판독문 머리줄 ※ 뒤) |
| `MARGIN_FOLIO` | 난외 장차(張次) | 예: ᄉᆞ십칠 (판독문 머리줄 〈 〉 안) |
| `MARGIN_UNKNOWN` | 난외 표기(미분류) | 판독문·사전으로 확정하지 못한 난외 글자열 |

## 글자 라벨

| 라벨 | 한국어 | 소속 영역 |
|---|---|---|
| `BODY_CHAR` | 본문 글자 | TEXT_COLUMN |
| `NOTE_CHAR` | 협주·작은 글자 | NOTE_LINE 또는 TEXT_COLUMN |
| `FIGURE_CHAR` | 도판 표기 글자 (`modern`에 현대어 표기) | FIGURE_LABEL |
| `MARGIN_CHAR` | 난외 글자 | MARGIN_TITLE / MARGIN_SECTION / MARGIN_FOLIO |

## 글자 값의 출처 (`text_source`)

- `transcript-aligned`: 판독문 본문과 정렬해 배정한 값
- `panshim-header`: 판독문 머리줄 `〈장차〉 ※ 편명`, 또는 서명 사전에서 배정한 값
- `figure-label-list(+manual)`: 판독문의 도판 표기 목록에서 배정한 값. `+manual`은 사람이 검수 보정 파일로 대응을 확정한 경우
- `ocr`: 판독문과 대응하지 못해 인식값을 그대로 둔 경우

## 판단 근거

- **FIGURE**: 판독문의 `지도 N` 줄을 `figure_id`로 씁니다. `figure_type`은 해설(지도해설)에서 ‘천체도’ 같은 낱말을 찾아 넣습니다. 이 자료집은 천체도도 ‘지도’로 번호를 매기므로 `subtype=MAP`으로 통일하고, 실제 유형은 `figure_type`으로 구분합니다.
- **FIGURE_LABEL의 현대어**: 해설의 `원문(현대어)` 쌍, 예를 들어 `머규리(수성)`을 우선 씁니다(`modern_source=commentary_gloss`). 쌍이 없으면 현대어역 표기 목록의 같은 순서 낱말을 씁니다(`label_list_order`).
- **MARGIN_* (난외 표기)**: 『사민필지』는 1889년 신식 활자본이어서, 목판본 판심(版心)에 해당하는 서명·편명·장차가 광곽 바깥 여백에 찍혀 있습니다. 그래서 ‘판심’ 대신 ‘난외 표기(판심 상당)’로 이름 붙였습니다. 다만 각 영역에 `note` 속성을 두어 판심과 호환되게 했습니다.

## FIGURE_LABEL 추가 속성 (v0.4)

| 속성 | 값 | 설명 |
|---|---|---|
| `writing_direction` | `vertical_ttb` | 세로쓰기(위→아래). 예: 북아메리까, 인도양 |
| | `vertical_spread` | 세로로 글자 간격을 넓게 벌려 씀. 예: 대 … 평 … 양 |
| | `horizontal_rtl` | 가로쓰기 오른쪽→왼쪽. 인쇄는 ‘극북’, 읽기는 북극 |
| | `horizontal_rtl_spread` | 오른쪽→왼쪽 가로쓰기인데 글자 간격이 넓음. 예: 아…프리까 |
| | `arc_rtl` | 극 둘레를 따라 호(弧)로 배치, 오른쪽→왼쪽. 예: 남빙양 |
| `in_transcript_list` | true/false | 판독문의 패널별 표기 목록에 있는가 (‘아시아’, ‘적도’는 목록에 없음) |
| `modern_source` | `commentary_gloss` / `label_list_order` / `commentary_text` / `manual` | 현대어 표기의 출처 |
| `n_chars`, `n_chars_visible` | 정수 | 가려져 보이지 않는 글자가 있으면 두 값이 다름 |
| `uncertain`, `note` | | 판독이나 용도가 불확실할 때 |

글자(`chars`) 항목에는 `visibility: visible|occluded`와 `bbox_tightened`를 둡니다. `occluded`는 해안선 등에 가려 보이지 않는 글자로 `bbox=null`입니다. `bbox_tightened=false`는 잉크로 조이지 못해 사람이 준 상자를 그대로 쓴 경우입니다.

**회전 인쇄 도판**: 지도가 쪽에 옆으로 인쇄된 경우(지도 2)에는 북쪽이 위로 오도록 회전한 이미지를 기준으로 좌표를 매깁니다. 회전 정보는 `image_orientation`에 적고, PDF 쪽 좌표는 `bbox_pt`로 함께 줍니다.


## v0.5 추가 사항

- **범례 연결**: 지도 위 색 점은 색으로 자동 검출합니다. 갈색·붉은 점은 `city_marker`(각 읍ᄂᆡ), 파란 점은 `capital_marker`(각국 도셩)이며, 각 점의 `legend_item`으로 해당 범례 항목과 연결됩니다.
- **`centerline`**: 비스듬하거나 휜 표기는 사람이 준 읽기 방향 중심선 `[x0,y0,x1,y1]`을 기록해 둡니다. 글자는 이 선 방향으로 투영해 분할합니다.
- **`writing_direction` 추가 값**: `diagonal`(비스듬한 표기), `diagonal_spread`(비스듬하고 글자 간격이 넓은 표기)
- **`bbox_estimated`**: 획이 선에 가려 잉크로 찾지 못해 기하적으로 추정한 글자 상자입니다.
- **`uncertain`**: 작은 글씨가 빗금·국경선과 겹쳐 글자 경계를 검수해야 하는 표기입니다.


## v0.6 — 소장자료총서 1 (김씨부인 한글상언 · 정조어필 한글편지첩 · 곤전어필)

```
DOCUMENT 고문서(1부 상언)       part, doc_title, image_segment (한 문서가 이미지 2장으로 나뉨)
├─ DOC_HEADER 머리 행             예: 튱쳥도 부여현 거 고 녕부ᄉᆞ 신 니이명 쳐 김시
├─ TEXT_COLUMN 본문 행            line_no=01~40, text(판독문), annotated_text(주석문 같은 번호 줄), modern(현대어역 같은 번호 줄)
└─ DATE_LINE 연기 행              예: 옹졍 오년 십월 일
LETTER 편지(2부)                  figure_id='편지 3', date='1755(?)~1759.2. 이전', caption{text,bbox_pt}, annotated_text, modern, commentary
├─ ENVELOPE_TEXT 피봉(봉투) 글씨   rotation=90|-90, reading_direction, text, modern
├─ TEXT_COLUMN 본문 행            elevated=true(대두), inserted_text
├─ SIGNATURE 발신자 표기          예: 元孫
└─ PAPER_DECORATION 시전지 문양·인장 (자동)
BOOK_PAGE 책 면(3부)              folio='1a' (이미지 아래 캡션), caption, transcript, modern
├─ SECTION_TITLE 책 속 제목       예: 만셕군뎐
└─ TEXT_COLUMN 본문 행
글자: BODY_CHAR / INSERTED_CHAR(삽입) / DELETED_MARK(∎) / ENVELOPE_CHAR(회전 글자, rotation)
```

| 라벨 | 한국어 | 설명 |
|---|---|---|
| `DOCUMENT` | 고문서(상언 등) | 한 장 이상 이미지로 나뉜 문서. part, doc_title, image_segment |
| `LETTER` | 편지 | 편지 한 통 이미지. figure_id='편지 3', date, caption, 주석문·현대어역·해설 |
| `BOOK_PAGE` | 책 면(面) | 책 한 면 사진. folio='1a' (이미지 아래 캡션) |
| `DOC_HEADER` | 문서 첫머리(발신·신원) | 상언 첫 줄처럼 행 번호 없는 머리 행 |
| `DATE_LINE` | 연기(年記)·날짜 행 | 예: 옹졍 오년 십월 일 |
| `SIGNATURE` | 발신자 표기(서명) | 편지 끝의 발신자 표기. 예: 元孫 |
| `ENVELOPE_TEXT` | 피봉(봉투) 글씨 | 편지에 오려 붙인 봉투 글씨. rotation(도)=글자가 돌아간 각도 |
| `SECTION_TITLE` | 책 속 제목 | 예: 만셕군뎐 |
| `PAPER_DECORATION` | 시전지 문양·인장 | 편지지의 채색 문양이나 인장. kind=decoration|seal (자동 검출) |
| `INSERTED_CHAR` | 삽입 글자 |  |
| `DELETED_MARK` | 삭제·판독불가 표시(∎) |  |
| `ENVELOPE_CHAR` | 봉투 글자 |  |

**캡션 보존 방식**
- 이미지 아래 캡션은 이미지 밖에 있으므로 영역 상자 대신 최상위 영역의 속성으로 둡니다: `caption: {text, bbox_pt}`
- 편지는 `figure_id`와 `date`(예: `<편지 13> 1796.12.20.` → `편지 13`, `1796.12.20.`), 책 면은 `folio`(예: `1b`)로 따로 저장합니다.


## v0.7 — 곤전어필 행 배정 속성

`TEXT_COLUMN`(BOOK_PAGE)에 다음 속성이 추가되었습니다.
- `line_no`: 행 번호. SECTION_TITLE 행은 빼고 1부터 셉니다.
- `detected_boxes`: 자동 검출된 글자 상자 수. 참고용이며, 최종 글자 수는 판독문 배정으로 정해집니다.
- `verification`: `line-start-anchored`는 사람이 확인한 행 글자열(`corrections/kimssi_gonjeon.json`)로 고정된 행입니다.
- `needs_review`: 잉크 잘림 적합도와 상자 수가 배정된 글자 수와 어긋나 검수를 권하는 행입니다.

글자의 `text_source`는 `transcript-column-allocated`로, 판독문을 행 단위로 배정한 뒤 행 안에서 글자 수만큼 나눈 값입니다.
