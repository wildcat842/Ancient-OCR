# HanOCR — Claude Code 프로젝트 지침

## 한 줄 요약
국립한글박물관 소장자료총서 4종의 글자 단위 바운딩박스 + 텍스트 데이터셋 구축.  
판독문 정렬 기반 자동 라벨링 → S2 편집기 검수 → HRCNet 미세조정 → 성능 보드 비교.

## 컨텍스트 파일 위치
`obsidian_context/` 폴더의 11개 `.md` 파일이 전체 프로젝트 맥락입니다.  
세션 시작 시 이 폴더를 먼저 읽으세요.

## 핵심 규칙 (반드시 지킬 것)

1. **제안과 정답 분리**: 기계 제안은 `proposed`로만 저장. 사람이 확정한 것만 gold.
2. **`text` vs `rec` 구분**: `text` = 최종값(판독문 우선), `rec` = 테서랙트 원본. `rec`를 `text`에 넣으면 안 됩니다.
3. **평가셋 불변**: `eval-v1`(편지 2, 30자)은 학습에 섞지 않습니다.
4. **계보 기록**: 학습 스냅샷 이름 + 내용 해시를 함께 남깁니다.

## 세션 시작 체크리스트

```bash
cat obsidian_context/08_현재_상태_진행현황.md   # 현재 상태 확인
cat corrections/samin_pilji.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(list(d['pages'].keys()))"  # 완료된 지도 쪽
python3 -c "import csv; rows=list(csv.DictReader(open('gold/gold-r02/chars.tsv'),delimiter='\t')); print(len(rows), 'chars')"
```

## 빠른 테스트
```bash
PYTHONPATH=. python3 -c "
from hanocr.cli import run
# 사민필지 60쪽 (텍스트 레이어 있는 본문)
# run('사민필지.pdf', 'out/test', pages='60')
print('import ok')
"
```

## 모듈 진입점
- `cli.py` → `run()`, `postprocess()`
- `kimssi.py` → `process_letter()`, `process_book_page()`
- `figure_manual.py` → `build()` (지도 수동 주석)
- `evaluate_gold.py` → HRCNet 결과 JSON 생성 (성능 보드 등록용)

## 현재 단계 (2026-09-19)
- gold-r02: 264자 (학습 234 / 평가 30), 편지 1·2·3·4
- HRCNet 평가 대기 (gold-r02 패키지 준비 완료)
- 다음: 편지 13 검수 → gold-r03 → HRCNet 기준선 평가
