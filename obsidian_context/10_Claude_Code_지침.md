# Claude Code 초기 지침

이 파일을 Claude Code 세션 시작 시 `/hanocr` 레포지토리 초기 컨텍스트로 사용합니다.

## 프로젝트 규칙

### 코드 수정 시
1. 기존 파이프라인(`pipeline.py`, `align.py`, `facsimile.py`)의 인터페이스를 바꾸지 않습니다.  
   새 기능은 파라미터 추가 or 별도 함수로 분리합니다.
2. 영인 글자 `text` 필드는 반드시 판독문 정렬 결과 or 사람 검수값이어야 합니다.  
   테서랙트 raw 결과를 `text`에 넣으면 안 됩니다. (`rec`에 보존)
3. 새 라벨을 추가할 때는 `labels.py`와 `LABELS.md`를 함께 업데이트합니다.

### 데이터 수정 시
1. 검수하지 않은 제안을 정답으로 바꾸지 않습니다.
2. gold 스냅샷은 내용 해시와 함께 저장합니다.
3. 평가셋(`split=eval`)은 학습 데이터와 섞지 않습니다.

### 성능 비교 시
1. 항상 같은 평가셋(현재 `eval-v1`, 편지 2, 30자)을 사용합니다.
2. 비교 수치는 "어떤 스냅샷으로 학습했는지"를 함께 기록합니다.
3. 편지 3처럼 학습셋 자기 평가 수치와 독립 검증 수치를 구분합니다.

## 주요 경로
```
/home/claude/hanocr/
  hanocr/          ← 패키지 소스
  corrections/     ← 수동 보정 JSON
  README.md        ← 이 세션에서 업데이트됨
  LABELS.md        ← v0.7 라벨 정의
  ROADMAP.md       ← 다음 과제

```

## 세션 시작 체크리스트
- [ ] `/home/claude/hanocr/` 코드 최신 버전 확인
- [ ] `gold-r02/manifest.json` 스냅샷 해시 확인
- [ ] 성능 보드 `champion` 필드 확인 (아직 없으면 HRCNet 기준선 등록 필요)
- [ ] `pending_items` 확인 (현재: 편지 13)

## 빠른 테스트
```bash
cd /home/claude/hanocr
PYTHONPATH=. python -c "
from hanocr.kimssi import process_letter
import pymupdf as fitz, glob, json, pickle
d = fitz.open(sorted(glob.glob('/mnt/user-data/uploads/*.pdf'))[3])
# 편지 4 (PDF 15) 빠른 테스트
res = process_letter(d, 15, '/tmp/test_letter4', split_wide=True)
print(len(res[0]['chars']), 'chars', res[0]['alignment'])
"
```
