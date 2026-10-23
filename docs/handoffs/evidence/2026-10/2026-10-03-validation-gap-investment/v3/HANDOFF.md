# 2026-10-03 20:49 KST · 정확특징2건 캐시/원본 추적

- 상태: 중단(사전등록 표본매핑 gate). 기록자: Codex하위검토.
- 요청·결정: v2보존, 정확해시2UID만 v3등록 후 readonly 비교. 원본18decode·40header·900초·512MiB 이내, labels/Gold/Report/prediction blind.
- 결과·파일: [cache비교](run/cache-comparison.json): 과거6x9x224와 현재64x336x336의 **두전체행/마스크가 각각완전동일**. 과거4394↔4404, 현재4397↔4407, fold1↔0. 동일모델입력 중복은 확인, 동일환자확정 아님.
- 검증·근거: [등록본](registration.json), [header영수증](run/header-receipt.json), [후속검증](verification.json), [상세보고](report.md). 첫7쌍 geometry/InstanceNumber일치, 8번째Axial대응은18↔12로 실패하여 교체없이중단.16header·0pixeldecode,0.110초·peak75.34MiB. PatientID원문미저장; 양쪽존재/hash다름, Name/DOB비어있음.
- 남은 일·다음: 최소후속은 이미geometry일치한7쌍만 새등록 후14decode. 먼저민감값예외/경고출력억제·adapter검증순서 보완. 현재입력교차fold중복을품질해석에표시하되 split/학습변경은없음. 중앙STATE/INDEX는루트담당.
