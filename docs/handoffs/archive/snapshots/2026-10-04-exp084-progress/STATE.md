# 현재 상태

2026-10-04 KST. [통합 인계](../HANDOFF.md) · [월별 이력](handoffs/monthly/2026-10.md).

## 범위와 운영

PC별 KneeOps localhost:8765·개인 DB/설정/제출이력. PC 종료/절전 시 중단·부팅 자동실행 없음. 공용서버·팀공유·RAG/MCP 제외. [범위](decisions/0002-personal-research.md) · [운영](runbooks/dashboard.md).

Layout v2·knee_workspace. [경로/복구](runbooks/workspace-layout.md). 앱·Kaggle은 .venv, 학습은 GPU Python. 로컬4070→무료Kaggle/보유Colab→비용제안. 제출·공개는 매번 승인, 유료는 별도승인. 중단지시 전 연구를 계속하며 일5회 기회를 근거 있게 활용한다. [지속 연구](runbooks/continuous-research.md).

## 구현·자료

- 개인 연구실: 자료·노트북/토론·실험/제출 검색, 출처/revision/NEW·LB·예측비교·diff·다운로드. 정기수집 꺼짐. [구현](architecture/implemented-dashboard.md) · [사용법](runbooks/dashboard.md).
- MRI 영상·ROI·EDA·원본/캐시·전처리 비교. 이전hashed assets 보존. [영상](runbooks/image-viewer.md).
- ZIP265GB, train4,407+test3. 전체CRC/픽셀감사 미완료. 원본/캐시 불변. [데이터](runbooks/training-data.md).
- 보고서그룹fold. Gold58 선택노출·Silver unknown/confidence loss 미적용. test Report 없음. [학습](runbooks/training-raptor.md) · [원격](runbooks/remote-gpu.md).

## 현재 연구·실행

1. 10-03 11:34 관측: 제출56773226=0.943,1,129/4,943위·선두0.961. 기준본 보존. R2는 teacher 간접누출·혼합라벨 epoch선택으로 독립 개선 확정 불가. EXP074~076 중단 유지.
2. [6시간 재점검](research/six-hour-reassessment-2026-10-03.md): 내부개선과0.943전이공백. EXP078~080은ImageNet초기화,노출weak개발fold·환자독립미검증. F2완료후추가fold/seed자동확대보류. 정확D01 독립검증 미확보;새group검증은ancestor포함 재학습필요.
3. EXP078~080: B입력임시·RN작은보완효과보존,골절손실반복으로확대보류. [F2판정](research/exp080-f2-result-2026-10-03.md). [EXP081](research/exp081-coronal-span-2026-10-03.md) max-span교체−.000684/CI0포함·미채택,coverage전체기각아님.
4. D01 **56796964**/candidate-v3 v1,10-03 20:46승인제출·20:46:32 PENDING. 통합133/공개3실행PASS,1300보수8.594h/9h·29.541/30GiB로full/개선미보장. 원hashFAIL/통합경로NO_GO보존. [상세](handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json).
5. EXP082혼합+.000534 CI[−.000268,+.001293]·작은개발후보보존/미채택. [EXP083](research/exp083-slot-bias-result-2026-10-03.md) v1실패보존·v2평가1회. slotbias−.0000246 CI[−.000215,+.000171],두refit모두원B보다−.01155로현설정미채택. 정확D01근거아님.
6. Silver raw3475 형식유효·1Baker누락. PFOA20/Synovitis16 가림대조 완료, Dread 자동교정/추가표본 근거부족·감독전체기각 아님. [근거](research/silver-other-research-and-pfoa-2026-10-03.md).
7. EXP084 새group T3455/V888·RNclip20/RNones20/B8,01:03순차GPU시작·ETA02:10–02:30. Gold/중간점수금지·3fit후1회평가. 기존B/RN6weights CPU패키지완료·입력/T4미검증. [현재실행](../HANDOFF.md).

[전략 누락 점검](runbooks/strategy-review.md)은 결정 전·주1회. weak/Gold/간접누출/LB/private를 구분하고 공개LB 사후 비율튜닝을 금한다. [평가](decisions/0004-validation-protocol.md) · [전체방향](research/route-to-first-2026-10-03.md). 매작업 기록·[관리](runbooks/handoff-maintenance.md).
