# 현재 상태

2026-10-03 KST. [인계 색인](handoffs/INDEX.md) · [월별 이력](handoffs/monthly/2026-10.md).

## 범위와 운영

PC별 KneeOps localhost:8765·개인 DB/설정/제출이력. KneeOps.cmd 시작·KneeOps-Stop.cmd 종료. PC 종료/절전 시 중단, 부팅 자동실행 없음. 공용서버·팀공유·RAG/MCP 제외. [범위](decisions/0002-personal-research.md) · [운영](runbooks/dashboard.md).

Layout v2·knee_workspace. [경로/복구](runbooks/workspace-layout.md). 앱·Kaggle은 .venv, 학습은 GPU Python. 로컬4070→무료Kaggle/보유Colab→비용제안. 제출·공개는 매번 승인, 유료는 별도승인. 중단지시 전 연구를 계속하며 일5회 기회를 근거 있게 활용한다. [지속 연구](runbooks/continuous-research.md).

## 구현·자료

- 개인 연구실: 수집자료·노트북/토론·실험/제출 검색, 출처/revision/NEW, 전체LB·예측비교·노트북diff·다운로드 관리. 정기수집 기본 꺼짐. [구현](architecture/implemented-dashboard.md) · [사용법](runbooks/dashboard.md).
- MRI 영상·ROI·EDA·원본/캐시·전처리 비교. 이전hashed assets 보존. [영상](runbooks/image-viewer.md).
- ZIP265GB, train4,407+test3, DICOM819,635. 전체CRC/픽셀감사 미완료. 원본/캐시 불변. [데이터](runbooks/training-data.md).
- Raptor corpus4,410·보고서그룹fold. Gold58은 노출진단셋, 테스트 Report 없음. 환자독립 미확인. [학습](runbooks/training-raptor.md) · [원격](runbooks/remote-gpu.md).

## 현재 연구·실행

1. 10-03 11:34 관측: 제출56773226=0.943,1,129/4,943위·선두0.961. 기준본 보존. R2는 teacher 간접누출·혼합라벨 epoch선택으로 독립 개선 확정 불가. EXP074~076 중단 유지.
2. [실행 우선순위](research/execution-priorities-2026-10-03.md): strict정밀도·stable RNG·학습/평가라벨 분리·fixed epoch 구현. 저장T4↔4070 R2F0 869건 max확률차0.0004814는 부분근거, 새T4 전체스택 미완료.
3. EXP078 입력A/B:224px·F0/F1·고정8epoch·4fit. A_F0/B_F0/A_F1완료,B_F1실행(16:03). [명세](../artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/manifest.json). 네fit 후판정,전체ETA16:18~16:25. 자동확대 없음.
4. D01 고정90/10·원본30셀보존. v1입력hash FAIL유지. 새reference검사 로컬/KaggleCPU107건PASS·codec압축fixture PASS. R2 T4 100건604초,anchor50 1,082초·49+순차경로완료. 단순1,300건합산9.785h,시간gate보류. Header v1좌표해시FAIL보존·별도v2 CPU재검사,anchor100 T4실행중. releasev2구현·최종T4/예산후후보생성. [상세](handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json).
5. [MCL120 동결후대조](handoffs/evidence/2026-10/2026-10-03-mcl-review/unblind-v1/report.md):음성후보55모두기존Dread이진음성,unknown56보존. 새수정근거/독립평가기준부족으로MCL2fit보류,라벨연구전체기각아님. 입력대조/PFOA표현후보우선,자동재라벨없음.
6. 장시간 작업은 같은채팅 heartbeat `exp078-a-b`, 첫 ETA20분 전·미완료면 갱신ETA. 실시간ID/로그/예약은 인계. 로컬 GPU 학습1개, 중복실행 금지.

[전략 누락 점검](runbooks/strategy-review.md)은 결정 전·주1회. weak/Gold/간접누출/LB/private를 구분하고 공개LB 사후 비율튜닝을 금한다. [평가](decisions/0004-validation-protocol.md) · [전체방향](research/route-to-first-2026-10-03.md). 매작업 기록·[관리](runbooks/handoff-maintenance.md).
