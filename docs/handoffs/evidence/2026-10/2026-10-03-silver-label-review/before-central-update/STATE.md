# 현재 상태

2026-10-03 KST. [인계 색인](handoffs/INDEX.md) · [월별 이력](handoffs/monthly/2026-10.md).

## 범위와 운영

PC별 KneeOps localhost:8765·개인 DB/설정/제출이력. KneeOps.cmd 시작·KneeOps-Stop.cmd 종료. PC 종료/절전 시 중단, 부팅 자동실행 없음. 공용서버·팀공유·RAG/MCP 제외. [범위](decisions/0002-personal-research.md) · [운영](runbooks/dashboard.md).

Layout v2·knee_workspace. [경로/복구](runbooks/workspace-layout.md). 앱·Kaggle은 .venv, 학습은 GPU Python. 로컬4070→무료Kaggle/보유Colab→비용제안. 제출·공개는 매번 승인, 유료는 별도승인. 중단지시 전 연구를 계속하며 일5회 기회를 근거 있게 활용한다. [지속 연구](runbooks/continuous-research.md).

## 구현·자료

- 개인 연구실: 자료·노트북/토론·실험/제출 검색, 출처/revision/NEW·LB·예측비교·diff·다운로드. 정기수집 꺼짐. [구현](architecture/implemented-dashboard.md) · [사용법](runbooks/dashboard.md).
- MRI 영상·ROI·EDA·원본/캐시·전처리 비교. 이전hashed assets 보존. [영상](runbooks/image-viewer.md).
- ZIP265GB, train4,407+test3, DICOM819,635. 전체CRC/픽셀감사 미완료. 원본/캐시 불변. [데이터](runbooks/training-data.md).
- Raptor corpus4,410·보고서그룹fold. Gold58은 노출진단셋, 테스트 Report 없음. 환자독립 미확인. [학습](runbooks/training-raptor.md) · [원격](runbooks/remote-gpu.md).

## 현재 연구·실행

1. 10-03 11:34 관측: 제출56773226=0.943,1,129/4,943위·선두0.961. 기준본 보존. R2는 teacher 간접누출·혼합라벨 epoch선택으로 독립 개선 확정 불가. EXP074~076 중단 유지.
2. [6시간 재점검](research/six-hour-reassessment-2026-10-03.md): 내부개선과0.943전이공백. EXP078~080은ImageNet초기화,노출weak개발fold·환자독립미검증. F2완료후추가fold/seed자동확대보류. 30건187series감사완료,6쌍영상범위차·대안우월미확정.
3. EXP078 B−A+.003202/CI0포함·F1골절−.0280. EXP079 RN보완+.0021175,EXP080 F2 RN+.001243이나B−A골절−.022337반복. B임시·작은RN효과보존·추가학습보류. [F2판정](research/exp080-f2-result-2026-10-03.md). [EXP081](research/exp081-coronal-span-2026-10-03.md) GPU2pass149초완료·834exact/34변경,macro−.000684 CI0포함. 현재가중치max-span교체미채택,커버리지가설전체기각아님.
4. D01 원v1hashFAIL·통합v1경로NO_GO보존. 통합133 PASS,보수1300추정8.594h/9h·메모리29.541/30GiB로full보장없음. 최종공개3/출력9종20:42검증PASS(401.6초). 사용자승인후20:46 제출**56796964**,kernel candidate-v3 v1·PENDING/점수없음·당일4회잔여. 정확도개선미입증. [상세](handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json).
5. EXP082 RN ones21:27완료691초/skip0·고정평가1회:혼합+.000534 CI[−.000268,+.001293],RN단독+.006828 CI0포함. 작은개발후보보존/미채택·D01근거아님. group계약4343건/4196components완료,새split없음·환자독립미확정. EXP083 slot-bias 두head CPU준비·GPU미시작,22:30목표/22:10확인·자동확대없음.
6. 장시간 작업은 같은채팅 heartbeat `exp078-a-b`, 첫 ETA20분 전·미완료면 갱신ETA. 실시간ID/로그/예약은 인계. 로컬 학습1개·중복금지.

[전략 누락 점검](runbooks/strategy-review.md)은 결정 전·주1회. weak/Gold/간접누출/LB/private를 구분하고 공개LB 사후 비율튜닝을 금한다. [평가](decisions/0004-validation-protocol.md) · [전체방향](research/route-to-first-2026-10-03.md). 매작업 기록·[관리](runbooks/handoff-maintenance.md).
