# 디스커션 재확인 후 실행 우선순위 — 2026-10-03

10-03 13시에는 로컬 소규모 대조까지 진행하도록 승인됐다. 제출·공개·유료 자원은 별도 승인이다. 장시간 작업은 첫 ETA 20분 전에 같은 채팅 heartbeat로 확인하고, 미완료이면 진행률로 갱신한 ETA에 재예약한다. 기다리는 반복문 대신 실제 종료/산출물 확인 후 허가된 후속을 진행한다.

2026-10-03 후속 지시: 중단 지시 전 [지속 연구](../runbooks/continuous-research.md)를 이어간다. EXP078의4fit·gate는 그대로 유지하고 결과 검토 후 허가 범위 안에서 다음 실험을 새로 등록한다. 서로 다른 사전 고정 가설과 완성 후보가 있으면 하루 최대5회 기회를 활용하되 제출·공개는 매번 승인받는다. 횟수 소진·LB에 맞춘 비율/소견 반복튜닝은 하지 않는다. [이번 변경 전 원문](../handoffs/archive/snapshots/2026-10-03-continuous-research/manifest.json).

## 판단 근거

13:05~13:08 KST 최근 목록60개·관련16개 원문/전체 댓글을 Kaggle 스킬의 SDK adapter로 다시 확인했다. **앱의 당일 수집본 대비 새 댓글·본문 변경은0건**이다. 아래는 새 성능 발견이 아닌 최신 재확인이다. [URL·시각·SHA·수집범위](../handoffs/evidence/2026-10/2026-10-03-execution/discussions/manifest.json), [판단과 메시지ID](../handoffs/evidence/2026-10/2026-10-03-execution/discussions/review.json).

| 근거 | 의미와 실행 결정 |
|---|---|
| [733491](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491) 주최 답변3509957 | 정답은 영상 재판독. 보고서 weak CV 향상을 임상 정답 향상으로 단정하지 않는다. |
| [745214](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214) 댓글3531864 등 | 단순 Qwen 라벨로low0.95 주장과 라벨 변경0.931→0.942 주장이 공존한다. 모두 참가자 자기보고이며 라벨만 병목이라는 증거가 아니다. |
| [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) 댓글3529062 | 50/50 혼합의 Gold/원라벨CV 상승이LB로 전이되지 않았다는 반례. 기존 R2와 nested 효과를 과대 해석하지 않는다. |
| 같은 글·[743148](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148) | 작은224 모델과 입력/학습 조건 개선 사례는 작은 통제 실험을 지지한다. 성능 수치는 자기보고다. |
| [742050](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050) 댓글3525875 | 독립 신호 없는 blend 비율 반복제출 경고.0.943 보존·가설 사전 고정·제출 제한 유지. |
| [744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230) | T4 BF16 에뮬레이션/시간초과/상수 fallback 사례. native BF16 확인·정밀도 무단변경 금지·실제장치 parity 우선. 우리FP16 실행에 동일 실패가 있었다는 뜻은 아니다. |
| [744961](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744961)·[744519](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744519) | 공개checkpoint 검증 노출·private plane 완전성 질문에 주최 답변이 없다. 미답변을 보증으로 해석하지 않는다. |

## 우선순위와 범위

1. **P0 — 비교의 기본 조건 보완.** 정밀도/TF32/cuDNN 기록, unsupported AMP·NaN/Inf·실패 즉시 중단, checkpoint norm/window 계약 준수, 샘플링 재현, 학습/평가 라벨 분리, 고정8epoch 선택, Gold 선택 금지, 실제optimizer update와FP16 skip 기록. 공개 기준 노트북은 그대로 보존한다.
2. **P1 — 기존 후보의 실제 입력 연결.** 신규 UID+test_series+DICOM에서 만드는 adapter와 기존corpus의 동일 study 입력/예측을 비교한다. Report·라벨을 추론 입력으로 쓰지 않는다. 대표표본 runtime·실패 상태까지 확인한 뒤 패키지를 검토하며, sample 성공으로 숨겨진 전체 실행을 보장하지 않는다.
3. **P2 —224px 입력 정책 A/B 한 가설.** 새 baseline A=`global-valid`, B=`within-slot-valid`. 두 정책 모두 mask상 비어 있지 않은3장만 허용하고, B는 동일slot 안의3장만 쓴다. candidate pool과 추출창 분포가 달라지므로 ‘경계창만 교체하고 나머지는 동일’은 아니다. 성능 개선은 아직 가설이다.
4. **P3 — 결과 기반 후속.** 두 개발fold에서 개선·target/group 안정성·실행품질을 함께 검토한다. 유망하면 추가seed 또는 다음fold 한 단계만 제안/진행한다. 불확실하면 원인 검토로 돌아가며 nested teacher·384·5fold×2seed를 자동 확대하지 않는다.

## 첫 대조의 사전 조건

- 같은 로컬4070·같은 코드/초기weights·원보고서 라벨·FP16 학습/평가·224px·8epoch·seed42. F0/F1 각각 A/B를 새로 학습한다. 기존 노출fold이므로 untouched holdout이 아닌 개발 대조다. 이전R1/R2 점수를 새로운baseline 대신 쓰지 않는다.
- 원래 그룹fold를 유지한다. mask상 유효3장이 전혀 없는1개 study(fold2)와 실데이터 parity에서 ZIP CRC 오류가 확인된1개 study(fold3)를 두arm 모두 사전 격리한다. F0/F1 평가869건씩에는 해당하지 않는다. 두 손상은 표준ZipFile로도 재현됐으며 원본을 수정하지 않는다. mask는 픽셀합>0 지표이지 실제 acquisition 보증이 아니다.
- epoch8은 사전에 고정하며 해당 결과만 평가한다. Gold58은 최종 오류 진단으로만 출력하며 채택·중단·hyperparameter 결정에 쓰지 않는다. report-group bootstrap은 고정 예측의 조건부 불확실성만 측정한다.
- 확대 검토 기준: 두fold의 고정reference macro Δ가 양수, 평균Δ≥0.002, target별 하락·그룹 편차 검토, nonfinite/fallback/계약 오류0. Δ기준은 운영상의 선별 규칙이며 임상 향상 증명이나 private 보장이 아니다. 큰 하락을 macro로 숨기지 않는다. 약한 근거를 더 많은LB 제출로 해결하지 않는다.
- FP16 optimizer skip 비율>1% 또는10회 연속skip은 수치 안정성 검토 대상이다. 시간/메모리/처리량은 첫 실제epoch로 갱신하며 배정 예산을 소진하는 것이 목표가 아니다.
- 연구 gate는 네 개의 사전 고정run 후 검토한다. 같은 대조 안의4fit 순차 실행 외에 자동 연구 확대는 없다. 예상 GPU3~5h는 기존throughput 기록 기반이므로 strict 모드 최초실측에 따라 바뀐다.

## 검증과 기록

수정 전 코드/문서는 [보존본](../handoffs/evidence/2026-10/2026-10-03-priority-execution/prechange/manifest.json)에 있다. smoke·실행명세·새 결과·예약ID는 [인계 색인](../handoffs/INDEX.md)의 해당 작업 기록을 따른다. 모든 신규 산출물은 새 경로를 사용한다. 코드 검사 통과와 모델 성능 개선을 구분한다.

앞선 실데이터 확인: 공개test3건의 DICOM→model 입력hash와 예측은 기존cache와 완전 일치했다. 대표train100건 중99건은volume/mask byte일치,1건은ZIP CRC오류로 strict중단했다. 별도 기존empty study는ZIP local-header오류였다. 따라서100건전체통과·완전한hidden-serving완료라고 주장하지 않는다. [sample3](../../artifacts/predictions/serving-20261003/sample3-comparison.json)·[대표100건](../../artifacts/predictions/serving-20261003/cpu-parity100.json)·[표준ZIP 재현](../handoffs/evidence/2026-10/2026-10-03-priority-execution/archive-failures.json).

[14:11 KST 후속 검사](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-package-verification.json):5fold decode공유 ensemble16개·기존serving10개·D01 package9개CPU 검사가 통과했다. D01 로컬stage 자산은1,469,070,653bytes(약1.469GB), anchor원본30셀을 보존한 후보32셀이다. [14:15 비공개 업로드 시작](../handoffs/evidence/2026-10/2026-10-03-continuous-research/upload-launch.json) 뒤 실제Kaggle T4 parity·대표표본/전체runtime·부분손상 처리정책 확인이 남았다. `0.9×rank(anchor)+0.1×rank(mean5(raw R2))`는 사전 고정 탐색이며 CPU 검사·sample3 성공을 점수 향상이나 private 안정성으로 해석하지 않는다.

## Biohub 회고를 반영한 판단 점검 (2026-10-03)

[누락 점검](../runbooks/strategy-review.md)을 EXP078 결과 해석과 다음 방향 결정에 적용한다. 사전등록 gate·실행 중 코드·4fit 예산은 유지한다. Δ0.002 미달은 자동 확대 기준 미충족이며 효과 없음의 증거가 아니다. 작은 양의 효과는 CI·소견/group 일관성·비용과 함께 보존/보류를 판단한다. 라벨 혼합 특정 실패를 라벨 전략 전체의 기각으로 확대하지 않는다. 네fit 종료 후 소견별 오류 기여와 입력/학습/감독신호/encoder의 큰 가설3~5개를 비교한다. 현재 대조만으로 우승 격차를 설명하거나 도구 검증을 성능 근거로 삼지 않는다.


## 15:23 재개 점검 — 다음 가설과 투자 순서

현재 성능을 올렸다는 새 근거는 없다. EXP078 F0의 양의 차이는 F1까지 고정 대조를 마친 뒤 판단한다. serving 검사는 점수 증거가 아니지만, 검증되지 않은 노트북에 제출 기회를 쓰지 않기 위한 선행 작업이다. [실시간 근거](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json).

| 가설 | 현재 근거·한계 | 가장 작은 다음 행동·비용 |
|---|---|---|
| 입력의 유효 창 분포 | EXP078 4fit 중2완료. F0만으로 정책 선택 금지 | 남은2fit 완료 후 기존 gate와 별도 소견/group 진단. 전체ETA16:20~16:40,후속추가학습 미배정 |
| MCL 감독 정의 정합성 | 저등급/과거손상 오류 반례. 가림20건은 음성후보12/unknown7/미언급1; 명시급성 부재를 음성으로 만들 수 없음 | 나머지100건을 같은규칙으로 동결. 정의의 coverage/모호성을 먼저 판단, 충분할 때만 별도 train-label 버전과2fit 대조(약1.7~2h). 임상 검증 아님 |
| 표현/encoder의 PFOA 보완 | 기존 ResNet의 PFOA 방향은5fold 일관하지만 lr·20epoch·OOF선택까지 달라 구조 효과로 분리되지 않음 | 기존 예측 상보성 CPU분석 후 필요시 새고정2fold 레시피. 과거 ResNet20epoch2fit은약80~100분,strict새조건은미측정. 소견별routing/LB튜닝 자동실행 없음 |
| 시리즈 선택·정보 손실 | 동일 plane/FS 다중후보·mask결측은 오류 원인 후보이며 오선택의 증거는 아님 | 학습 없이 series metadata/선택receipt의 극단20~30건 비교,분포·선정규칙 고정까지약30~60분 예상. 개선원인이구체화되면2fit을별도배정 |
| R2와 anchor 오류 보완 | 기존R2는간접누출·혼합라벨선택으로 개선효과 독립추정 불가. 고정90/10은전이확인용 가설 | 새입력계약·codec·whole runtime을통과한private후보 후매회승인. 비율검색안함. 실패시라벨/독립모델전략전체를기각하지않음 |

우선순위는 **실행 중4fit과 가림검토 완료 → 비용 대비 다음 성능 가설 하나 선택**이다. 동시에 진행하는 D01은 제출 가능한지 판단할 최소 실행 검증까지만 한다. 헤더 최적화10건은 읽기70.5% 감소에도 로컬9.48→10.01초로 빨라지지 않았다. 동일성과 원격 실측 전에는 이 최적화를 채택하거나 시간 절감으로 계산하지 않는다.

새 [소견/group 보조 도구](../../tools/input_ab_diagnostics.py)는 모든4fit과 primary report를 요구한다. P/N·12소견별 AUC/Δ·fold별장비/결측·report-group CI를 기록하되 사전 gate를 바꾸지 않는다. 작성시 F0결과는이미관측됐으므로 새사전등록 성능검정으로 부르지 않는다. CPU3검사통과,실제4fit 실행은대기중이다.


15:35 결과 반영: [MCL120 동결후 대조](../handoffs/evidence/2026-10/2026-10-03-mcl-review/unblind-v1/report.md)에서 음성후보55건의 Dread 이진양성은0건,다른소스는31~36건이었다. Dread .2→0은이진오류수정이아니라감독강도변경이다. 현재 MCL2fit은보류하며 입력대조·표현가설을우선한다. 이전반례를표본에서제외했으므로라벨연구전체기각으로확대하지 않는다.

동시 serving 결과: 새수치계약 로컬/Kaggle CPU107건모두PASS,원본anchor50의4CoAt순차경로완료1,082초. 단순1,300건환산은anchor+R2=9.785h로시간gate미충족이다. 고정시작비용을여러번계산하는보수적추정이며full실패증명은아니지만현재후보공개/제출준비완료라할수없다. CPU헤더대조100건을실행해최소한의속도보완가능성을확인하고최종T4실행은그결과뒤결정한다. 원래v1FAIL은보존했다.


16:03 검토: [PFOA 상보성](pfoa-complementarity-proposal-2026-10-03.md)에서 ResNet은 PFOA pooled +0.0120이나 전체 macro −0.0118, F1 PFOA는 단2순위쌍 차이다. 구조/LR/epoch/선택/환경이 혼재해 구조 효과로 단정하지 않는다. EXP078 원래 gate로 arm을 정한 뒤 그 baseline과 기존 ResNet F0/F1을 90/10 평균동순위 rank로 단 한 번 CPU 선별한다. 유망할 때만 동일 baseline을 재사용하는 새 strict ResNet20epoch 두fit(잠정1.5~2h)을 검토한다. 자동학습·PFOA routing·비율검색은 하지 않는다. anchor100 T4/헤더v2 CPU는 별도 실행시간 연구이며 성능 근거가 아니다.


16:38 실행: [EXP078](exp078-input-ab-result-2026-10-03.md)은mean+.003202·두fold양수이나CI0포함/F1골절−.0280이남아B를개발임시기준으로만선택했다. [EXP079](exp079-r34-complement-2026-10-03.md)의고정90/10 CPU선별은mean+.001593·두fold/PFOA/다른11평균양수로확인됐다. 새strict ResNet20epoch2fit만배정하고B를재사용한다. 최초epoch40초라기존T4기반예산1.5~2h보다짧은전체ETA17:02~17:10으로갱신했다. 신규대조는레시피보완검사이며독립검증/구조효과/제출채택은아니다.
