# RSNA Knee 상위권 전략과 실행 계획 — 2026-10-03

17:40 재점검: 현재 우선순위·검증 해석은 [6시간 재평가](six-hour-reassessment-2026-10-03.md)를 따른다. 아래 시각별 내용은 당시 관측/결정이다. F2만 완료하고 D01통합검증1회·가림30건감사를 우선하며 추가fold/seed는 자동확대하지 않는다.

갱신: 2026-10-03 11:34 KST 리더보드/제출 직접 확인, 이후 방향 정리. 목표는 최종 private LB의 최대한 높은 순위다. **공개 LB 과적합을 피하면서 0.943 기준본에 검증 가능한 보완 신호를 더한다.** 이것이 현재 실행 계획이다. [노트북12개 비교](notebook-starting-point-2026-10-03.md), [전략 결정](../decisions/0003-baseline-and-route-to-first.md), [검증 규칙](../decisions/0004-validation-protocol.md).

10-03 13시 실행 갱신: 토론 재확인 후 로컬 소규모 대조가 승인됐다. 현재 상세 기준은 [실행 우선순위](execution-priorities-2026-10-03.md)이며, 아래 과거 LB 수치와 감사 관측은 해당 시각 자료다. 이전 계획은 [변경 전 원문](../handoffs/archive/snapshots/2026-10-03-execution/manifest.json)에 보존했다.

10-03 후속 갱신: 중단 지시 전 [지속 연구](../runbooks/continuous-research.md)를 이어가고, 근거 있는 서로 다른 가설에 하루 최대5회 제출 기회를 활용한다. 매회 제출·공개 승인은 유지한다. 14:11 KST ensemble CPU 검사·D01 로컬stage 확인 뒤14:15 비공개 업로드를 시작했다. 실제T4·전체runtime은 미검증이다. 아래11:34의LB/순위 관측은 새 수치로 덮어쓰지 않았다. [이번 변경 전 원문](../handoffs/archive/snapshots/2026-10-03-continuous-research/manifest.json).

## 1. 현재 위치와 확보한 것

| 항목 | 확인한 상태 | 해석 |
|---|---|---|
| 실제 최고 제출 | 56773226 COMPLETE, **0.943** | Jiweiliu 공유 스택의 비공개 복사본, 자체 모델 점수와 구분 |
| 순위 | **1,129/4,943, 상위22.84%** | 직전10:49의1,127/4,939에서 점수 불변·순위 하락. 상위권 유지는 아직 달성한 상태가 아님 |
| 현재 참고선 | 1위0.961, 10위0.958, 50위0.954, 100위0.950, 200위0.945 | public snapshot이며 마감 목표선·private 점수 보장 아님. 표시 동점 때문에0.943=동메달이라고 해석하지 않음 |
| 마감/제약 | **10/23 08:59 KST**, 약20일. 12-target macro AUC, offline notebook≤9h, 하루5회·최종2개 | 테스트 Report 없음. 제출 후 채점 대기하지 않음 |
| 데이터 | train4,407, 공식라벨58, 미표기4,349. ZIP265GB, Raptor corpus·기존2종 캐시 보유 | 추가 대용량 데이터 확보보다 계약/라벨/실전 추론 연결이 우선 |
| 분할 | Gold58+보고서중복4건 제외,4,345=869×5 | 보고서 그룹은 환자 독립성의 대리. Gold58은 노출된 진단셋 |
| 자체 실제 제출 | EXP045 ConvNeXt-S standalone **0.909** | 최근 Raptor/R2의 LB는 아직 없음 |
| 완료 모델 | EXP060 Raptor384 F0/F1; EXP061 Raptor224 5fold; EXP071 ResNet34 5fold; EXP072 R2 5fold; EXP073 F0~2 | 21개 최근 summary와 주요 코드4개 hash가 오전 재감사 때와 동일 |
| 개발 지표 | 원보고서 라벨·best checkpoint 재채점: R1 0.878263, ResNet34 0.866471, R2 0.892580 | R2 +0.014317은 teacher 간접 누출 때문에 독립 개선 확정 불가. 이 weak 점수들을 LB와 직접 비교하지 않음 |
| 실행 상태 | 오전 EXP074~076 중단 유지; 새 EXP078 소규모 입력대조 승인 | 현재 프로세스/예약은 최신 인계에서 확인. 과거 큐 자동재개 없음 |
| 자원 | RTX4070 Ti SUPER16GB, 무료 Kaggle·보유 Google Plus Colab 조건부 활용 | 실제 원격 할당량은 실행 전에 확인. 유료는 비용·기대효과를 제안한 뒤 결정 |

앱의 과거 imported 실험 개수를 전체 디스크 실험 개수로 해석하지 않는다. 21개는 이번 hash 재확인 대상이며 전체 실험 수가 아니다.

## 2. 방향을 바꾸는 판단

**확보한 강점:** 0.943 재현본, 원본 데이터와 빠른 corpus, 여러 자체 checkpoint, 학습/수집/GPU 관리 도구가 있다. 처음부터 전체 공개 스택을 다시 만들 이유가 없다.

**현재 병목:** 새 가설을 올바르게 평가하고 숨겨진 DICOM 추론·제출 후보까지 연결하는 과정이다. raptor_infer.py는 캐시용이고 raptor_serve.py는 신규 UID DICOM을 지원한다. 14:11 갱신 기준5fold decode공유 ensemble과 패키저의CPU 검사는 끝났지만 실제Kaggle T4·전체runtime 검증은 남아 있다.

**우선 바꿀 순서:** 모든 검증 기능과 새 학습을 끝낸 뒤 첫 실전 후보를 만드는 순서를 피한다. 기존 checkpoint로 고정한 후보를 준비하는 작업과 작은 clean 대조 실험을 함께 진행한다. R2의 간접 누출은 CV 해석 문제이지 테스트 데이터 누출이 확인됐다는 의미가 아니다. 기존 R2를 탐색 제출 후보로 쓰되 성능 개선을 확정하지 않는다.

공개0.945~0.946 notebook은 추가 자산을 확보하지 못했고, 다른0.943 복사본은 대부분 같은 구성이다. 새 가중치 공개나 유의미한 변경이 없으면 이 조사를 반복하지 않는다. 앱 기능 추가도 모델 실험에 직접 필요한 것만 한다.

## 3. 현재 우선순위와 첫 대조

토론 원문/댓글16개 재확인 결과 라벨 혼합이 weak CV/Gold에서 좋아도 LB에 전이되지 않은 사례가 있다. nested teacher 반복을 첫 실험으로 삼지 않는다.0.943는 유지하고 비교 조건·실제 입력 연결·작은 영상 가설 순서로 진행한다.

| 순서 | 실행 | 판정/남은 일 |
|---|---|---|
| P0 | strict 정밀도/백엔드·stable RNG·norm/window 계약·학습/평가 라벨 분리·fixed epoch·실제 optimizer update 기록 구현 | 스모크/관련 검사 완료. 기존 R2를 소급해 독립 OOF로 바꾸지는 않음 |
| P1 | 기존 R2를 신규 UID DICOM·5fold ensemble·D01 패키지에 연결 | 앞선F0 test3 일치·대표100 중99 byte일치/1 CRC실패는 보존.14:11 기준 decode공유·CPU ensemble16+serving10+package9 통과, 실제T4·전체runtime 후속 |
| P2 | EXP078:224px global-valid A vs within-slot-valid B, F0/F1 총4fit | 원라벨·seed42·FP16·고정8epoch·공통손상2건 제외. GPU3~5h 초기예상, 실제 처리량으로 재산정 |
| P3 | 사전 gate·소견/group 편차·정밀도/실전 runtime 검토 | 추가seed/다음fold/라벨가설 중 근거 있는 하나만.384·nested 반복·대량seed 자동확대 없음 |

[사전 명세](../../artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/manifest.json)의 코드·라벨·분할·입력/초기weights hash와 비교 조건을 확인한 뒤 해석한다. 두fold Δ양수·평균≥0.002·skip≤1%는 개발 선별 기준이다. bootstrap도 고정 weak 예측의 조건부 불확실성이며 임상/anchor/private 이득을 보장하지 않는다. Gold는 최종 진단 전용이다.

D01은 LB 전에 고정한 `0.9×rank(anchor)+0.1×rank(mean5(raw R2))` 탐색 후보다. [14:11 로컬stage 검사](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-package-verification.json): 자산1,469,070,653bytes(약1.469GB), anchor30셀 불변·후보32셀·CPU ensemble16+serving10+package9 통과. 이후[14:15 비공개 업로드 시작](../handoffs/evidence/2026-10/2026-10-03-continuous-research/upload-launch.json), 실제T4 parity·대표표본/전체runtime 검증은 대기 중이다. 패키징은 제출 준비 완료를 뜻하지 않는다. R2의 teacher 간접누출·혼합라벨 epoch선택 이력 때문에 우월성을 확정하지 않으며 신규 입력 대조도 anchor의 독립 검증을 대신하지 않는다. 검사를 마친 구체안에 매회 제출 승인을 받는다.

T4↔4070의 기존869건 비교는 최대확률차0.0004814·macro절대차0.00000594였다. 과거T4 출력과 현재로컬 비교이며 장치·라이브러리·batch 차이가 섞여 있다. 같은checkpoint/입력/batch/precision의 신규실제T4 검사는 남았다. 공개0.943 설정은 보존한다. [운영 절차](../runbooks/training-raptor.md).

## 4. 1등 도전과 상위권 확보를 위한 실험 축

다음 두 축만 우선 열고 효과가 없는 축은 중단한다. 큰 모델·많은 seed 자체를 목표로 삼지 않는다.

### A. 정답 정의에 맞는 감독 신호

대회 정답은 영상 전문의 재판독이므로 Report 문장 추출과 다르다. 공식 ACL/MCL 중증도, 표면에 닿는 반월판 파열, effusion 정도 등과 현재 라벨의 의미가 맞는지 확인한다. 보유 Qwen v2 raw의 mentioned/negated/confidence를 이용해 언급 없음·명시적 음성·불확실·과거력/수술·중증도 오류를 층화해 표본 검토한다. LLM confidence를 정답 확률로 무비판적으로 사용하지 않는다.

첫 비교는 보고서-only vs nested 혼합0.5 하나다. 유망할 때만 실제 미언급·uncertain에 국한한 soft supervision이나 소견별 loss mask를 추가한다. 현재 silence≤0.1 구현은 mentioned와 의미가 달라 보류한다. Gold58로 규칙을 선택하지 않는다.

전문가 재판독을 확보할 수 있으면 출처/접근조건·비용을 검토해 대표 표본의 blind 평가를 제안한다. 모델 불일치 사례만 모은 학습용 검토셋과 무작위/층화 검증셋은 구분한다. 새로 읽은 study도 기존 모델이 학습했다면 그 모델의 독립 holdout은 아니다.

### B. 기준본이 놓치는 영상 정보

공개 pretrained arm의 기존 변환은 유지한다. **자체 재학습 후보에서** series 내부 유효 인접3장·결측 mask·plane/series 구분·slot별 최소 관찰 수를 시험한다. 오전 감사에서 기존 selector의 k24 창12.5%가 slot 경계를 넘고5.0%가 masked slice를 포함했지만, 그 수치만으로 점수 손실이 측정된 것은 아니다.

원래 변환 대비 변경1개를 같은 라벨/seed/분할로 비교한다. 소견별 효과를 확인한 뒤 coronal/axial 또는 medial/lateral/PF 구획에 특화된 pooling/ROI를 검토한다. 무조건 입력 장수/해상도를 늘리면 좋아진다고 가정하지 않는다.

다른 계열은 보유 ResNet34·ConvNeXt부터 확인한다. 의료 사전학습/OrthoDiffusion은 다음 후보이며, 전처리 parity·학습노출·현재 자산 조건·실측비용을 통과한 뒤 작은 실험으로 시작한다. 낮은 상관만으로 좋은 보완 모델이라고 판정하지 않는다.

## 5. 공개 LB 과적합을 줄이는 운영

**0.943 기준본 자체에도 공개 튜닝 위험이 있다.** public 재현 성공은 private 일반화의 증거가 아니다. 기준본을 보존하는 이유는 동일 조건의 비교와 실행 복구이지 최종 성능의 보장이 아니다.

- 제출 전 바꿀 요소·checkpoint·고정 비율·코드/자산 hash·중단 기준을 등록한다. LB를 본 후 가설/기준을 소급 변경하지 않는다.
- LB는 미리 만든 후보가 실제 평가환경으로 전이되는지 확인하는 제한된 피드백이다. 라우팅·12소견 비율·seed 선택을 공개 점수에 맞춰 반복하지 않는다. 기존 public routing을 새 가설 없이 확대하지 않는다.
- 서로 다른 사전 고정 가설·검사된 후보가 있으면 **하루 최대5회 기회를 적극 활용**한다. 가설당1~2회가 기본이며 매번 잔여 한도·pending을 확인하고 승인받는다. 총6~10회 제한은 후속 지시로 대체했지만 횟수 소진·점수 상승까지의 비율/소견 반복튜닝은 하지 않는다.
- 표시0.001 상승/하락만으로 결정하지 않는다. 큰 public 상승도 내부 결과·domain 안정성·추론 품질과 함께 판단한다. 1위0.961을 맞추기 위한 탐색 루프를 만들지 않는다.
- public 모델이 train을 본 예측과 자기 OOF를 합쳐 가중치를 최적화하지 않는다. 자체 모델끼리의 비율은 가능한 범위에서 nested OOF로 선택한다.
- weak 라벨CV 향상도 임상 정답 향상의 충분조건이 아니다. Gold58은 큰 오류 진단에만 쓰며 정상 범위의 점수 변화로 채택/탈락시키지 않는다. CV와 LB 불일치는 두 평가라벨/분포/계약을 점검하는 근거다.
- scanner별 OOF와 scanner를 제외하고 학습한 holdout은 구분한다. 최종 유망2개 레시피에만 추가seed·사전 지정 group holdout을 적용한다. 기존 노출이 있는 셋을 새 독립 holdout이라고 부르지 않는다.
- report-group bootstrap은 고정 예측의 불확실성 진단이다. CI가0을 포함하면 이득 입증 부족이며, CI가양수라도 누출/라벨 오류/반복 탐색을 해결해 주지 않는다.

**최종2개**는 검증 근거가 가장 균형적인 주력 조합과 오류/구조가 다르고 안정성 근거가 있는 후보로 정한다. 최고public점과 flat-0.60을 자동 선택하던 이전 지침은 폐기한다. 근거 부족 시0.943를 예비안으로 남기되 private 안전모델이라고 부르지 않는다.

## 6. 남은 일정과 중단 기준

| KST | 목표 | 방향을 바꿀 조건 |
|---|---|---|
| 10/03~06 | 고정 추가 후보1개, adapter/parity/runtime, 입력 A/B 4fit | 계약/누출 문제면 그 경로만 고치고 무관한 앱 개발은 하지 않음 |
| 10/06~11 | 라벨 또는 입력에서 재현되는 개선1개, 다른 encoder 후보1개 | 내부 이득·안정성·제한된 실전 전이가 모두 불명확하면 해상도/seed 확대 중단 |
| 10/11~16 | 유망한 최대2개 레시피만5fold, 필요시 추가seed/그룹holdout | public만 오르고 내부 근거가 약하면 주력 교체 보류. 검증값만 오르면 라벨/계약/평가 정합성 재점검 |
| 10/16~20 | 전체학습 fixed epoch 또는 fold ensemble 선택, 오프라인 패키지·자산 명세·실측 runtime | 근거 없는 새 대형 구조/외부 데이터 도입 종료 |
| 10/21~22 | 최종2개 후보 결정·제출·복구 여유 | 실패나 채점 지연을 감안해10/22 KST 내 제출 목표. 실제 마감10/23 08:59 |

최종 full-data fit은 CV에서 정한 epoch를 사용하고 Gold checkpoint 선택 fallback을 제거한 뒤 실행한다. 5fold×2seed×384를 자동 실행하지 않는다. **상위권 확보 경로는 확실한 작은 개선을 쌓는 것**, 1등 도전 경로는 그 과정에서 정답 정의/영상 정보의 재현 가능한 큰 개선이 나타날 때 자원을 집중하는 것이다. 현재 근거로 우승 가능성을 수치화할 수 없다.

## 7. 자원과 시간 배분

기존 총91~154 GPUh는 필요시의 계획 범위이며 소진 목표나 비용 승인이 아니다. 먼저 소규모 실험을 통과하고 실제 남은 시간으로 다시 산정한다. 로컬 GPU 우선, 필요시 무료 Kaggle·보유 Google Plus Colab, 그 뒤 유료 제안 순서다.

연구·구현 시간의 초기 배분은 약50% 성능 가설/작은 대조,30% serving·평가·제출 패키지,20% 라벨/실패 원인 감사다. 막힌 경로가 확인되면 재배분한다. 대규모 full DICOM 감사·모든 공개노트북 재조사·새 UI 작업을 먼저 끝내는 일정을 만들지 않는다.

유료 자원은 검증된 실험이16GB 메모리 또는 실제 가용시간에 막히는 경우에 한해 GPU 종류·예상배속·필요시간·최대비용·종료조건을 제안한다. 라벨/검증 문제가 해결되지 않은 상태에서 GPU 증설을 우선하지 않는다.

## 8. 기록 정합성과 근거

- [누적 작업/handoff 타당성 감사](progress-audit-2026-10-03.md): 기반 구축·0.943 재현·작은 학습 실험은 타당. R2 효과 확정·반복증류 확대, ‘라벨만 병목/모델 상한’ 단정은 근거 부족.224/384 비교는8/12epoch 차이도 있어 해상도 단독 효과로 해석하지 않는다.
- 장시간 작업은 ID/ETA/로그와 같은 채팅 완료 확인 heartbeat를 남기고 반환한다. 실제 종료·산출물 확인 후 허가된 후속만 재개하고 예약을 해제한다. [운영 절차](../runbooks/tools.md). 예약ID·현재 실행 상태는 [인계 색인](../handoffs/INDEX.md)의 해당 기록에서 확인한다. 첫 확인은 ETA 20분 전, 미완료면 갱신 ETA에 재예약한다.
- **변경한 지침:** R2 채택 확정→탐색/clean 재검증; Raptor 전면교체→후보 추가/부분교체; 공개 routing 전면금지→기준본 보존·새 LB튜닝 금지; public 최고/flat0.60 자동선택→안정성 종합판단; CI포함0=동일→이득 입증 부족; sample3건 runtime 외삽→대표표본 실측; anchor bf16 일괄금지→자체/공유 precision 계약 구분.
- **구현과 한계(14:11 갱신):** 독립 eval-label 입력·fixed epoch·full-data Gold선택 차단·stable RNG·strict precision·DICOM adapter·5fold decode공유와D01 로컬stage를 구현했다. CPU 검사 통과는 성능·GPU runtime 근거가 아니다. EXP078 전용 report-group bootstrap과 저장OOF 오류기여 감사는 개발 진단용이다. ancestor-aware nested labels·Kaggle 실제T4 serving·전체runtime 및 label_mix의 teacher 계보 검사는 미완료다.
- [최신 직접 LB](../handoffs/evidence/2026-10/2026-10-03-direction/leaderboard.json)·[제출8건](../handoffs/evidence/2026-10/2026-10-03-direction/submissions.json)·[앱 briefing](../handoffs/evidence/2026-10/2026-10-03-direction/briefing.json).
- [코드4개·summary21개 hash 재확인](../handoffs/evidence/2026-10/2026-10-03-direction/local-verification.json). 원 재채점/teacher 경로/selector/라벨 진단은 [오전 CPU 감사](../handoffs/evidence/2026-10/2026-10-03-replan/local-audit.json)이며 이번에 모델을 다시 평가한 것이 아니다.
- [주최·토론 근거](../handoffs/evidence/2026-10/2026-10-03-replan/kaggle-public.json)·[추가 토론/조건](../handoffs/evidence/2026-10/2026-10-03-replan/kaggle-additional.json). 공식 주최 설명과 참가자의0.950+ 주장을 구분한다. 13시 재확인16개 원문/댓글과 관측범위는 [실행 우선순위](execution-priorities-2026-10-03.md)에 별도 기록했다.
- 공개 자산의 학습 코드/사용 조건/수상 제출 의무는 최종 후보별로 확인한다. NC 표기만으로 대회 사용 금지를 단정하지 않는다.
