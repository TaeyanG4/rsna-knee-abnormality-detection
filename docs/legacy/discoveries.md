# 배운 점과 closed ideas

2026-10-05 형식 정리 후 문헌 조사·비용·개선 범위 교훈을 보완했다. 기존 M01–M11의 관측·판단·재검토 조건·근거를 유지했다. [형식 정리 전 원문](docs/handoffs/archive/snapshots/2026-10-05-agent-rules-v2/manifest.json) · [이번 보완 전 원문/hash](docs/handoffs/archive/snapshots/2026-10-05-discoveries-recording-v1/manifest.json). 진행은 [HANDOFF](HANDOFF.md), 다음 가설은 [plan](plan.md), 지시는 [AGENTS](AGENTS.md).

## 요약·색인

기준 스택과 평가 노출을 확인하고, 특정 설정의 실패와 전략 전체를 구분한다. 완료 결과는 재사용한다. 아래 색인과 Closed ideas를 먼저 읽고 이번 작업에 해당하는 ID만 상세 조회한다.

| ID | 주제 |
|---|---|
| [M01](#m01) | 기준본 |
| [M02](#m02) | 검증 노출 |
| [M03](#m03) | 그룹 독립성 |
| [M04](#m04) | 작은 효과·골절 |
| [M05](#m05) | 특정 입력·표현 실패 |
| [M06](#m06) | Silver |
| [M07](#m07) | 실패·철회 |
| [M08](#m08) | 부분 실행 검증 |
| [M09](#m09) | 종료 판정 |
| [M10](#m10) | LB·시간 배분 |
| [M11](#m11) | 입력 보존·분모 |
| [D12](#d12) | 기록의 중복과 상태 분리 |
| [D13](#d13) | 문헌 사실과 운영 제안 구분 |
| [D14](#d14) | 과거 상위권의 입력·감독 개선 |
| [D15](#d15) | 개선 적용 범위·재학습 비용·결합 검증·유사도 축소 |
| [D16](#d16) | 공개 전처리의 구현·효능과 우리 진행 범위 |

## Closed ideas

닫힌 것은 특정 계약의 반복이며 전략 계열 전체가 아니다. 상세 관측·재검토 조건은 위 M행이 원본이다.

| 상태 | 범위 | 다시 열 조건 |
|---|---|---|
| 완료·재사용 | [M06](#m06) Silver 동일 가림 감사, [M09](#m09) 통과한 입력/수치 계약 | 새 정의 충돌·독립 근거·계약 변경 |
| 특정 recipe 미채택 | [M04](#m04) EXP084 ones, [M05](#m05) EXP081/083/086 설정 | 다른 메커니즘·맞춘 대조·정해진 비용 |
| 구현 실패·효능 미평가 | [M07](#m07) EXP087 full, EXP088 RSS 실패, D01 동일 실패 제출 | 원인/수정 근거와 새 버전·검사 계약 |
| 보류·불확실 | [M04](#m04) 작은 효과, [M05](#m05) EXP085, [M10](#m10) 기준본 전이 | 같은 비교의 추가 근거·소견 손실·비용 재판정 |
| 해석 철회 | [M07](#m07) 초기 race 의심, [M10](#m10) OOF 부재 주장·비교 혼동 | 새 근거 없는 재인용 대상 아님 |

## 교훈 상세

<a id="m01"></a>

### M01 · 기준본

- 관측·판단: 공개 스택 0.943는 실제 제출 성과. 코드의 A5 비중은 52%로 소개의 45%와 달랐고 Gold 기반 epoch 선택·stacker 계보 공백이 있었다.
- 반복 금지·재검토 조건: 소개글만으로 recipe를 복원하지 않는다. 정확 코드·revision·가중치·계산식을 확인하고 자체 weak 개선량을 0.943에 더하지 않는다.
- 근거: [기준본 직접 분석](docs/research/anchor-notebook-study-2026-10-03.md)

<a id="m02"></a>

### M02 · 검증 노출

- 관측·판단: R2 teacher 간접누출, 전체 fold 모델 평균의 train in-sample, Gold58 선택 노출은 서로 다른 문제다. EXP078 이후 ImageNet 초기화 실험에 R2 누출을 일괄 적용하지 않는다.
- 반복 금지·재검토 조건: Gold를 뒤늦게 제외·재분할·새 판독해도 기존 노출은 지워지지 않는다. 새 V의 관련 encoder/teacher/student/stacker/calibrator ancestor를 모두 제외해 재학습해야 한다. 그것도 기존 D01 가중치의 독립 검증은 아니다.
- 근거: [Gold 노출 이후 검증](docs/research/gold-exposure-validation-plan-2026-10-03.md)

<a id="m03"></a>

### M03 · 그룹 독립성

- 관측·판단: 보고서 그룹만으로 환자 독립성을 보장할 수 없고 cross-fold 동일 영상 후보가 실제 저장 배열/픽셀 일부에서 확인됐다.
- 반복 금지·재검토 조건: 알려진 연결을 새 group 계약에 반영하되 기존 OOF를 소급해 독립 OOF로 부르지 않는다. 유사도만으로 동일 환자를 확정하거나 미발견을 독립성 증명으로 쓰지 않는다.
- 근거: [그룹 계약](artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/contract.json), [조사·투자 기록](docs/handoffs/evidence/2026-10/2026-10-04-next-investment/root-decision.json)

<a id="m04"></a>

### M04 · 작은 효과·골절

- EXP088의 384·12epoch 레시피도 노출 weak G1에서 작은 양의 불확실 후보였다. 외측 반월판/골절 이득과 MCL 손실·field/mask 그룹 차이를 함께 보존한다. 해상도·generic·epoch·precision·microbatch가 함께 달라 해상도 단독 효과가 아니다. 약4.7배의 서로 다른 경계 학습 비용과 영상 정답으로의 전이 미측정을 고려해 자동5fold보다 두 완료 G1 단독 후보의 고정 전이 비교 실행성부터 검토한다. [고정 평가·투자 판정](docs/handoffs/evidence/2026-10/2026-10-05-exp088-raptor384-v1/evaluation-v4-root-result-decision.json).

- 관측·판단: EXP080 F2 RN 보완 +.001243, B−A 골절 −.022337. EXP082 loss 혼합 +.000534는 CI에 0 포함. 작은 양의 효과와 반복된 골절 손실을 함께 보존한다. EXP084 새group ones−clip−.000096/CI0포함으로 전환미채택,clip RN+.000997·골절−.003111도 보존.
- 반복 금지·재검토 조건: CI가 0을 포함한다고 동등/효과 없음으로 단정하지 않는다. macro만 보고 추가 fold/seed/384를 자동 확대하지 않는다. 같은 비교의 12소견 P/N·group·skip·조건부 CI와 비용으로 재판정한다.
- 근거: [EXP080](docs/research/exp080-f2-result-2026-10-03.md), [EXP082](docs/research/exp082-loss-screen-2026-10-03.md), [EXP084](docs/research/exp084-group-loss-result-2026-10-04.md)

<a id="m05"></a>

### M05 · 특정 입력·표현 실패

- 관측·판단: EXP081 max-span 직접교체 −.000684/CI 0 포함. EXP083 slot-bias−control 약−.0000246이며 두 refit 모두 frozen B보다 약−.01155. 해당 설정 미채택. EXP086 V2는 정상완료635초/3448updates/skip0 뒤 DINO−B−.064575(CI음수)/12소견하락으로 현8epoch전이미채택·B유지. DINO전체기각아님. [결과](docs/research/exp086-dinov2s-result-2026-10-04.md). EXP085 full 상대위치+.001852/원B+.002154는CI0포함, Baker하락·group편차와 함께 개발후보로보존/교체보류. [결과](docs/research/exp085-relative-position-result-2026-10-04.md).
- 반복 금지·재검토 조건: coverage·sequence·표현 전략 전체를 기각하지 않는다. 다른 메커니즘·동일 입력/학습량 대조·예산을 사전 고정하면 다시 검토한다. 처리군이 대조보다 높아도 둘 다 기존 B보다 낮은 사실을 숨기지 않는다.
- 근거: [EXP081](docs/research/exp081-coronal-span-2026-10-03.md), [EXP083](docs/research/exp083-slot-bias-result-2026-10-03.md)

<a id="m06"></a>

### M06 · Silver

- 관측·판단: PFOA20/Synovitis16 가림 대조와 train3475 소스 비교 완료. 현재 Dread 자동교정의 직접 근거는 부족했다. Qwen의 넓은 정의 문제를 Dread 오류로 전용할 수 없다.
- 반복 금지·재검토 조건: 같은 가림 검토를 다시 하지 않는다. 미언급/proxy/불확실을 임상 음성으로 바꾸지 않는다. 새 정의 충돌·독립 영상 근거가 있으면 감독 개입을 재검토한다. pos_weight/soft BCE는 라벨 오류 해결의 증거가 아니다.
- 근거: [Silver 감사·외부 연구](docs/research/silver-other-research-and-pfoa-2026-10-03.md)
- 감독 개선의 적용 단위·재학습 비용은 [D15](#d15), 과거 대회의 데이터 선택·품질별 학습 사례와 한계는 [D14](#d14)를 따른다. 기존 감사 완료를 라벨 교정 완료로 확대하지 않는다.
- 10-05 공개 Knee의 누락·mask·OOF 혼합 사례와 반대 근거, 우리 적용/미적용 범위는 [D16](#d16)에 기록했다.

<a id="m07"></a>

### M07 · 실패·철회

- EXP089 T4 첫 notebook은 kernelspec 누락으로 papermill이 셀 실행 전에 거부했다. nbformat 형식/코드 compile만으로 원격 실행 metadata를 검증한 것은 아니다. 새 notebook metadata만 교정하고 code/worker/protocol/기존 자산은 재사용하며 별도 delta·예산 검토 뒤 진행한다. API SSL 실패와 로그 수집의 로컬 parsing 오류도 모델 실행 실패와 구분한다. [원 실패·변경 조건](docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/root-t4-v3-failure-decision.json).

- EXP088 V4는 private cap32GiB/호스트 available8GiB 하한에서 정상 완료했다. summed RSS는96GiB를 넘었으나 private/가용 RAM 계약은 지켰다. 이전 RSS guard 종료를 물리 RAM 부족/OOM이나 recipe 성능 실패로 부르지 않는다. 원 실패는 유지하고 새 완료 수용과 구분한다. [실 완료·표본 자원](docs/handoffs/evidence/2026-10/2026-10-05-exp088-raptor384-v1/full-v4-root-acceptance.json).

- supervisor가 새 telemetry/tmp 파일을 만들 때 child의 시작 allowlist도 맞춰야 한다. EXP088 V2는 이 누락으로 모델/학습 전4.234초 FAIL했다. 정확 파일명만 허용한 실제 AST 시작 조건2검사로 교정하며 기존 출력의 덮어쓰기 거부는 유지한다. [시작 실패](docs/handoffs/evidence/2026-10/2026-10-05-exp088-raptor384-v1/root-v2-startup-failure.json).

- EXP088 384 본 학습은 9epoch/3888updates skip0 뒤 96GiB summed RSS cap을 약37MiB 넘어 종료했다. shared mmap 중복 가능성이 있어 고유 RAM/OOM/성능 실패로 단정하지 않는다. 마지막에만 저장해 복구 가중치가 없었다. 자원 감시는 private commit와 호스트 가용 RAM을 구분하고 epoch 복구 상태를 보존한다. 같은 cap의 자동 재실행은 하지 않는다. [원 실패](docs/handoffs/evidence/2026-10/2026-10-05-exp088-raptor384-v1/root-full-failure-review.json).

- 관측·판단: EXP087 full140.110초는 gradient norm 비유한으로 FAIL·성능미평가. 후속104.313초진단attempt74/완료73:forward유한,stem gradient136개(NaN54/+Inf27/−Inf55)비유한. 단순유한원소norm합산overflow아님·발생연산/해결미확정·원full실패step미확정. [실진단](docs/handoffs/evidence/2026-10/2026-10-04-exp087-numerical-diagnostic-v1/root-diagnostic-result-review.json). EXP086 DINO pilot83.235초는CUDA bicubicAA 결정론backward미지원으로FAIL,성능기각아님. V2CPU미분보간으로정상full/평가1회완료했으나해당recipe성능은하락. 평가절대경로첫CLI entry거부→상대경로교정/실평가1회,treeRSS공유mmap중복가능성을고유RAM으로단정금지. [결과](docs/research/exp086-dinov2s-result-2026-10-04.md). [실패·CPU후속](docs/handoffs/2026-10-04-0534-exp086-pilot-final-config.md). 기준본의 재실행 RecursionError·이전 receipt 잔존·fallback에 따른 조합 변경을 관측. 초기 JSON race 의심은 최종 lock 확인 후 철회했다. D01 원 hash FAIL·통합 v1 경로 NO_GO도 보존했다. 10-04 D01제출56796964는COMPLETE/점수공백뒤exactref에서hidden unhandled error확인·원인미확정. [결과](docs/handoffs/2026-10-04-0506-d01-error-exp086.md).
- 반복 금지·재검토 조건: 철회된 race를 확정 결함으로 재인용하지 않는다. 실패·부재·정상·저하 실행을 구분하고 실패를 PASS로 덮지 않는다. 교정은 새 ID/버전·정확 원인·비용을 기록한 뒤 필요한 범위만 실행한다.
- 근거: [기준본 인계](docs/handoffs/2026-10-03-1233-anchor-study.md), [D01 상세](docs/handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json)

<a id="m08"></a>

### M08 · 부분 실행 검증

- 관측·판단: 10-04 제출56809595 actualv2의실제채점COMPLETE/0.913 확인. 이는해당hidden실행완료근거이나hiddenN/실시간·RSS상세는미확인.  B/RN최종설정actualv2 공개3은95.117초/root·독립PASS이나hidden시간/정확도는미확정. [후보](docs/research/b-rn3f-submission-candidate-2026-10-04.md). D01 공개3/통합133 PASS에도 1300건 보수 추정8.594h/9h·RAM29.541/30GiB였다. 다른 경로·hidden 전체 성공이나 정확도 향상은 입증하지 않았다.
- 반복 금지·재검토 조건: kernel COMPLETE·exit0·sample PASS만으로 최종 성공을 선언하지 않는다. T4/4070·입력·모델·코호트가 달라지면 해당 계약을 새로 검증한다. B local100과thread교정뒤 T4 원DICOM100 수치/자원PASS를확보했으나,동적test/압축구문/hidden 전체·정확도는별도다.
- 근거: [D01](docs/research/d01-exploratory-candidate-2026-10-03.md), [B 입력 계약](docs/handoffs/evidence/2026-10/2026-10-04-b-rn3f-parity-v1/HANDOFF.md), [T4 실제결과](docs/handoffs/2026-10-04-0330-b-rn3f-t4-result.md)

<a id="m09"></a>

### M09 · 종료 판정

- 관측·판단: B parity V1은 parent FAILED와 수치 FAILED/exit0를 자체 차단하지 않았다. 수치 오류로 확인된 것은 아니며 별도 acceptance adapter로 보완했다. local100 정수입력exact/normalized100해시FAIL은 CPU보간thread1/12 차이로1건재현. 별도V2 thread맞춤100+합성6에서원수치한도/producer해시연결PASS,원FAIL보존.
- 반복 금지·재검토 조건: 미래 비교는 성공한 parent receipt·worker hash·명시적 numerical PASS를 함께 확인한다. gpu_job/SUPERVISOR/cleanup도 별도 확인한다. 이미 통과한 모델추론은 재실행하지 않는다. test release의 CSV게시 후 metadata/ack 실패와 deadline도 rollback 경계 안에서 검사한다. [10-04 교정](docs/handoffs/2026-10-04-0423-release-review.md). 입력재구성은 producer의 실제CPU threads도 기록·일치시켜야 하며 수치허용치와exact연결을 분리하지 않는다.
- 근거: [반례](docs/handoffs/evidence/2026-10/2026-10-04-b-rn3f-parity-review-v1/review.json), [보완](docs/handoffs/evidence/2026-10/2026-10-04-b-rn3f-acceptance-v1/HANDOFF.md)

<a id="m10"></a>

### M10 · LB·시간 배분

- 10-05 비용 판단 보완: 추가 모델을 우선하기 전에 기준본과 후보의 공동 추론 예산을 확인한다. 기존 실측 기반 1,300건 가정에서 anchor 약7.334h, D01 약8.594h, anchor+B/RN6 약10.503h였다. hidden 실제 건수·전체 실행시간이 아니며 RN3 비용은 미측정이다. D01 원응답은 unhandled error로 원인 미확정이므로 시간초과로 확정하지 않는다. 시간 여유가 입증되지 않으면 같은 비용의 교체·학습 신호 개선·효율 개선을 먼저 검토한다. [비용 근거](docs/handoffs/evidence/2026-10/2026-10-04-feedback-scale-control/anchor-rn-feasibility.md) · [D01 원응답](docs/handoffs/evidence/2026-10/2026-10-04-d01-first-score/exact-submission.json).

- 관측·판단: 16시정정:EXP079/080동일V실결합OOF와기존최종재학습계획은존재. 전체3fold배포/0.943공통미사용평가와구분·Gold미사용원인미입증. D01은이미anchor+옛R2결합실행실패. [정정/후속](docs/research/feedback-investment-review-2026-10-04.md). 11:38 사용자질문으로비교연결정정: manifest anchor없음, 내부B+RN−B와공개새스택−0.943는다른질문. RN효과반전단정불가. 정확제출ensemble과독립검증/기준본의연결공백을[기록](docs/handoffs/evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-validation-comparator-reconciliation.json).  10-04 B/RN actualv2 제출56809595는Public0.913으로기준0.943보다−0.030·현후보미채택. 서로다른스택이므로RN혼합의인과효과로단정금지. [결과](docs/handoffs/2026-10-04-1127-brn-result.md). D01(anchor+옛R2)과 EXP079/080(B+RN)은 다른 가설이다. 실행 도구 검증 자체는 성능 개선 성과가 아니다.
- 반복 금지·재검토 조건: Public 결과를 보고 비율·소견 routing을 반복 탐색하지 않는다. 하루5회는 상한이다. 작은 이득을 보존하면서 정보/표현/감독 등 3~5가설의 근거·비용을 비교한다. 10-04 [사용자 상시승인](docs/research/submission-authorization-2026-10-04.md)으로 검증후보 제출은 재질문 없이 진행하되 새근거/한도/중복검사를 유지한다. D01 재제출금지·공개/유료 별도승인은 유지한다.
- 근거: [6시간 재점검](docs/research/six-hour-reassessment-2026-10-03.md), [후속3안](docs/handoffs/evidence/2026-10/2026-10-04-post084-priority/brief.md), [지속 연구](docs/runbooks/continuous-research.md)

<a id="m11"></a>

### M11 · 입력 보존·분모

- 관측·판단: cache c±1은native연속보장없고전역center샘플러때문에한slot mask변경이타slot을바꿀수있다. 이전1936/4407 출처/정의미확인, broad1985/slot2357은다른정의다. EXP087 strict T252/V63는315입력stage57.797초/독립검토로유효alt/fallback0확인. V126창(0.5912%)은예정치환이며효능미검증. T8 CPU192장/11.281초는원replay·RNG보존을확인했으나대표표본/임상정보/효능증명아님. [실결과](docs/handoffs/2026-10-04-0736-exp087-review.md).
- 반복 금지·재검토 조건: 현G1치료가용률로이숫자를쓰지않는다. 기존center/RNG를먼저고정·다른window보존,같은series내triplet만사용. 다른FoR/SeriesUID·pixel차이는임상추가정보의증명이아니다.
- 근거: [EXP087 CPU계약](docs/handoffs/2026-10-04-0656-exp087-cpu.md), [제안/원근거](docs/handoffs/evidence/2026-10/2026-10-04-post086-priority/coverage-proposal.md)
- 공개 노트북의 원본 연속 triplet과 표본 캐시의 인접 plane을 구분한 코드 조사 및 진행 범위는 [D16](#d16)을 따른다.

<a id="d12"></a>

### D12 · 기록의 중복과 상태 분리

- 관측·판단: 상태를 복제하면 취소된 ETA와 보류 상태가 함께 남을 수 있다.
- 10-05 기록 감사: 기존 실험 교훈은 있었지만 이번 과거 대회 조사·시간 제약 판단·재학습 범위 설명이 답변과 별도 근거에만 남아 있었다. M06/M10과 D14/D15에 보완하고, 재사용할 교훈·중요 정정은 반환 전 기존 ID와 색인에 반영하도록 [갱신 절차](docs/runbooks/handoff-maintenance.md#discoveries-갱신)를 명시했다.
- 적용·재검토 조건: 진행/결과는 HANDOFF, 다음 가설은 plan, 교훈은 이 파일을 원본으로 삼는다. 구조를 바꿀 때 기존 원문을 보존한다.
- 근거: [문서 역할 분리 기록](docs/handoffs/2026-10-04-2227-document-roles-agent-cli.md).

<a id="d13"></a>

### D13 · 문헌 사실과 운영 제안 구분

- 관측·판단: AI Relay 원문의 릴레이·코드 검토 권고와 우리 운영안을 구분한다. 제품별 고정 역할·모든 아이디어의 사전 승인·토큰 소진에 따른 교대는 원문에서 확인되지 않았다.
- 적용·재검토 조건: 운영 제안을 문헌의 확정 사실로 인용하지 않는다. 새로운 원문 근거가 있을 때 재검토한다.
- 근거: [문헌·사실 대조](docs/research/ai-relay-applicability-2026-10-04.md).

<a id="d14"></a>

### D14 · 과거 상위권의 입력·감독 개선

- 관측: 10-05 완료 대회4개·상위 솔루션9개 원문을 수집했다. RSNA2024 MRI 1위는 위치·슬라이스 선택/crop과 MIL·LSTM을 결합했고 큰 모델·긴 학습은 도움이 안 됐다고 보고했다. 2위는 ensemble OOF와 라벨의 큰 충돌 사례를 제외해 재학습한 뒤 public/private 약1% 개선을 보고했다. RSNA2023 CT 2위는 장기 가시성·crop으로 입력을 선별하고 슬라이스 절반을 추론했으며 추가 앙상블이 최강 파이프라인의 private 점수를 넘지 못했다고 보고했다.
- 반대·조건부 근거: HMS2024 1위는 전체 데이터 학습 후 고투표 라벨로 추가 학습했다. 3위는 pretrained 2D에 저투표/pseudo가 도움이 안 됐지만 scratch 1D에는 품질별 감쇠 가중으로 일부 도움이 됐다고 보고했다. RSNA 유방암1위 사후 대조는 외부 데이터 효과를 보고했으나 soft-positive 라벨 기법의 명확한 이득은 확인하지 못했다.
- 판단·적용 범위: 입력 정보 보존·감독 품질과 적합한 모델 구조를 함께 개선할 근거다. 우리 후보 한 개의 입력/감독 대조와 비용을 먼저 확인하고 이후 교체·추가를 검토할 수 있다. 이는 현재 실행 계획 변경이나 Knee 성능 개선의 입증이 아니다.
- 반복 금지·재검토 조건: 모두 작성자 보고이며 이번 재실행0. 서로 다른 loss/F1/KL 수치를 Knee AUC 예상 이득으로 환산하거나 데이터가 항상 모델보다 중요하다고 일반화하지 않는다. 높은 loss만으로 희귀 양성/어려운 사례를 오류로 삭제하지 않는다. 고투표 데이터와 Gold58의 규모·선택 노출은 다르다. 새로운 라벨 오류 근거 또는 고정 평가·계보·비용의 자체 대조 결과가 있으면 투자 순위를 재판정한다. 기존 M06 감사·M05 입력 실패의 범위는 유지한다.
- 근거: [9개 원문·출처·SHA256](docs/handoffs/evidence/2026-10/2026-10-05-past-competition-strategy/source-index.json). 원문/댓글은 같은 폴더의 topic-ID JSON에 보존했다. 상세 출처는 [MRI1위](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091), [MRI2위](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452), [CT2위](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453), [HMS1위](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560), [HMS3위](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471), [유방암1위](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449).

<a id="d15"></a>

### D15 · 개선 적용 범위·재학습 비용

- 관측·판단: 0.943의5개는 단일 모델5개가 아닌 구성 그룹이다(DINOv2만20 members). 정제 라벨 CSV 변경은 기존 가중치의 예측을 바꾸지 않는다. 정제 신호를 전체 구성에 반영하려면 각 학습 모델의 재학습/미세조정과 결합 재검증이 필요하다. 기초 사전학습부터 전부 다시 할 필요는 없다.
- 적용 범위: 후보 하나에만 정제·입력 개선을 적용해 대조한 뒤 특정 구성의 교체/추가를 검토할 수 있다. 이때 기존 전체 스택이 정제 데이터로 학습됐다고 표현하지 않는다. 공개 체크포인트 추론 재현과 원본 학습 과정의 완전 재현은 다른 주장이다.
- 10-05 재학습·앙상블 판단: 검증된 물리 정렬·입력 유효성·라벨 정의는 공통 기반으로 개선할 수 있지만, DINO의 member별 crop/slot, A5의 16채널, Rad의 특징/head, Raptor·CoAt의 view/장수/정규화를 일괄 통일하지 않는다. 새 입력에 맞춘 재학습은 학습/추론 불일치를 줄이는 방법이며 성능 향상의 증명은 아니다. 같은 전처리가 오류를 비슷하게 만들 수 있으므로 단독 AUC·낮은 예측 상관만으로 앙상블 기여를 판정하지 않는다.
- 결합 재검증: 기준본은 단순 5개 평균이 아니라 88-feature 학습 stacker와 소견별 혼합을 포함한다. 구성 재학습 후 기존 결합의 성능을 다시 확인하고 필요하면 적절한 OOF로 결합기를 다시 학습한다. 원 stacker 학습 계보·일부 원 학습 recipe는 미확인이므로 전체 재현을 보장하지 않는다. 우선 재현 가능한 한 구성의 입력/감독 개입을 고정 평가에서 대조하고, 동일 검증에서 앙상블 교체의 추가 이득과 추론 비용을 확인할 때만 다른 구성으로 확대한다는 전략 판단이다. 공개 가중치의 train 예측·노출 Gold58을 독립 OOF로 쓰지 않는다. [입력 조사와 미검증 범위](#d16)·[평가 노출](#m02).
- 10-05 유사도 기반 축소 판단: 동일 study·12소견·입력 유효성 범위의 예측 순위 상관(Spearman)을 그룹/member/view별 후보 선별에 쓸 수 있다. 높은 상관도 일부 양성/음성 쌍·소견의 보완을 숨길 수 있고 낮은 상관은 저품질 예측일 수도 있다. 사전 지정 제거 후보를 하나씩 제외한 전체 결합의 paired macro/소견별 AUC·표본수·불확실성과 실제 절약 시간을 확인한다. CI가0을 포함한다고 성능 동등이 입증되지는 않으며 보편적 상관 삭제선을 두지 않는다. stacker 의존 경로와 제거 후 결합 정책을 명시하고, 개별 제거 효과를 여러 개 동시 제거에 합산하지 않는다. DINO shared-prefix·Raptor view/reader 공유 때문에 제거 개수와 비용 절약도 비례하지 않는다. 독립성이 확인된 OOF/holdout과 선택 뒤 확인 자료를 확보할 때 축소를 재판정하며, 노출 Gold58/train 예측이나 Public 반복 탐색만으로 삭제를 확정하지 않는다. 현재 특정 member 제거의 유사도·효능·비용은 측정하지 않은 전략 제안이다. [구성·의존·공유 계산 근거](docs/research/anchor-notebook-study-2026-10-03.md)·[기존 상관 진단 도구](monitoring/kneeops/predictions.py).
- 반복 금지·재검토 조건: 동일 구조·입력의 라벨 개선은 추론 비용을 늘리지 않을 수 있지만 정제·재학습·검증 개발 비용은 발생한다. 해상도·입력 장수·ROI 검출기 추가는 별도 추론 비용도 측정한다. 개선 범위·필요 코드/라벨·평가 계보·비용이 구체화될 때 한정 대조부터 재판정하며 전체5개/모든fold를 자동 재학습하지 않는다.
- 근거: [기준본 구성·stacker 계보](docs/research/anchor-notebook-study-2026-10-03.md), [현재 감독 대응](docs/research/gold-silver-status-2026-10-03.md), [비용 판단](#m10).

<a id="d16"></a>

### D16 · 공개 전처리의 구현·효능과 우리 진행 범위

- 관측·범위: 10-05 이 대회의 디스커션11개(반환 댓글 포함)와 새 공개 노트북6개의 코드를 조사했다. 전체 게시물 전수조사·모델 실행·원본/가중치 다운로드는 하지 않았다. 원응답·코드 bytes·SHA256·수집 시각을 보존했고 notebook 숫자 revision과 점수 연결은 미확인이다. [출처/hash](docs/handoffs/evidence/2026-10/2026-10-05-preprocessing-survey-v1/source-index.json) · [영역별 비교·코드 근거](docs/handoffs/evidence/2026-10/2026-10-05-preprocessing-survey-v1/findings.json).
- 영상 기본 처리: 공개 코드는 물리 정렬·mm crop·series percentile·MONOCHROME1·uint8 cache를 구현한다. 최근 [파일명→IPP 정렬](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745759)의 +0.0276은 작성자의 한 fold weak CV다. 우리 corpus는 이미 IOP/IPP 물리 정렬·140mm/336px·64plane·p2/p98를 사용하므로 이 수치를 새 기대 이득으로 옮기지 않는다. 캐시 속도 개선과 AUC 개선을 구분한다.
- 입력의 남은 차이: Steven/Rayan 코드는 anchor 주변 원 DICOM 연속3장을 묶지만 우리 within-slot-valid는 같은 slot의 표본 cache3장을 묶는다. 원본 연속성을 보장하지 않는다. Wguesdon의 tissue bbox는 병변 검출기가 아니며 Dread의 ROI 옵션은 기본 OFF·별도 helper/boxes가 필요하다. 주석의 adjacent/physical crop와 실제 index/crop 경로를 구분한다. 공개5그룹의 입력을 바로 통일하지 않는다.
- 우리 진행: 기본 corpus·입력 계약·알려진 중복 연결·소규모 라벨 감사는 완료했다. EXP081 max-span 직접교체는 -0.000684/CI0포함으로 미채택, EXP087의315건 입력은 수용했지만 full이 gradient 비유한으로 실패해 효능 미평가다. EXP088의336→384 확대 결과는 학습 레시피 묶음 비교이며 native384 영상 정보·전처리 단독 효과가 아니다. 실제 병변 ROI·native 연속 입력의 검증된 이득과 전체 라벨 교정은 완료하지 않았다.
- 감독의 조건부 근거: [미언급 조사](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)는 소견별 누락 구조와 표적/일괄 보간 차이를 보고했다. [라벨 최적화](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214)에서 한 작성자는 라벨 변경 LB0.931→0.942를, 다른 작성자는 단순 Qwen으로 low0.95·추가 정제의 작은 효과를 주장한다. [single-model 댓글](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)은 단순 mask가 도움 없던 사례와 OOF soft 혼합이 도움 된 사례를 모두 포함한다. 독립 재현0이며 Gold58 점수·보고서 라벨 품질·영상 모델 LB를 같은 지표로 쓰지 않는다.
- 적용·재검토 조건: MCL120/PFOA20/Synovitis16 감사와 train3475 source 비교는 라벨 수정 완료가 아니다. 현재 Dread 학습에 소견별 언급/불확실 loss mask·신뢰도 가중은 미적용이다. 새 명시적 충돌 근거, 검증 노출을 통제한 한 후보의 입력/감독 대조, 추론·개발 비용이 구체화될 때 재판정한다. 미언급을0으로 강제하거나 proxy를 정답으로 확정하지 않는다. 전체 재라벨·5그룹 재학습·진행 중 실험 변경은 자동 실행하지 않는다. [M06](#m06)·[M11](#m11)·[D15](#d15)의 한계를 유지한다.
