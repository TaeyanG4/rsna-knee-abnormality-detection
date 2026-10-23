# 다음 작업과 실험 계획

다음 가설·우선순위·비용·판정 조건을 관리한다. 실제 진행과 결과는 [HANDOFF](HANDOFF.md), 교훈과 재검토 조건은 [discoveries](discoveries.md), 공통 지시는 [AGENTS](AGENTS.md). [이전 원문/hash](docs/handoffs/archive/snapshots/2026-10-04-agent-docs-v1/manifest.json).

## 우선순위

| 우선 | 가설·목표 | 다음 단계·담당 역할 | 비용·완료 조건 |
|---|---|---|---|
| P1 | 자체 224/384 G1 레시피 차이가 대회 영상 정답에도 전이되는가 | root가 완료 두 가중치를 재사용한 단독 후보의 실행성을 준비 | CPU 준비 최대 20분, 다음 단계는 별도 검토·실행 비용 판정 |
| 보류 | 기준본 + RN3 추가 비용 측정 | [중단 checkpoint](docs/handoffs/evidence/2026-10/2026-10-04-rn3-cost-v1/HANDOFF.md) 보존 | 실측 미완료. EXP088 결과와 배포 비용을 본 뒤 재판정 |

## 실험 계약

### EXP088_RAPTOR384_G1

- 변경 근거: EXP060 384 F0/F1 weak AUC .896286/.893828과 EXP061 224 .880645/.877222는 더 강한 레시피 투자를 지지한다. split·epoch·선택·runtime 차이가 있으므로 해상도 단독 효과가 아니다. 10-04 사용자 시간 배분 지적을 적용해 RN3보다 우선한다. M02/M03/M04/M07/M10을 적용한다.
- 변경 변수: generic CoAtNet 384, 12epoch, BF16, microbatch 4/accumulation 2. 기존 EXP084_B_G1(224/8epoch/FP16)과 실용 레시피를 비교한다. LR 3e-5/1e-3, weight decay .02, clipped positive weights, seed 42, within-slot-valid, train K12/eval K24, 224와 동일 G1 T3455/V888·12target·weak 라벨·그룹·평가 지표를 고정한다. 원 336px cache를 384로 확대하며 native384 corpus 재현이라고 하지 않는다.
- 선택: 최종 epoch 12 고정, 중간 V 지표/Gold/SWA 없음. V 라벨 파일은 기존 loader에서 읽고 0으로 치환한다. full은 generic에서 재시작하며 pilot/B 가중치를 이어 쓰지 않는다. scheduler 전체 horizon 5184 updates, 매 epoch 863 microbatches/432 updates, 마지막 accumulation group 1을 유지한다.
- 예산: pilot V2 32 updates/600초, full 18,000초(최종 추론·저장·cleanup 포함), pilot 표본 tree RSS 24GiB / full 96GiB, 12 workers/내부 thread 1, eval workers 2/batch 2. pilot V1은 startup 포함 240초에 걸려 32update 완료 없이 종료했다. 원 실패를 보존하고 같은 레시피의 비용 측정 V2는 준비시간·8update마다 진행 기록을 추가한다. 모델 실제 초기 state SHA를 완료 pilot에서 기록하고 full과 일치시킨다. 초기 파일 SHA/source/ordered UID는 manifest로 결합한다.
- 이전 메모리 근거: 기존 EXP086 full의 표본 tree RSS 88.57GiB는 공유 mmap 중복 가능성이 있다. full 24GiB cap은 실제 고유 RAM 부족을 뜻하지 않는 중단을 만들 수 있어 96GiB tree RSS로 변경한다. 로컬 RAM 약95GiB/시작 가용량을 확인하고 root가 수용한다. OS hard cap/고유 RAM으로 주장하지 않는다. 완료 pilot의 모델·UID·라벨·source 계약은 그대로 재사용하며 재시험하지 않는다. 새 full budget/protocol은 별도 검토한다.
- 복구 준비: M07에 따라 원 RSS cap 실패를 보존한다. 모델·학습·평가 조건/12 workers를 바꾸지 않고 새 helper/runner 버전에서 Windows private commit 합계 cap 32GiB와 호스트 available RAM 하한 8GiB를 감시한다. tree RSS는 진단값으로만 기록한다. 100ms 표본·OS hard cap 아님, I/O/측정 실패는 FAIL. epoch마다 원자적 복구 상태(model/optimizer/scheduler/RNG/순서)를 저장하되 최종 epoch 12 선택은 유지하고 자동 resume/중간 평가하지 않는다. 원 full은 체크포인트가 없어 generic에서 새 fit가 필요하다. CPU 합성 감시·snapshot 검사와 별도 자원/동등성 검토 뒤 최대 18000초의 새 fit 1개를 root가 판정한다. 원 실패까지 GPU 감독 시간은 9406.905초(첫 pilot242.031 + 완료 pilot97.593 + full9067.281)이며 새 최대18000초를 더하면 누적27406.905초다(추가 startup FAIL4.234초 보존). 새 fit 실측 예상은 약12000초이며 대안보다 먼저 자체 강한 레시피의 효능을 알아내는 데 사용한다. 24GiB 초기안은 실측24.827GiB에서 종료돼 host 가용46.7GiB·main private18GiB/RSS3.96GiB를 회수했다. 32GiB는 실측 초기값 +12worker×prefetch2×batch4×K12×RGB384float32 명목1.898GiB +snapshot 약1GiB +4GiB 여유를 위한 한정 조정이다. full 피크는 미측정이므로 host available8GiB 하한은 유지하고 추가 cap 확대/fit는 자동으로 하지 않는다. 실행 decision에 원 RSS 및 최근 private 실패 해시·검토 ID를 모두 결합한다. 초기 prefetch 2.025GiB 표기는 보수적 초안이며 정확 tensor 명목값은1.8984375GiB, root decision에 정정한다. 새 한도도 실패하면 재판정한다. 완료 pilot/모델/원 corpus 재검사·추가 pilot 없음. 통과 못하면 실행하지 않는다. 새 supervisor telemetry startup 파일의 allowlist 누락으로 V2는 학습 전 실패했다. V3/V6은 정확 metadata/tmp만 추가하고 합성2검사 및 별도 delta 검토로 교정한다.
- 진입: 별도 검토와 pilot parent/child 성공, 32 updates·skip 0·cleanup·finite 검사, `1.3 × last24 whole-update 평균 × 5184 + 실제 setup 초 + 300 ≤ 18000`일 때 root가 full을 별도 실행한다. 실패 시 원 로그 보존, 자동 retry/precision fallback/skip 완화 없음.
- 판정: 동일 V888의 저장 B 예측을 재사용해 raw candidate−B와 12소견 P/N·Baker/Fracture·그룹·공유 component 500회 조건부 CI를 한 번 계산한다. 양의 효과와 비용/소견 손실을 함께 보고 추가 fold를 결정한다. CI가 0을 포함한 작은 이득도 보존하되 자동 확대하지 않는다. .943/private/임상 개선을 주장하지 않는다.
- 제출: 준비된 서로 다른 가설의 전이를 상시승인 범위에서 확인한다. 일일 횟수 소진, Public 비율/소견/seed 반복 튜닝은 하지 않는다. 표시 자릿수로 .002~.003 잡음선을 정하지 않으며 사전 등록만으로 선택 편향이 사라진다고 하지 않는다.
- 담당: root 구현·실행·기록, 별도 검토자는 읽기 전용 source/예산 검토만 한다. 새 라벨·split·teacher·공개·유료·remote·5fold 자동 실행 없음. [진행·재개](HANDOFF.md).
- 기준본 결합과 내부 B+RN은 다른 질문이다. 정확한 기준 스택/hash·결합 수식·평가 노출·runtime을 고정한 후 제한적 전이 탐색을 판단한다.
- EXP085·작은 loss/RN 효과·입력/표현·감독 가설은 discoveries M04–M06의 조건으로 비교한다. 추가 fold/seed/해상도를 자동 확대하지 않는다. 재실행 제외 범위는 discoveries의 closed ideas를 따른다.
- 준비된 검증후보 제출은 기존 상시승인 범위다. 새 공개/유료 작업은 계획에 없다.

### EXP089_RECIPE_TRANSFER_G1 · CPU 실행성 준비

- 질문: 완료 `EXP084_B_G1` 224px/8epoch와 `EXP088_RAPTOR384_G1_V4` 384px/12epoch의 실용 레시피 차이가 대회 영상 정답으로 전이되는가. 동일 T3455/V888·G1·seed42·clipped weak 라벨이고, 해상도·generic 변형·epoch·precision·microbatch는 함께 달라진다. 하나의 레시피 묶음 비교이며 해상도 단독 효과로 해석하지 않는다.
- 후보는 각각 기존 최종 가중치 1개, raw sigmoid 확률·동일 원 DICOM reader/336 cache 재현·K24·ImageNet·within-slot-valid·strict FP16·TF32 off로 고정한다. anchor/RN/혼합/새 학습은 없다. 두 예측을 서로 다른 해상도 때문에 같은 multi-member 호출에 넣지 않고 각각 처리한다.
- 먼저 root CPU 준비 20분 이내로 기존 passive loader/reader와 공개 배포 도구의 최소 재사용 가능성, 필요한 자산·입력·batch/환경 검사·정확 argv·자원 예산을 정한다. 모델/배열/라벨/OOF 값 로드·GPU/remote/등록/제출은 준비 단계에서 0이다. 준비가 끝나지 않으면 미완료 범위를 반환하며 자동 연장하지 않는다.
- 기존 100건 DICOM/cache 일치와 passive reader 근거를 재사용한다. 새 384 가중치·resize·배포 batch/device 조합은 기존 224 parity의 적용 범위 밖이므로 필요한 차이만 검사한다. 이미 계산한 V888 AUC/통계·원 100 DICOM 감사·완료 pilot은 재실행하지 않는다.
- 실행/제출 경로는 별도 최소 검토 후 root가 단계별 결정한다. 실제 무료 할당·현재 일일 한도·중복·새 후보 runtime을 확인하며, 준비만으로 실행 허가가 되지 않는다. 최대 두 사전 고정 단독 후보의 전이 비교를 고려하고 대회 점수 차이로 소견/비율/seed를 재조정하지 않는다.
- 사전 예상은 weak 개발군에서 작은 양의 효과를 보인 384가 영상 정답에서도 개선할 가능성이 있다는 방향 가설이다. 효과 크기·잡음선·승률은 근거 없이 정하지 않으며 하락도 가능하다. 기존 6weights .913이나 anchor .943와의 차이를 이 레시피의 인과효과로 부르지 않는다. Public 관측은 선택 편향과 private 불확실성을 남긴다.
- 투자/결과 별도 검토는 CPU 구현 준비만 PASS했다. root가 얇은 `tools/exp089_single_recipe.py`와 V1 protocol/local-plan을 준비했으며, source/실행 경로 검토가 남았다. 첫 local 제안은 기존 engineering100과 겹치는 V8의 각 모델 batch1 확률을 저장 최종 확률과 비교한다(라벨/점수/새 DICOM 없음). 240초·GPU1·내부 thread1·retry0, RSS16GiB는 표본 사후 수용값이며 online kill cap이 아니다. T4/공개3/제출은 별도 미준비다.
- local 검사는 완료·수용했다. 다음 root CPU 준비는 20분 이내에 두 완료 가중치·기존 passive source/offline wheel·동일 engineering100·새 V8 참조를 묶은 private T4 계약만 만든다. 두 새 모델의 다른 기기/실 reader 조합은 기존 6weights 검사의 범위 밖이다. 제안은 모델마다 새 child에서 100건 추론 최대900초, 두 child 순차·platform2100초·worker0/내부 thread는 실제 CPU 할당 이내·retry0, 표본 RSS8GiB 및 output64MiB 사후 수용이다. 새 PID를 cold process라 부르되 OS 파일 캐시 초기화로 주장하지 않는다. 원100 입력 일치 결과를 참조하며 decoder 감사 자체를 반복하는 목적이 아니다. CPU 준비·별도 source/예산 검토·실제 무료 quota와 SDK 시간 계약 확인 뒤 root가 remote 1회 여부를 결정한다. CPU 단계는 upload/GPU/remote/제출 0이며 공개·유료 작업은 없다.
- T4 첫 실행은 notebook kernelspec 누락으로 셀 실행 전에 실패했다. 원 버전과 자산을 보존하고 새 notebook metadata만 교정한다. 같은 코드·protocol·dataset v1을 재사용하며 별도 delta/예산 검토·현재 무료 quota 뒤 새 push 1회를 판정한다. 추가 platform 2100초/T4x2 1.1667h, 첫 예약 상한까지 합친 보수적 누적 2.3333h다. 원 실패의 실제 할당 비용은 미확정이며 자동 retry는 없다. [실패·새 CPU 범위](docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/root-notebook-metadata-repair-preparation.json).
- 후보 채택·추가 fold는 결과/소견 손실/비용과 별도 검토 후 판정한다. 새 seed/5fold/Gold/라벨 교정·anchor arm 교체는 자동으로 시작하지 않는다. [완료 비교와 대안 비용 판단](docs/handoffs/evidence/2026-10/2026-10-05-exp088-raptor384-v1/evaluation-v4-root-result-decision.json).
- T4 완료 뒤 다음은 root CPU20분의 test-only release 준비다. 원 dataset v1의 passive reader/runtime/weights를 재사용하며 새 wrapper만 만든다. 공식 sample 순서·전체 UID/12열·유한 확률·FP16/TF32 off·source·자연 종료/cleanup·원자적 CSV 게시를 요구한다. 두 단독 후보를 Public 결과 전에 고정하며 1번째 점수로 2번째를 바꾸지 않는다. 별도 최소 source/예산 검토 뒤 private Public3부터 진행한다. 제안은 각 platform900초/내부600초·RSS8GiB 사후 수용·retry0, 2개 순차 T4x2 최대1GPUh. full은 별도 판정하며 platform32400초/내부32100초·30분 여유 screen30600초를 유지한다. warmed R384 속도 대신 둘 모두 더 느린 B224 감독652.503초/100과1.3배+300초로 운영 비용을 보수적으로 선별한다. hiddenN/실시간 보장은 아니며 screen/cap 실패 시 자동 완화·재시도하지 않는다. CPU 준비는 모델/라벨/OOF/pixel/GPU/network0, 이미 완료한 engineering100/V8 재검사0이다.

## 기록 형식

작업ID·우선순위 / 가설·새 근거 / 기준 스택·예측 hash / 변경 요소 / study·12target·label·split·mask·변환 계약 / 실행자·검토자 / 재사용 산출물 / 비용·중단·채택·보류 조건 / 결과·검토 링크.

후보 목록은 실행 예약이 아니다. [10-03 전략](docs/research/route-to-first-2026-10-03.md)은 당시 근거이며 현재 상태·승인은 루트 문서를 따른다.
