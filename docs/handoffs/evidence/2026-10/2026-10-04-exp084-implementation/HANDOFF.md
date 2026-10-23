# 2026-10-04 00:53 KST · EXP084 CPU 구현·등록

- 상태: CPU 준비 완료, GPU 미실행. 기록자: Codex `/root/anchor_inputs`.
- 요청·결정: root가 RN clip20/RN ones20/공통 B8을 선택. 새 group hash 규칙을1회 배정하고 fixed-final·seed42·90/10·strictFP16·workers12를 고정했다.
- 결과: [protocol](workspace://artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json) SHA `fe3a4e3c97f7ae5f99530f780e4a3551637dc9772f02c6d75a1b2212bf3d51de`. T3455/V888, component3323/873·교차0. reference12소견 모두P/N 존재(Fracture P29). 새 split은 untouched/환자독립 검증이 아니다.
- 코드: [helper](workspace://tools/exp084_group_loss.py), [runner](workspace://training/runners/exp084_group_loss.py), [tests](workspace://tests/test_exp084_group_loss.py). Gold·epoch/final AUC·SWA선택 제거, final 예측만 저장. 원 optimizer inner-loop AST 동일·원4source 불변. 기존4source/원protocol/중앙문서 수정없음.
- 검증: [CPUverification](CPUverification.json), [명령](commands.md), [초기값](initial-assets.json). synthetic9 PASS·20binding+원4source SHA PASS·세arm 인자gate PASS. CPU offline fresh RN/B full-state가 기존 기대SHA와 exact; CUDA 초기화false. B snapshot/revision은 로컬cache 근거·원격미조회.
- 실행조건: [queue](workspace://artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/queue.json)는 RN→RN→B. RN각1800초/B3600초·전체7200초·retry0·실패시후속중단. workers12·내부thread1. 실제처리속도/GPU초기값은 미검증. skip1%미만PASS도0skip/양RN동일updates를 뜻하지 않는다.
- 남은일: root가 evaluator 등록과 CPU계약을 검토한뒤GPUqueue 시작·ID/log/ETA/heartbeat 기록. 세fit완료전 점수개봉금지. evaluator 담당에게 manifest SHA·스키마 전달. 기존D01 조기조회없음.
- 누락점검: Dread계보/노출weak/환자연결미확인 유지. 새imageancestor V제외를 전체독립성이나exactD01/private증거로 확대하지 않는다. 작은효과·골절손실·실효updates를후속판정에보존.
