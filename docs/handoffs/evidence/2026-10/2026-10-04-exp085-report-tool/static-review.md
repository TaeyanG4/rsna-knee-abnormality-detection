# EXP085 정적 검토 체크포인트 · 2026-10-04 02:59 KST

`/root/anchor_inputs`, M02–05/M10 적용. source/PLAN/기존metadata만 읽었고 실제 영상·새 예측·라벨값·Gold·GPU·재평가0.

- 확인된 blocker: V1 `contextualize`의 q.float()@k.float(), attention@v.float()는 외부 CUDA autocast 안에서 FP16으로 다시 cast될 수 있다. `.float()`만으로 PLAN의 FP32 행렬곱을 보장하지 않는다. root가 발견한 이 문제에 동의하며 explicit autocast-disabled 블록을 넣은 새 source/protocol 확인 전 평가계약 동결 보류.
- 나머지 읽기범위에서 확인된 추가 blocker는 없음. PositionWindows는 원 StudyWindows 픽셀을 사용하고 같은 UID/epoch/seed로 centers를 재현한다. intensity jitter는 별도 stream이고 window 순서를 바꾸지 않는다. 빈 query는 finite 자기attention 뒤 residual0, 원blank pooling은 보존한다.
- 공통 초기B 복원·expanded state hash·각arm seedreset·전체optimizer소유·3448updates·skip즉시실패·모든 minibatch UID/center/position/RNG trace 비교는 실험의 대조 조건에 부합한다. initial zero projection으로 treatment 효과는 후속 update에서 학습하며 input-A 비교가 아니므로 과거 골절문제 해소를 주장할 수 없다.
- pilot은 T16만 추론하며 예측/weights를 저장하지 않는다. 원B는 weights-only 초기값이고 새 optimizer의 추가8epoch이므로 B16 연속학습으로 부르지 않는다. 외부 supervisor가 pilot+full active7200초 상한을 맡는다. per-fit soft deadline은 summary/cleanup 직전에 검사하므로 마지막 정리시간까지 개별fit3600초의 독립 hard-timeout을 입증하는 것은 아니다.
- 원B/RNclip 재사용 후보: EXP084 평가와 새runner 모두 동일 V888, 같은 StudyWindows source·within-slot-valid·K24/224/ImageNet, eval batch4·workers2/internalCPU1, no_grad·strictFP16→float32 sigmoid다. 새attention은 계획된 모델개입이며 입력계약 변경이 아니다. evaluator는 source/protocol/UID/라벨/group/reference 산출물hash를 다시 연결한 경우에만 기존 예측을 사용한다. 실제 재추론/동등성 실측을 추가로 했다는 주장은 없음.
- 예정 evaluator 계약: 양arm summary PASS+전체phase PASS+gpu_job SUCCESS+resource COMPLETE/clean/exit0를 모두 먼저 확인. 그 전 새predictions/라벨값 접근 금지. 초기B/provenance·최종checkpoint hash·pairedtrace/skip·source·group·UID·label 계약을 확인. primary treatment−control, 두arm−원B mandatory, 고정RNclip90/10 secondary;500회 동일component draws·12소견 전체유효 조건부CI. 단일class 소견은 macro 무효, subgroup 탐색용/no CI. 수정 V2의 정확CLI/phase tag는 최신 실행계획에서 동결한다.

## V2 확인 및 동결 · 2026-10-04 03:12 KST

- 위 V1 blocker는 V2에서 해소됐다. logits/softmax/value reduction을 explicit autocast-disabled 블록으로 분리하고 실제 logits/attention/context/bias FP32 gate를 추가한 source를 확인했다. V1 기록은 보존한다. protocol SHA `58110be852792f8849ce051f83e12e0a2a3c12282cf2b9a971364dca6a5d7539`.
- 새 evaluator와 합성검사 32개 PASS(1.82초), 실제 metadata-only prepare PASS. [registration](registration.json) SHA `3251ab5b0be7f60b0d9d0e01ebc57ac1b88609d181738a33430b913c9bf9b954`. 03:11:56 KST source binding 17개 재확인 일치, prepare 출력과 registration byte hash 일치, evaluation-v1 미생성.
- 수정 source owner가 보고한 52검사는 이번 agent가 재실행하지 않았다. 새 예측·라벨값·Gold·model state·GPU·EXP084 재평가 접근/실행 없음. 실제 pilot/full 종료나 성능 PASS를 이 문서가 선언하지 않는다. 이후 root는 [실행계획](execution-plan.json)과 완료 gate에 따라 1회 평가한다.
