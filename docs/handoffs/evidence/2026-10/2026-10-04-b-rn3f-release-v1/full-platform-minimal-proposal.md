# Full 플랫폼 설정 최소안 · 제안만

> 05:17 구현 갱신: 아래는 당시 제안 원문이다. 이후 [최종 결정](root-final-config-decision.json)이 원 release source의 byte 불변을 요구하여, 실제로는 별도 `tools/b_rn3f_full_config.py`로 새 stage를 만들었다. 원900 필드/결정은 과거 V3 검증 근거로 보존하고 새 `timeout_contract`와 실제 SDK 요청32400을 명시했다. source 변경 제안은 적용하지 않았다. [완료 인계](full-config-HANDOFF.md)·[CPU 검증](full-config-verification.json)·[root 실행계획](full-config-execution-plan.json)이 구현된 현황이다. 원격 실행은 아직 하지 않았다.

2026-10-04 05:09 KST · `/root/anchor_inputs`. [public3 독립 수용](independent-result-review.json)·[root 시간 결정](root-timeout-decision.json)·[최소 경로 요청](root-final-route-request.json) 적용. 신규 구현·push·제출은 하지 않았다.

추천은 **계산 코드를 바꾸지 않는 별도 설정 revision 1회**다. release source `4bca01aa…356ae`를 출발점으로 exact6·고정 집계·codec·입력·run/publish/ack 함수는 재사용하고, 포장·manifest/운영 기록의 실제 플랫폼 한도를32400초로 정합화한다. 설정 문자열 변경도 새 source/notebook/manifest SHA를 만들므로 이전 source hash와 같다고 쓰지 않는다. 새100건benchmark·모델추론 비교·기존52검사 반복은 필요 없다. 공개3건은 해당 새 version의 통상 실행으로 한 번만 확인하고 그 정확 version을 이후 제출 후보로 검토한다.

실제 public3 post-ack는99.7676초였고, 두T4를 모두 계산해도 이 구간은 약0.0554GPUh다. 컨테이너 시작·과금단위는 포함하지 않으므로 계획은0.1–0.2GPUh로 잡되 **이는 예상이며0.5GPUh hard cap이 아니다**. 내부600초는 정상적인 worker 감독/게시/ack deadline이다. parent·파일시스템이 멈춘 상황까지 notebook 전체를600초 안에 종료시키는 강제 장치는 아니다.

새 watchdog을 만들지 않는 최소 운영은 이 차이를 비용으로 명시하는 것이다. 플랫폼32400초·두T4 기준 비정상 최악노출은18GPUh다. 마지막 quota는04:37의27.70h이며 이번 약100초 실행 전 관측이다. root가 실행 직전 잔여량과 다른 예약을 확인해 **18GPUh 이상을 남겨 두는 한 번의 설정 실행**을 별도 승인한다면 quota 안에서 가능하다. 정상적으로 약2분에 끝나면 실제 사용분만 차감된다. 18GPUh 노출을 허용하지 않으면 full HOLD를 유지한다. 작은 예상시간만으로0.5GPUh 보장을 붙이지 않는다.

기존 같은-chat 완료 예약으로 약5분에 한 번 확인한다. 여전히 실행 중이면 실제 로그/상태로 이상을 판단하고 검증된 기존 취소 수단이 있을 때만 중단한다. 반복 확인·자동 재시도·새 감시 서비스는 추가하지 않는다. 앱/PC가 꺼져도900초 중단이 보장된다는 주장은 하지 않는다. 최종 안전 한도는 실제 플랫폼32400초다.

정확한 변경 지도는 `tools/b_rn3f_release.py:36`의 시간 정책 명칭/값, `:456` manifest timeout 설명, `:503–505` stage/push 설정, `:524–529` SDK request/기록이다. 과거 public qualification900과 미래 실제platform32400을 같은 필드로 모호하게 두지 않는다. `run:357`, `publish_final:266`, `finish_record:325`, worker/codec 및 frozen producer는 변경하지 않는 안이다. 기존 root-timeout 결정은 보존하고 별도 새 결정에 연결한다. 확인은 config diff·내장source/manifest·실제SDK timeout/metadata에 한정하며 CPU15–20분을 상한으로 제안한다.

만약 **총 public900초/두T4≤0.5GPUh가 반드시 강제 조건**이면 이 설정-only 안은 부적합하다. 첫cell bootstrap부터 전체 notebook 자손을 소유한 외부 supervisor가900초(정리 포함) 전에 종료시켜야 한다. Python signal만 추가하거나 현재 내부600초를 hard cap으로 부르는 것으로 해결되지 않는다. 기존 supervisor를 바깥으로 옮기는 별도 wrapper도 firstcell/추출/종료/게시 순서가 바뀌므로 표적 failure 검증과 새 실행이 필요하다. 이는 작은 timeout 설정 변경을 넘는 작업이다. 현 quota에서 최악18GPUh 한 번의 노출을 허용할 수 있다면 먼저 설정-only를 선택하고, 그 노출을 허용할 수 없을 때만 별도30분 설계/검증 예산으로 검토한다. 지금 구현하지 않는다.

완료되면 공개3의 source/설정·raw/CSV·parent/resource/ack만 연결하고, 실제 hidden N에는 기존 고정 `1.3*(877.354403408*N/100)+300<=30600` 조건을 유지한다. Public3의100초를 hidden N에 선형 투영하지 않는다. 숨김 채점의 quota 처리·시간상속을 추정하지 않으며, 대회 제출은 정확한 새 version·가설·비용을 제시한 뒤 사용자 별도 승인을 따른다.
