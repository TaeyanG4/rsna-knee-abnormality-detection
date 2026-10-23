# 2026-10-04 · EXP086 파일럿 독립 실패 검토

- 상태: 완료. 담당 `/root/anchor_inputs`, 마지막 관측은 [검토 JSON](pilot-independent-review.json)의 `observed_at`. 중앙문서 변경 없이 root에 반환했다.
- 요청·결과: 사전 고정 metadata로 실패를 확인했다. 실제 종료05:34:36.286 KST·83.235초, CUDA strict deterministic에서 bicubic AA backward 미지원. helper의 resource_gate 실패는 전파된 결과다.
- 검증: 등록 source21개·helper/worker argv·initial/provenance 불변. worker/supervisor/helper exit1, cleanup actions/errors/survivors0.702개 표본의 host tree RSS최대8.205681GiB 재계산 일치. GPU peak와 다르다.
- 제한: pilot receipt가 없어 skip/steady/full예상값은 없다. 첫 optimizer update 전 실패는 고정224입력과 매회 PE보간·backward→step 순서의 소스 추론이며, 지속 counter에 의한 실측0 주장과 구분했다. 성공gate/모델/가중치/예측/라벨/GPU/재실행0.
- 다음·기존결정: M02/M03/M05/M08/M09/M10 적용, 원실패 보존. root의 별도 CPU수정안 검토를 기다리며 새pilot/full은 미허가. 이 실행 오류는 DINO 경쟁성능 기각 근거가 아니다.
