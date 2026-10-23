# 2026-10-04 02:43 KST · producer thread replay V2 준비

- 상태: CPU 구현/합성검증 완료, 실제100 미실행·root 검토대기. 담당 Codex `/root/anchor_inputs`. V1 입력math PASS와 producer연결FAIL을 모두 보존. [단일UID 원인](../local-input-binding-diagnostic/HANDOFF.md).
- 결정: 새 [helper](../../../../../../tools/b_rn3f_thread_input_v2.py)는 완료 producer parent→worker SHA·상태·100UID/입력기록·actual selected_threads를 확인한다. passive_sources의 초기1thread 후 실제producer thread를 명시적 적용·기록, CPU strict/inference_mode로 재구성한다. 모델·CUDA·labels/예측배열0.
- 계약: [새protocol](protocol.json) SHA `820be3d9d047f8e746c0753c0bbcc59a1171e008a22286ba832df5ceba579b5f`. helper SHA `1d6bc7c2a8d0cd152b784d9f26e196dad41746423733ef63efc65dabbb89e21e`. 실제100 전에 동결했고 선행1UID 진단을 명시. 원V1 helper/protocol/producer/stage·수치허용치 변경0.
- 출력: V1 호환 report에 `thread_adapter`/실제 `cpu_threads`를 추가. V1 helper SHA는 재사용한 math 출처이며 V2 실행주체 SHA·새protocol·producer binding은 별도 명시한다. math explicitPASS + V1 validate_input_report의 exact100 hash/volume/mask/centers 연결 + 종료시 producer receipt불변을 요구. 별도receipt `PASS_INPUT_AND_PRODUCER_LINK_ONLY`만 exit0, 실패exit1.
- 검증: [verification](verification.json). 새 synthetic20검사0.58초PASS: 부모실패/hash·thread0/bool/초과·interop·불완전/역순입력·실패math·exact hash/mask/center·actualthread·protocol변조·CLI실패. 기존suite 재실행0. 첫실행은 Windows 기본pytest temp접근거부(1PASS/19setup오류), 새evidence basetemp로 해결. 기준 완화0.
- 재개: [execution-plan](execution-plan.json)에 정확local argv/remote template. root 검토후 CPU100 1회 예상60초·cap120초/retry0. local output `.local/b-rn3f-runs/20261004-v2-local-input.json`, receipt 같은prefix `-acceptance.json`. 원 producer/GPU 재실행 없음. 원격은 producer완료·offline환경·관측경로 확인 후 같은bridge 사용.
- 연동: dicom_header_io가 CLI/출력schema를 확인하고 T4 stage에 V2protocol/helper·localacceptance 추가 예정. 기존 `b_rn3f_acceptance.py evaluate`는 그대로 호환. 새bridge acceptance도 별도검사해야 함. completed producer 입력metadata만 검사하며 prediction결과 전체감사는 기존 paired evaluator가 맡는다.
- 제한: 합성20PASS는 실제100연결/T4/전체runtime/경쟁성능PASS가 아니다. 실제100/root 승인·후속예약은 중앙담당. 현재 이agent 진행job/예약 없음.
