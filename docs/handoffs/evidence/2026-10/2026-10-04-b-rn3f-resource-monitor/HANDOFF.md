# 2026-10-04 01:39 KST · B_RN3F RSS supervisor

- 상태: 구현·CPU 합성검사 완료. 담당 `/root/precision_code_review`; 실제 모델 실행은 미실행.
- 요청·결정: 기존 producer/stage를 바꾸지 않는 외부 프로세스 RSS 기록. [명령](execution-plan.json)은 root가 EXP084 종료와 활성 GPU를 확인한 뒤 실행한다.
- 결과: `tools/process_resource_receipt.py`, `tests/test_process_resource_receipt.py`. fresh output·정확 argv·100ms root/recursive descendant RSS·PID+creation time·샘플/접근실패/지연 횟수·Windows 가용RAM·실제 Linux cgroup 경로 시작/종료 읽기 전용 snapshot을 기록한다. producer 재시도0, 실패/timeout/남은 자손은 관측된 소유 프로세스만 종료하고 최종 receipt를 동결한다.
- 검증: [verification](verification.json)의 CPU20검사 통과. 무관한 sentinel 생존, 손자 프로세스 정리, PID 재사용 보호, 인자 보존/덮어쓰기 거부, cgroup 경로·미변경 fixture 확인. Linux 실제 host와 GPU는 미검증. EXP084 evaluator SHA 및 frozen producer/manifest SHA 유지 확인.
- 제한: RSS 표본은 순간peak를 놓치고 공유메모리를 중복합산한다. 관측 전에 분리된 자손은 놓칠 수 있다. cgroup peak는 공유세션 문맥이며 reset하지 않는다. v1에는 memory.events가 없어 failcnt를 별도 기록하고 telemetry 불완전을 표시한다. hard kill/PC 종료는 receipt 완료를 보장하지 못한다.
- 다음: root가 명령·새 output/tag를 검토하고 허가된 reference를 배정한다. wrapper COMPLETE는 임상성능·T4 수치동등성·full-runtime PASS가 아니다. 중앙 문서는 root 담당. Linux live 확인은 실제 T4 실행 시 필요.
