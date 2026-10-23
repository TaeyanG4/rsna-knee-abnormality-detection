# 2026-10-04 01:01 KST · EXP084 고정 그룹 loss 평가기

- 상태: 완료(구현·사전등록; 실제평가미실행). 기록자: Codex / precision_code_review. 소유: tools/exp084_group_loss_report.py·tests/test_exp084_group_loss_report.py·이 evidence. 중앙문서/학습 source 미수정.
- 구현: prepare→해시동결→3fit SUCCESS/계약확인→평가1회. 전체V average rank/N·90:10·4대조,동일component500draw·seed20261003·한target single-class면 전체draw제외. P/N/AUC/Δ/12·골절/PFOA/MCL·scanner/mask·실효update/skip·한계 출력.
- 검증: CPU합성23통과(2.76초),adapter 독립AST14통과. 실제label값/예측/Gold/점수/GPU 접근없음. Temp권한/상위경로 부재 fixture오류 기록보존. [검증](verification.json), [검토](adapter-independent-review.json).
- 사전등록: 실제T3455/V888·component3323/873,manifest fe3a4e3c… 검증. [등록](registration.json) SHA18baee07…·27개source/file/queue binding·기존prelaunch재검사없음. 실제평가출력은미생성.
- 다음: root만3fit완료후 [고정명령](run-command.txt) 1회실행. 등록된label/UID/source/path 계약 변경·출력덮어쓰기·자동재평가 금지. 실패는원인/버전별보존. 새로운그룹도weak/노출자료이며정확D01검증아님.
