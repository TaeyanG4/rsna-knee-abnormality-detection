# 2026-10-03 22:35 KST · EXP083 고정 평가기

- 상태: 완료(구현·CPU 검사·root 최종등록 명령 준비). 기록자: Codex `/root/anchor_inputs`.
- 요청·결정: 새점수 개봉 전 EXP083 전용 평가기 준비. V1 FP16 가정 실패로 초안 보존 후 root가 확정한 V2 nativeFP32·실패시간 포함2700초 계약에 맞춤. 원protocol/source·중앙문서 불변, 채점/GPU/Gold 미실행.
- 결과·파일: `tools/exp083_slot_bias_report.py`, 전용tests; 본폴더README/verification/v1초안. 첫등록은실제tag를명시해성공했고, root요청으로기본tag수정후 `registration-before-default-fix.json`에보존. 최종sourceSHA `d857ec4d9b15ccf0c6d6edbd9314dbe75355f1acfcd5ef9a7d2b7c08a3e0223a`; V2protocolSHA `2ae04b602dd7688c6c06625ca3c9e6e257ffb93113fdf478a0b6200db5c2a55b`.
- 검증·근거: 최종합성CPU10passed/2.32초. P/N·primary/secondary·부분macro거부·동일component draw·single-class전체무효·가중AUC=study복제·skip/RNG/scale/중간채점거부·CSV↔NPY·registration변경·수동import검사. 독립 소스검토의hash/필수binding/runner지적반영. 기본tag수정전prepare0.57초·35bindings 성공, 예측/라벨값 미열람.
- 계약: primary treatment−재학습control, common-cache frozenB8는secondary.12소견·Fracture·scanner/mask 탐색 진단,500component bootstrap/seed20261003. 동일draw를모든arm·target에 공유, 단일class면macro draw전체무효. zero-skip/1100update·완료job/pilot/root authority·V1실패6.984초 포함 예산을 채점 전에 검증. 과거B replay·독립임상/private 주장은 금지.
- 남은 일·다음: root가 README의 `--prepare`로최종source동결, 실제full성공확인후고정 `--run`1회. 기본값/실제job tag 모두 `EXP083_SLOT_BIAS_V2_FULL`. 노출weakF2·한seed·Gold ancestry한계유지. 실행 후현실검증/판정/중앙STATE/INDEX는root 담당.
