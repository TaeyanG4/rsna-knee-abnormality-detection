# 2026-10-04 02:45 KST · EXP085 상대위치 CPU 준비

- 상태: CPU 구현·프로토콜 초안·합성검사 완료, GPU 미실행. 담당 `/root/precision_code_review`; 출력은 신규 helper/runner/tests와 이 evidence. 중앙기록은 root 담당.
- 결정: M02/M03의 T-only B8·같은group T3455/V888 계보 유지. M04/M05에 따라 treatment−control과 두arm−원B를 함께 판정한다. M10에 따라 Gold/LB/epoch/비율탐색 없이 상대위치 유무1쌍만.
- 결과: 원B backbone/head 보존, 추가262,208파라미터의 같은 residual attention. 출력 projection0으로 원B에서 출발; control좌표0/treatment상대좌표. bs8·K12/24·224px·같은픽셀/정규화/마스크·전체backbone8epoch. weights-only 초기값의 **optimizer재시작 추가적응**이다. [상세](PLAN.md).
- 검증: [verification](verification.json)의 CPU47검사 통과. 실제checkpoint는 hash만 확인; 실제영상·라벨·예측·Gold·GPU·원격 접근 없음. 기존source4개와 신규source bindings7개 확인. 음수/NaN pilot시간 허용 결함과 fixture누락은 수정·회귀검사 통과로 기록. reviewer의 임시폴더 삭제는 자동검토가 차단해 보존했다.
- 계약: `artifacts/experiments/EXP085_RELATIVE_POSITION_G1_PROTOCOL/manifest.json`, SHA `73661946b88d804abea397c08629d31a78500e933a42f10c0944725f3cc43228`. [실행계획](execution-plan.json)은 미실행이며 root phase decision이 필요하다.
- 다음: paired pilot각32updates/T16·600초cap → 실제속도×1.25로 각fit≤3600초·pilot+full≤7200초면 root가 full배정. retry0. final-only 두arm과 원B·고정RNclip90/10 secondary를 평가담당자가1회 판정. source초안은 실제GPU/Windows worker 실행과 T4수치동등성 미검증이다.
