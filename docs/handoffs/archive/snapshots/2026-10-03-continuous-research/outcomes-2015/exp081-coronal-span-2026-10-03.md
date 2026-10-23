# EXP081 시리즈 선택 대조 — CPU 준비 결과

2026-10-03 19:32 KST. 동일 EXP080-B fixed-final8 가중치로 입력만 비교하는 별도 가설이다. EXP080 F2 결과를 본 뒤 실행을 결정했으며, 노출된 보고서 라벨 개발 F2에서의 탐색이다. 골절 손실을 해결한다는 근거는 아직 없다.

- 전체 F2 868건 중34건에서 coronal-fluid 슬롯12장을 변경,834건 원본 유지. 패치46,061,568바이트. 최대 중심 간 물리길이를 선택했으며 해부학적 품질 우월을 뜻하지 않는다.
- 129시리즈·3,305헤더 검사4.33초,parser에 반환한5,084,039바이트. 디스크/ZIP 실제 물리 I/O와 구분한다. 원본대조·패치까지전체21.68초였으며, 준비ETA에는 코드 구현·검토 시간이 포함됐다.
- 원래 선택의63슬롯을 다시 생성해 모두byteexact. 변경34건의다른4슬롯·mask가 같고,미변경834건은원본readonly메모리를재사용한다. 소스23개·결과11개·가중치·UID·기하/선택지도는root와독립리뷰에서대조했다. 합성검사78통과.
- 34건 중1건은변경슬롯mask와K24중심목록도변했다. 다른슬롯의픽셀이같아도창배분은달라질수있으며사전계약에명시된결과다.

19:28 root가사용자의기존소규모로컬실험허가안에서GPU대조진행을결정했다. 과거CPU등록의GPU금지는그준비단계범위로보존하고,별도GPU실행명세에서변경허가와소스/hash를동결한다. 아직GPU추론은실행하지않았다. 동일868건·동일checkpoint·batch8·strictFP16으로두군을새로추론한후고정평가한다. 원학습OOF는batch4여서동등한대조로재사용하지않는다. 구현·CPU검토목표19:50~20:00,완료후root가명령과계약을확인하고gpu_job으로1회실행한다.

양군이완료되고입력/출력계약이통과하기전에는정답점수를읽지않는다. 12소견/P·N/골절/전체macro·조건부CI와사전변경군을검토하되,ratio·소견routing·새학습은자동추가하지않는다. 4070/T4일치는별도실측문제이며CPU준비PASS또는같은FP16설정만으로보장하지않는다.

[CPU 인계](../../artifacts/logs/EXP081_CORONAL_SPAN_FIXED_F2/HANDOFF.md) · [등록](../../artifacts/experiments/EXP081_CORONAL_SPAN_FIXED_F2/registration.json) · [실제 결과](../../artifacts/experiments/EXP081_CORONAL_SPAN_FIXED_F2/receipt.json) · [진행 상태](../handoffs/evidence/2026-10/2026-10-03-continuous-research/exp081-status.json)
