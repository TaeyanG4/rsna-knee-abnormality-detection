# EXP084 독립 결과 해석 · 2026-10-04 02:26 KST

담당 `/root/anchor_inputs`. evaluation-v1 읽기만; 재평가·학습·GPU0. [report](../2026-10-04-exp084-report-tool/evaluation-v1/report.json) SHA256 `0051229e543564b8cebaec27277c300954c9f1e0ba3b376d933460c3686f1f10`.

**권고: ones 전환 미채택, clip 유지.** loss 동등성 증명은 아니다. 추가 fold/seed·ratio/소견별 선택을 자동 확대하지 않는다. EXP082의 이전 작은 양의 결과(+.000534, 구간[−.000268,+.001293])도 반대근거로 보존하며 재검증하지 않았다.

|고정 비교|macro12 차이|조건부 구간|
|---|---:|---|
|blend ones−clip|−.00009579|[−.00100886,+.00075570]|
|RN ones−clip|−.00390390|[−.01164946,+.00399352]|
|blend clip−B|+.00099742|[+.00010133,+.00186504]|
|blend ones−B|+.00090163|[−.00014915,+.00183245]|

RN 작은 보완효과는 보존한다. B .89339653→clip blend .89439394이나 이는 secondary·조건부 내부결과다. ones의 MCL/Medial OA/Effusion/Baker 손실을 평균 뒤에 숨기지 않는다.

|소견|P/N|blend ones−clip|
|---|---|---:|
|ACL|122/766|+.001771|
|MCL|45/843|−.001147|
|Medial Meniscus|358/530|+.000229|
|Lateral Meniscus|118/770|+.001810|
|Medial OA|195/693|−.001580|
|Lateral OA|127/761|−.000548|
|PF OA|298/590|−.000407|
|Effusion|225/663|−.000858|
|Synovitis|217/671|−.000464|
|Baker's|110/778|−.000900|
|Contusion|136/752|+.000020|
|Fracture|29/859|+.000923|

- clip blend−B: PFOA +.004340, 골절 −.003111(구간[−.008792,+.001175]). ones 골절손실도 −.002188.29양성의 불확실성을 보존하며 input-A가 없어 과거 B−A 골절문제 해소로 해석 불가.
- 그룹은 탐색용22개·CI 없음. ones−clip은1.5T535건 −.000455,3T315건 +.000091; zero-mask234건 +.000224/all-nonzero654건 −.000238로 방향혼재. Canon42건 골절양성1, Fujifilm1건 등 작은/불완전12소견 그룹으로 routing하지 않는다. zero-mask≠임상음성.
- bootstrap873components·500/500유효·동일draw/12소견. ranks 재계산 없이 고정scores를 재표집한 조건부 구간이며 seed·라벨·선택·private분포 불확실성은 제외한다. 소견/그룹 다중비교 보정 근거 없음.
- RN20epoch 각8620updates/skip0(726/724초), B8epoch3448/skip0(2610초), fixed-final·Gold차단·평가유예. 유효update 불균형은 이번 RN차이 설명이 아니다.
- protocol/initial-assets/ancestor-exclusions hash 연결은 보존돼 있으나 본 리뷰는 원계보 재인증이 아니다. weak라벨·기존연구자노출·Dread계보·미확인환자/근접중복 한계가 남는다. 새 임상 holdout/정확D01/0.943/private 개선 주장 금지. 다음 투자는 이 loss 비교 반복보다 사전 고정한 다른 신호 가설을 root가 선택한다.
