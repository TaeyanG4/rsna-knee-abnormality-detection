# EXP083 · 시리즈별 attention bias 대조 준비

2026-10-03 21:50 KST. EXP082의 작은 불확실한 효과는 보존하고, 다음 하나의 개발 가설로 **시리즈 종류를 알려주는 최소 attention bias**를 선택했다. 현재 CPU 구현/검토 중이며 GPU 추출·학습은 시작하지 않았다.

기존 EXP080 B F2의 encoder를 고정하고 train3475/eval868의 동일한 K24 특징을 한 번 추출한다. 두 head는 같은 B8 head weights에서 출발해 둘 다 재학습한다. treatment만5개slot×12소견의60개 bias를 attention logits에 더한다. zero 초기값,empty bias0,clip BCE,20epoch/batch64/seed42/strictFP16은 고정하며 창 선택·라벨·backbone·혼합비율을 바꾸지 않는다. train특징도eval변환으로고정하므로원B의K12+jitter학습과같다고부르지않는다.

현재head가영상에서시리즈종류를이미암묵적으로구분할수있으므로명시ID부재를성능병목으로단정하지않는다. 영상손실이나골절정보를복원하는실험도아니다. 상대개입을분리하는최소대조다. 추가60parameter의용량효과와slot정보를완전히분리하지는못한다.

주 비교는재학습한두head의macro12차이이며원B대비변화는보조다.12소견P/N·골절·장비/결측과고정component bootstrap500회를확인한다. noGold·중간epoch/F2선택·비율튜닝없음. 결과가양수여도노출weakF2/한seed/미확인환자중복/조건부CI한계는남으며D01개선근거로전용하지않는다.

새group분할에서RN두fit만반복하면EXP082의혼합주비교를재검증하지못한다. 공통B까지새학습하면66~82GPU분추정이며weak라벨/과거노출은그대로다. 따라서지금은표현가설의새정보를우선하고,group확인은유망후보확대전단계로보존한다. MCL감독가설도새수정근거가확보되면재검토한다.

CPU 준비목표22:30,상한22:45. 첫예약22:10에실제진척을확인한다. GPU예산상한45분은계획이며추출/두head실측은아직없다. root가CPU계약을검토한뒤기존허가의로컬소규모대조실행을판정한다. 원4학습source/기존cache/모델은불변이고추가제출·원격·유료는없다.

[선택·고정조건](../handoffs/evidence/2026-10/2026-10-03-exp083-preparation/root-decision.json) · [비용/가설 비교](../handoffs/evidence/2026-10/2026-10-03-validation-gap-investment/v7-next-action/report.md) · [EXP082 결과](exp082-loss-screen-2026-10-03.md).
