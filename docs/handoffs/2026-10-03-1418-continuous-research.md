# 2026-10-03 14:18 KST · 지속 연구 재개

- 상태: 진행,17:27갱신. Codex. 제출/공개매회승인·유료별도승인·이번제출0회.
- [EXP078](../research/exp078-input-ab-result-2026-10-03.md)4fit완료·B임시기준. [EXP079](../research/exp079-r34-complement-2026-10-03.md)2fit17:00완료·총25분/skip0. 고정90/10 B대비mean+.0021175·CI[+.001462,+.002731],12소견평균양수. A대비F1골절−.022666·F1Philips3T−.000496보존. 노출weak 개발후보,채택/제출아님. 전용평가기75검사·실계약/평가PASS.
- [EXP080](../research/exp080-f2-confirmation-2026-10-03.md) F2 A/B8epoch·RN20epoch3fit17:14시작. train3475/val868·공통제외2건·seed42·학습4소스동결. A실행/나머지순차대기,runner35608/train19980. 2/8epoch323/296초·skip1·7.8GiB·초기/코드/UID일치. ETA18:50~19:05. [ID/로그](evidence/2026-10/2026-10-03-continuous-research/exp080-status.json). F2전용평가기구현중,3fit후원판정·골절/그룹/CI. report/series교차중복0이나환자독립미검증.
- D01[상태/정확경로](evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json):v1입력FAIL·CPU107/codec3 PASS유지. anchor1002031초·원30셀/49+·fallback0. R2분할100비공개2T4 v1은17:10:56완료·349초·input100/5모델확률exact. 단순1300공동8.594h(+5분8.677h/+30분9.094h)는산술이며full보장아님. 최종helper v3는준비중,별도T4/공동시간/메모리전후보생성없음. header v2입력PASS나cache/order혼재·미채택.
- [시리즈목록](evidence/2026-10/2026-10-03-series-inventory/v2/report.json):4405건선택replay일치·다중후보1985/axial대안450,예측가린30건고정. 오선택증거아님,다음coverage/품질감사후가설선택. MCL2fit보류는라벨전략전체기각아님.
- Biohub:작은RN효과보존·입력골절분리·F2/seed/series/감독비용비교. 토론성별prior는로컬라벨에서재현안돼미적용. 원격다운로더SDK오류비노출21검사PASS. [축약전](archive/snapshots/2026-10-03-continuous-research/checkpoint-1709/manifest.json). 같은채팅exp078-a-b 17:45재개확인. [진행중하위작업](evidence/2026-10/2026-10-03-continuous-research/active-worker-checkpoint.json)을먼저회수,중복구현금지. 긴작업만남으면반환.
