# 2026-10-03 14:18 KST · 지속 연구 재개·D01 원격 준비

- 상태: 진행,16:38갱신. Codex. 제출/공개매회승인·유료별도승인. 이번제출0회.
- EXP078 네fit16:17:20완료. [고정판정](../../artifacts/predictions/EXP078_INPUT_AB_report.json)·[소견/그룹](../../artifacts/predictions/EXP078_INPUT_AB_diagnostics.json)·[해석/결정](../research/exp078-input-ab-result-2026-10-03.md). B−A F0+.003772/F1+.002632,평균+.003202·수치gate통과. CI[−.000480,+.007110],F1골절P30/N839·−.0280하락유지. B는CPU후속임시기준,전체확대/제출적용보류. 학습4소스동결유지.
- [EXP079](../research/exp079-r34-complement-2026-10-03.md):고정90/10 CPU평균+.001593·PFOA+.003622/다른11+.001408. 두fold양수라별도명세strict RN34 고정20epoch2fit16:35시작,B재사용. 러너29016/학습26476,F0실행/F1대기. 첫epoch40초·skip0·초기/코드/UID일치,전체ETA17:02~17:10. [ID/로그/명세](evidence/2026-10/2026-10-03-continuous-research/exp079-status.json). 완료후새프로토콜용평가기필요,과거RN screen오용금지. 비율/routing/Gold선택없음. MCL2fit보류는전체라벨전략기각아님.
- D01고정90/10·원본30셀·v1입력FAIL보존. [실시간ID/경로/결과](evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json). 수치계약CPU107/codec3fixture PASS. R2 T4100건604초,anchor50 1,082초. 1,300건합산9.785h·고정비일부보정9.555h;시간gate미해결. 원본anchor100 비공개2T4 v1실행중·최대1h/2GPUh·ETA16:35~16:45. 모델로딩과DICOM준비가겹쳐로드시간합산차감금지.
- [Header v2](evidence/2026-10/2026-10-03-continuous-research/dicom-header-remote-v2-handoff.md)16:08완료·100건volume/mask/현재receipt/과거geometry/실제순서/spacing PASS. 205.89→199.96초지만두번째실행이100건모두빨라캐시혼재·속도불확실,미채택. 72%논리I/O감소≠시간단축. v1FAIL유지.
- releasev2구현완료·최종helperT4/시간gate전후보생성없음. remote출력은tools/kaggle_output_file.py로정확version/file만회수,CLI전체output금지. kernelCOMPLETE만으로PASS아님.
- 같은채팅exp078-a-b 다음16:45예약확인,미완료면갱신ETA. Biohub:골절손실/작은불확실효과보존,입력변형반복대신표현레시피검토;도구PASS를점수근거로쓰지않음. [축약전](archive/snapshots/2026-10-03-continuous-research/checkpoint-1630/manifest.json). INDEX36건유지.
