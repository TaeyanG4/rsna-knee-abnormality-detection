# 2026-10-03 14:18 KST · 지속 연구 재개·D01 원격 준비

- 상태: 진행,16:03갱신. Codex. 제출/공개매회승인·유료별도승인. 이번제출0회.
- EXP078 A_F0/B_F0/A_F1완료,B_F1실행. 러너20764,[명세](../../artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/manifest.json)·[실측](evidence/2026-10/2026-10-03-continuous-research/progress-checkpoint.json). 전체ETA16:18~16:25. 학습4파일동결,네fit 후input_ab_report→input_ab_diagnostics와Biohub점검. F0만으로판정없음.
- D01고정90/10·원본30셀보존. [ID·실행·결과](evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json). v1입력hash FAIL유지. 별도수치계약 로컬/KaggleCPU107 PASS·codec3압축fixture PASS,최종T4 전체스택미완료.
- R2 T4 100건604초,anchor50 1,082초. 단순1,300건합산9.785h;검증된고정비일부제외도9.555h,시간gate보류. 원본anchor100 비공개2T4 v1실행확인16:03,최대1h/보수2GPUh·ETA16:35~16:45. 50/100두점외삽은full시간/메모리보증아님. releasev2구현완료지만최종T4·예산통과전후보생성없음.
- Header v1첫건FAIL보존. 좌표내적차2.84e−14mm로5slot해시차재현;픽셀/실제순서동일. [별도v2계약](evidence/2026-10/2026-10-03-continuous-research/dicom-header-order-v2-handoff.md)은원geometry·전체순서/spacing·나머지receipt exact유지. CPU20검사/100reference/1통합PASS. 비공개CPU v2를15:59시작,ETA16:10~16:20;원격결과전채택없음. 로컬100건속도2.47%느림도보존.
- [MCL120대조](evidence/2026-10/2026-10-03-mcl-review/unblind-v1/report.md):음성후보55모두Dread이진음성,unknown56. 새수정근거부족으로2fit보류,라벨연구전체기각안함. [PFOA상보성](../research/pfoa-complementarity-proposal-2026-10-03.md)검토완료:PF+.012/전체−.0118,레시피혼재. 네fit판정후고정90/10 CPU선별→유망시새RN2fit검토. [5가설/비용](../research/execution-priorities-2026-10-03.md).
- SDK정확version/file회수용 tools/kaggle_output_file.py 추가·실제v1실패receipt SHA일치검사PASS. CLI전체output의dependency/429경로사용금지. 연구실API·보고서갱신.
- 같은채팅 exp078-a-b 다음16:20확인,미완료면갱신ETA. [축약전보존](archive/snapshots/2026-10-03-continuous-research/checkpoint-1604/manifest.json). INDEX36건유지. Biohub:도구검증을성능이득으로보지않고4fit후큰가설선택;작은효과/특정실패를전략전체기각으로확대하지않음.
