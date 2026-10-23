# EXP087 최종 학습 구현 독립 검토 · 2026-10-04 09:14 KST

- **소스·계약 PASS. 실제 실행은 root의 phase/RSS 결정 전 HOLD.** [상세](independent-final-review.json), [실측source6/protocol/plans 결박](independent-final-binding-checks.json). 현재 확정 학습계약 결함0, 수정/실행0.
- actualrunner diff가원EXP084 B8과일치한다. 원인수에는mode만추가, generic초기hash·G1·loss·optimizer·3448scheduler·FP16strict/TF32off·batch8K12/evalbatch4K24가같다. wrapper만입력을치환하며기존B대조재사용조건에부합한다.
- Windowsgate는full3455 dataset·baseline.corpus=None·별도parent비교·8T sampler·12persistent spawn·epoch0→1·disabled/forced/enabled·worker종료를검사한다. 1batch이므로12worker전체동시처리량까지증명하지않으며실행은아직0.
- pilot는원3448horizon으로32updates/skip0,체크포인트·V추론없이종료한다. full은generic부터새로시작,3448/skip0·trainworker정리후888최종추론1회다. deliveredUID/치환계획trace와최소1실제tensor대조가있으며모든tensor를대조한것은아니다.
- phase별정확protocol/이전산출물SHA/root결정이필요하고자동연쇄·재시도없다. 최종elapsed초과·잔존/실패worker·감독예외는FAIL. supervisorPASS만으로산출물완료를판정하지않는다.
- **24GiB는미수용제안이다.** guard는동결protocol의24GiB를사용하고decision의rss_bytes/seconds_cap을override하지않는다. root가24GiB를명시수용하거나,다른한도를원하면새protocol/source결박후실행해야한다. 시간은부모preflight제외·RSS는100ms감시다.
- 합성12PASS/0.081초와완료stage검사는재사용했다. 실제Windows/model/GPU/label/OOF/decode/평가0. GPU실행시에는weak label/evalCSV를읽은뒤V라벨을0으로바꾸므로라벨미접근으로표현하면안된다.
- 비차단metadata: pilot의상속time_limit3600과logs=EXP084_B_G1은실제pilot180초/새tag경로와다르다. 실제guard·최종phase schema를따른다. root가기록해혼동을막는다.
- 다음: root의정확Windows자원/phase결정→실Windows결과인수→별도pilot/full결정. B/RN제출허가는본구현실행허가가아니며,weakG1·private/효능한계는그대로다.
