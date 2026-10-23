# 비교 검증 정정·EXP087 짧은 진단

2026-10-04 11:38 KST. root 중앙기록. M02/04/07/08/10 적용.

- 사용자 질문으로 원manifest·완료평가/anchor분석 대조. **anchor_included=false**:0.913은0.943소폭수정본아닌별도B/RN6weights. 내부EXP080 B .885460→.886703, EXP084새B .893397→.894394는서로다른weak계약이며후자weights는제출에없음.
- 작은RN효과가공개에서반전됐다는설명은미입증. B-only3fold의같은public성적없음. 정확제출ensemble의독립성능·0.943대비경쟁력을연결하지못한검증/판단공백을인정. metric버그새증거없고완료검사재실행0. [근거·후속규칙](evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-validation-comparator-reconciliation.json).
- weak라벨/노출개발셋/한seed·조건부CI/환자·ancestor공백유지. 모든fold예측을train에평균해독립OOF라부르지않는다. 다음full투자전동일대조/홀드아웃/라벨/production계산을명시. exactanchor독립검증이없으면전이탐색이라고표시·성능개선확정금지.0.943유지/비율튜닝없음.
- EXP087진단 R1경합수정새5PASS·기존11재사용. anchor_inputs11:33독립deltaPASS·원HOLD/pre-r1보존. root가새protocol fcec7360…/plan64cdb6b0…의1회실행수락.
- **EXP087_NUMERICAL_DIAGNOSTIC_V1** 11:37:48실행·wrapper40856. [영수증](evidence/2026-10/2026-10-04-exp087-numerical-diagnostic-v1/root-diagnostic-launch.json). artifacts/logs/同tag/run.log와experiments/同tag/{resource-supervisor,current-attempt,numerical-failure,diagnostic-receipt,worker-cleanup}.json. max200attempt/240초/retry0/RSS24GiB/12worker/internal1/metadata16MiB·원recipe유지. full/평가승인아님.
- 정상후속 exp078-a-b **11:42**에1회완료회수. 실패그대로보존·telemetry/child/cleanup/잔존확인. prefix200은full안정성아님. rsna-2는5시간토큰중단복구전용·정상연구중복없음. B/RN채점조회종료.

11:41회수: 진단11:39:32종료/104.313초·child1/FAIL유지. attempt74(완료73)에서input/logits/loss유한,scale1024/found_inf1,stem.conv1.weight gradient136개(NaN54/+inf27/−inf55). 원소비유한이므로단순norm합산overflow설명배제·발생연산/낮은scale해결여부미확정.12worker정상종료/13rootcheck/잔존0. [실결과](evidence/2026-10/2026-10-04-exp087-numerical-diagnostic-v1/root-diagnostic-result-review.json). 자동retry/full없음·다음은검증계약/투자재판정.11:42예약갱신도구2회Transport closed로실패·확정아님.
