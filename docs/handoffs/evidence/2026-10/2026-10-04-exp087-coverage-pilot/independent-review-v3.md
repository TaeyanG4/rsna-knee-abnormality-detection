# EXP087 v3 독립 검토 · 2026-10-04 07:42 KST

- **PASS — root의 구체 실행 판단 준비 완료. 실행 승인은 아니다.** [상세/실측5SHA](independent-review-v3.json). v2 HOLD·미실행 이력은 보존한다.
- 원07:40 상한에서 검토 미완료를 보고했고, [root의추가180초승인](root-review-cap-decision.json) 안에서 최종binding/기록만 완료했다. 자동연장은하지않았다.
- 두차단해결: cleanup뒤elapsed≥600은FAIL이며exit0도예외없다. Popen이후예외는finally에서자식·알려진후손종료/회수, 정리실패·생존·기록실패는성공으로숨기지않는다. 감독기록실패는cleanup후stderr에구조화FAIL을남긴다.
- 변경범위: 보존v2와실제v3의AST 비교에서`run`만변경. 입력·원C/RNG·K12/K24·selector·renderer·geometry는불변이다. 기존검토를재사용한다.
- 실제5파일SHA가repair-v3-bindings와모두일치했다. v2/v3의동일8pair전체구조가같고canonical digest도일치한다. source/tests→fixture/plan→launch연결·정확argv·환경·예산·출력·retry0을확인했다. 07:41:46 출력경로는없었다.
- 합성25PASS/1.665초XML을재사용했다. 새5경계검사는600·600.001FAIL/599.9PASS/job기록·monitor예외정리를검사한다. 테스트재실행·실제pixel/header/model/GPU/label/OOF·fixture재열거·pilot실행은0.
- 제한:600초는preflight이후supervisor시작부터cleanup까지이며whole-command 하드제한은아니다. 100msRSS감시는OS hardquota가아니다. 실제영상replay/codec/대안창사용/시간은후속실행전미확인이다.
- 다음: root가`freeze-v3/launch.json`1회실행여부를별도결정한다. 기존B8조건부재사용·batch4/K24평가계약과노출weakG1한계유지. fullcache/학습/GPU/성능개선은이PASS의범위가아니다.
