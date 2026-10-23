# EXP087 scale CPU 독립 검토 · 2026-10-04 08:21 KST

- **PASS: root의 1회 bounded stage 실행 판단 준비 완료. 실행·학습 승인은 아니다.** [상세](independent-review.json), [실측7SHA/결박검사](independent-binding-checks.json). 확정 차단결함0.
- 실제driver/adapter/tests/helper/fixture/plan/launch SHA가모두일치한다. 원v3 helper불변·supervisor AST동일. v1/v2의315 map전체동일, T252/V63·재사용8/신규307·G1소속/순서결박·exactargv·예산·retry0을확인했다. 원eligibility census와pilot8 검사는재실행하지않았다.
- source/missing/parse/decode/budget오류는FAIL, geometry/sharedSOP/sampledduplicate/validtriplet없음은원본fallback이다. 다음후보search없다. 원307은header만읽고대안12plane만decode하며원307 pixel replay성공을주장하지않는다.
- crossG1 SOP는pilot8과신규307의선택된원본/대안사이만검출한다. fallback전등록순서는타당하다. 다른study/series·near-duplicate·환자독립을검증한것은아니다.
- adapter는원StudyWindows의x/y/uid를받아slot2지정창만clone후치환한다. 라벨객체·길이·순서·epoch·원C/RNG·다른window를보존하며B8 recipe/corpus/G1을검사한다. 합성15PASS/1.831초를재사용했다.
- Windows검사는Store의cached array/module없는pickle범위다. 실제DataLoader spawn·전체wrapper/원Corpus상태·학습통합은후속검증이다. Store의manifest상태만으로학습하면안되며supervisorPASS+완료315receipt+manifestSHA 인수가필요하다.
- 시간/메모리:900초는부모preflight이후부터자식preflight·처리·저장·cleanup까지다. whole-command 하드제한이아니며RSS8GiB도100ms감시다. 최종초과·정리오류는기존v3대로FAIL.
- 실제header/pixel/cache/ZIP/model/GPU/label/OOF·검사재실행·stage실행0. 각row에양쪽mode일정을기록하므로B8집계는T의train/V의eval만취한다. scheduled창수는실제tensor차이·drop_last후분모·성능이득이아니다. 노출weakG1·private/T4한계와원B8조건부비교계약을유지한다.
- 다음: root가최종정확명령을별도승인하고실결과를인수한다. 현PASS는추가입력staging까지만; full학습등록/GPU/효능판정은포함하지않는다.
