# 2026-10-04 06:33 KST · EXP086 V2 저장 결과 독립 검토

- 상태: 검토 완료, 저장 결과 정합성 PASS·현재 후보 채택 부적합. 담당 `/root/anchor_inputs`. [검토 JSON](independent-result-review.json) · [원 report](../2026-10-04-exp086-report-tool-v2/evaluation-v1/report.json).06:28:57 시작,10분상한 내 종료. 중앙문서 변경0.
- 성능: candidate **0.8288212437**, B **0.8933965283**, 차이 **−0.0645752846**, 조건부95% CI **[−0.0735314175,−0.0539888691]**.500/500 draw,12소견 점추정 모두 하락. MCL−.144405, Baker−.105597, Fracture−.025290. Contusion·Fracture CI는0포함; 나머지10개는 전구간음수.
- 산술: 소견별AUC차·Δ/12·12개합과macro·targets.csv 일치.22그룹/4분류 각각888건이며 소견별P/N합도전체와일치.15그룹은12소견macro유효,7그룹은한쪽class부재로macro null유지; 그룹CI는모두null.11소견 재정규화 없음.
- 완료: 8epoch·3448update·skip0·fixed final, strict runtime/초기값/위치정책·job/decision/argv·trace 일치. full635.438초, 이전실패+pilot포함750.314초≤5580. 자연정리오류/잔존0. 작은완료metadata hash와결과4파일SHA확인. model/OOF hash는receipt↔plan↔report의저장값만대조했다.
- 실행이력: 최초absolute경로호출은contained()입구에서exit1, scoring전거부. 상대경로교정호출exit0의stdout·SUCCESS·report일치. **실채점1회**와입구거부1회를구분했다. report.md/receipt.json은평가폴더산출물이아니다.
- 자원: 기존3.06MB/3218표본만읽어15프로세스합RSS peak **88.573551GiB** 재계산. 공유/mmap중복집계가능성이있지만실제중복량·고유물리RAM peak는미측정. 시작/끝available RAM으로peak사용량을단정할수없다.
- 제한·다음: raw예측/라벨·모델·source/weight전수재해시·AUC/bootstrap/평가재실행·재모니터링0. 고정된결과의산술/계약확인이며원예측에서점수재계산은아니다. 노출weakG1/seed1/환자독립미확정한계유지. M02/M05/M07–10에따라실패와음성결과보존, 이recipe비채택이며DINO전체기각/자동후속학습근거로확대하지않는다.
