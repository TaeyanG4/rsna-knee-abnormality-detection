# D01 고정 가설 — 승인 제출과 검증 범위

2026-10-03 20:55 KST. 사용자 승인으로 `taeyangg4/knee-d01-candidate-v3` 실제 version1을 **20:46:11에 1회 제출**했다. 제출ID **56796964**,20:46:32 관측 PENDING·점수 없음, CLI가 보고한 당일 잔여4회. 추가 제출·유료 지출·노트북 공개는 없다. **기존0.943보다 정확도가 높다는 독립 근거는 아직 없다.**

가설은 기존0.943 anchor의 소견별 순위90%에 기존R2 다섯 모델 평균확률의 순위10%를 더해 오류를 보완할 수 있는지 확인하는 것이다. cohort 전체 average-tie rank/n, 최종 재순위화 없음. EXP079/080의 새B+RN과 다른 모델이므로 그 개선량을 D01 예상효과로 사용하지 않는다. 기존R2의 teacher 노출·혼합라벨/epoch선택 한계는 유지된다.

| 확인한 내용 | 실제 근거 | 뜻하지 않는 것 |
|---|---|---|
| 최종후보 실행 | 공개3UID·12소견,401.603초,출력9종/hash 검증·OOM/잔존자손0 | 숨겨진 전체 성공·정확도 개선 |
| 입력과GPU차이 | public3 volume/mask exact;F0 로컬/T4 max차이0.00015955,mean0.00004492로고정gate통과 | 모든GPU/모델/hidden분포의완전일치 |
| 혼합계산 | 5모델평균exact,고정90/10순위혼합재계산오차1.11e−16 | 조합이baseline보다우수함 |
| 통합133 | 생산2399.726초,원100입력/정규화/5모델확률exact,추가33volume/maskexact,DINOmemmap실경로 | 추가33별도정규화/확률기준·full1300보장 |
| 자원추정 | 보수1300건8.594h/9h,여유24.36분·최소5분통과 | 30분·30%여유목표는실패;queue/최대건수불명 |
| 메모리 | 공개3 peak21.96GiB,통합133 peak29.541/30GiB | Rad669/892캐시전환과full메모리는미측정 |

원래anchor30셀,기록된자산/helper,mean-before-rank계약을유지했다. 모든서버자산의resolvedrevision은확정하지못했다. 이전D01v1입력hashFAIL과통합v1경로NO_GO는보존하며,새버전검증으로소급PASS로바꾸지않는다. 20:40에는하위영수증검토중이어서HOLD를기록했고,20:42기존출력검토완료뒤승인받아제출했다. 시한뒤추가추론/benchmark는없다.

이번제출은성능이입증된개선안의배포가아닌고정가설의전이시험이다. 기준0.943과상승/동점/하락을모두기록하며public점수로비율·소견을반복튜닝하지않는다. public상승도private향상의증명이아니고하락도학습형모델전체기각근거가아니다. 기존기준본을보존한다.

검증독립성추가조사에서fold0↔1의서로다른2UID가과거/현재캐시와마스크전체에서정확히같았다. 모델입력전체와원본7쌍의PixelData/저장배열이정확히같음을확인했다. 환자/원본전체동일성은아직미확정이다. 해당쌍은F2에서는둘다학습쪽이지만다른중복이나weak/노출F2문제는배제되지않는다. 이결과를기존OOF의독립성보증으로쓰지않는다.

채점대신기다리는반복문은없다. 제출20:46+보수8.594h로잠정ETA10-04 05:22,첫확인05:02 KST를계획했다. 실제hidden시작·queue는미확인이라완료약속이아니다. 독립CPU작업은이보다앞서완료확인후연구를이어간다.

[전체상태·실패보존](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json) · [9개출력검사](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-candidate-actual-v1-audit.json) · [입력중복조사](../handoffs/evidence/2026-10/2026-10-03-validation-gap-investment/v4/report.md).

원격노트북SHA `75ae6a56d5e7bb0993e6d026648169240b1f41f918ebb1b261d92f0e08ed9e2e`; 공개3 CSV SHA `180848bb37be7c7b7be01a8c2e00c46ef68b1974ff9b875c2e6df6460121786f`. 승인·제출명령·실제응답은workspace://.local/d01-candidate-v3/20261003-v2/competition-submission-receipt.json와submission-approval-request.json에보존했다.
