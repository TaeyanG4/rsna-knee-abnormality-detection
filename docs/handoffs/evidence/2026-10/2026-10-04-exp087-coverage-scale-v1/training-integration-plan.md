# EXP087 B8 통합 계획

- 완료·`/root/precision_code_review`·2026-10-04 08:38 KST. 읽기전용 계획이며 학습허가 아님. [상세·정확 계약/소스 위치](training-integration-plan.json). 실행중 stage 산출물은 읽지 않았다. 중앙문서root담당.
- **최소경로는 기존 B8 대조 재사용 + 입력wrapper만 바꾸는 새 후보runner다.** 기존등록/소스·15합성검사·T8결과를 보존한다. 단계완료와 Windows실통합/Pilot을 통과한 뒤root가 학습을 결정한다.

## 보존·연결점

`training/runners/exp084_group_loss.py`117–125행의 train/eval `StudyWindows`를 `CoverageStudyWindows`로 감싼다. 양쪽 모두 같은 accepted store를 사용한다. 원seed/init·loader/order·loss/optimizer/schedule·MIL·최종평가는 유지한다. 옛guard는3arm/tag/schema를고정하므로 그대로 새tag를넣지말고 별도좁은guard/runner를작성한다. 기존실험을수정하거나 학습runner를import하지않는다.

**train bs8/K12/accum1**, T3455·drop_last로431step×8=3448update(매epoch7study제외). **eval bs4/K24**, V888·worker2. seed42·trainworker12/internal1·224/ImageNet/jitter.20·generic초기state `e4ea3a…5b38e`·strictFP16/TF32off·최종8epoch를보존한다. B8학습완료checkpoint에서시작하지않는다. oldcompletion은1%미만skip을허용하므로 새정확update비교는3448실update/skip0을명시한다.

## 학습 전 필요한 새 근거

1. stage종료후 supervisorPASS/exit0/cleanup오류없음·900초/RSS예산,315완료receipt(재사용8/신규307),manifestSHA/정확G1·policy/fixture결박을root가인수한다. Store는supervisor를확인하지않으므로 별도완료인수기록이필수다. 실제유효치환수·비용을검토하고 재census/decode하지않는다.
2. 실제Windows spawn/persistent worker에서 새wrapper를검사한다. 전체T3455dataset은유지하고 bounded sampler로유효대안/미교체사례,epoch0→1공유전달,원본/disabled/force-original의UID·tensor·비교비대상창/RNG보존을확인한다. 생산dataset은부모getitem전 `corpus=None` 상태로spawn한다. 부모비교는별도객체로하며 warm Corpus/memmap을넘기지않는다. 완료후생산loader/generator를새로만든다.
3. 별도허가된T-only32update pilot: 원bs8/K12/FP16·3448step scheduler로finite/skip0·메모리/속도·실제치환경로확인. full은generic초기값에서재시작한다. `train=False`에T ID를넣으면현재wrapper의V888검사에걸리므로편법변경하지않는다. 실제eval경로는V888/zeroY·bs4/K24·no_grad/float sigmoid·순서검사를유지하고중간채점하지않는다.

## 비용·한계·다음

준비30–60분,Windows CPU1–3분,GPU pilot1–3분/full45–60분은추정이다. full근거는완료B8 2610초이며새I/O비용은미측정이다. 입력외recipe가바뀌면먼저차이를제거하고,의도된변경이면그때matched control을판단한다. 자동B재학습/재추론없음. M02/05/09/11적용·노출weakG1/private·T4한계유지. stage인수→최소구현/gate→root단일학습판정 순서이며,이번GPU/모델/라벨/픽셀/검사재실행/등록0.
