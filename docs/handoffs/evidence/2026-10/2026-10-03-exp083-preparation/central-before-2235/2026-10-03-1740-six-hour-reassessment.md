# 2026-10-03 17:40 KST · 6시간 방향·검증 재점검

- 상태:진행,22:20갱신. D01 승인제출1회·공개/유료0. 개발대조→계보분리 재학습→제한LB;일5회는상한·Public최고선택편향잔존.
- 사용자Gold노출 질문:토론5개재수집·노트북3개코드·과거RSNA확인. anchor는Gold epoch선택확인/직접학습미확정;옛R2 teacher누출과mean5의train in-sample별도. **정확D01 독립로컬임상검증 미확보.** Gold제외·새판독·재분할은과거노출을지우지않음. [근거/검증계획](../research/gold-exposure-validation-plan-2026-10-03.md).
- 다음확대는group V를고정하고baseline/variant/encoder/teacher/stacker에서모두제외해재학습. 새recipe검증이지정확D01검증아님. group4343UID/4196components·crossfold중복edge1·환자독립미확정,새split없음. weak/연구자노출한계보존.
- EXP082 21:27완료691초/skip0·21:29평가1회:혼합+.000534467 CI[−.000268,+.001293],RN+.006828 CI0포함. 작은개발후보보존·미채택/추가fold없음. [결과](../research/exp082-loss-screen-2026-10-03.md). EXP080 골절손실반복·B임시,EXP081 max-span미채택/coverage전체기각아님.
- EXP083 CPU11검사/23binding확인→22:18 GPU16건pilot **6.984초native FP16 gate FAILED**,첫batch·학습/평가0. wrapper22480종료·artifacts/logs/EXP083_SLOT_BIAS_PILOT. 원manifest/source/cache/실패보존. [준비·판정](evidence/2026-10/2026-10-03-exp083-preparation/). native float32무손실cache 별도v2 CPU준비dicom_header_io(목표22:30/상한22:34),독립검토precision_code_review. root재검토후pilot1회결정,full미허가. anchor_inputs평가기준비22:32/상한22:42. 실제dtype/parity미검증·자동재시도없음·총2700초에실패시간포함.
- D01 **56796964**,20:46:11승인제출·20:46:32 PENDING/점수없음·4회잔여는과거관측. candidate-v3실제v1·재제출금지. public3/통합133PASS이나1300보수8.594h/9h·memory29.541/30GiB·full보장없음. 원hashFAIL/경로NO_GO보존. 첫채점조회10-04 **05:02**,조기반복조회금지. [상태](evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json).
- Biohub:soft BCE/pos_weight≠Silver교정;unknown/confidence loss미적용. 작은효과·반대증거보존·MCL보류로감독전체폐기금지. [감사](../research/gold-silver-status-2026-10-03.md). GPU실행PASS≠개선. 긴작업만남으면동일heartbeat를다음ETA에맞추고반환. 제출/공개/유료별도승인.
