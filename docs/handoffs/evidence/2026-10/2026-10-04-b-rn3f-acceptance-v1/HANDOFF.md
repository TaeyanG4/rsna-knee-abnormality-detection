# B/RN 완료 판정 보완 · 2026-10-04

- root가 독립리뷰의 2개 완료판정 공백을 별도 `tools/b_rn3f_acceptance.py`로 보완했다. 원 parity V1/helper/protocol/producer/stage/실제local100 출력은 불변. 수치 허용치·확률·rank 계산은 바꾸지 않았다.
- 두 host의 producer `run-receipt.json` 성공상태와 worker receipt hash를 먼저 확인한다. worker-failure marker/누락·잘못된 종료기록을 거부한다. V1 numeric/input JSON이 명시적으로 통과하지 않으면 실패영수증을 남기고 exit1이다.
- CPU 12개 + 실제CLI 조기거부1개 =13개 검사 PASS. 실제양host 비교/GPU/원격0. [검증/고정hash](verification.json). [기존local100 재사용](reused-input-receipt.json), 새점수·예측·허용치 탐색 없음.
- 입력명령: 기존 V1 `input` 인자를 새adapter에 전달하고 `--parity-helper tools/b_rn3f_parity.py --acceptance-receipt NEW.json` 추가. 이미통과한local100은재실행하지 않는다. T4 경로는 실제mount관측후 고정.
- 평가명령: 기존 `evaluate` 인자를 새adapter에 전달하고 같은2개flag 추가. 미래 출력은 새경로만. V1 CLI를 단독 실행하고 exit0만으로 PASS 처리하지 않는다.
- 별도로 gpu_job SUCCESS, RSS supervisor의 COMPLETE/child_exit_code0/cleanup을 확인한다. numerical PASS만으로 메모리·전체시간·제출준비/임상성능을 주장하지 않는다. root GPUreference결정은 EXP084 종료·활성작업확인 후이며30분상한/retry0이다.
