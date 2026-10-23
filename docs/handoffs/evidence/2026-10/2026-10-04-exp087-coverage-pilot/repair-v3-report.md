# EXP087 실행 전 최소 보완 v3

2026-10-04 07:36 KST. root의 `root-pre-pilot-repair-decision.json` 두 항목만 보완했다. 실제 데이터 픽셀/GPU/model/label 접근과 pilot 실행은 0이며 독립 재검토·root 실행 결정을 기다린다.

- 수정 전 source/tests의 **정확한 bytes**를 `pre-repair-v2/`에 보존하고 이전 SHA와 일치 확인했다.
- child가 마지막 poll 사이에 정상 종료해도 **최종 elapsed≥600초면 FAIL**이다. 최종 시각은 cleanup 후 판정한다.
- `Popen` 이후 job.json 저장·모니터·종료 등 예외를 `BaseException` 경계로 수집하고 `finally`에서 재조회한 child/descendant와 이전에 추적한 process를 정리한다. direct Popen kill/wait도 별도로 시도하며 cleanup 오류/생존 PID는 FAIL receipt에 기록한다.
- supervisor.json 쓰기 자체가 실패하면 cleanup을 이미 완료한 뒤 구조화된 FAIL을 stderr에 남기고 예외 종료한다. 파일시스템이 기록을 거부하는 상황에서 파일 생성 성공을 주장하지 않는다.
- 추가 반례: exit0/elapsed600, exit0/600.001의 FAIL, 정상599.9 PASS, job.json 쓰기 예외와 모니터 예외 각각에서 direct child+descendant 종료 및 FAIL receipt 확인. fake Popen/psutil로 검사했으며 실제 subprocess·데이터는 실행하지 않았다.
- **합성 테스트25개 통과, 1.66초**. `focused-tests-repair-v3.xml` 보존.
- `freeze-v3`는 v2의 동일8쌍 UID/series/header를 그대로 복사하고 source/tests 및 root repair 결정만 새로 결박했다. metadata 재열거·표본 교체 없음. 원192planes/K12/K24/예산/출력 경로 불변. v1/v2는 미실행 이력으로 보존한다.

정확한 source/test/fixture/plan/launch SHA는 `repair-v3-bindings.json`; 실행 명령은 `freeze-v3/launch.json`. 실제 `.local/exp087-coverage/pilot-v1`은 아직 없다. **v2 실행 금지, v3도 root 결정 전 실행 금지.**

RSS는 기존100ms 샘플 감시이며 OS hard memory quota는 아니다. 알려진/재조회된 process 정리 실패는 성공으로 숨기지 않는다. 이 수정은 입력 정책이나 학습·성능 판단을 바꾸지 않는다.
