# 2026-10-03 18:18 KST · 통합시험 PASS 후 후보 연결 읽기 검토

- 상태: 검토 완료. 기록자: Codex 하위 작업 dicom_header_io. 실행 소스·helper·후보·원격 상태 변경 없음. wrapper `599e7d08…`, helper `3800aec2…`, 원anchor `cd5b8b69…` 및 stage notebook/protocol SHA가 동결본과 같음.
- 결론: PASS일 때 별도 작은 candidate adapter를 추가하면 된다. 동결 trial의 `run_phase`로 anchor subprocess→자손 종료/정리→v3 R2 subprocess 순서 유지, 원 `blend`를 전체 test UID에 한 번 적용. 원anchor30셀은 그대로 두고 동결 v2의 일반 `anchor_worker_notebook` suffix만 재사용 가능. 기존 v2 build/execute는 R2-first·옛 gate·단일 worker·1300 hard cap이므로 직접 사용하지 않는다.
- 입력·시간: train133 proxy를 실제 test.csv/test_series 경로로만 교체. 49+와 작은 public3의 원래 anchor 분기는 유지한다. 대회약1300은 상한이 아니므로 실제N과 계산N을 기록하고 9h−5min 공통 deadline으로 판단한다. 133건 전용 cap/gate/public3 후행검사를 후보 전체실행에 복사하지 않는다. shard별rank 금지,5raw평균→전체cohort rank90/10→원UID순서→receipt/atomic publish.
- 남은 근거: 현재 dataset16·kernel2 ref는 version 미지정이다. 설치 CLI는 owner/slug/version을 지원한다. 정확 통합시험v1 metadata/input revision을 회수해 실제 시험된 버전에 고정할 수 있는지 확인해야 하며, latest를 과거 mounted version으로 추정하지 않는다. DINO20 독립 fileSHA가 없는 기존 제한도 보존.
- 다음: 정확v1 PASS와 연결 receipt를 SHA로 회수/검증한 뒤 제안파일 `tools/d01_candidate_v3.py`·전용 CPU검사·새stage만 추가(예상20~35분). 새검사범위는 gate binding·원30셀/payload·호출순서/실패미게시·N예산·전체rank/UID뿐. 새GPU리허설·100반복 불필요. [구체 명령/파일/한계](d01-candidate-route-review.json). 실패면 20:40까지 보류 결론, 추가 engineering loop 없음.
