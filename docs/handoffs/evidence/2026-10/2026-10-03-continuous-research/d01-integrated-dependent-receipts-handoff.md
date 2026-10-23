# 2026-10-03 20:04 KST · D01 v2 dependent 영수증 검증

- 상태: 완료. 기록자: Codex(dicom_header_io). root의 정확한 kernel version2 하위 영수증 검토 요청 수행. 후보 제작·GPU·추가 원격 실행 없음.
- 결과: `taeyangg4/knee-d01-integrated133` v2의 요청 영수증7개 회수. 전부 exact-version 다운로드 SHA 일치, 6개 final result byte SHA 연결, 3phase JSON 값 일치. public3 phase는 final 내 개별 byte SHA가 없어 exact 다운로드와 embedded JSON으로만 연결한다.
- 직접 검증: original100 전체 receipt/정규화 tensor hash·extra33 volume/mask hash가 frozen 기준과 exact. 두 shard67/66건·각5모델·133개 중복/누락0·fallback/neutral0. 확률100 gate는 hash로 연결된 result 보고값이며 이 검토에서 NPZ를 별도 재계산하지 않았다. extra33 독립 GPU 확률 기준 없음.
- 판정: 탐색 제출 후보 준비 가능. 정확도 개선·hidden full1300 PASS 아님. core2399.726초, 보수적1300예상8.594h/9h여유24.36분; 최소5분 통과·30분/30% 여유 실패. anchor kernel cgroup peak29.541/30GiB(여유469.7MiB), OOM0. sample 성공이나 full 메모리 안전성 미확정. page cache를 전부 회수 가능하다고 단정하지 않는다.
- 근거: [검증 JSON](d01-integrated-dependent-receipts-review.json) SHA `fadc0dee86c89da2bdf4c3d305e63e0f2ec30c49c1b42a4c20e1b3a69f1350a4`; [재현 검사](d01-integrated-dependent-receipts-review.py), `.venv/Scripts/python.exe -X utf8 <검사>` PASS. 첫 기록 시 tzdata 부재를 UTC+09:00 표현으로 수정 후 완료.
- 남은 일: root가 최소 production adapter 제작·검사. trial train proxy/root mask 제외. `production-core` RUNNING은 public3 전 snapshot이며 final PASS/hash로 완료 확인. 기존 v1경로FAIL·입력hashFAIL 보존. 제출 승인은 별도이며 STATE/INDEX/main handoff는 root가 갱신한다.
